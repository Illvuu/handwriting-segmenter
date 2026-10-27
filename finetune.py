"""
Fine-tune the pretrained Inoob/DIS-Handwriting-Remover IS-Net on real (clean, answered-scan) pairs
produced by scripts/make_real_pairs.py, optionally mixed with synthetic data from data/train.

Differences from train.py:
  - Architecture: ISNetDISOriginal (stride-2 stem, same as the pretrained checkpoint, 3x faster)
  - Init: full pretrained backbone + side heads lifted 1 -> 4 channels so epoch 0 == pretrained model
  - Loss: multi-scale CE with ignore_index=255 (uncertain pixels from the pair-diff labelling)
  - Sampling: N random augmented crops per page per epoch (pages are few, crops are many)
  - Validation: full held-out pages at inference scale (long edge 1536, like the Java pipeline),
    IoU per class ignoring 255; best checkpoint by handwriting IoU
  - Export: ONNX with the standard contract (input [N,3,H,W] x/255-0.5, logits [N,4,H,W])

Usage (from the repo root):
    python finetune.py --holdout "*_p3" --epochs 15 --export export/finetuned.onnx
    python finetune.py --data data/real data/train --crops-per-page 40
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.dis_net import ISNetDISOriginal, ISNetONNXExport, load_binary_dis_weights
from train import get_device
from utils.augmentations import TrainAugmentation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IGNORE = 255
NUM_CLASSES = 4
CLASS_WEIGHTS = [0.2, 1.6, 3.0, 5.0]


# ----------------------------------------------------------------------------- data

def find_pairs(data_dirs: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for d in data_dirs:
        for img in sorted(glob.glob(os.path.join(d, "image_*.png"))):
            msk = img.replace("image_", "mask_")
            if os.path.exists(msk):
                pairs.append((img, msk))
    return pairs


def split_holdout(pairs, holdout_glob: str | None):
    """holdout_glob: comma-separated fnmatch patterns on the sample stem (basename minus image_/.png)."""
    if not holdout_glob:
        return pairs, []
    import fnmatch
    patterns = [g.strip() for g in holdout_glob.split(",") if g.strip()]
    val = [p for p in pairs if any(fnmatch.fnmatch(os.path.basename(p[0])[len("image_"):-4], g) for g in patterns)]
    train = [p for p in pairs if p not in val]
    return train, val


class CropDataset(Dataset):
    """Each page is sampled `crops_per_page` times per epoch, with a fresh random crop each time."""

    def __init__(self, pairs, crops_per_page: int, crop_size: int):
        self.pairs = pairs
        self.crops_per_page = crops_per_page
        self.aug = TrainAugmentation(crop_size=crop_size)
        self._cache: dict[int, tuple[Image.Image, Image.Image]] = {}

    def __len__(self):
        return len(self.pairs) * self.crops_per_page

    def _load(self, i):
        if i not in self._cache:
            img, msk = self.pairs[i]
            self._cache[i] = (Image.open(img).convert("RGB"), Image.open(msk).convert("L"))
        return self._cache[i]

    def __getitem__(self, idx):
        image, mask = self._load(idx % len(self.pairs))
        return self.aug(image.copy(), mask.copy())


def load_full_page(img_path: str, msk_path: str, long_edge: int):
    """Page at inference scale: image tensor [1,3,h,w], mask tensor [h,w] (nearest-resized)."""
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(msk_path).convert("L")
    scale = long_edge / max(image.size)
    w, h = (round(image.width * scale), round(image.height * scale))
    # Pad to a multiple of 64 so all 6 pooling levels divide evenly
    w64, h64 = (-(-w // 64) * 64, -(-h // 64) * 64)
    image = image.resize((w, h), Image.BILINEAR)
    mask = mask.resize((w, h), Image.NEAREST)
    canvas = Image.new("RGB", (w64, h64), (255, 255, 255))
    canvas.paste(image, (0, 0))
    mcanvas = Image.new("L", (w64, h64), IGNORE)
    mcanvas.paste(mask, (0, 0))
    x = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0 - 0.5).permute(2, 0, 1)[None]
    y = torch.from_numpy(np.asarray(mcanvas, dtype=np.int64))
    return x, y


# ----------------------------------------------------------------------------- loss / metrics

def multi_scale_ce(outputs, target, weights):
    total = 0.0
    for out in outputs:
        if out.shape[2:] != target.shape[1:]:
            out = F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
        total = total + F.cross_entropy(out, target, weight=weights, ignore_index=IGNORE)
    return total / len(outputs)


def iou_per_class(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    valid = target != IGNORE
    ious = []
    for c in range(NUM_CLASSES):
        p = (pred == c) & valid
        t = (target == c) & valid
        union = (p | t).sum().item()
        ious.append(float("nan") if union == 0 else ((p & t).sum().item() / union))
    return ious


def ink_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Precision / recall of 'ink' (class 2 or 3) against labelled handwriting, ignoring 255."""
    valid = target != IGNORE
    p = ((pred == 2) | (pred == 3)) & valid
    t = (target == 2) & valid
    tp = (p & t).sum().item()
    prec = tp / max(1, p.sum().item())
    rec = tp / max(1, t.sum().item())
    return prec, rec


@torch.no_grad()
def validate(model, val_pairs, device, long_edge: int):
    model.eval()
    all_ious, precs, recs = [], [], []
    for img, msk in val_pairs:
        x, y = load_full_page(img, msk, long_edge)
        logits = model(x.to(device))[0]
        pred = logits.argmax(1)[0].cpu()
        all_ious.append(iou_per_class(pred, y))
        pr, rc = ink_metrics(pred, y)
        precs.append(pr)
        recs.append(rc)
    ious = np.nanmean(np.array(all_ious), axis=0) if all_ious else np.full(NUM_CLASSES, np.nan)
    return ious, float(np.mean(precs)) if precs else float("nan"), float(np.mean(recs)) if recs else float("nan")


# ----------------------------------------------------------------------------- export

def export_onnx(model, path: str):
    import onnx
    import onnxruntime as ort

    model = model.cpu()
    wrapper = ISNetONNXExport(model).eval()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dummy = torch.randn(1, 3, 768, 768)
    torch.onnx.export(wrapper, dummy, path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                                    "logits": {0: "batch", 2: "height", 3: "width"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 512, 640) * 0.3
    with torch.no_grad():
        pt = wrapper(x).numpy()
    rt = sess.run(["logits"], {"input": x.numpy()})[0]
    diff = float(np.abs(pt - rt).max())
    agree = float((pt.argmax(1) == rt.argmax(1)).mean())
    print(f"Exported {path} ({os.path.getsize(path) / 1e6:.1f} MB)  parity max_diff={diff:.2e} argmax_agree={agree:.5f}")
    if diff > 1e-3 or agree < 0.999:
        print("WARNING: parity check failed")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=[os.path.join(BASE_DIR, "data", "real")])
    ap.add_argument("--holdout", default=None, help="comma-separated fnmatch patterns on sample stem, e.g. '*_p3' or 'quizA_*,quizB_*'")
    ap.add_argument("--pretrained", default=os.path.join(BASE_DIR, "pretrained", "inoob_isnet.pth"))
    ap.add_argument("--resume", default=None, help="continue from a finetune checkpoint instead of pretrained")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--crops-per-page", type=int, default=40)
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5, help="backbone learning rate")
    ap.add_argument("--head-lr", type=float, default=3e-4, help="side-head learning rate")
    ap.add_argument("--freeze-epochs", type=int, default=2, help="train heads only for the first N epochs")
    ap.add_argument("--val-long-edge", type=int, default=1536)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "checkpoints", "finetune_best.pth"))
    ap.add_argument("--export", default=None, help="write ONNX of the best checkpoint here")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--class-weights", default=",".join(str(w) for w in CLASS_WEIGHTS),
                    help="CE class weights bg,printed,hw,overlap (default from train.py)")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")

    pairs = find_pairs(args.data)
    train_pairs, val_pairs = split_holdout(pairs, args.holdout)
    if not train_pairs:
        print("no training pairs found", file=sys.stderr)
        return 2
    print(f"train pages: {len(train_pairs)}   val pages: {len(val_pairs)}")
    for p in train_pairs:
        print("  train", os.path.basename(p[0]))
    for p in val_pairs:
        print("  val  ", os.path.basename(p[0]))

    ds = CropDataset(train_pairs, args.crops_per_page, args.crop)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=args.workers > 0)

    model = ISNetDISOriginal(in_ch=3, out_ch=NUM_CLASSES)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"resumed from {args.resume} (epoch {ck.get('epoch')})")
    else:
        sd = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        info = load_binary_dis_weights(model, sd)
        print(f"pretrained: {info['loaded']} backbone tensors, {info['heads']} head tensors lifted 1->{NUM_CLASSES}")
    model = model.to(device)

    head_params = [p for n, p in model.named_parameters() if n.startswith("side")]
    body_params = [p for n, p in model.named_parameters() if not n.startswith("side")]
    optimizer = torch.optim.AdamW([{"params": body_params, "lr": args.lr},
                                   {"params": head_params, "lr": args.head_lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    cw = [float(v) for v in args.class_weights.split(",")]
    print(f"class weights: {cw}")
    weights = torch.tensor(cw, dtype=torch.float32, device=device)

    if val_pairs:
        ious, prec, rec = validate(model, val_pairs, device, args.val_long_edge)
        print(f"epoch  0 (pretrained)      | IoU bg {ious[0]:.3f} printed {ious[1]:.3f} hw {ious[2]:.3f} "
              f"| ink prec {prec:.3f} rec {rec:.3f}")

    best, best_epoch = -1.0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        freeze = epoch <= args.freeze_epochs
        for p in body_params:
            p.requires_grad_(not freeze)
        model.train()
        total, n = 0.0, 0
        for images, masks in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = multi_scale_ce(model(images), masks, weights)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        scheduler.step()
        train_loss = total / max(1, n)

        if val_pairs:
            ious, prec, rec = validate(model, val_pairs, device, args.val_long_edge)
            score = ious[2] if not np.isnan(ious[2]) else -1.0
            msg = (f"| IoU bg {ious[0]:.3f} printed {ious[1]:.3f} hw {ious[2]:.3f} "
                   f"| ink prec {prec:.3f} rec {rec:.3f}")
        else:
            score = -train_loss
            msg = ""
        print(f"epoch {epoch:2d}{' [heads only]' if freeze else '             '} | loss {train_loss:.4f} {msg} "
              f"| {time.time() - t0:.0f}s")

        if score > best:
            best, best_epoch = score, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "score": score,
                        "args": vars(args)}, args.out)
            print(f"   -> saved {args.out}")

    print(f"best score {best:.4f} at epoch {best_epoch}")
    if args.export:
        ck = torch.load(args.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        export_onnx(model, args.export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
