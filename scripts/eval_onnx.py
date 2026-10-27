"""
Evaluate one or more exported ONNX segmenters on real pairs (image_*.png + mask_*.png) with the
same metrics finetune.py reports, so different models can be compared on the same held-out pages.

Reports per model:
  - argmax IoU per class (bg / printed / hw), ignoring 255   (what finetune.py prints)
  - ink precision / recall under the pipeline decision rule  p_ink = softmax[2] + softmax[3] > threshold
  - the same split by source (_pr printer / _ph phone) and per page with --per-page

Usage (from the repo root):
    python -m scripts.eval_onnx --models export/inoob_isnet.onnx export/finetuned_c.onnx export/finetuned_d.onnx \
        --holdout "quizA_*,quizB_*"
    python -m scripts.eval_onnx --models export/finetuned_d.onnx --holdout "*" --data data/real_legacy   # every pair in a dir
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORE = 255
NUM_CLASSES = 4
CLASS_NAMES = ["bg", "printed", "hw", "overlap"]


def find_pairs(data_dirs):
    pairs = []
    for d in data_dirs:
        for img in sorted(glob.glob(os.path.join(d, "image_*.png"))):
            msk = img.replace("image_", "mask_")
            if os.path.exists(msk):
                pairs.append((img, msk))
    return pairs


def stem(img_path):
    return os.path.basename(img_path)[len("image_"):-4]


def select(pairs, patterns):
    pats = [g.strip() for g in patterns.split(",") if g.strip()]
    return [p for p in pairs if any(fnmatch.fnmatch(stem(p[0]), g) for g in pats)]


def load_full_page(img_path, msk_path, long_edge):
    """Same preprocessing as finetune.validate: resize long edge, pad to multiple of 64, x/255-0.5."""
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(msk_path).convert("L")
    scale = long_edge / max(image.size)
    w, h = (round(image.width * scale), round(image.height * scale))
    w64, h64 = (-(-w // 64) * 64, -(-h // 64) * 64)
    image = image.resize((w, h), Image.BILINEAR)
    mask = mask.resize((w, h), Image.NEAREST)
    canvas = Image.new("RGB", (w64, h64), (255, 255, 255))
    canvas.paste(image, (0, 0))
    mcanvas = Image.new("L", (w64, h64), IGNORE)
    mcanvas.paste(mask, (0, 0))
    x = (np.asarray(canvas, dtype=np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
    y = np.asarray(mcanvas, dtype=np.int64)
    return np.ascontiguousarray(x), y


def softmax(logits):
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def page_metrics(logits, y, threshold):
    """logits [4,h,w] -> dict of IoUs (argmax) and ink prec/rec (argmax and pipeline rule)."""
    valid = y != IGNORE
    pred = logits.argmax(0)
    ious = []
    for c in range(NUM_CLASSES):
        p = (pred == c) & valid
        t = (y == c) & valid
        union = (p | t).sum()
        ious.append(np.nan if union == 0 else (p & t).sum() / union)
    t = (y == 2) & valid

    def pr(p):
        p = p & valid
        tp = (p & t).sum()
        return tp / max(1, p.sum()), tp / max(1, t.sum())

    prec_a, rec_a = pr((pred == 2) | (pred == 3))
    prob = softmax(logits)
    prec_t, rec_t = pr((prob[2] + prob[3]) > threshold)
    return {"iou": ious, "prec_argmax": prec_a, "rec_argmax": rec_a, "prec_thr": prec_t, "rec_thr": rec_t}


def run_model(path, pairs, long_edge, threshold, providers):
    sess = ort.InferenceSession(path, providers=providers)
    name = sess.get_inputs()[0].name
    rows = []
    for img, msk in pairs:
        x, y = load_full_page(img, msk, long_edge)
        logits = sess.run(None, {name: x})[0][0]
        m = page_metrics(logits, y, threshold)
        m["stem"] = stem(img)
        m["source"] = "phone" if m["stem"].endswith("_ph") else ("printer" if m["stem"].endswith("_pr") else "legacy")
        rows.append(m)
        print(f"    {m['stem']}", file=sys.stderr, flush=True)
    return rows


def summarise(rows):
    iou = np.nanmean(np.array([r["iou"] for r in rows]), axis=0)
    return {
        "n": len(rows),
        "iou": iou,
        "prec_argmax": np.mean([r["prec_argmax"] for r in rows]),
        "rec_argmax": np.mean([r["rec_argmax"] for r in rows]),
        "prec_thr": np.mean([r["prec_thr"] for r in rows]),
        "rec_thr": np.mean([r["rec_thr"] for r in rows]),
    }


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def fmt(label, s, thr):
    return (f"{label:<22} n={s['n']:<3} IoU bg {s['iou'][0]:.3f} printed {s['iou'][1]:.3f} hw {s['iou'][2]:.3f} | "
            f"argmax prec {s['prec_argmax']:.3f} rec {s['rec_argmax']:.3f} | "
            f"p_ink>{thr:g} prec {s['prec_thr']:.3f} rec {s['rec_thr']:.3f} F1 {f1(s['prec_thr'], s['rec_thr']):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="ONNX files (4-class contract)")
    ap.add_argument("--data", nargs="+", default=[os.path.join(BASE_DIR, "data", "real")])
    ap.add_argument("--holdout", required=True, help="comma-separated fnmatch patterns on the pair stem; '*' = all")
    ap.add_argument("--long-edge", type=int, default=1536)
    ap.add_argument("--threshold", type=float, default=0.5, help="pipeline ink threshold on p[2]+p[3]")
    ap.add_argument("--per-page", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPUExecutionProvider")
    args = ap.parse_args()

    pairs = select(find_pairs(args.data), args.holdout)
    if not pairs:
        sys.exit("no pairs matched")
    print(f"{len(pairs)} pages: " + ", ".join(stem(p[0]) for p in pairs))
    providers = ["CPUExecutionProvider"] if args.cpu else ort.get_available_providers()

    results = {}
    for m in args.models:
        print(f"\n== {m}", file=sys.stderr)
        results[m] = run_model(m, pairs, args.long_edge, args.threshold, providers)

    print()
    for m, rows in results.items():
        label = os.path.basename(m)
        print(fmt(label, summarise(rows), args.threshold))
        for src in ("printer", "phone", "legacy"):
            sub = [r for r in rows if r["source"] == src]
            if sub:
                print(fmt(f"  {src}", summarise(sub), args.threshold))
        if args.per_page:
            for r in rows:
                print(f"    {r['stem']:<70} hw IoU {r['iou'][2]:.3f} prec {r['prec_thr']:.3f} rec {r['rec_thr']:.3f}")
        print()


if __name__ == "__main__":
    main()
