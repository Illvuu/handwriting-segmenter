"""
Training script for handwriting segmentation IS-Net.

Features:
- IS-Net with out_ch=4 (background, printed, handwriting, overlap)
- Warm-start from inoob/DIS-Handwriting-Remover checkpoint (shape-matching tensors only)
- AdamW lr=1e-4, cosine annealing
- Multi-scale CE over all 6 side outputs
- Class weights [0.2, 1.6, 3.0, 5.0]
- Best checkpoint by mean IoU of ink classes (2 and 3)
- Supports MPS (Apple Silicon), CUDA, or CPU

Usage:
    python train.py
"""

import os
import sys
import glob
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm

from models.dis_net import ISNetDIS
from utils.augmentations import TrainAugmentation, ValAugmentation

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "train")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Hyperparameters
NUM_CLASSES = 4
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
CLASS_WEIGHTS = [0.2, 1.6, 3.0, 5.0]  # background, printed, handwriting, overlap
CROP_SIZE = 512


class SegmentationDataset(Dataset):
    """Dataset that loads image/mask pairs from data/train/."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Find all image files and their corresponding masks
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "image_*.png")))
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = img_path.replace("image_", "mask_")
            if os.path.exists(mask_path):
                self.mask_paths.append(mask_path)
            else:
                self.mask_paths.append(None)

        # Filter out samples without masks
        valid = [(i, m) for i, m in zip(self.image_paths, self.mask_paths) if m is not None]
        self.image_paths = [v[0] for v in valid]
        self.mask_paths = [v[1] for v in valid]

        print(f"Dataset: {len(self.image_paths)} valid image/mask pairs from {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Default: just convert to tensor
            import torchvision.transforms.functional as TF
            image = TF.to_tensor(image) - 0.5
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask


def get_device():
    """Select best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_pretrained_weights(model, device):
    """
    Warm-start from inoob/DIS-Handwriting-Remover checkpoint.
    Load only shape-matching tensors; ignore mismatched side heads.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. Skipping pretrained weight loading.")
        print("Install with: pip install huggingface_hub")
        return model

    try:
        print("Downloading pretrained checkpoint from inoob/DIS-Handwriting-Remover...")
        checkpoint_path = hf_hub_download(
            repo_id="inoob/DIS-Handwriting-Remover",
            filename="isnet.pth",
        )

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Remap key names: pretrained uses conv_s1/bn_s1, our model uses conv/bn
        remapped = {}
        for k, v in state_dict.items():
            new_key = k.replace('.conv_s1.', '.conv.').replace('.bn_s1.', '.bn.')
            remapped[new_key] = v
        state_dict = remapped

        # Filter to only load matching shapes
        model_dict = model.state_dict()
        pretrained_filtered = {}
        skipped = []

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    pretrained_filtered[k] = v
                else:
                    skipped.append(f"{k}: pretrained {v.shape} vs model {model_dict[k].shape}")
            else:
                skipped.append(f"{k}: not in model")

        model_dict.update(pretrained_filtered)
        model.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_filtered)}/{len(state_dict)} pretrained tensors")
        if skipped:
            print(f"Skipped {len(skipped)} mismatched tensors (side heads with different out_ch)")

    except Exception as e:
        print(f"WARNING: Could not load pretrained weights: {e}")
        print("Training from scratch.")

    return model


def multi_scale_ce_loss(outputs, target, class_weights, device):
    """
    Compute multi-scale cross-entropy loss over all 6 side outputs.
    Each side output is weighted equally.
    """
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    total_loss = 0.0

    for i, output in enumerate(outputs):
        # output shape: [B, C, H, W], target: [B, H, W]
        # Resize output to target size if needed
        if output.shape[2:] != target.shape[1:]:
            output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=False)

        loss = F.cross_entropy(output, target, weight=weights)
        total_loss += loss

    return total_loss / len(outputs)


def compute_iou(pred, target, num_classes):
    """Compute per-class IoU."""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    return ious


def train_one_epoch(model, dataloader, optimizer, class_weights, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)

        loss = multi_scale_ce_loss(outputs, masks, class_weights, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, class_weights, device):
    """Validate and compute per-class IoU."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_ious = []

    for images, masks in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device).long()

        outputs = model(images)
        loss = multi_scale_ce_loss(outputs, masks, class_weights, device)

        total_loss += loss.item()
        num_batches += 1

        # IoU from main output (d1)
        pred = outputs[0].argmax(dim=1)
        for b in range(pred.shape[0]):
            ious = compute_iou(pred[b], masks[b], NUM_CLASSES)
            all_ious.append(ious)

    avg_loss = total_loss / max(num_batches, 1)

    # Average IoU per class
    all_ious = np.array(all_ious)
    mean_ious = np.nanmean(all_ious, axis=0)

    return avg_loss, mean_ious


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Dataset
    print("Loading dataset...")
    train_transform = TrainAugmentation(crop_size=CROP_SIZE)
    val_transform = ValAugmentation(crop_size=CROP_SIZE)

    full_dataset = SegmentationDataset(DATA_DIR, transform=None)

    if len(full_dataset) == 0:
        print("ERROR: No training data found. Run data generation scripts first:")
        print("  python -m scripts.make_templates")
        print("  python -m scripts.make_strokes")
        print("  python -m scripts.generate_synthetic_overlays")
        sys.exit(1)

    # Split 90/10 train/val
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size

    train_dataset = SegmentationDataset(DATA_DIR, transform=train_transform)
    val_dataset = SegmentationDataset(DATA_DIR, transform=val_transform)

    # Use index-based split
    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=use_pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=use_pin_memory
    )

    print(f"Train: {len(train_subset)} samples, Val: {len(val_subset)} samples")

    # Model
    print("Building IS-Net (out_ch=4)...")
    model = ISNetDIS(in_ch=3, out_ch=NUM_CLASSES)
    model = load_pretrained_weights(model, device)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training loop
    best_ink_iou = 0.0
    best_epoch = 0

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Class weights: {CLASS_WEIGHTS}")
    print("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, CLASS_WEIGHTS, device)

        # Validate
        val_loss, val_ious = validate(model, val_loader, CLASS_WEIGHTS, device)

        scheduler.step()

        elapsed = time.time() - start_time

        # Mean IoU of ink classes (class 2=handwriting, class 3=overlap)
        ink_iou = np.nanmean([val_ious[2], val_ious[3]])

        # Logging
        iou_str = " | ".join([f"C{i}:{val_ious[i]:.4f}" for i in range(NUM_CLASSES)])
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"IoU: [{iou_str}] | "
              f"Ink IoU: {ink_iou:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {elapsed:.0f}s")

        # Save best checkpoint (by ink IoU)
        if ink_iou > best_ink_iou:
            best_ink_iou = ink_iou
            best_epoch = epoch
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_ink_iou": best_ink_iou,
                "val_ious": val_ious.tolist(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  -> New best! Saved checkpoint (Ink IoU: {best_ink_iou:.4f})")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            periodic_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:02d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_ious": val_ious.tolist(),
                "val_loss": val_loss,
            }, periodic_path)

    print("-" * 70)
    print(f"Training complete. Best Ink IoU: {best_ink_iou:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    main()
