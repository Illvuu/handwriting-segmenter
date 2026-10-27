"""
Composite strokes onto templates to produce training image/mask pairs.

Produces:
  - input image (template + overlaid strokes)
  - 4-class label map: 0=background, 1=printed, 2=handwriting, 3=overlap

Usage:
    python -m scripts.generate_synthetic_overlays
"""

import json
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "data", "templates")
STROKE_DIR = os.path.join(BASE_DIR, "data", "strokes")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "train")

# Color distribution for ink
# Pencil=40%, Blue=28%, Black=22%, Red=10%
COLOR_WEIGHTS = [0.40, 0.28, 0.22, 0.10]
COLOR_CHOICES = ["pencil", "blue", "black", "red"]

# Color tints (R, G, B) for each ink type
COLOR_TINTS = {
    "blue": (30, 50, 180),
    "black": (15, 15, 20),
    "red": (180, 30, 30),
    "pencil": None,  # Special handling
}


def load_templates():
    """Load all template images and their cell JSON sidecars."""
    templates = []
    if not os.path.exists(TEMPLATE_DIR):
        raise FileNotFoundError(f"Template directory not found: {TEMPLATE_DIR}")

    for fname in sorted(os.listdir(TEMPLATE_DIR)):
        if fname.endswith(".png"):
            base = fname.replace(".png", "")
            json_path = os.path.join(TEMPLATE_DIR, base + ".cells.json")
            cells = []
            if os.path.exists(json_path):
                with open(json_path) as f:
                    cells = json.load(f)
            templates.append({
                "image_path": os.path.join(TEMPLATE_DIR, fname),
                "cells": cells,
            })

    return templates


def load_strokes():
    """Load all stroke image paths."""
    if not os.path.exists(STROKE_DIR):
        raise FileNotFoundError(f"Stroke directory not found: {STROKE_DIR}")

    strokes = []
    for fname in sorted(os.listdir(STROKE_DIR)):
        if fname.endswith(".png"):
            strokes.append(os.path.join(STROKE_DIR, fname))
    return strokes


def choose_color():
    """Choose ink color based on distribution weights."""
    r = random.random()
    cumulative = 0
    for color, weight in zip(COLOR_CHOICES, COLOR_WEIGHTS):
        cumulative += weight
        if r <= cumulative:
            return color
    return "black"


def tint_stroke(stroke_img, color_type):
    """
    Apply color tint to a grayscale stroke.
    For pencil: translucent gray + per-pixel graphite noise, alpha floor 0.35.
    Returns RGBA image.
    """
    arr = np.array(stroke_img, dtype=np.float32)

    if arr.shape[2] != 4:
        raise ValueError("Stroke must be RGBA")

    alpha = arr[:, :, 3] / 255.0
    gray = arr[:, :, 0]  # Original grayscale ink value

    if color_type == "pencil":
        # Pencil: translucent gray with graphite noise
        # Per-pixel graphite noise
        noise = np.random.normal(0, 15, gray.shape).astype(np.float32)
        gray_noisy = np.clip(gray + noise, 0, 255)

        # Make it lighter/more translucent (pencil look)
        pencil_gray = np.clip(gray_noisy * 0.7 + 60, 40, 180)

        out = np.zeros_like(arr)
        out[:, :, 0] = pencil_gray
        out[:, :, 1] = pencil_gray
        out[:, :, 2] = pencil_gray

        # Alpha: floor at 0.35 for visible pixels, scale by original alpha
        pencil_alpha = alpha * 255
        # Apply alpha floor for pixels that have some alpha
        mask = alpha > 0.05
        pencil_alpha[mask] = np.maximum(pencil_alpha[mask], 0.35 * 255)
        out[:, :, 3] = pencil_alpha

    else:
        tint = COLOR_TINTS[color_type]
        out = np.zeros_like(arr)

        # Intensity variation
        intensity = random.uniform(0.7, 1.0)
        out[:, :, 0] = tint[0] * intensity
        out[:, :, 1] = tint[1] * intensity
        out[:, :, 2] = tint[2] * intensity
        out[:, :, 3] = alpha * 255

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def place_stroke_in_cell(stroke_img, cell):
    """Scale and position stroke to fit within a cell (0.55-1.15x so some cross borders)."""
    cx, cy, cw, ch = cell

    sw, sh = stroke_img.size
    if sw == 0 or sh == 0:
        return None, None, None

    # Scale to fit cell with factor 0.55-1.15
    scale_factor = random.uniform(0.55, 1.15)
    target_w = max(1, int(cw * scale_factor))
    target_h = max(1, int(ch * scale_factor))

    # Maintain aspect ratio roughly
    aspect = sw / max(sh, 1)
    if aspect > target_w / max(target_h, 1):
        new_w = target_w
        new_h = max(1, int(target_w / aspect))
    else:
        new_h = target_h
        new_w = max(1, int(target_h * aspect))

    stroke_resized = stroke_img.resize((new_w, new_h), Image.BILINEAR)

    # Position: centered in cell with slight random offset
    offset_x = cx + (cw - new_w) // 2 + random.randint(-cw // 8, cw // 8)
    offset_y = cy + (ch - new_h) // 2 + random.randint(-ch // 8, ch // 8)

    return stroke_resized, offset_x, offset_y


def place_stroke_random(stroke_img, page_w, page_h):
    """Place stroke at random position on page with random scale/rotation."""
    sw, sh = stroke_img.size
    if sw == 0 or sh == 0:
        return None, None, None

    # Random scale
    scale = random.uniform(0.5, 2.0)
    new_w = max(1, int(sw * scale))
    new_h = max(1, int(sh * scale))
    stroke_resized = stroke_img.resize((new_w, new_h), Image.BILINEAR)

    # Random rotation
    angle = random.uniform(-15, 15)
    if abs(angle) > 2:
        stroke_resized = stroke_resized.rotate(angle, expand=True, resample=Image.BILINEAR)

    # Random position
    x = random.randint(-new_w // 4, page_w - new_w * 3 // 4)
    y = random.randint(-new_h // 4, page_h - new_h * 3 // 4)

    return stroke_resized, x, y


def generate_sample(template_info, stroke_paths, sample_idx):
    """Generate one training sample: input image + mask."""
    # Load template
    template = Image.open(template_info["image_path"]).convert("RGB")
    page_w, page_h = template.size
    cells = template_info["cells"]

    # Create mask: start with printed pixels from template
    template_arr = np.array(template)
    template_gray = np.mean(template_arr, axis=2)

    # Label 1 = printed (dark pixels from template, luma < 160)
    mask = np.zeros((page_h, page_w), dtype=np.uint8)
    mask[template_gray < 160] = 1

    # Composite image (start with template)
    composite = template.copy().convert("RGBA")

    # Choose number of strokes to overlay
    num_strokes = random.randint(5, 25)

    # ~53% placed inside cells
    num_in_cells = int(num_strokes * 0.53) if cells else 0
    num_random = num_strokes - num_in_cells

    # Handwriting alpha accumulator for mask
    hw_alpha = np.zeros((page_h, page_w), dtype=np.float32)

    def overlay_stroke(stroke_resized, ox, oy):
        """Overlay a stroke onto composite and accumulate alpha for mask."""
        nonlocal composite
        if stroke_resized is None:
            return

        sw, sh = stroke_resized.size
        # Clip to page bounds
        src_x1 = max(0, -ox)
        src_y1 = max(0, -oy)
        src_x2 = min(sw, page_w - ox)
        src_y2 = min(sh, page_h - oy)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return

        dst_x1 = max(0, ox)
        dst_y1 = max(0, oy)

        # Paste onto composite
        region = stroke_resized.crop((src_x1, src_y1, src_x2, src_y2))
        paste_target = Image.new("RGBA", composite.size, (0, 0, 0, 0))
        paste_target.paste(region, (dst_x1, dst_y1))

        # Alpha composite
        composite = Image.alpha_composite(composite, paste_target)

        # Update handwriting alpha mask
        region_arr = np.array(region, dtype=np.float32)
        if region_arr.shape[2] == 4:
            alpha_region = region_arr[:, :, 3]
            rh, rw = alpha_region.shape
            hw_alpha[dst_y1:dst_y1 + rh, dst_x1:dst_x1 + rw] = np.maximum(
                hw_alpha[dst_y1:dst_y1 + rh, dst_x1:dst_x1 + rw],
                alpha_region
            )

    # Place strokes in cells
    for _ in range(num_in_cells):
        stroke_path = random.choice(stroke_paths)
        try:
            stroke = Image.open(stroke_path).convert("RGBA")
        except Exception:
            continue

        color = choose_color()
        stroke = tint_stroke(stroke, color)

        cell = random.choice(cells)
        result = place_stroke_in_cell(stroke, cell)
        overlay_stroke(*result)

    # Place strokes randomly
    for _ in range(num_random):
        stroke_path = random.choice(stroke_paths)
        try:
            stroke = Image.open(stroke_path).convert("RGBA")
        except Exception:
            continue

        color = choose_color()
        stroke = tint_stroke(stroke, color)

        result = place_stroke_random(stroke, page_w, page_h)
        overlay_stroke(*result)

    # Build final mask
    # Class 2 = handwriting (alpha > 80/255)
    hw_mask = hw_alpha > 80

    # Class 3 = overlap (handwriting AND printed)
    overlap = hw_mask & (mask == 1)

    # Apply classes in priority order
    final_mask = mask.copy()
    final_mask[hw_mask] = 2
    final_mask[overlap] = 3

    # Convert composite to RGB for output
    output_img = composite.convert("RGB")

    # Save
    img_path = os.path.join(OUTPUT_DIR, f"image_{sample_idx:05d}.png")
    mask_path = os.path.join(OUTPUT_DIR, f"mask_{sample_idx:05d}.png")

    output_img.save(img_path, "PNG")
    # Save mask as single-channel image (values 0-3)
    mask_img = Image.fromarray(final_mask, mode="L")
    mask_img.save(mask_path, "PNG")

    return img_path, mask_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading templates...")
    templates = load_templates()
    print(f"  Found {len(templates)} templates")

    print("Loading strokes...")
    stroke_paths = load_strokes()
    print(f"  Found {len(stroke_paths)} stroke images")

    if not templates:
        print("ERROR: No templates found. Run make_templates.py first.")
        return
    if not stroke_paths:
        print("ERROR: No strokes found. Run make_strokes.py first.")
        return

    # Generate ~3-5 samples per template for varied overlay patterns
    samples_per_template = random.randint(3, 5)
    total_samples = len(templates) * samples_per_template
    print(f"Generating {total_samples} training samples ({samples_per_template} per template)...")

    sample_idx = 0
    for t_idx, template_info in enumerate(templates):
        for _ in range(samples_per_template):
            try:
                generate_sample(template_info, stroke_paths, sample_idx)
            except Exception as e:
                print(f"  WARNING: Failed sample {sample_idx}: {e}")
            sample_idx += 1

        if (t_idx + 1) % 100 == 0:
            print(f"  Processed {t_idx + 1}/{len(templates)} templates ({sample_idx} samples)")

    print(f"Done. Generated {sample_idx} training samples to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
