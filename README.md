# Handwriting Segmenter

Training pipeline for a 4-class handwriting segmentation model (IS-Net architecture) that separates
student handwriting from printed content on scanned quizzes and tests. The exported ONNX model is
used to erase answers so a worksheet can be redone for practice.

Pretrained ONNX weights: https://huggingface.co/guoguo27/handwriting-segmenter

## Quick inference

```python
import numpy as np, onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession("finetuned_e.onnx")
img = Image.open("page.png").convert("RGB")
img.thumbnail((1536, 1536))                                   # long edge 1536 px, as trained
x = (np.asarray(img, np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
logits = sess.run(["logits"], {"input": x})[0][0]             # [4, H, W]
p = np.exp(logits - logits.max(0)); p /= p.sum(0)
ink = (p[2] + p[3]) > 0.6                                     # handwriting mask
```

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Background | White/empty areas |
| 1 | Printed | Template text and form elements (luma < 160) |
| 2 | Handwriting | Ink strokes from student writing (alpha > 80/255) |
| 3 | Overlap | Handwriting overlapping printed content |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Full Regeneration Pipeline

Run these commands in order from the repo root:

```bash
# 1. Generate synthetic form templates (1275-1650 pages)
python -m scripts.make_templates

# 2. Render handwriting stroke images
python -m scripts.make_strokes

# 3. Composite strokes onto templates to create training pairs
python -m scripts.generate_synthetic_overlays

# 4. Train the model
python train.py

# 5. Export to ONNX
python export_onnx.py
```

## Fine-tuning on Real Scans (recommended path)

Synthetic data alone misreads real pencil (dark graphite gets labelled *printed*). Real pairs fix this.

1. Put pairs in `samples/`: `clean_NNNNN.pdf` (the original, unanswered quiz, vector PDF) and
   `answered_NNNNN.pdf` (the scanned, answered copy, same page order).
2. Build labels:
   ```bash
   python -m scripts.make_real_pairs          # -> data/real/image_*.png + mask_*.png + debug/
   ```
   The clean page is registered onto the scan (SIFT + RANSAC homography, optional ECC refinement)
   and the mask is derived by differencing: printed from the clean render, handwriting = scan ink
   away from printed, **255 = ignore** (3 px band around printed glyphs, scanner streaks, page
   borders). Check `data/real/debug/*_align.png` (green printed should sit on grey glyphs) and the
   printed hit rate printed per page (~0.9 = good alignment).
3. Fine-tune from the pretrained Inoob/DIS-Handwriting-Remover checkpoint:
   ```bash
   python finetune.py --holdout "*_p3" --epochs 15 --export export/finetuned.onnx
   ```
   Uses `ISNetDISOriginal` (the pretrained topology, 3x faster than `ISNetDIS`), lifts the binary
   side heads to 4 classes so epoch 0 reproduces the pretrained model, trains with
   `ignore_index=255`, validates on full held-out pages at inference scale and exports ONNX with
   the standard contract. `--data data/real data/train` mixes in synthetic pages.
4. Compare models on held-out pairs with `python -m scripts.eval_onnx` (below).

### Many answered copies per original (scan bundles)

When scans arrive as bundles (one PDF holding several quizzes and writers, pages in any order,
printer and phone sources) match pages first, then build pairs from the manifest:

```bash
python -m scripts.match_scans --originals DIR_OF_ORIGINAL_PDFS --printer DIR_OF_SCAN_PDFS \
    --phone phone_scans.pdf                      # -> data/match/manifest.{csv,json}, summary.txt, thumbs/
python -m scripts.make_real_pairs --manifest data/match/manifest.csv [--status matched,low_confidence]
```

`match_scans.py` ranks every original page per scan page by SIFT inliers (coarse), re-registers
the best candidates and scores them by printed hit rate + the fraction of scan ink the page
explains; look-alike pages (continuation sheets, same sheet from another year) are separated by
correlating the glyphs that differ at full scan resolution. Status per page: `matched`,
`low_confidence` (inliers < 40 or hit rate < 0.7), `blank`, `unmatched`; non-matched pages get a
thumbnail under `thumbs/` so a missing original can be identified. `rot` records the rotation
that makes the scan upright. Outputs are named `image_<original>_<scanfile>_p<n>_<pr|ph>.png`;
phone pages get a coarse-to-fine ECC pyramid and pages still below `--min-hit` after ECC are
skipped and listed. Large grey fills (chessboards, shaded boxes) are labelled printed on an
absolute grey test and ink on them handwriting, since the local-background darkness is blind to
them.

Hold out whole quizzes, and both sources (phone + printer) of the same sheet together, since they are
the same physical pages:

```bash
python finetune.py --class-weights 1,1,1,1 --epochs 40 --crops-per-page 40 --export export/finetuned_d.onnx \
    --holdout "quizA_*,quizB_*"
```

`--holdout` takes comma-separated fnmatch patterns on the stem.

Compare exported models on the same held-out pages (argmax IoU like finetune.py, plus ink precision/recall
under the pipeline rule `p[2]+p[3] > threshold`, split by printer/phone, `--per-page` for detail):

```bash
python -m scripts.eval_onnx --models export/inoob_isnet.onnx export/finetuned_c.onnx export/finetuned_d.onnx \
    --holdout "quizA_*,quizB_*" --threshold 0.6 --per-page
```

Released model (`finetuned_e.onnx`): 111 real scan pairs, 12 crops/page, 40 epochs, best epoch 21. On 15
held-out pages at threshold 0.6: ink precision 0.848, recall 0.834, F1 0.841; IoU printed 0.732, handwriting 0.726.
Known weakness: smooth printed curves (function graphs) are often classified as handwriting, regardless of
colour and threshold. Next lever: oversample pages with printed curves, or add synthetic printed-curve overlays.

Pretrained-only export (no training): `python export_pretrained_onnx.py` -> `export/inoob_isnet.onnx`.

## Hardware Requirements

| Hardware | Training Time (est.) | Notes |
|----------|---------------------|-------|
| Apple Silicon (M1/M2/M3) MPS | ~2 hours | Recommended for development |
| NVIDIA GPU (RTX 3080+) | ~45 minutes | CUDA required |
| CPU only | ~12+ hours | Not recommended |

Memory: 16 GB RAM minimum, 32 GB recommended. The model uses ~800 MB VRAM; batch size 4 at 768x768 crops requires ~6-8 GB.

## Data Generation Details

- **Templates** (`data/templates/`): Synthetic form pages at 150 DPI with field rows, numbered boxes, tables, checkboxes, signature blocks, answer tables, chemical notation, inverted headers, lettered lines, corner-label cells, bold years, and dense small-print pages.

- **Strokes** (`data/strokes/`): RGBA grayscale ink on transparent alpha. Generated from macOS handwriting fonts (Snell Roundhand, Bradley Hand, etc.) plus procedural marks (Bezier signatures, scribbles, checkmarks, X's, circles, arrows, underlines, Lewis structures, formula tokens).

- **Training pairs** (`data/train/`): Each pair is an input image + 4-class mask. ~53% of strokes placed inside cells (scaled 0.55-1.15x), remainder placed randomly. Ink colors: pencil 40%, blue 28%, black 22%, red 10%.

## Training Details

- Architecture: IS-Net (from DIS - Dichotomous Image Segmentation)
- Warm-start: `inoob/DIS-Handwriting-Remover` pretrained weights (shape-matching tensors only)
- Optimizer: AdamW, lr=1e-4, cosine annealing to 1e-6
- Loss: Multi-scale cross-entropy over 6 side outputs
- Class weights: [0.2, 1.6, 3.0, 5.0] (background, printed, handwriting, overlap)
- Best checkpoint selected by mean IoU of ink classes (class 2 + class 3)
- Augmentations: scale 0.35-1.6x, rotation +/-15 deg, color jitter, shadow ramps, Gaussian noise, JPEG artifacts, random crop 768x768

## ONNX Export Contract

- Input name: `input` - float32 [N, 3, H, W] RGB normalized as `pixel_value / 255.0 - 0.5`
- Output name: `logits` - float32 [N, 4, H, W] raw class logits
- Dynamic batch, height, width axes
- Opset 17
- Parity verified: atol < 1e-4 and argmax agreement vs PyTorch

## Iterating on Quality

1. **More diverse templates**: Add new section types in `scripts/make_templates.py`
2. **Better strokes**: Add more fonts, adjust procedural mark parameters
3. **Augmentation tuning**: Adjust scale range, noise levels in `utils/augmentations.py`
4. **Class balance**: Modify `CLASS_WEIGHTS` in `train.py` if certain classes underperform
5. **Longer training**: Increase `NUM_EPOCHS` or add learning rate warmup
6. **Resolution**: Increase `CROP_SIZE` for finer detail (requires more VRAM)
7. **Post-hoc analysis**: Check per-class IoU in training logs to identify weak areas

## Directory Structure

```
.
├── README.md
├── LICENSE / NOTICE
├── MODEL_CARD.md         # Hugging Face model card (upload as README.md on the Hub)
├── requirements.txt
├── train.py              # Synthetic-data training script
├── finetune.py           # Fine-tune pretrained DIS on real pairs (data/real)
├── export_onnx.py        # ONNX export + parity check (train.py checkpoints)
├── export_pretrained_onnx.py  # Export pretrained Inoob weights with the 4-class contract
├── scripts/
│   ├── make_templates.py           # Synthetic form page generator
│   ├── make_strokes.py             # Handwriting stroke renderer
│   ├── generate_synthetic_overlays.py  # Compositing pipeline
│   ├── make_real_pairs.py          # clean+answered PDF pairs -> aligned real labels (also --manifest)
│   ├── match_scans.py              # scan bundles -> manifest of (scan page, original page)
│   └── eval_onnx.py                # compare ONNX models on held-out pairs
├── models/
│   └── dis_net.py        # IS-Net architecture (modified for 4-class)
├── utils/
│   └── augmentations.py  # Training augmentations
├── samples/              # clean_NNNNN.pdf + answered_NNNNN.pdf real pairs (input)
├── pretrained/           # downloaded inoob_isnet.pth
├── data/
│   ├── templates/        # Generated form pages + cell JSON
│   ├── strokes/          # Generated stroke RGBA PNGs
│   ├── train/            # Synthetic image/mask pairs
│   └── real/             # Real image/mask pairs from make_real_pairs.py (+ debug/)
├── checkpoints/          # Saved model checkpoints
└── export/               # ONNX model output
```

## License

Apache License 2.0, see `LICENSE` and `NOTICE`. The IS-Net architecture is from
[xuebinqin/DIS](https://github.com/xuebinqin/DIS) and the warm-start weights from
[Inoob/DIS-Handwriting-Remover](https://huggingface.co/Inoob/DIS-Handwriting-Remover), both Apache-2.0.

`scripts/make_strokes.py` renders strokes from fonts installed on macOS; fonts are not included here and
are subject to their own licenses.
