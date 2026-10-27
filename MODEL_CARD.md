---
license: apache-2.0
library_name: onnx
pipeline_tag: image-segmentation
base_model: Inoob/DIS-Handwriting-Remover
tags:
  - handwriting
  - document-image
  - segmentation
  - is-net
  - onnx
---

# Handwriting Segmenter (IS-Net, 4-class, ONNX)

Segments scanned quiz/test pages into background, printed, handwriting and handwriting-over-print, so student
answers can be erased and the sheet redone for practice. Fine-tuned from
[Inoob/DIS-Handwriting-Remover](https://huggingface.co/Inoob/DIS-Handwriting-Remover) (IS-Net, from
[xuebinqin/DIS](https://github.com/xuebinqin/DIS)).

Training code: https://github.com/Illvuu/handwriting-segmenter

## Files

| file | sha256 |
|------|--------|
| `finetuned_e.onnx` (176 MB, opset 17) | `e396488615dff875476934b2d43c94ecc1d1545ad608e67be79bf8ef9d0b0087` |

## Contract

- Input `input`: float32 `[N, 3, H, W]`, RGB, `pixel / 255 - 0.5`. Dynamic N/H/W. Trained for 1536 px long edge.
- Output `logits`: float32 `[N, 4, H, W]`, raw logits for `{0 = background, 1 = printed, 2 = handwriting, 3 = overlap}`.
- Recommended decision: softmax over the 4 logits, ink = `p[2] + p[3] > 0.6`.

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

path = hf_hub_download("guoguo27/handwriting-segmenter", "finetuned_e.onnx")
sess = ort.InferenceSession(path)
img = Image.open("page.png").convert("RGB")
img.thumbnail((1536, 1536))
x = (np.asarray(img, np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
logits = sess.run(["logits"], {"input": x})[0][0]
p = np.exp(logits - logits.max(0)); p /= p.sum(0)
ink = (p[2] + p[3]) > 0.6
```

## Training

- Warm start from Inoob/DIS-Handwriting-Remover; binary side heads lifted to 4 classes.
- 111 real pairs: scanned answered pages (printer and phone) aligned to the clean original (SIFT + RANSAC + ECC),
  labels by differencing; a band around printed glyphs is ignored (label 255).
- 768 px crops, 12 per page, 40 epochs, AdamW; best epoch 21 by held-out ink IoU.

## Evaluation (15 held-out pages, whole quizzes, threshold 0.6)

| ink precision | ink recall | ink F1 | IoU printed | IoU handwriting |
|---------------|-----------|--------|-------------|-----------------|
| 0.848 | 0.834 | 0.841 | 0.732 | 0.726 |

The ignored band around printed glyphs means faint strokes and stroke edges are partly invisible to these metrics.

## Limitations

- Smooth printed curves (function graphs) are often classified as handwriting, independent of colour or threshold.
- Trained on a small set of high-school math/science quizzes; other layouts, languages and scripts are untested.
- Output is a mask. Erase deterministically (e.g. fill with local background median); do not inpaint generatively,
  which can fabricate content on a study sheet.

## License

Apache-2.0. See `NOTICE` in the training repository for attribution to DIS and Inoob/DIS-Handwriting-Remover.
