"""
Export the pretrained Inoob/DIS-Handwriting-Remover IS-Net (binary, 1-channel) to ONNX
using the SAME contract as our 4-class model, so the Java pipeline and MlCleanCli can load
it with zero changes.

Contract (see export_onnx.py):
  - Input  "input":  float32 [N, 3, H, W], RGB, x/255 - 0.5   (identical to DIS preprocessing)
  - Output "logits": float32 [N, 4, H, W]

The pretrained net emits ONE logit l per pixel (p_ink = sigmoid(l)). We lift it to 4 fake class
logits so that softmax(p)[2] + softmax(p)[3] == sigmoid(l) exactly:

    logits = [ 0, -1e4, l, -1e4 ]     # bg, printed, handwriting, overlap

  softmax -> p_hw = e^l / (1 + e^l) = sigmoid(l);  p_bg = 1 - p_hw;  p_printed = p_overlap ~ 0
  argmax  -> handwriting iff l > 0  (same decision as the original DIS thresholding at 0.5)

Usage (from the repo root):
    python export_pretrained_onnx.py [--weights pretrained/inoob_isnet.pth] [--out export/inoob_isnet.onnx]
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from models.dis_net import ISNetDIS, ISNetDISOriginal, remap_dis_state_dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "Inoob/DIS-Handwriting-Remover"
REPO_FILE = "isnet.pth"


class BinaryToFourClass(nn.Module):
    """Wrap a 1-channel IS-Net into the 4-class logits contract."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        d1 = self.model(x)[0]                      # [N, 1, H, W] raw logit
        neg = torch.full_like(d1, -1e4)
        zero = torch.zeros_like(d1)
        return torch.cat([zero, neg, d1, neg], dim=1)  # [N, 4, H, W]


def resolve_weights(path: str | None) -> str:
    if path and os.path.exists(path):
        return path
    default = os.path.join(BASE_DIR, "pretrained", "inoob_isnet.pth")
    if os.path.exists(default):
        return default
    print(f"Weights not found locally, downloading {REPO_ID}/{REPO_FILE} ...")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO_ID, filename=REPO_FILE)


def load_pretrained(weights_path: str) -> ISNetDIS:
    model = ISNetDISOriginal(in_ch=3, out_ch=1)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = remap_dis_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {len(state) - len(unexpected)}/{len(state)} tensors from {weights_path}")
    if missing:
        print(f"  MISSING in checkpoint ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  UNEXPECTED in checkpoint ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if missing:
        print("ERROR: architecture mismatch, refusing to export a partially initialised model.")
        sys.exit(1)
    return model


def export(wrapper: nn.Module, onnx_path: str):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    wrapper.eval()
    dummy = torch.randn(1, 3, 768, 768)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "logits": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"Exported {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")


def parity_check(wrapper: nn.Module, onnx_path: str):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ok = True
    for shape in [(1, 3, 768, 768), (1, 3, 512, 640)]:
        x = torch.randn(*shape) * 0.3
        with torch.no_grad():
            pt = wrapper(x).numpy()
        rt = session.run(["logits"], {"input": x.numpy()})[0]
        max_diff = np.abs(pt - rt).max()
        agree = (pt.argmax(1) == rt.argmax(1)).mean()
        # sanity: softmax ink prob must equal sigmoid of the raw logit
        l = pt[:, 2]
        e = np.exp(pt - pt.max(1, keepdims=True))
        p_ink = (e[:, 2] + e[:, 3]) / e.sum(1)
        sig = 1 / (1 + np.exp(-l))
        contract_ok = np.abs(p_ink - sig).max() < 1e-5
        passed = max_diff < 1e-4 and agree > 0.999 and contract_ok
        ok &= passed
        print(f"  {shape}: max_diff={max_diff:.2e} argmax_agree={agree:.5f} "
              f"p_ink==sigmoid={contract_ok} [{'PASS' if passed else 'FAIL'}]")
    if not ok:
        print("Parity check FAILED")
        sys.exit(1)
    print("Parity check PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="path to isnet.pth (default: pretrained/inoob_isnet.pth or HF download)")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "export", "inoob_isnet.onnx"))
    args = ap.parse_args()

    weights = resolve_weights(args.weights)
    model = load_pretrained(weights)
    wrapper = BinaryToFourClass(model)
    export(wrapper, args.out)
    parity_check(wrapper, args.out)
    print("\nCompare with:  python -m scripts.eval_onnx --models", os.path.relpath(args.out, BASE_DIR))


if __name__ == "__main__":
    main()
