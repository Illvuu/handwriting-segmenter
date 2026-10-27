"""
Export best checkpoint to ONNX format.

Contract:
  - Input: float32 NCHW RGB normalized (x/255 - 0.5)
  - Output: [N, 4, H, W] raw logits
  - Dynamic batch/H/W axes
  - Opset 17

Includes mandatory ONNX-Runtime vs PyTorch parity check.

Usage:
    python export_onnx.py
"""

import os
import sys

import numpy as np
import torch
import onnx
import onnxruntime as ort

from models.dis_net import ISNetDIS, ISNetONNXExport

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
EXPORT_DIR = os.path.join(BASE_DIR, "export")
ONNX_PATH = os.path.join(EXPORT_DIR, "model.onnx")


def load_model():
    """Load best checkpoint into ISNetDIS."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        print("Train the model first with: python train.py")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = ISNetDIS(in_ch=3, out_ch=4)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, "
          f"Ink IoU: {checkpoint.get('best_ink_iou', '?'):.4f}")

    return model


def export_to_onnx(model):
    """Export model to ONNX with dynamic axes."""
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Wrap for export (returns only d1 logits)
    wrapper = ISNetONNXExport(model)
    # CRITICAL: call .eval() on THE WRAPPER (not just inner model)
    # torch.onnx.export restores train/eval mode on root module
    wrapper.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 768, 768)

    # Dynamic axes for batch, height, width
    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch", 2: "height", 3: "width"},
    }

    print(f"Exporting ONNX to {ONNX_PATH}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Validate ONNX model
    print("Validating ONNX model structure...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    return wrapper


def parity_check(pytorch_wrapper):
    """
    Mandatory parity check: ONNX-Runtime vs PyTorch.
    Checks atol 1e-4 and argmax agreement on generated samples.
    """
    print("\nRunning ONNX-Runtime vs PyTorch parity check...")

    pytorch_wrapper.eval()

    # Load ONNX model
    session = ort.InferenceSession(ONNX_PATH)

    # Test with multiple input sizes
    test_shapes = [
        (1, 3, 768, 768),
        (1, 3, 512, 512),
        (2, 3, 384, 384),
    ]

    all_passed = True

    for shape in test_shapes:
        # Generate random test input (simulating normalized image data)
        test_input = torch.randn(*shape) * 0.3  # Typical range after normalization

        # PyTorch inference
        with torch.no_grad():
            pt_output = pytorch_wrapper(test_input).numpy()

        # ONNX Runtime inference
        ort_inputs = {"input": test_input.numpy()}
        ort_output = session.run(["logits"], ort_inputs)[0]

        # Check numerical parity (atol 1e-4)
        max_diff = np.abs(pt_output - ort_output).max()
        atol_passed = max_diff < 1e-4

        # Check argmax agreement
        pt_argmax = pt_output.argmax(axis=1)
        ort_argmax = ort_output.argmax(axis=1)
        argmax_agreement = (pt_argmax == ort_argmax).mean()
        argmax_passed = argmax_agreement > 0.999  # Allow tiny floating point differences

        passed = atol_passed and argmax_passed
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(f"  Shape {shape}: max_diff={max_diff:.2e}, "
              f"argmax_agreement={argmax_agreement:.6f} [{status}]")

        if not atol_passed:
            print(f"    WARNING: atol exceeded! max_diff={max_diff:.2e} > 1e-4")
        if not argmax_passed:
            print(f"    WARNING: argmax disagreement! agreement={argmax_agreement:.6f}")

    if all_passed:
        print("\nParity check PASSED. ONNX model matches PyTorch output.")
    else:
        print("\nParity check FAILED. Review differences above.")
        sys.exit(1)

    # Print model info
    file_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"\nExported model: {ONNX_PATH}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Input: float32 NCHW RGB (x/255 - 0.5)")
    print(f"Output: [N, 4, H, W] raw logits")


def main():
    model = load_model()
    wrapper = export_to_onnx(model)
    parity_check(wrapper)
    print("\nDone. Model ready for Java/ONNX inference.")


if __name__ == "__main__":
    main()
