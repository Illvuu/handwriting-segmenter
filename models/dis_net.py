"""
IS-Net (Intermediate Supervision Network) for handwriting segmentation.
Architecture from xuebinqin/DIS (Apache-2.0 license).
Modified: out_ch=4, raw logits output (no sigmoid), no GT-encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


# ----- RSU (Residual U-block) building blocks -----

class REBNCONV(nn.Module):
    """Conv-BN-ReLU block."""

    def __init__(self, in_ch=3, out_ch=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RSU7(nn.Module):
    """RSU-7: U-Net-like block with depth 7."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dilation=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(nn.Module):
    """RSU-6: U-Net-like block with depth 6."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dilation=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(nn.Module):
    """RSU-5: U-Net-like block with depth 5."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilation=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(nn.Module):
    """RSU-4: U-Net-like block with depth 4."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilation=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(nn.Module):
    """RSU-4F: Dilated RSU block (no pooling), for bottleneck stages."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilation=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilation=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilation=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilation=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


def _upsample_like(src, tar):
    """Upsample src to the spatial size of tar."""
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)


# ----- IS-Net main architecture -----

class ISNetDIS(nn.Module):
    """
    IS-Net for Dichotomous Image Segmentation.
    Modified for 4-class handwriting segmentation:
      0=background, 1=printed, 2=handwriting, 3=overlap
    Returns raw logits (no sigmoid) from 6 side outputs + 1 main output.
    """

    def __init__(self, in_ch=3, out_ch=4):
        super().__init__()
        self.out_ch = out_ch

        # Encoder stages
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # Decoder stages
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # Side output heads (produce logits at each stage)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def forward(self, x):
        hx = x

        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # Decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # Side outputs (logits at input resolution)
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        return [d1, d2, d3, d4, d5, d6]


class ISNetDISOriginal(ISNetDIS):
    """
    Original DIS IS-Net topology (xuebinqin/DIS), as used by the Inoob/DIS-Handwriting-Remover
    checkpoint: a stride-2 ``conv_in`` (in_ch -> 64) in front of stage1 (which therefore takes 64
    channels) and side outputs upsampled back to the input resolution. Runs ~3x faster than
    ISNetDIS at the same input size because every stage works at half resolution.
    """

    def __init__(self, in_ch=3, out_ch=4):
        super().__init__(in_ch=in_ch, out_ch=out_ch)
        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.stage1 = RSU7(64, 32, 64)

    def forward(self, x):
        hx = self.conv_in(x)

        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        d1 = _upsample_like(self.side1(hx1d), x)
        d2 = _upsample_like(self.side2(hx2d), x)
        d3 = _upsample_like(self.side3(hx3d), x)
        d4 = _upsample_like(self.side4(hx4d), x)
        d5 = _upsample_like(self.side5(hx5d), x)
        d6 = _upsample_like(self.side6(hx6), x)
        return [d1, d2, d3, d4, d5, d6]


def remap_dis_state_dict(state_dict: dict) -> dict:
    """Original DIS checkpoints name REBNCONV layers conv_s1/bn_s1; ours use conv/bn."""
    return {k.replace(".conv_s1.", ".conv.").replace(".bn_s1.", ".bn."): v for k, v in state_dict.items()}


@torch.no_grad()
def load_binary_dis_weights(model: ISNetDISOriginal, state_dict: dict, strict_backbone: bool = True) -> dict:
    """
    Load a 1-channel DIS checkpoint into an ISNetDISOriginal with any out_ch.

    Backbone tensors are copied as-is. For the 6 side heads (1 -> out_ch channels) the single
    pretrained channel becomes the *handwriting* logit (class 2), background gets a zero logit,
    and the other classes start with a strongly negative bias. With out_ch = 4 this makes the
    freshly initialised network reproduce the binary model exactly:
        softmax(logits)[2] == sigmoid(pretrained_logit)
    so fine-tuning starts from the pretrained behaviour instead of random heads.

    Returns {"loaded": n, "heads": n_heads, "missing": [...], "unexpected": [...]}.
    """
    sd = remap_dis_state_dict(state_dict)
    own = model.state_dict()
    out_ch = model.out_ch
    loaded, heads = 0, 0
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k].copy_(v)
            loaded += 1
        elif k.startswith("side") and v.shape[0] == 1 and own[k].shape[0] == out_ch:
            if k.endswith(".weight"):
                own[k].zero_()
                own[k][2] = v[0]
            else:  # bias
                own[k].fill_(-5.0)
                own[k][0] = 0.0
                own[k][2] = v[0]
            heads += 1
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    model.load_state_dict(own)
    if strict_backbone and missing:
        raise RuntimeError(f"checkpoint missing backbone tensors: {missing[:5]} ...")
    return {"loaded": loaded, "heads": heads, "missing": missing, "unexpected": unexpected}


class ISNetONNXExport(nn.Module):
    """
    ONNX export wrapper for IS-Net.
    Returns only the d1 (main) side output logits.
    """

    def __init__(self, model: ISNetDIS):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs[0]  # d1 only
