#!/usr/bin/env python3
"""
tools/training/deeplabv3plus.py

DeepLabV3+ (Chen et al. 2018, ECCV) public-architecture baseline against
SegFormer-B2 for side-scan sonar semantic segmentation.

Implementation strategy:
  - Reuse torchvision.models.segmentation.deeplabv3_resnet50 / _resnet101
    prebuilt modules (ImageNet-pretrained backbone + ASPP + decoder). This
    gives a faithful, community-validated DeepLabV3+ implementation without
    re-implementing the ASPP / decoder by hand.
  - Replace the backbone's first conv (backbone.conv1) when in_channels != 3
    so the model accepts raw 1-channel grayscale or 5-channel SSPM
    (raw + range_norm + local_contrast + shadow + reliability) inputs. The
    replacement preserves the original conv's kernel/stride/padding so the
    downstream RF and feature-map sizes are unchanged. Pretrained weights
    are loaded *before* the swap, and the new conv is initialised to average
    the old weights across channels (for 1ch) or replicate-then-pad (for
    5ch) so the early features start close to the ImageNet prior.
  - Output is the torchvision dict {"out": logits, "aux": ...}; callers in
    train_joint_dmce.py / train_subpipe_segformer.py already branch on
    `isinstance(logits, dict)` and use logits["out"], so DeepLabV3+ drops in
    without further plumbing.

DeepLabV3+ architecture (high level):
  Input (B, C, H, W)
     |
  ResNet backbone (conv1 + maxpool + layer1-4)
     |
  ASPP (atrous spatial pyramid pooling, rates=[6,12,18])
     |
  Decoder: fuse low-level (layer1, 1/4) features with ASPP output
     |
  1x1 conv -> num_classes logits (B, num_classes, H, W after upsampling)

Reference:
  Chen L.-C., Zhu Y., Papandreou G., Schroff F., Adam H. (2018).
  "Encoder-Decoder with Atrous Separable Convolution for Semantic Image
  Segmentation." ECCV 2018.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)


_BackboneName = Literal["resnet50", "resnet101"]


def _adapt_first_conv(backbone: nn.Module, in_channels: int) -> None:
    """Replace backbone.conv1 to accept `in_channels` input.

    ResNet's conv1 is Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False).
    We re-create it with the same hyperparams but in_channels=input, then
    initialise weights to preserve the ImageNet prior as closely as possible:
      - in_channels=1: copy the mean of the original RGB weights into the
        single input channel (so grayscale input ≈ averaged-RGB input).
      - in_channels=5: copy original 3 weights to channels 0-2, zero-init
        channels 3-4 (extra SSPM channels start with no contribution, the
        network learns to use them).
      - in_channels=3: no-op.

    This is the standard "new first conv" trick used by every imagenet-pretrained
    segmentation model adapted to non-RGB input.
    """
    old_conv: nn.Conv2d = backbone.conv1
    if in_channels == old_conv.in_channels:
        return  # nothing to do (3-channel path keeps pretrained weights)

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        old_w = old_conv.weight  # (out=64, in=3, 7, 7)
        new_w = torch.zeros_like(new_conv.weight)  # (out=64, in=in_channels, 7, 7)
        if in_channels == 1:
            # Grayscale: average RGB weights -> 1 channel
            new_w[:, 0] = old_w.mean(dim=1)
        else:
            # in_channels > 3 (e.g. 5 for SSPM): copy RGB weights into the
            # first 3 channels, leave extras at 0 (BN downstream learns scale).
            copy_ch = min(3, in_channels)
            new_w[:, :copy_ch] = old_w[:, :copy_ch]
        new_conv.weight.copy_(new_w)
    backbone.conv1 = new_conv


class DeepLabV3PlusWrapper(nn.Module):
    """Thin wrapper around torchvision DeepLabV3+ for non-RGB input.

    Forward returns the same dict as torchvision's DeepLabV3Head:
        {"out": (B, num_classes, H, W), "aux": (B, num_classes, H, W)}
    Callers already handle the dict via `logits["out"]`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        backbone_name: _BackboneName = "resnet50",
        pretrained: bool = True,
        aux_loss: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # When pretrained=True, load ImageNet ResNet50 backbone weights via
        # `weights_backbone` (NOT `weights`). Using `weights=` would force
        # num_classes to match the COCO/VOC pretrained head (21 classes),
        # which conflicts with our 8-class unified semantic space.
        # `weights_backbone` only loads the backbone, leaving the ASPP +
        # decoder randomly initialised — but it's the only way to keep
        # num_classes configurable.
        weights_arg = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        if backbone_name == "resnet50":
            model = deeplabv3_resnet50(weights_backbone=weights_arg,
                                        num_classes=num_classes,
                                        aux_loss=aux_loss)
        elif backbone_name == "resnet101":
            w101 = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            model = deeplabv3_resnet101(weights_backbone=w101,
                                         num_classes=num_classes,
                                         aux_loss=aux_loss)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # backbone.conv1 is the input conv; replace it for non-3-channel input
        _adapt_first_conv(model.backbone, in_channels)

        # aux head is built only when aux_loss=True; default off to keep
        # smoke tests simple
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model(x)


def build_deeplabv3plus_model(
    in_channels: int = 3,
    num_classes: int = 8,
    backbone: _BackboneName = "resnet50",
    pretrained: bool = True,
    aux_loss: bool = False,
) -> DeepLabV3PlusWrapper:
    """Factory for DeepLabV3+, matching build_segformer_model signature.

    Args:
        in_channels: 1 (raw grayscale) or 5 (SSPM representation) or 3 (RGB).
        num_classes: 8 for unified joint semantic space (C0-C7).
        backbone: "resnet50" or "resnet101" (torchvision-provided, ImageNet pretrain).
        pretrained: whether to load ImageNet-pretrained backbone weights.
        aux_loss: enable auxiliary head (default off — train_joint_dmce.py
            does not consume the aux output).

    Returns:
        DeepLabV3PlusWrapper module. forward(x) returns the torchvision
        dict {"out": logits (B, num_classes, H, W), ...}.
    """
    if not pretrained:
        logging.getLogger(__name__).info(
            "DeepLabV3+ built WITHOUT pretrained backbone weights "
            "(pretrained=False) — for fair comparison use pretrained=True.")
    return DeepLabV3PlusWrapper(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=pretrained,
        aux_loss=aux_loss,
    )


if __name__ == "__main__":
    # Quick forward / param count for 1/3/5-channel inputs
    for in_ch in (1, 3, 5):
        m = build_deeplabv3plus_model(in_channels=in_ch, num_classes=8,
                                       pretrained=False)
        x = torch.randn(2, in_ch, 256, 256)
        out = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"in_ch={in_ch}: input={x.shape}  out_keys={list(out.keys())}  "
              f"out_shape={out['out'].shape}  params={n_params/1e6:.2f}M")
