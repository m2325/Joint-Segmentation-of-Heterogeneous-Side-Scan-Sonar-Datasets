#!/usr/bin/env python3
"""
tools/training/sspm_segformer.py

SegFormer-B2 with a 5-channel SSPM input stem.

Architecture:
  (5, H, W) SSPM input
        |
  3x3 conv + BN + GELU   (5 -> 3 channels, preserves receptive field)
        |
  SegFormerEncoder (hardcoded in_channels=3, reused unchanged)
        |
  SegFormerHead (multi-level feature fusion)
        |
  (num_classes, H/4, W/4) logits

Design notes:
  - We wrap (not modify) the existing SegFormerEncoder / SegFormerHead so all
    existing 3-channel checkpoints stay loadable and the legacy
    train_segformer.py main() is untouched.
  - 3x3 conv (not 1x1) per D.18 review item (e): a 1x1 adapter cannot mix
    neighboring pixels, defeating the point of spatially-structured channels
    like local_contrast and shadow. 3x3 + padding=1 preserves spatial size.
  - BN + GELU follows the ImprovedSegFormerSonar input_proj convention but
    adds a nonlinearity so the fusion is not purely linear.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

SSPM_NUM_CHANNELS = 5


class SegFormerSonarSSPM(nn.Module):
    """SegFormer-B2 backbone with a 5-channel SSPM input adapter."""

    def __init__(
        self,
        num_classes: int = 8,
        input_channels: int = SSPM_NUM_CHANNELS,
        embed_dims=None,
        num_heads=None,
        depths=None,
        sr_ratios=None,
        decoder_embed_dim: int = 256,
    ):
        super().__init__()
        # Import locally so this module does not hard-depend on train_segformer
        # at import time (keeps unit tests that only check channels light).
        from train_segformer import SegFormerEncoder, SegFormerHead

        if embed_dims is None:
            embed_dims = [64, 128, 320, 512]
        if num_heads is None:
            num_heads = [1, 2, 5, 8]
        if depths is None:
            depths = [3, 4, 6, 3]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]

        self.input_channels = input_channels
        self.num_classes = num_classes

        # 3x3 conv fusion stem: 5ch -> 3ch so the SegFormer patch_embed1
        # (Conv2d(3, 64, 7, 4, 3)) is reused verbatim.
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )

        self.encoder = SegFormerEncoder(
            embed_dims, num_heads, depths, sr_ratios
        )
        self.decoder = SegFormerHead(
            in_channels_list=embed_dims,
            num_classes=num_classes,
            embed_dim=decoder_embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.input_proj(x)
        outs = self.encoder(x)
        # SegFormerEncoder returns token sequences (B, N, C) per stage.
        # SegFormerHead expects (B, C, H, W) feature maps. Reshape assuming
        # square feature maps (true for square inputs at each stage).
        feat_maps = []
        for out in outs:
            Bn, N, C = out.shape
            H = W = int(math.sqrt(N))
            feat_maps.append(out.transpose(1, 2).reshape(Bn, C, H, W))
        return self.decoder(feat_maps)


def build_sspm_model(num_classes: int = 8, input_channels: int = SSPM_NUM_CHANNELS) -> SegFormerSonarSSPM:
    """Factory matching the build_segformer_model / build_improved_model convention."""
    return SegFormerSonarSSPM(
        num_classes=num_classes,
        input_channels=input_channels,
    )


if __name__ == "__main__":
    model = build_sspm_model(num_classes=8, input_channels=5)
    x = torch.randn(2, 5, 512, 512)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input  : {x.shape}")
    print(f"output : {out.shape}")
    print(f"params : {n_params/1e6:.2f}M")
    print(f"input_proj: {model.input_proj}")
