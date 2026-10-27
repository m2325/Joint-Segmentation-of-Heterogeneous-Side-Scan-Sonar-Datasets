#!/usr/bin/env python3
"""
tools/training/sspm_channels.py

SSPM-5ch (SLAM-aware Semantic Preprocessing Module, 5-channel) generator.

Produces a 5-channel representation from a single 2D side-scan sonar image,
designed to preserve raw acoustic cues while exposing structure that serves
semantic segmentation and SLAM-oriented reliability modeling.

Channels:
  C1 raw intensity          — original grayscale, [0,1]
  C2 range-normalized       — across-track brightness trend removed
  C3 local contrast map     — local std in a sliding window
  C4 shadow likelihood map  — dark regions weighted by adjacent contrast
  C5 reliability map         — high where structured & unsaturated, low where flat

Design rationale (D.18/D.19):
  - Classical enhancement (TVG/bilateral/CLAHE) actively hurts segmentation on
    AI4 mosaics and is unstable across datasets. SSPM keeps raw as C1 and
    ADDS geometric/semantic cues instead of replacing the image.
  - 3x3 conv adapter + BN + GELU fuses 5ch -> 3ch so the pretrained
    SegFormer-B2 encoder (hardcoded in_channels=3) is reused unchanged.

Inputs accept either uint8 [0,255] or float32 [0,1] 2D arrays.
All outputs are float32 in [0,1].
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

SSPM_NUM_CHANNELS = 5
SSPM_CHANNEL_NAMES = [
    "raw",
    "range_norm",
    "local_contrast",
    "shadow",
    "reliability",
]
SSPM_LOCAL_WINDOW = 15  # local contrast / reliability window (pixels)


def _to_unit_float(img: np.ndarray) -> np.ndarray:
    """Coerce a 2D image to float32 in [0, 1]."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {img.shape}")
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    if img.max() > 1.5:  # uint8-range float
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def generate_sspm_channels(
    img_2d: np.ndarray,
    local_window: int = SSPM_LOCAL_WINDOW,
    eps: float = 1e-6,
) -> np.ndarray:
    """Generate the 5-channel SSPM representation from a 2D sonar image.

    Args:
        img_2d: (H, W) uint8 or float32 grayscale. uint8 is auto-normalized.
        local_window: window size (pixels) for local contrast / reliability.
        eps: numerical floor for safe division.

    Returns:
        (5, H, W) float32 array, each channel in [0, 1].
    """
    img = _to_unit_float(img_2d)
    H, W = img.shape

    # --- C1: raw intensity (identity, already [0,1]) -----------------------
    c1_raw = img.copy()

    # --- C2: range-normalized intensity ------------------------------------
    # Across-track brightness varies strongly with slant range in SSS.
    # For a (H, W) image with W = across-track, estimate a per-column robust
    # brightness profile (median along the along-track axis) and divide it out.
    col_profile = np.median(img, axis=0, keepdims=True)  # (1, W)
    col_profile = np.maximum(col_profile, eps)
    range_norm = np.clip(img / col_profile, 0.0, 1.0)
    c2_range = range_norm.astype(np.float32)

    # --- C3: local contrast map (local standard deviation) -----------------
    local_mean = uniform_filter(img, size=local_window, mode="reflect")
    local_sq = uniform_filter(img * img, size=local_window, mode="reflect")
    local_var = np.maximum(local_sq - local_mean * local_mean, 0.0)
    local_std = np.sqrt(local_var)
    std_max = local_std.max()
    if std_max < eps:  # completely flat image
        c3_contrast = np.zeros_like(img)
    else:
        c3_contrast = np.clip(local_std / std_max, 0.0, 1.0)

    # --- C4: shadow likelihood (dark + adjacent contrast) ------------------
    # Shadow regions in SSS are dark AND lie next to bright reflectors, so
    # their boundary shows high local contrast. Shadow likelihood = darkness
    # weighted by local contrast.
    darkness = 1.0 - img
    shadow_raw = darkness * c3_contrast
    sh_max = shadow_raw.max()
    if sh_max < eps:
        c4_shadow = np.zeros_like(img)
    else:
        c4_shadow = np.clip(shadow_raw / sh_max, 0.0, 1.0)

    # --- C5: reliability map (structured & unsaturated) ---------------------
    # High reliability where the image carries texture in a usable intensity
    # range. Suppressed in (a) flat water-column / nadir (low contrast) and
    # (b) saturated bright returns (no usable gradient). This channel is the
    # one a SLAM backend would use to down-weight unreliable semantic regions.
    saturation = 1.0 - np.abs(img - 0.5) * 2.0  # peak at 0.5, 0 at 0/1
    saturation = np.clip(saturation, 0.0, 1.0)
    reliability_raw = c3_contrast * saturation
    rel_max = reliability_raw.max()
    if rel_max < eps:
        c5_reliability = np.zeros_like(img)
    else:
        c5_reliability = np.clip(reliability_raw / rel_max, 0.0, 1.0)

    return np.stack(
        [c1_raw, c2_range, c3_contrast, c4_shadow, c5_reliability],
        axis=0,
    ).astype(np.float32)


def sspm_channels_from_3ch(img_3ch: np.ndarray) -> np.ndarray:
    """Convenience wrapper: derive SSPM channels from a (3, H, W) triplicated
    grayscale image (as returned by the existing dataset loaders).

    Channel 0 of the triplicated stack is the resized/preprocessed grayscale;
    passing it through generate_sspm_channels reproduces all 5 SSPM channels.
    """
    img_3ch = np.asarray(img_3ch)
    if img_3ch.ndim != 3 or img_3ch.shape[0] != 3:
        raise ValueError(f"expected (3, H, W), got {img_3ch.shape}")
    return generate_sspm_channels(img_3ch[0])


if __name__ == "__main__":
    # Smoke test on a synthetic image
    rng = np.random.default_rng(0)
    synthetic = (rng.random((128, 128)) * 255).astype(np.uint8)
    out = generate_sspm_channels(synthetic)
    assert out.shape == (5, 128, 128), out.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    print(f"SSPM smoke test OK: shape={out.shape} range=[{out.min():.3f}, {out.max():.3f}]")
    for i, name in enumerate(SSPM_CHANNEL_NAMES):
        print(f"  C{i+1} {name:14s}: mean={out[i].mean():.4f} std={out[i].std():.4f}")
