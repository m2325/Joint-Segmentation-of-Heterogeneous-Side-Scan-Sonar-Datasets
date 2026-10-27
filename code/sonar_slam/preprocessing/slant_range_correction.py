#!/usr/bin/env python3
"""
sonar_slam/preprocessing/slant_range_correction.py

侧扫声纳slant-range到ground-range的几何校正。
将斜距图像转换为地距图像，更适合匹配和分割。
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
import cv2


def slant_to_ground(img: np.ndarray,
                    altitude_m: float,
                    range_m: float,
                    sample_resolution_m: Optional[float] = None,
                    output_height: Optional[int] = None) -> np.ndarray:
    """
    将斜距(slant-range)图像转换为地距(ground-range)图像。

    几何原理：
        ground_range = sqrt(slant_range^2 - altitude^2)

    使用线性插值实现平滑的地距映射。

    Args:
        img: 输入图像 (H, W)，每行是一个ping，每列是一个采样点
        altitude_m: 平台离底高度（米）
        range_m: 单侧量程（米）
        sample_resolution_m: 每个采样点的距离分辨率（米），如果为None则从range_m和宽度推断
        output_height: 输出图像高度，如果为None则与输入相同

    Returns:
        地距校正后的图像
    """
    if img.ndim != 2:
        raise ValueError(f"图像必须是2维，当前维度: {img.ndim}")

    h, w = img.shape

    if sample_resolution_m is None:
        sample_resolution_m = range_m / w

    output_h = output_height if output_height is not None else h
    output_w = w

    ground_img = np.zeros((output_h, output_w), dtype=img.dtype)

    slant_indices = np.arange(w, dtype=np.float32)
    slant_ranges = slant_indices * sample_resolution_m

    valid = slant_ranges > altitude_m
    ground_ranges = np.sqrt(np.maximum(slant_ranges ** 2 - altitude_m ** 2, 0))

    max_ground_range = np.max(ground_ranges[valid]) if np.any(valid) else range_m
    ground_indices = (ground_ranges / max_ground_range * (output_w - 1)).astype(np.float32)

    for row in range(h):
        src_row = img[row].astype(np.float32)
        dst_row = np.zeros(output_w, dtype=np.float32)

        if np.any(valid):
            valid_src_idx = np.where(valid)[0]
            valid_ground_idx = ground_indices[valid]
            valid_src_val = src_row[valid]

            sorted_order = np.argsort(valid_ground_idx)
            sorted_ground = valid_ground_idx[sorted_order]
            sorted_vals = valid_src_val[sorted_order]

            for i in range(len(sorted_ground) - 1):
                g0 = sorted_ground[i]
                g1 = sorted_ground[i + 1]
                v0 = sorted_vals[i]
                v1 = sorted_vals[i + 1]

                if g1 - g0 < 0.5:
                    idx = int(round(g0))
                    if 0 <= idx < output_w:
                        dst_row[idx] = max(dst_row[idx], v0)
                    continue

                n_pixels = max(1, int(round(g1 - g0)))
                for p in range(n_pixels + 1):
                    idx = int(round(g0 + p))
                    if 0 <= idx < output_w:
                        t = p / n_pixels
                        dst_row[idx] = v0 * (1 - t) + v1 * t

        out_row_idx = int(row * output_h / h) if output_h != h else row
        if 0 <= out_row_idx < output_h:
            if img.dtype == np.uint8:
                ground_img[out_row_idx] = np.clip(dst_row, 0, 255).astype(np.uint8)
            else:
                ground_img[out_row_idx] = dst_row

    return ground_img


def slant_to_ground_fast(img: np.ndarray,
                          altitude_m: float,
                          range_m: float,
                          sample_resolution_m: Optional[float] = None,
                          output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    快速版slant-range到ground-range校正，使用OpenCV remap。

    Args:
        img: 输入图像
        altitude_m: 平台离底高度（米）
        range_m: 单侧量程（米）
        sample_resolution_m: 采样分辨率（米）
        output_size: 输出尺寸 (height, width)

    Returns:
        地距校正后的图像
    """
    if img.ndim != 2:
        raise ValueError(f"图像必须是2维，当前维度: {img.ndim}")

    h, w = img.shape

    if sample_resolution_m is None:
        sample_resolution_m = range_m / w

    out_h, out_w = output_size if output_size is not None else (h, w)

    if out_w > 32767 or out_h > 32767 or w > 32767 or h > 32767:
        return slant_to_ground(img, altitude_m, range_m,
                               sample_resolution_m, out_h)

    ground_x = np.arange(out_w, dtype=np.float32) * range_m / out_w
    slant_x = np.sqrt(ground_x ** 2 + altitude_m ** 2)
    src_x = slant_x / sample_resolution_m

    map_x = np.tile(src_x, (out_h, 1))
    map_y = np.tile(np.arange(out_h, dtype=np.float32) * h / out_h, (out_w, 1)).T

    try:
        ground_img = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    except cv2.error:
        return slant_to_ground(img, altitude_m, range_m,
                               sample_resolution_m, out_h)

    return ground_img


class SlantRangeCorrector:
    """斜距校正器类。"""

    def __init__(self,
                 altitude_m: float = 5.0,
                 range_m: float = 50.0,
                 sample_resolution_m: Optional[float] = None,
                 method: str = 'fast'):
        self.altitude_m = altitude_m
        self.range_m = range_m
        self.sample_resolution_m = sample_resolution_m
        self.method = method

    def process(self,
                img: np.ndarray,
                altitude_m: Optional[float] = None,
                output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        处理图像。

        Args:
            img: 输入图像
            altitude_m: 覆盖默认的高度
            output_size: 输出尺寸 (height, width)

        Returns:
            校正后的图像
        """
        alt = altitude_m if altitude_m is not None else self.altitude_m

        if self.method == 'fast':
            return slant_to_ground_fast(
                img, alt, self.range_m,
                self.sample_resolution_m, output_size)
        else:
            out_h = output_size[0] if output_size else None
            return slant_to_ground(
                img, alt, self.range_m,
                self.sample_resolution_m, out_h)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """直接调用。"""
        return self.process(img)
