#!/usr/bin/env python3
"""
sonar_slam/preprocessing/contrast_enhance.py

侧扫声纳图像对比度增强与归一化。
包括CLAHE和百分位归一化。
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
import cv2


def normalize_percentile(img: np.ndarray,
                         low_pct: float = 1.0,
                         high_pct: float = 99.0) -> np.ndarray:
    """
    百分位归一化。

    Args:
        img: 输入图像
        low_pct: 低百分位
        high_pct: 高百分位

    Returns:
        归一化后的图像 (uint8, 0-255)
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img

    lo = np.percentile(img_float, low_pct)
    hi = np.percentile(img_float, high_pct)

    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)

    normalized = ((np.clip(img_float, lo, hi) - lo) / (hi - lo) * 255.0)

    if img.dtype == np.uint8:
        return normalized.astype(np.uint8)
    else:
        return normalized


def enhance_clahe(img: np.ndarray,
                  clip_limit: float = 2.0,
                  tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    CLAHE对比度增强。

    Args:
        img: 输入图像 (uint8)
        clip_limit: 对比度限制
        tile_grid_size: 网格大小 (列, 行)

    Returns:
        增强后的图像
    """
    if img.dtype != np.uint8:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_uint8)


def enhance_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    全局直方图均衡化。

    Args:
        img: 输入图像

    Returns:
        增强后的图像
    """
    if img.dtype != np.uint8:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    return cv2.equalizeHist(img_uint8)


def enhance_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma校正。

    Args:
        img: 输入图像
        gamma: Gamma值 (>1变暗, <1变亮)

    Returns:
        增强后的图像
    """
    img_float = img.astype(np.float32) / 255.0
    img_gamma = np.power(img_float, gamma)
    return (img_gamma * 255.0).astype(np.uint8)


class ContrastEnhancer:
    """对比度增强器类。"""

    def __init__(self,
                 method: str = 'clahe',
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_width: int = 8,
                 clahe_tile_height: int = 8,
                 norm_low_pct: float = 1.0,
                 norm_high_pct: float = 99.0,
                 gamma: float = 1.0):
        """
        初始化对比度增强器。

        Args:
            method: 增强方法
                - 'clahe': CLAHE
                - 'histogram': 全局直方图均衡化
                - 'percentile': 百分位归一化
                - 'gamma': Gamma校正
                - 'clahe_percentile': CLAHE + 百分位归一化
            clahe_clip_limit: CLAHE对比度限制
            clahe_tile_width: CLAHE网格宽度
            clahe_tile_height: CLAHE网格高度
            norm_low_pct: 百分位归一化低百分位
            norm_high_pct: 百分位归一化高百分位
            gamma: Gamma值
        """
        self.method = method
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = (clahe_tile_width, clahe_tile_height)
        self.norm_low_pct = norm_low_pct
        self.norm_high_pct = norm_high_pct
        self.gamma = gamma

    def process(self, img: np.ndarray) -> np.ndarray:
        """
        处理图像。

        Args:
            img: 输入图像

        Returns:
            增强后的图像
        """
        if self.method == 'clahe':
            return enhance_clahe(
                img, self.clahe_clip_limit, self.clahe_tile_size)
        elif self.method == 'histogram':
            return enhance_histogram_equalization(img)
        elif self.method == 'percentile':
            return normalize_percentile(
                img, self.norm_low_pct, self.norm_high_pct)
        elif self.method == 'gamma':
            return enhance_gamma(img, self.gamma)
        elif self.method == 'clahe_percentile':
            img_clahe = enhance_clahe(
                img, self.clahe_clip_limit, self.clahe_tile_size)
            return normalize_percentile(
                img_clahe, self.norm_low_pct, self.norm_high_pct)
        else:
            raise ValueError(f"未知的增强方法: {self.method}")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """直接调用。"""
        return self.process(img)
