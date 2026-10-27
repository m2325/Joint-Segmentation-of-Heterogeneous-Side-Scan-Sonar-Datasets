#!/usr/bin/env python3
"""
sonar_slam/preprocessing/despeckle.py

侧扫声纳图像去噪。
抑制speckle噪声，保留结构边缘。
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import cv2
from scipy.ndimage import median_filter, uniform_filter


def despeckle_median(img: np.ndarray,
                     window_size: int = 5) -> np.ndarray:
    """
    中值滤波去噪。

    Args:
        img: 输入图像
        window_size: 滤波窗口大小

    Returns:
        去噪后的图像
    """
    return median_filter(img, size=window_size)


def despeckle_lee(img: np.ndarray,
                  window_size: int = 7,
                  num_looks: int = 1) -> np.ndarray:
    """
    Lee滤波去噪（适用于SAR/SSS图像）。

    Args:
        img: 输入图像，uint8或float32
        window_size: 滤波窗口大小
        num_looks: 视数（用于估计噪声方差）

    Returns:
        去噪后的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img

    h, w = img_float.shape
    result = np.zeros_like(img_float)

    half_win = window_size // 2

    for y in range(half_win, h - half_win):
        for x in range(half_win, w - half_win):
            window = img_float[y - half_win:y + half_win + 1,
                               x - half_win:x + half_win + 1]

            mean_win = np.mean(window)
            var_win = np.var(window)

            if var_win > 0:
                var_noise = var_win / (1 + num_looks)
                weight = max(0, 1 - var_noise / var_win)
            else:
                weight = 0

            result[y, x] = mean_win + weight * (img_float[y, x] - mean_win)

    if img.dtype == np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def despeckle_lee_fast(img: np.ndarray,
                       window_size: int = 7,
                       num_looks: int = 1) -> np.ndarray:
    """
    快速Lee滤波，使用卷积实现。

    Args:
        img: 输入图像
        window_size: 滤波窗口大小
        num_looks: 视数

    Returns:
        去噪后的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img

    mean_win = uniform_filter(img_float, size=window_size)
    mean_sq_win = uniform_filter(img_float ** 2, size=window_size)
    var_win = mean_sq_win - mean_win ** 2

    var_noise = var_win / (1 + num_looks)
    weight = np.maximum(0, 1 - var_noise / (var_win + 1e-10))

    result = mean_win + weight * (img_float - mean_win)

    if img.dtype == np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def despeckle_bilateral(img: np.ndarray,
                        d: int = 9,
                        sigma_color: float = 75,
                        sigma_space: float = 75) -> np.ndarray:
    """
    双边滤波去噪（保边去噪）。

    Args:
        img: 输入图像
        d: 滤波直径
        sigma_color: 颜色空间标准差
        sigma_space: 坐标空间标准差

    Returns:
        去噪后的图像
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def despeckle_gaussian(img: np.ndarray,
                       kernel_size: int = 5,
                       sigma_x: float = 0) -> np.ndarray:
    """
    高斯滤波去噪。

    Args:
        img: 输入图像
        kernel_size: 核大小
        sigma_x: X方向标准差

    Returns:
        去噪后的图像
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_x)


class Despeckler:
    """去噪器类。"""

    def __init__(self,
                 method: str = 'lee_fast',
                 window_size: int = 7,
                 num_looks: int = 1,
                 bilateral_d: int = 9,
                 bilateral_sigma_color: float = 75,
                 bilateral_sigma_space: float = 75):
        """
        初始化去噪器。

        Args:
            method: 去噪方法
                - 'median': 中值滤波
                - 'lee': Lee滤波
                - 'lee_fast': 快速Lee滤波
                - 'bilateral': 双边滤波
                - 'gaussian': 高斯滤波
            window_size: 滤波窗口大小
            num_looks: Lee滤波视数
            bilateral_d: 双边滤波直径
            bilateral_sigma_color: 双边滤波颜色空间标准差
            bilateral_sigma_space: 双边滤波坐标空间标准差
        """
        self.method = method
        self.window_size = window_size
        self.num_looks = num_looks
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    def process(self, img: np.ndarray) -> np.ndarray:
        """
        处理图像。

        Args:
            img: 输入图像

        Returns:
            去噪后的图像
        """
        if self.method == 'median':
            return despeckle_median(img, self.window_size)
        elif self.method == 'lee':
            return despeckle_lee(img, self.window_size, self.num_looks)
        elif self.method == 'lee_fast':
            return despeckle_lee_fast(img, self.window_size, self.num_looks)
        elif self.method == 'bilateral':
            return despeckle_bilateral(
                img, self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space)
        elif self.method == 'gaussian':
            return despeckle_gaussian(img, self.window_size)
        else:
            raise ValueError(f"未知的去噪方法: {self.method}")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """直接调用。"""
        return self.process(img)
