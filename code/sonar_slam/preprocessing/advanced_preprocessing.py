#!/usr/bin/env python3
"""
sonar_slam/preprocessing/advanced_preprocessing.py

高级侧扫声纳预处理方法。
包括：TVG增益补偿、条纹噪声去除、同态滤波、小波去噪等。
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
import cv2
from scipy import ndimage
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def apply_tvg(img: np.ndarray,
              alpha: float = 2e-4,
              beta: float = 2.0) -> np.ndarray:
    """
    时间增益补偿(Time-Varying Gain, TVG)。
    补偿声波随距离的衰减。

    Args:
        img: 输入图像
        alpha: 衰减系数
        beta: 幂次系数

    Returns:
        TVG补偿后的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img

    h, w = img_float.shape
    r = np.arange(1, w + 1, dtype=np.float32)

    tvg = (r ** beta) * np.exp(alpha * r)
    tvg = tvg / tvg.mean()

    return img_float * tvg[np.newaxis, :]


def remove_stripe_noise_fft(img: np.ndarray,
                            threshold_factor: float = 3.0) -> np.ndarray:
    """
    基于FFT的条纹噪声去除。
    参考 sidescantools 的方法。

    Args:
        img: 输入图像
        threshold_factor: 阈值因子

    Returns:
        去条纹后的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img

    fft = np.fft.fft2(img_float)
    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)
    magnitude_log = np.log(1 + magnitude)

    mean_mag = np.mean(magnitude_log)
    std_mag = np.std(magnitude_log)
    threshold = mean_mag + threshold_factor * std_mag

    mask = magnitude_log > threshold

    rows, cols = img_float.shape
    center_row, center_col = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

    mask = mask & (dist_from_center > 5)

    fft_shift[mask] = fft_shift[mask] * 0.1

    fft_ishift = np.fft.ifftshift(fft_shift)
    img_filtered = np.fft.ifft2(fft_ishift)
    img_filtered = np.real(img_filtered)

    if img.dtype == np.uint8:
        img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

    return img_filtered


def homomorphic_filter(img: np.ndarray,
                       sigma_low: float = 60,
                       sigma_high: float = 5,
                       gamma_low: float = 0.4,
                       gamma_high: float = 1.6) -> np.ndarray:
    """
    同态滤波。
    用于增强对比度同时压缩动态范围。

    Args:
        img: 输入图像
        sigma_low: 低频高斯 sigma
        sigma_high: 高频高斯 sigma
        gamma_low: 低频增益
        gamma_high: 高频增益

    Returns:
        同态滤波后的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img

    img_log = np.log1p(img_float)

    fft = np.fft.fft2(img_log)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = img_float.shape
    center_row, center_col = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

    gaussian_low = np.exp(-(dist_from_center ** 2) / (2 * sigma_low ** 2))
    gaussian_high = 1 - np.exp(-(dist_from_center ** 2) / (2 * sigma_high ** 2))

    filter_func = gamma_low * gaussian_low + gamma_high * gaussian_high

    fft_filtered = fft_shift * filter_func

    fft_ishift = np.fft.ifftshift(fft_filtered)
    img_filtered = np.fft.ifft2(fft_ishift)
    img_filtered = np.real(img_filtered)

    img_exp = np.expm1(img_filtered)

    if img.dtype == np.uint8:
        img_exp = np.clip(img_exp, 0, 255).astype(np.uint8)

    return img_exp


def despeckle_wavelet(img: np.ndarray,
                      wavelet: str = 'db4',
                      level: int = 2,
                      soft_threshold: bool = True) -> np.ndarray:
    """
    小波去噪。

    Args:
        img: 输入图像
        wavelet: 小波类型
        level: 分解层数
        soft_threshold: 是否使用软阈值

    Returns:
        小波去噪后的图像
    """
    if not HAS_PYWT:
        raise ImportError("pywt not installed. Install with: pip install PyWavelets")

    img_float = img.astype(np.float32) if img.dtype != np.float32 else img

    coeffs = pywt.wavedecn2(img_float, wavelet, level=level)

    cA = coeffs[0]
    cDs = coeffs[1:]

    threshold = np.median(np.abs(cDs[-1]['d'])) / 0.6745 * np.sqrt(2 * np.log(img_float.size))

    new_cDs = []
    for cD in cDs:
        new_cD = {}
        for key in cD:
            if soft_threshold:
                new_cD[key] = pywt.threshold(cD[key], threshold, mode='soft')
            else:
                new_cD[key] = pywt.threshold(cD[key], threshold, mode='hard')
        new_cDs.append(new_cD)

    new_coeffs = [cA] + new_cDs
    img_denoised = pywt.waverecn2(new_coeffs, wavelet)

    if img.dtype == np.uint8:
        img_denoised = np.clip(img_denoised, 0, 255).astype(np.uint8)

    return img_denoised


def despeckle_bilateral_advanced(img: np.ndarray,
                                 d: int = 9,
                                 sigma_color: float = 75,
                                 sigma_space: float = 75,
                                 iterations: int = 1) -> np.ndarray:
    """
    高级双边滤波（可迭代）。

    Args:
        img: 输入图像
        d: 滤波直径
        sigma_color: 颜色空间标准差
        sigma_space: 坐标空间标准差
        iterations: 迭代次数

    Returns:
        去噪后的图像
    """
    result = img.copy()
    for _ in range(iterations):
        result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
    return result


def enhance_unsharp_mask(img: np.ndarray,
                         kernel_size: int = 5,
                         sigma: float = 1.0,
                         amount: float = 1.0) -> np.ndarray:
    """
    非锐化掩蔽(Unsharp Mask)增强。

    Args:
        img: 输入图像
        kernel_size: 高斯核大小
        sigma: 高斯sigma
        amount: 增强量

    Returns:
        增强后的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img

    blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
    sharpened = img_float + amount * (img_float - blurred)

    if img.dtype == np.uint8:
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def normalize_minmax(img: np.ndarray) -> np.ndarray:
    """
    最小-最大归一化到[0, 255]。

    Args:
        img: 输入图像

    Returns:
        归一化后的图像
    """
    img_min = img.min()
    img_max = img.max()

    if img_max <= img_min:
        return np.zeros_like(img, dtype=np.uint8)

    normalized = ((img - img_min) / (img_max - img_min) * 255.0)

    if img.dtype == np.uint8:
        return normalized.astype(np.uint8)
    else:
        return normalized


def remove_water_column(img: np.ndarray,
                        nadir_threshold: float = 0.15,
                        min_column_width: int = 5,
                        fill_method: str = 'interpolate') -> np.ndarray:
    """
    侧扫声纳水柱去除与插值填充修复。

    检测声纳图像中央的 nadir 区域（水柱），
    并使用行向线性插值填充，避免信息丢失。

    Args:
        img: 输入声纳图像（灰度）
        nadir_threshold: nadir区域亮度阈值（相对于图像最大值的比例）
        min_column_width: 最小水柱宽度（像素），小于此值不处理
        fill_method: 填充方法 'interpolate'(线性插值) 或 'mirror'(镜像)

    Returns:
        水柱去除后的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img.copy()
    h, w = img_float.shape

    column_mask = _detect_water_column(img_float, nadir_threshold, min_column_width)
    if column_mask is None:
        return img

    result = img_float.copy()
    for row in range(h):
        mask_row = column_mask[row]
        if not np.any(mask_row):
            continue
        col_indices = np.where(mask_row)[0]
        if len(col_indices) == 0:
            continue

        left_col = col_indices[0] - 1
        right_col = col_indices[-1] + 1

        left_val = result[row, max(left_col, 0)]
        right_val = result[row, min(right_col, w - 1)]

        if fill_method == 'interpolate':
            for c in col_indices:
                t = (c - left_col) / max(right_col - left_col, 1)
                result[row, c] = left_val * (1 - t) + right_val * t
        elif fill_method == 'mirror':
            for c in col_indices:
                dist_from_left = c - col_indices[0]
                mirror_col = col_indices[0] - 1 - dist_from_left
                if 0 <= mirror_col < w:
                    result[row, c] = img_float[row, mirror_col]
                else:
                    t = (c - left_col) / max(right_col - left_col, 1)
                    result[row, c] = left_val * (1 - t) + right_val * t

    if img.dtype == np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def _detect_water_column(img: np.ndarray,
                         threshold: float,
                         min_width: int) -> Optional[np.ndarray]:
    h, w = img.shape
    if w < min_width * 3:
        return None

    col_means = np.mean(img, axis=0)
    max_val = np.max(col_means)
    if max_val < 1e-6:
        return None

    normalized_means = col_means / max_val

    center = w // 2
    search_half = w // 4

    low_mask = normalized_means < threshold
    center_low = low_mask[center - search_half:center + search_half]

    if not np.any(center_low):
        return None

    column_mask = np.zeros((h, w), dtype=bool)
    for row in range(h):
        row_center = img[row, center - search_half:center + search_half]
        row_max = np.max(img[row])
        if row_max < 1e-6:
            continue
        row_norm = row_center / row_max
        row_low = row_norm < threshold
        if np.sum(row_low) >= min_width:
            local_indices = np.where(row_low)[0]
            start = local_indices[0] + center - search_half
            end = local_indices[-1] + center - search_half + 1
            column_mask[row, start:end] = True

    if np.sum(column_mask) < min_width * h * 0.1:
        return None

    return column_mask


class AdvancedPreprocessor:
    """高级预处理器类。"""

    def __init__(self):
        pass

    def process(self,
                img: np.ndarray,
                enable_tvg: bool = True,
                enable_stripe_removal: bool = False,
                enable_homomorphic: bool = False,
                enable_wavelet: bool = False,
                enable_unsharp: bool = False,
                tvg_alpha: float = 2e-4,
                tvg_beta: float = 2.0,
                stripe_threshold: float = 3.0,
                homo_sigma_low: float = 60,
                homo_sigma_high: float = 5,
                homo_gamma_low: float = 0.4,
                homo_gamma_high: float = 1.6,
                wavelet_name: str = 'db4',
                wavelet_level: int = 2,
                unsharp_amount: float = 1.0) -> Tuple[np.ndarray, dict]:
        """
        应用高级预处理流水线。

        Returns:
            (processed_img, info_dict)
        """
        current_img = img.copy()
        info = {}

        if enable_tvg:
            current_img = apply_tvg(current_img, tvg_alpha, tvg_beta)
            info['tvg_applied'] = True

        if enable_stripe_removal:
            current_img = remove_stripe_noise_fft(current_img, stripe_threshold)
            info['stripe_removed'] = True

        if enable_homomorphic:
            current_img = homomorphic_filter(current_img, homo_sigma_low, homo_sigma_high, homo_gamma_low, homo_gamma_high)
            info['homomorphic_applied'] = True

        if enable_wavelet and HAS_PYWT:
            current_img = despeckle_wavelet(current_img, wavelet_name, wavelet_level)
            info['wavelet_applied'] = True

        if enable_unsharp:
            current_img = enhance_unsharp_mask(current_img, amount=unsharp_amount)
            info['unsharp_applied'] = True

        current_img = normalize_minmax(current_img)

        return current_img, info
