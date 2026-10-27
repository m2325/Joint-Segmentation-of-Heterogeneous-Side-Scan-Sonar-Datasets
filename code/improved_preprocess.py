#!/usr/bin/env python3
"""
tools/training/improved_preprocess.py

改进版声纳预处理模块 — 基于公开研究改进水柱处理。

改进策略（基于消融实验结果和公开资料）:
  1. 弃用 zero-fill 水柱去除（导致mIoU下降1.36%, Pipeline IoU下降2.64%）
  2. 改用 interpolate 填充（保持亮度连续性）
  3. 自适应检测 nadir 位置（不假设在中心）
  4. 保留水柱mask作为额外信息

参考文献:
  - SW-Net (Sensors 2026): 方向注意力处理nadir
  - SonarNet (Remote Sens 2025): 全局特征混合注意力
  - Bodine et al. (JGR 2024): 分割而非去除水柱
  - DSM-Seg (Remote Sens 2025): 物理先验语义引导
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ImprovedPreprocessConfig:
    """改进版预处理配置。"""
    enable_tvg: bool = False
    enable_homomorphic: bool = False
    enable_water_column: bool = False  # 改进版：默认使用interpolate填充
    enable_slant_correction: bool = False
    enable_despeckle: bool = False
    enable_enhance: bool = False

    # 水柱处理改进参数
    wc_fill_method: str = 'interpolate'  # 'interpolate' | 'mean' | 'zero' | 'preserve'
    wc_detection: str = 'adaptive'       # 'adaptive' | 'center'
    wc_preserve_mask: bool = False       # 是否生成水柱mask作为额外通道

    # 斜距/高度参数
    altitude_m: float = 5.0
    range_m: float = 100.0

    # v41.3改进: TVG截断阈值（避免远场暗区底噪异常放大）
    tvg_max_gain: float = 3.0            # TVG增益上限（相对于归一化基准1.0）
    tvg_alpha: float = 2e-4              # 衰减系数
    tvg_beta: float = 2.0               # 幂次系数
    # v41.4: TVG归一化方法 ('mean'|'median'|'p80')，median推荐用于异构数据集
    tvg_norm_method: str = 'median'

    # v41.3改进: 降噪方法选择（默认bilateral保边降噪替代Lee）
    despeckle_method: str = 'bilateral'  # 'bilateral' | 'lee' | 'frequency' | 'none'
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    # v41.4: 频域降噪衰减系数 (None=自适应，固定值=覆盖自适应)
    frequency_decay_factor: float = None

    # v41.4改进: 传感器自适应路由 (Domain Routing)
    # 不同传感器声学特性不同，需匹配不同TVG/降噪参数
    # 'klein3500': SubPipe Klein 3500 浅水亮信号
    # 'edgetech' : EdgeTech 深水增益缺失，需更大max_gain
    # 'generic'  : 通用预设
    sensor_profile: str = 'generic'

    @property
    def enabled_steps(self):
        steps = []
        if self.enable_tvg: steps.append(f"tvg(clip={self.tvg_max_gain},norm={self.tvg_norm_method})")
        if self.enable_homomorphic: steps.append("homomorphic")
        if self.enable_water_column: steps.append(f"wc({self.wc_fill_method})")
        if self.enable_slant_correction: steps.append("slant")
        if self.enable_despeckle: steps.append(f"denoise={self.despeckle_method}")
        if self.enable_enhance: steps.append("enhance")
        if self.sensor_profile != 'generic': steps.append(f"profile={self.sensor_profile}")
        return steps

    def apply_sensor_profile(self) -> None:
        """根据 sensor_profile 字段自动覆盖TVG/降噪参数（5数据集联合训练用）。"""
        profile = SENSOR_PROFILES.get(self.sensor_profile, {})
        if not profile:
            return
        if 'tvg_max_gain' in profile: self.tvg_max_gain = profile['tvg_max_gain']
        if 'tvg_alpha'     in profile: self.tvg_alpha     = profile['tvg_alpha']
        if 'tvg_beta'      in profile: self.tvg_beta      = profile['tvg_beta']
        if 'despeckle_method' in profile: self.despeckle_method = profile['despeckle_method']
        if 'frequency_decay_factor' in profile:
            self.frequency_decay_factor = profile['frequency_decay_factor']


# v41.4 传感器自适应路由表（应对5数据集异构性）
# 引用依据:
#   - SubPipe (Klein 3500, 900kHz HF): 浅水(2-30m)，高对比，max_gain=3.0足够
#   - AI4Shipwrecks (EdgeTech 2050): 深水(5-50m)，远场增益缺失，max_gain=4.0
SENSOR_PROFILES = {
    'klein3500': {
        'tvg_max_gain': 3.0, 'tvg_alpha': 2e-4, 'tvg_beta': 2.0,
        'despeckle_method': 'bilateral',
        'frequency_decay_factor': 0.3,  # 高对比，中等衰减
    },
    'edgetech': {
        'tvg_max_gain': 4.0, 'tvg_alpha': 3e-4, 'tvg_beta': 2.0,
        'despeckle_method': 'frequency',
        'frequency_decay_factor': 0.2,  # 中等对比，较弱衰减
    },
    'seafloor_smooth': {
        'tvg_max_gain': 3.0, 'tvg_alpha': 2e-4, 'tvg_beta': 2.0,
        'despeckle_method': 'frequency',
        'frequency_decay_factor': 0.1,  # 平滑底质，最小衰减保留沙波纹
    },
    'generic': {
        'tvg_max_gain': 3.0, 'tvg_alpha': 2e-4, 'tvg_beta': 2.0,
        'despeckle_method': 'bilateral',
        'frequency_decay_factor': None,  # 自适应
    },
}


def apply_tvg_clipped(img: np.ndarray,
                      alpha: float = 2e-4,
                      beta: float = 2.0,
                      max_gain: float = 3.0,
                      norm_method: str = 'median') -> np.ndarray:
    """带截断阈值的TVG增益补偿（v41.4鲁棒归一化升级）。

    解决远场暗区底噪被异常放大的问题：
    - 原版 apply_tvg 在远场(r大)时增益无上限，会把暗区噪声放大成"假信号"
    - 改进版对增益做clip: tvg = min(tvg, max_gain)
    - max_gain=3.0 表示远场最大增益不超过归一化基准的3倍

    v41.4归一化升级（应对5数据集联合训练）：
    - 'mean'  : 原版，tvg / mean(tvg) — 极端异常值会拉偏归一化基准
    - 'median': tvg / median(tvg) — 鲁棒，不受极端值影响（推荐）
    - 'p80'   : tvg / percentile(tvg, 80) — 保留更多远场增益余量

    Args:
        img: 输入声纳图像 (H, W) 或 (H, W, C)
        alpha: 衰减系数
        beta: 幂次系数
        max_gain: 增益上限（相对于归一化基准1.0）
        norm_method: 归一化方法 ('mean'|'median'|'p80')

    Returns:
        TVG补偿后（带截断）的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img.copy()

    if img_float.ndim == 2:
        h, w = img_float.shape
        r = np.arange(1, w + 1, dtype=np.float32)
        tvg = (r ** beta) * np.exp(alpha * r)
        # v41.4: 鲁棒归一化（防极端异常值拉偏）
        if norm_method == 'median':
            norm_base = np.median(tvg)
        elif norm_method == 'p80':
            norm_base = np.percentile(tvg, 80)
        else:  # 'mean'
            norm_base = tvg.mean()
        tvg = tvg / (norm_base + 1e-6)
        # 关键改进: 截断增益上限
        tvg = np.clip(tvg, 0, max_gain)
        return img_float * tvg[np.newaxis, :]
    else:
        # 多通道：对每个通道分别处理
        results = []
        for c in range(img_float.shape[2]):
            channel = apply_tvg_clipped(img_float[:, :, c], alpha, beta,
                                        max_gain, norm_method)
            results.append(channel)
        return np.stack(results, axis=2)


def despeckle_frequency(img: np.ndarray,
                         cutoff_ratio: float = 0.3,
                         preserve_edges: bool = True,
                         decay_factor: float = None) -> np.ndarray:
    """频域分解降噪（论文PhysD-Enhance的简化版，v41.4自适应衰减）。

    使用高斯径向滤波器分解为低频（结构）+高频（细节+噪声）：
    - 低频保留：图像整体结构
    - 高频处理：使用自适应阈值压制噪声，但保留边缘响应

    v41.4自适应衰减系数（应对5数据集异构性）：
    - decay_factor=None 时，根据图像全局方差自动选择衰减强度
    - 高对比目标图（沉船/管道）→ 衰减加强（0.4-0.5），强化保边
    - 低对比纹理图（SeafloorAI泥沙/沙波纹）→ 衰减减弱（0.1-0.2），保留微弱信号

    Args:
        img: 输入图像 (H, W)
        cutoff_ratio: 截止频率比例（0-1），越小保留越多低频
        preserve_edges: 是否保留高频边缘
        decay_factor: 弱响应衰减系数；None=自适应，固定值=覆盖自适应

    Returns:
        降噪后的图像
    """
    img_float = img.astype(np.float32) if img.dtype != np.float32 else img.copy()
    if img_float.ndim == 3:
        img_float = cv2.cvtColor(img_float.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = img_float.shape
    # 2D FFT
    fft = np.fft.fft2(img_float)
    fft_shift = np.fft.fftshift(fft)

    # 高斯径向滤波器（论文式13-15的简化版）
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    sigma = max_dist * cutoff_ratio
    gaussian_low = np.exp(-(dist ** 2) / (2 * sigma ** 2))

    # 分解
    low_freq = fft_shift * gaussian_low
    high_freq = fft_shift * (1 - gaussian_low)

    if preserve_edges:
        # 高频中保留强响应（边缘），仅压制弱响应（噪声）
        high_mag = np.abs(high_freq)
        threshold = np.mean(high_mag) + 2 * np.std(high_mag)
        high_mask = (high_mag > threshold).astype(np.float32)
        # v41.4: 自适应衰减系数
        if decay_factor is None:
            global_std = np.std(high_mag)
            # 低对比纹理图衰减小(0.1)，高对比目标图衰减大(0.5)
            decay = float(np.clip(0.1 + 0.5 * (global_std / 128.0), 0.1, 0.5))
        else:
            decay = float(decay_factor)
        # 强响应保留，弱响应衰减到 decay 比例
        high_freq = high_freq * (high_mask + decay * (1 - high_mask))

    # 重建
    filtered_fft = low_freq + high_freq
    filtered_fft = np.fft.ifftshift(filtered_fft)
    result = np.real(np.fft.ifft2(filtered_fft))

    if img.dtype == np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def detect_nadir_adaptive(img: np.ndarray, min_width: int = 10,
                          search_ratio: float = 0.2) -> Tuple[int, int]:
    """自适应检测nadir水柱位置。

    在图像中心区域搜索最亮列，适应不同图像的水柱位置。
    SubPipe图像特征：左舷近场(亮)|远场(暗)|nadir(亮)|远场(暗)|右舷近场(亮)
    因此搜索范围限制在中心附近(search_ratio=0.2表示中心±20%范围)。

    Args:
        img: 输入声纳图像 (H, W)
        min_width: 最小水柱宽度
        search_ratio: 中心搜索范围比例(0-0.5)

    Returns:
        (left, right): 水柱区域的左右边界
    """
    h, w = img.shape
    col_means = img.mean(axis=0).astype(np.float32)

    # 平滑列均值（kernel必须为奇数）
    kernel = max(1, w // 200)
    if kernel % 2 == 0:
        kernel = max(1, kernel + 1)  # 确保奇数
    col_means = cv2.GaussianBlur(col_means.reshape(1, -1), (kernel, 1), 0).ravel()

    # 限制搜索范围在中心附近
    center = w // 2
    search_half = int(w * search_ratio)
    search_left = max(0, center - search_half)
    search_right = min(w, center + search_half)

    # 在中心区域找最大亮度
    search_region = col_means[search_left:search_right]
    max_offset = np.argmax(search_region)
    max_idx = search_left + max_offset
    max_val = col_means[max_idx]

    # 从最大位置向两侧扩展，直到亮度下降到max_val的50%以下
    threshold = max_val * 0.5
    left = max_idx
    right = max_idx

    while left > search_left and col_means[left] > threshold:
        left -= 1
    while right < search_right - 1 and col_means[right] > threshold:
        right += 1

    # 确保最小宽度
    if right - left < min_width:
        left = max(0, max_idx - min_width // 2)
        right = min(w, max_idx + min_width // 2)

    return left, right


def remove_water_column_improved(img: np.ndarray,
                                  fill_method: str = 'interpolate',
                                  detection: str = 'adaptive',
                                  width_px: int = 50) -> np.ndarray:
    """改进版水柱去除。

    Args:
        img: 输入图像 (H, W)
        fill_method: 填充方法
            - 'interpolate': cv2.inpaint插值（推荐，保持亮度连续）
            - 'mean': 邻域均值填充
            - 'zero': 置零（原方法，不推荐）
        detection: 检测方法
            - 'adaptive': 自适应检测nadir位置
            - 'center': 固定中心（原方法）
        width_px: 固定模式下的水柱宽度

    Returns:
        处理后的图像
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    h, w = img_gray.shape

    # 检测水柱位置
    if detection == 'adaptive':
        left, right = detect_nadir_adaptive(img_gray, min_width=width_px)
    else:
        center = w // 2
        left = max(0, center - width_px // 2)
        right = min(w, center + width_px // 2)

    # 创建mask
    mask = np.zeros_like(img_gray)
    mask[:, left:right] = 255

    # 填充
    if fill_method == 'zero':
        result = img.copy()
        if result.ndim == 2:
            result[:, left:right] = 0
        else:
            result[:, left:right, :] = 0
        return result

    elif fill_method == 'mean':
        result = img.copy()
        for row in range(h):
            if result.ndim == 2:
                non_water = np.concatenate([
                    result[row, :left],
                    result[row, right:]
                ])
                if len(non_water) > 0:
                    result[row, left:right] = int(np.mean(non_water))
            else:
                for c in range(result.shape[2]):
                    non_water = np.concatenate([
                        result[row, :left, c],
                        result[row, right:, c]
                    ])
                    if len(non_water) > 0:
                        result[row, left:right, c] = int(np.mean(non_water))
        return result

    elif fill_method == 'interpolate':
        # 使用cv2.inpaint进行插值填充
        mask_uint8 = mask.astype(np.uint8)
        if img.ndim == 2:
            if img.dtype != np.uint8:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img_uint8 = img
            result = cv2.inpaint(img_uint8, mask_uint8,
                                  inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        else:
            # 对每个通道分别处理
            if img.dtype != np.uint8:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img_uint8 = img
            result = img_uint8.copy()
            for c in range(img_uint8.shape[2]):
                result[:, :, c] = cv2.inpaint(
                    img_uint8[:, :, c], mask_uint8,
                    inpaintRadius=5, flags=cv2.INPAINT_TELEA
                )
        return result

    else:
        return img.copy()


def generate_nadir_mask(img: np.ndarray, width_px: int = 50) -> np.ndarray:
    """生成nadir水柱区域的mask。

    不去除水柱，而是生成mask作为额外信息。
    模型可以学习如何利用nadir特征。

    Args:
        img: 输入图像 (H, W)
        width_px: 水柱宽度

    Returns:
        mask: (H, W) 二值mask，水柱区域为1，其他为0
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    left, right = detect_nadir_adaptive(img_gray, min_width=width_px)
    mask = np.zeros_like(img_gray, dtype=np.float32)
    mask[:, left:right] = 1.0
    return mask


def preprocess_sss_improved(img: np.ndarray,
                              config: ImprovedPreprocessConfig) -> np.ndarray:
    """改进版SSS预处理流水线。"""
    # v41.4: 应用传感器自适应路由
    config.apply_sensor_profile()
    work = img.copy().astype(np.float32)

    # Step 1: TVG增益补偿（v41.3截断阈值 + v41.4鲁棒归一化）
    if config.enable_tvg:
        work = apply_tvg_clipped(
            work,
            alpha=config.tvg_alpha,
            beta=config.tvg_beta,
            max_gain=config.tvg_max_gain,
            norm_method=config.tvg_norm_method,
        )

    # Step 2: 同态滤波
    if config.enable_homomorphic:
        from sonar_slam.preprocessing.advanced_preprocessing import homomorphic_filter
        work = homomorphic_filter(work)

    # Step 3: 水柱处理（改进版）
    if config.enable_water_column:
        if config.wc_fill_method == 'preserve':
            # 不处理，保留原始水柱
            pass
        else:
            work = remove_water_column_improved(
                work,
                fill_method=config.wc_fill_method,
                detection=config.wc_detection,
            )

    # Step 4: 斜距校正
    if config.enable_slant_correction:
        from sonar_slam.preprocessing.slant_range_correction import slant_to_ground_fast
        work = slant_to_ground_fast(
            work, altitude_m=config.altitude_m, range_m=config.range_m
        )

    # Step 5: 去噪（v41.3改进: 默认bilateral保边降噪，可选frequency频域分解）
    if config.enable_despeckle:
        if config.despeckle_method == 'bilateral':
            from sonar_slam.preprocessing.despeckle import despeckle_bilateral
            # bilateral需要uint8输入
            work_uint8 = np.clip(work, 0, 255).astype(np.uint8)
            if work_uint8.ndim == 2:
                work = despeckle_bilateral(
                    work_uint8,
                    d=config.bilateral_d,
                    sigma_color=config.bilateral_sigma_color,
                    sigma_space=config.bilateral_sigma_space,
                ).astype(np.float32)
            else:
                # 多通道：对每个通道分别bilateral
                channels = []
                for c in range(work_uint8.shape[2]):
                    ch = despeckle_bilateral(
                        work_uint8[:, :, c],
                        d=config.bilateral_d,
                        sigma_color=config.bilateral_sigma_color,
                        sigma_space=config.bilateral_sigma_space,
                    )
                    channels.append(ch.astype(np.float32))
                work = np.stack(channels, axis=2)
        elif config.despeckle_method == 'frequency':
            work = despeckle_frequency(
                work, cutoff_ratio=0.3, preserve_edges=True,
                decay_factor=config.frequency_decay_factor,
            )
        elif config.despeckle_method == 'lee':
            from sonar_slam.preprocessing.despeckle import despeckle_lee_fast
            work = despeckle_lee_fast(work)
        # 'none' 不做任何去噪

    # Step 6: 对比度增强
    if config.enable_enhance:
        from sonar_slam.preprocessing.contrast_enhance import (
            normalize_percentile, enhance_clahe
        )
        work = normalize_percentile(work)
        work = enhance_clahe(work)

    return work


def get_improved_ablation_configs() -> dict:
    """获取改进版消融实验配置。

    v41.3新增: 3组对比配置 A/B/C 对应论文核心实验
      - config_A_raw: Raw Data (无预处理，基线参考)
      - config_B_basic: 基础预处理 (斜距校正 + 简单TVG无截断)
      - config_C_physd_final: PhysD-Enhance最终版 (斜距+截断TVG+改进水柱+保边降噪+CLAHE)
    """
    return {
        # === 论文核心3组对比 (v41.3) ===
        # 配置A: Raw Data (无预处理)
        "config_A_raw": ImprovedPreprocessConfig(),

        # 配置B: 基础预处理 (斜距校正 + 简单TVG无截断 + 简单Lee去噪)
        "config_B_basic": ImprovedPreprocessConfig(
            enable_tvg=True,
            tvg_max_gain=1e6,  # 无截断（等价于原版apply_tvg）
            enable_slant_correction=True,
            enable_despeckle=True,
            despeckle_method='lee',
        ),

        # 配置C: PhysD-Enhance最终版 (斜距+截断TVG+改进水柱+保边降噪+CLAHE)
        "config_C_physd_final": ImprovedPreprocessConfig(
            enable_tvg=True,
            tvg_max_gain=3.0,                  # 截断阈值
            tvg_alpha=2e-4,
            tvg_beta=2.0,
            enable_water_column=True,
            wc_fill_method='interpolate',      # 改进水柱填充
            wc_detection='adaptive',
            enable_slant_correction=True,
            enable_despeckle=True,
            despeckle_method='bilateral',      # 保边降噪
            bilateral_d=9,
            bilateral_sigma_color=75,
            bilateral_sigma_space=75,
            enable_enhance=True,
        ),

        # 配置C-alt: PhysD-Enhance频域分解版 (使用frequency降噪替代bilateral)
        "config_C_physd_frequency": ImprovedPreprocessConfig(
            enable_tvg=True,
            tvg_max_gain=3.0,
            enable_water_column=True,
            wc_fill_method='interpolate',
            wc_detection='adaptive',
            enable_slant_correction=True,
            enable_despeckle=True,
            despeckle_method='frequency',      # 频域分解降噪（论文式13-15简化）
            enable_enhance=True,
        ),

        # === 原有改进版消融配置（保留） ===
        # 原始配置（用于对比）
        "baseline": ImprovedPreprocessConfig(),

        # 改进版水柱处理
        "+wc_interpolate": ImprovedPreprocessConfig(
            enable_water_column=True,
            wc_fill_method='interpolate',
            wc_detection='adaptive',
        ),
        "+wc_mean": ImprovedPreprocessConfig(
            enable_water_column=True,
            wc_fill_method='mean',
            wc_detection='adaptive',
        ),
        "+wc_preserve": ImprovedPreprocessConfig(
            enable_water_column=False,  # 不去除水柱
        ),

        # 完整改进版流水线（v41.3更新为bilateral降噪）
        "improved_full": ImprovedPreprocessConfig(
            enable_tvg=True,
            tvg_max_gain=3.0,
            enable_homomorphic=True,
            enable_water_column=True,
            wc_fill_method='interpolate',
            wc_detection='adaptive',
            enable_slant_correction=True,
            enable_despeckle=True,
            despeckle_method='bilateral',
            enable_enhance=True,
        ),
    }


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from subpipe_dataset import load_pbm
    from pathlib import Path

    img_dir = Path("datasets/SubPipe/SubPipe/DATA/Chunk0/SSS_HF_images/Image")
    pbm_files = sorted(img_dir.glob("*.pbm"))
    img = load_pbm(pbm_files[0])

    print(f"图像尺寸: {img.shape}")
    print(f"原始亮度: mean={img.mean():.2f}")

    # 测试自适应检测
    left, right = detect_nadir_adaptive(img)
    print(f"自适应nadir检测: x=[{left}, {right}] (宽度={right-left})")

    # 测试改进版水柱去除
    for method in ['zero', 'mean', 'interpolate']:
        result = remove_water_column_improved(img, fill_method=method)
        print(f"{method}填充后: mean={result.mean():.2f}")

    print("\n改进版配置:")
    for name, cfg in get_improved_ablation_configs().items():
        print(f"  {name}: {cfg.enabled_steps}")
