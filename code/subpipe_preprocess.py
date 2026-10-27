#!/usr/bin/env python3
"""
tools/training/subpipe_preprocess.py

SubPipe SSS图像训练时预处理模块 — 用于消融实验。

6步预处理流水线（与SLAM运行时pipeline一致，但轻量化用于训练）：
  1. TVG增益补偿        (apply_tvg)
  2. 同态滤波           (homomorphic_filter)
  3. 水柱去除           (remove_water_column)
  4. 斜距校正           (slant_to_ground_fast)
  5. 散斑去噪(Lee)      (despeckle_lee_fast)
  6. 对比度增强(CLAHE)  (enhance_clahe + normalize_percentile)

消融实验配置：
  通过 enable_* 参数控制每一步的开关，实现增量式消融。
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Make the bundled sonar_slam preprocessing package importable
_SONAR_PKG = Path(__file__).resolve().parent / "sonar_slam"
if str(_SONAR_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_SONAR_PKG.parent))


# ---------------------------------------------------------------------------
# 预处理配置
# ---------------------------------------------------------------------------

class PreprocessConfig:
    """预处理配置 — 控制每一步的开关。

    用于消融实验：逐步启用预处理步骤，对比分割性能。
    v41.3扩展: 支持TVG截断阈值、bilateral/frequency降噪方法选择。
    v41.4扩展: TVG鲁棒归一化、频域降噪自适应衰减、传感器自适应路由。
    """

    def __init__(self,
                 enable_tvg: bool = False,
                 enable_homomorphic: bool = False,
                 enable_water_column: bool = False,
                 enable_slant_correction: bool = False,
                 enable_despeckle: bool = False,
                 enable_enhance: bool = False,
                 # SubPipe SSS参数
                 altitude_m: float = 5.0,
                 range_m: float = 100.0,
                 # 水柱处理改进参数
                 wc_fill_method: str = 'zero',
                 wc_detection: str = 'center',
                 # v41.3新增: TVG截断阈值
                 tvg_max_gain: float = 1e6,     # 1e6=无截断(等价原版apply_tvg), 3.0=典型截断
                 tvg_alpha: float = 2e-4,
                 tvg_beta: float = 2.0,
                 # v41.3新增: 降噪方法选择
                 despeckle_method: str = 'lee',  # 'lee'|'bilateral'|'frequency'
                 bilateral_d: int = 9,
                 bilateral_sigma_color: float = 75,
                 bilateral_sigma_space: float = 75,
                 frequency_cutoff_ratio: float = 0.3,
                 # v41.4新增: 鲁棒归一化 + 自适应衰减 + 传感器路由
                 tvg_norm_method: str = 'mean',  # 'mean'|'median'|'p80'
                 frequency_decay_factor: float = None,  # None=自适应
                 sensor_profile: str = 'generic'):  # 'generic'|'klein3500'|'edgetech'|'seafloor_smooth'
        self.enable_tvg = enable_tvg
        self.enable_homomorphic = enable_homomorphic
        self.enable_water_column = enable_water_column
        self.enable_slant_correction = enable_slant_correction
        self.enable_despeckle = enable_despeckle
        self.enable_enhance = enable_enhance
        self.altitude_m = altitude_m
        self.range_m = range_m
        self.wc_fill_method = wc_fill_method  # 'zero'|'mean'|'interpolate'
        self.wc_detection = wc_detection      # 'center'|'adaptive'
        # v41.3字段
        self.tvg_max_gain = tvg_max_gain
        self.tvg_alpha = tvg_alpha
        self.tvg_beta = tvg_beta
        self.despeckle_method = despeckle_method
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.frequency_cutoff_ratio = frequency_cutoff_ratio
        # v41.4字段
        self.tvg_norm_method = tvg_norm_method
        self.frequency_decay_factor = frequency_decay_factor
        self.sensor_profile = sensor_profile
        # 应用传感器路由（覆盖默认参数）
        self._apply_sensor_profile()

    def _apply_sensor_profile(self) -> None:
        """根据 sensor_profile 自动覆盖TVG/降噪参数（5数据集联合训练用）。"""
        # 引用 SENSOR_PROFILES 表（同模块函数get_improved_ablation_configs内引用）
        try:
            from improved_preprocess import SENSOR_PROFILES
        except ImportError:
            return
        profile = SENSOR_PROFILES.get(self.sensor_profile, {})
        if not profile:
            return
        if 'tvg_max_gain'         in profile: self.tvg_max_gain         = profile['tvg_max_gain']
        if 'tvg_alpha'            in profile: self.tvg_alpha            = profile['tvg_alpha']
        if 'tvg_beta'             in profile: self.tvg_beta             = profile['tvg_beta']
        if 'despeckle_method'     in profile: self.despeckle_method     = profile['despeckle_method']
        if 'frequency_decay_factor' in profile:
            self.frequency_decay_factor = profile['frequency_decay_factor']

    @property
    def enabled_steps(self) -> list:
        """返回已启用的步骤名称列表。"""
        steps = []
        if self.enable_tvg:
            if self.tvg_max_gain < 100:
                norm = f",norm={self.tvg_norm_method}" if self.tvg_norm_method != 'mean' else ""
                steps.append(f"tvg(clip={self.tvg_max_gain}{norm})")
            else:
                steps.append("tvg")
        if self.enable_homomorphic:      steps.append("homomorphic")
        if self.enable_water_column:     steps.append(f"wc({self.wc_fill_method})")
        if self.enable_slant_correction: steps.append("slant_correction")
        if self.enable_despeckle:        steps.append(f"despeckle={self.despeckle_method}")
        if self.enable_enhance:          steps.append("enhance")
        if self.sensor_profile != 'generic':
            steps.append(f"profile={self.sensor_profile}")
        return steps

    @property
    def name(self) -> str:
        """返回配置名称（用于日志和输出目录）。"""
        if not self.enabled_steps:
            return "baseline"
        return "+".join(self.enabled_steps)

    def __repr__(self):
        return f"PreprocessConfig({self.name})"


# ---------------------------------------------------------------------------
# 预设配置（消融实验用）
# ---------------------------------------------------------------------------

def get_ablation_configs() -> Dict[str, PreprocessConfig]:
    """返回消融实验的预设配置。

    增量式消融：从基线(无预处理)开始，逐步添加每一步预处理。
    每个配置在前一个的基础上增加一步，共7个配置。
    """
    return {
        "baseline": PreprocessConfig(
            enable_tvg=False, enable_homomorphic=False,
            enable_water_column=False, enable_slant_correction=False,
            enable_despeckle=False, enable_enhance=False,
        ),
        "+tvg": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=False,
            enable_water_column=False, enable_slant_correction=False,
            enable_despeckle=False, enable_enhance=False,
        ),
        "+tvg+homomorphic": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=True,
            enable_water_column=False, enable_slant_correction=False,
            enable_despeckle=False, enable_enhance=False,
        ),
        "+tvg+homomorphic+wc": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=True,
            enable_water_column=True, enable_slant_correction=False,
            enable_despeckle=False, enable_enhance=False,
        ),
        "+tvg+homomorphic+wc+slant": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=True,
            enable_water_column=True, enable_slant_correction=True,
            enable_despeckle=False, enable_enhance=False,
        ),
        "+tvg+homomorphic+wc+slant+denoise": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=True,
            enable_water_column=True, enable_slant_correction=True,
            enable_despeckle=True, enable_enhance=False,
        ),
        "full_pipeline": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=True,
            enable_water_column=True, enable_slant_correction=True,
            enable_despeckle=True, enable_enhance=True,
        ),
        # 改进版水柱处理配置（基于消融实验结果）
        "+tvg+homomorphic+wc_interp": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=True,
            enable_water_column=True, enable_slant_correction=False,
            enable_despeckle=False, enable_enhance=False,
            wc_fill_method='interpolate', wc_detection='adaptive',
        ),
        "+tvg+homomorphic+wc_mean": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=True,
            enable_water_column=True, enable_slant_correction=False,
            enable_despeckle=False, enable_enhance=False,
            wc_fill_method='mean', wc_detection='adaptive',
        ),
        "improved_full": PreprocessConfig(
            enable_tvg=True, enable_homomorphic=True,
            enable_water_column=True, enable_slant_correction=True,
            enable_despeckle=True, enable_enhance=True,
            wc_fill_method='interpolate', wc_detection='adaptive',
        ),

        # === v41.3 论文核心3组对比 (PhysD-Enhance) ===
        # 配置A: Raw Data (无预处理，基线参考)
        "config_A_raw": PreprocessConfig(
            enable_tvg=False, enable_homomorphic=False,
            enable_water_column=False, enable_slant_correction=False,
            enable_despeckle=False, enable_enhance=False,
        ),

        # 配置B: 基础预处理 (斜距校正 + TVG + 去噪)
        # NOTE: 源码字面值 tvg_max_gain=1e6 / despeckle='lee' 会被 runtime
        # SENSOR_PROFILES['generic'] 覆盖为 tvg_max_gain=3.0 / despeckle='bilateral'.
        # 实际 runtime 行为: TVG clip=3.0, norm=mean; slant correction; bilateral despeckle.
        # AI4/MCM_fixed/SubPipe B/C 均走同一覆盖机制，跨数据集对比具有内部一致性.
        # See: improved_preprocess.py SENSOR_PROFILES.
        "config_B_basic": PreprocessConfig(
            enable_tvg=True, tvg_max_gain=1e6,             # 字面值=无截断; runtime被generic profile覆盖为3.0
            enable_slant_correction=True,
            enable_despeckle=True, despeckle_method='lee',  # 字面值=lee; runtime被generic profile覆盖为bilateral
        ),

        # 配置C: PhysD-Enhance最终版 (斜距+截断TVG+改进水柱+保边降噪+CLAHE)
        # v41.4: 默认使用median鲁棒归一化（应对5数据集联合训练）
        # NOTE: sensor_profile='klein3500' 在 runtime 会覆盖 tvg_alpha/beta 与 despeckle 参数
        # 实际 runtime 行为: TVG clip=3.0, norm=median; adaptive water-column; slant correction;
        # bilateral despeckle; CLAHE; klein3500 profile.
        "config_C_physd_final": PreprocessConfig(
            enable_tvg=True, tvg_max_gain=3.0,              # 截断阈值
            tvg_alpha=2e-4, tvg_beta=2.0,
            tvg_norm_method='median',                       # v41.4: 鲁棒归一化
            enable_water_column=True,
            wc_fill_method='interpolate', wc_detection='adaptive',
            enable_slant_correction=True,
            enable_despeckle=True, despeckle_method='bilateral',
            bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
            enable_enhance=True,
            sensor_profile='klein3500',                     # v41.4: SubPipe传感器
        ),

        # 配置C-alt: 频域分解降噪版 (使用frequency替代bilateral)
        # v41.4: 鲁棒归一化 + 自适应衰减系数(由sensor_profile决定)
        "config_C_physd_frequency": PreprocessConfig(
            enable_tvg=True, tvg_max_gain=3.0,
            tvg_norm_method='median',
            enable_water_column=True,
            wc_fill_method='interpolate', wc_detection='adaptive',
            enable_slant_correction=True,
            enable_despeckle=True, despeckle_method='frequency',
            frequency_cutoff_ratio=0.3,
            enable_enhance=True,
            sensor_profile='klein3500',
        ),
    }


# ---------------------------------------------------------------------------
# 预处理函数
# ---------------------------------------------------------------------------

def preprocess_sss(img: np.ndarray,
                   config: PreprocessConfig) -> np.ndarray:
    """对SubPipe SSS图像应用预处理。

    Args:
        img: 2D uint8灰度图 (H, W)，原始PBM加载后
        config: 预处理配置

    Returns:
        处理后的2D uint8灰度图
    """
    if config.name == "baseline":
        return img

    # 确保是2D灰度图
    if img.ndim == 3:
        img = img[:, :, 0]

    # 转float32处理
    work = img.astype(np.float32)

    # Step 1: TVG增益补偿（v41.3: 当tvg_max_gain<100时使用带截断版）
    if config.enable_tvg:
        try:
            if config.tvg_max_gain < 100:
                # v41.3改进: 带截断阈值的TVG + v41.4鲁棒归一化
                from improved_preprocess import apply_tvg_clipped
                work = apply_tvg_clipped(
                    work, alpha=config.tvg_alpha, beta=config.tvg_beta,
                    max_gain=config.tvg_max_gain,
                    norm_method=config.tvg_norm_method,
                )
            else:
                from sonar_slam.preprocessing.advanced_preprocessing import apply_tvg
                work = apply_tvg(work, alpha=config.tvg_alpha, beta=config.tvg_beta)
        except Exception:
            pass

    # Step 2: 同态滤波
    if config.enable_homomorphic:
        try:
            from sonar_slam.preprocessing.advanced_preprocessing import homomorphic_filter
            work = homomorphic_filter(work)
        except Exception:
            pass

    # Step 3: 水柱去除（v41.3: 当fill_method=interpolate+detection=adaptive时使用改进版）
    if config.enable_water_column:
        try:
            if (config.wc_fill_method == 'interpolate'
                    and config.wc_detection == 'adaptive'):
                # v41.3改进: 自适应检测nadir + 插值填充
                from improved_preprocess import remove_water_column_improved
                work = remove_water_column_improved(
                    work,
                    fill_method=config.wc_fill_method,
                    detection=config.wc_detection,
                    width_px=max(10, work.shape[1] // 100),
                )
            else:
                from sonar_slam.preprocessing.water_column_removal import remove_water_column
                work = remove_water_column(work, method=config.wc_detection,
                                           width_px=max(10, work.shape[1] // 100),
                                           fill_method=config.wc_fill_method)
        except Exception:
            pass

    # Step 4: 斜距校正
    if config.enable_slant_correction:
        try:
            from sonar_slam.preprocessing.slant_range_correction import slant_to_ground_fast
            # SubPipe HF: 5000宽, 约100m量程
            work = slant_to_ground_fast(work,
                                        altitude_m=config.altitude_m,
                                        range_m=config.range_m)
        except Exception:
            pass

    # Step 5: 散斑去噪（v41.3: 支持bilateral/frequency降噪方法）
    if config.enable_despeckle:
        try:
            if config.despeckle_method == 'bilateral':
                # v41.3改进: bilateral保边降噪
                from sonar_slam.preprocessing.despeckle import despeckle_bilateral
                work_u8 = np.clip(work, 0, 255).astype(np.uint8)
                if work_u8.ndim == 2:
                    work = despeckle_bilateral(
                        work_u8, d=config.bilateral_d,
                        sigma_color=config.bilateral_sigma_color,
                        sigma_space=config.bilateral_sigma_space,
                    ).astype(np.float32)
                else:
                    channels = []
                    for c in range(work_u8.shape[2]):
                        ch = despeckle_bilateral(
                            work_u8[:, :, c], d=config.bilateral_d,
                            sigma_color=config.bilateral_sigma_color,
                            sigma_space=config.bilateral_sigma_space,
                        )
                        channels.append(ch.astype(np.float32))
                    work = np.stack(channels, axis=2)
            elif config.despeckle_method == 'frequency':
                # v41.3改进: 频域分解降噪 + v41.4自适应衰减
                from improved_preprocess import despeckle_frequency
                work = despeckle_frequency(
                    work, cutoff_ratio=config.frequency_cutoff_ratio,
                    preserve_edges=True,
                    decay_factor=config.frequency_decay_factor,
                )
            else:
                # 原版Lee滤波
                from sonar_slam.preprocessing.despeckle import despeckle_lee_fast
                work = despeckle_lee_fast(work, window_size=7, num_looks=1)
        except Exception:
            pass

    # Step 6: 对比度增强(CLAHE + 百分位归一化)
    if config.enable_enhance:
        try:
            from sonar_slam.preprocessing.contrast_enhance import (
                normalize_percentile, enhance_clahe)
            work = normalize_percentile(work, low_pct=1.0, high_pct=99.0)
            work_u8 = np.clip(work, 0, 255).astype(np.uint8)
            work_u8 = enhance_clahe(work_u8, clip_limit=2.0,
                                    tile_grid_size=(8, 8))
            return work_u8
        except Exception:
            pass

    return np.clip(work, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from subpipe_dataset import load_pbm

    # 加载测试图像
    test_img = Path("datasets/SubPipe/SubPipe/DATA/Chunk0/SSS_HF_images/Image")
    pbm_files = sorted(test_img.glob("*.pbm"))
    if pbm_files:
        img = load_pbm(pbm_files[0])
        print(f"原始图像: shape={img.shape}, dtype={img.dtype}, "
              f"range=[{img.min()}, {img.max()}]")

        configs = get_ablation_configs()
        for name, cfg in configs.items():
            processed = preprocess_sss(img, cfg)
            print(f"  {name:40s}: shape={processed.shape}, "
                  f"range=[{processed.min()}, {processed.max()}], "
                  f"mean={processed.mean():.1f}")
