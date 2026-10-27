#!/usr/bin/env python3
"""
tools/training/sonar_preprocess_adapter.py

跨数据集预处理适配器 — 把 SubPipe 专用的预处理配置映射到各数据集。

设计原则 (A1b):
  - 不把其他数据集硬塞进 SubPipe 逻辑
  - 只复用确实通用的增强模块 (TVG / 同态滤波 / 去噪 / CLAHE 对比度增强)
  - 水柱去除 / 斜距校正是 SubPipe 几何假设 (5000宽瀑布图 + klein3500 传感器 +
    altitude/range 元数据), 对其它数据集未验证, 第一轮显式禁用
  - sensor_profile 对非 SubPipe 数据集一律用 'generic', 避免 klein3500 专属参数

三组论文配置 (raw / basic / physd) 在各数据集的映射:

  | dataset       | raw      | basic                    | physd                                   |
  |--------------|----------|--------------------------|-----------------------------------------|
  | subpipe      | 原配置   | config_B_basic (原样)     | config_C_physd_final (原样, 含WC/slant)  |
  | ai4shipwrecks | 全关     | TVG无截断 + Lee           | 截断TVG + bilateral + CLAHE (无WC/slant)  |
  | mcm          | 全关     | TVG无截断 + Lee           | 截断TVG + bilateral + CLAHE (无WC/slant)  |

注意: AI4Shipwrecks 图像为 1728(宽)×可变高(~5579) 竖向, 与 SubPipe 5000×500
横向瀑布图方向相反; 水柱/斜距几何假设不适用, 故禁用.

Paper1 §0.A.2 单操作消融策略 (Task #7):
  审稿人可能质疑 CLAHE / Lee / bilateral 在 PhysD 内未单独消融, 故新增 4 个
  独立单操作策略, 与 raw/basic/physd 共同构成 7 组对比 (Table 1 行):
    - clahe_only     : 仅 CLAHE (Zuiderveld 1994), clipLimit=2.0, tile=(8,8)
    - lee_only        : 仅 Lee 滤波 (Lee 1980), 7x7 窗口, num_looks=1
    - bilateral_only  : 仅 bilateral 滤波 (Tomasi 1998), d=9, sigma=75
    - hist_eq         : 标准直方图均衡化 (Gonzalez & Woods 2002)
  单操作策略与数据集无关 (无几何步骤), 直接作用于灰度图; raw/basic/physd 仍走
  原 PreprocessConfig 流水线以保证向后兼容.

用法:
  from sonar_preprocess_adapter import preprocess_for_dataset, build_config, build_preprocess_adapter
  out = preprocess_for_dataset(img, dataset="ai4shipwrecks", config_name="physd")
  fn = build_preprocess_adapter("clahe_only")      # 单操作, 无需 dataset
  out = fn(img)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

# 让本模块可独立 import subpipe_preprocess
_TOOLS_TRAINING = Path(__file__).resolve().parent
if str(_TOOLS_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TOOLS_TRAINING))

from subpipe_preprocess import PreprocessConfig, preprocess_sss, get_ablation_configs  # noqa: E402

# 支持的数据集
SUPPORTED_DATASETS = ("subpipe", "ai4shipwrecks", "mcm")

# 数据集别名映射 (论文术语 → 正式数据集名)
# 用于 FCN/PSPNet 等通过 build_preprocess_adapter 入口调用时, 自动归一化别名
DATASET_ALIASES = {
    "ai4": "ai4shipwrecks",
}

# 支持的配置名 (论文三组, 向后兼容)
SUPPORTED_CONFIGS = ("raw", "basic", "physd")

# Paper1 §0.A.2 单操作消融策略 (Task #7):
# 审稿人可能质疑 CLAHE/Lee/bilateral 在 PhysD 内未单独消融, 故新增 4 个独立单操作
# 策略, 与 raw/basic/physd 共同构成 7 组对比 (Table 1 行).
SUPPORTED_SINGLE_OPS = ("clahe_only", "lee_only", "bilateral_only", "hist_eq")

# V6 物理启发算法:
#   - piaonet    : PIAE-Net 物理强度锚定增强 (SRAD + 物理残差), 解决传统增强 F1=0
#   - taad_sspm  : TAAD-SSPM 拓扑感知各向异性扩散 (Frangi + CED), 解决 SSPM 抹杀 C2
#   - srad       : 裸 SRAD (仅各向异性扩散, 无物理锚定), PIAE 的公开锚点消融
#                 对照链 Raw → SRAD → PIAE, 防"提升来自 SRAD 而非物理锚定"质疑
# 三者均为数据集无关的纯图像增强, 与单操作策略同接口.
V6_ALGO_OPS = ("piaonet", "taad_sspm", "srad")

# B4-B6 策略:
# 定义冻结:
#   tvg_only       : Raw → TVG (无截断) → min-max normalization
#   tvg_truncated  : Raw → min(TVG gain, g_max=3.0) → min-max normalization
#   all_physd      : metadata-conditional pipeline (需 dataset 参数):
#     AI4:     normalization + bilateral + CLAHE (TVG 默认禁用, range axis 不可靠)
#     MCM:     normalization + truncated TVG + bilateral + CLAHE
#     SubPipe: water-column removal + slant-range + truncated TVG + bilateral + CLAHE
B456_OPS = ("tvg_only", "tvg_truncated", "tvg_truncated_bilateral", "all_physd")

# 完整 2^3 析因消融新增策略:
# 用于补充 8-cell 析因矩阵的缺失 4 cell, 与已有 A0/A1/A2/A3 合并形成完整 2^3 设计.
# 三因素: R=range compensation (TVG截断), F=bilateral filter, C=CLAHE contrast enhancement
#   C2 = bilateral_only       : R=0, F=1, C=0 (单操作, 已存在于 SUPPORTED_SINGLE_OPS)
#   C3 = clahe_only           : R=0, F=0, C=1 (单操作, 已存在于 SUPPORTED_SINGLE_OPS)
#   C5 = tvg_truncated_clahe  : R=1, F=0, C=1 (新增)
#   C6 = bilateral_clahe      : R=0, F=1, C=1 (新增)
# 操作顺序固定为 R → F → C, 缺组件跳过不改变顺序.
# 参数严格匹配 physd (TVG: g_max=3.0, alpha=2e-4, beta=2.0, norm=median;
#   bilateral: d=9, sigma_color=75, sigma_space=75; CLAHE: clipLimit=2.0, tile=(8,8))
FACTORIAL_NEW_OPS = ("tvg_truncated_clahe", "bilateral_clahe")

# 全部支持的策略名 (工厂函数 / 训练脚本 choices 用)
SUPPORTED_STRATEGIES = (SUPPORTED_CONFIGS + SUPPORTED_SINGLE_OPS
                        + V6_ALGO_OPS + B456_OPS + FACTORIAL_NEW_OPS)


# ---------------------------------------------------------------------------
# 单操作预处理函数
# ---------------------------------------------------------------------------
# 设计: 每个函数仅做 1 个操作, 不叠加 TVG/归一化之外的步骤, 以保证纯净消融.
# 引用: Zuiderveld 1994 (CLAHE), Lee 1980 (Refined Filter),
#       Tomasi 1998 (Bilateral), Gonzalez & Woods 2002 (Histogram Equalization).
# ---------------------------------------------------------------------------

def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """转 2D uint8 灰度图 (单操作策略的统一前置)."""
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def clahe_only(img: np.ndarray,
               clip_limit: float = 2.0,
               tile_grid_size=(8, 8)) -> np.ndarray:
    """仅 CLAHE 对比度增强 (Zuiderveld 1994).

    应用于百分位归一化后的 uint8 图像 (1%-99% 拉伸到 0-255),
    随后 cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).
    """
    import cv2
    from sonar_slam.preprocessing.contrast_enhance import (
        normalize_percentile, enhance_clahe)
    work = _to_uint8_gray(img).astype(np.float32)
    work = normalize_percentile(work, low_pct=1.0, high_pct=99.0)
    work_u8 = np.clip(work, 0, 255).astype(np.uint8) if work.dtype != np.uint8 else work
    return enhance_clahe(work_u8, clip_limit=clip_limit, tile_grid_size=tile_grid_size)


def lee_only(img: np.ndarray,
             window_size: int = 7,
             num_looks: int = 1) -> np.ndarray:
    """仅 Lee 滤波去散斑 (Lee 1980 Refined Filter).

    公式: y = m + (v - noise_var) / v * (x - m), 窗口 7x7, num_looks=1.
    noise_var = v / (1 + num_looks); weight = max(0, 1 - noise_var / v).
    """
    from sonar_slam.preprocessing.despeckle import despeckle_lee_fast
    work = _to_uint8_gray(img)
    return despeckle_lee_fast(work, window_size=window_size, num_looks=num_looks)


def bilateral_only(img: np.ndarray,
                   d: int = 9,
                   sigma_color: float = 75.0,
                   sigma_space: float = 75.0) -> np.ndarray:
    """仅 bilateral 保边滤波 (Tomasi & Manduchi 1998).

    cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75).
    """
    import cv2
    work = _to_uint8_gray(img)
    return cv2.bilateralFilter(work, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def hist_eq(img: np.ndarray) -> np.ndarray:
    """标准全局直方图均衡化 (Gonzalez & Woods 2002).

    应用于百分位归一化后的 uint8 图像, cv2.equalizeHist.
    """
    import cv2
    from sonar_slam.preprocessing.contrast_enhance import normalize_percentile
    work = _to_uint8_gray(img).astype(np.float32)
    work = normalize_percentile(work, low_pct=1.0, high_pct=99.0)
    work_u8 = np.clip(work, 0, 255).astype(np.uint8) if work.dtype != np.uint8 else work
    return cv2.equalizeHist(work_u8)


# 单操作策略名 → 函数 的查表 (build_preprocess_adapter / preprocess_for_dataset 共用)
_SINGLE_OP_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "clahe_only": clahe_only,
    "lee_only": lee_only,
    "bilateral_only": bilateral_only,
    "hist_eq": hist_eq,
}

# V6 物理启发算法函数查表 (延迟导入避免循环依赖)
def _get_v6_funcs() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """V6 算法函数查表 (延迟导入)."""
    from v6_algorithms import piaenet_enhance, taad_sspm_enhance, srad_enhance
    return {
        "piaonet": piaenet_enhance,
        "taad_sspm": taad_sspm_enhance,
        "srad": srad_enhance,
    }

# ---------------------------------------------------------------------------
# B4-B6 预处理策略
# ---------------------------------------------------------------------------

def _min_max_normalize_uint8(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 255] uint8."""
    work = _to_uint8_gray(img).astype(np.float64)
    lo, hi = work.min(), work.max()
    if hi - lo < 1e-8:
        return np.zeros_like(work, dtype=np.uint8)
    work = (work - lo) / (hi - lo) * 255.0
    return np.clip(work, 0, 255).astype(np.uint8)


def tvg_only(img: np.ndarray) -> np.ndarray:
    """TVG-only: Raw -> TVG (无截断) -> min-max normalization.

    仅做 range-dependent gain compensation (无截断), 最后归一化到 [0, 255].
    用于 Phase 3 B4 消融: 分离 TVG 的独立贡献.
    """
    cfg = PreprocessConfig(
        enable_tvg=True, tvg_max_gain=1e6,
        sensor_profile='generic',
    )
    out = preprocess_sss(img, cfg)
    return _min_max_normalize_uint8(out)


def tvg_truncated(img: np.ndarray) -> np.ndarray:
    """Truncated-TVG-only: Raw -> min(TVG gain, 3.0) -> min-max normalization.

    仅做 gain-limited TVG (g_max=3.0), 最后归一化到 [0, 255].
    用于 Phase 3 B4 消融: 对比截断 vs 无截断 TVG 的效果.
    顺序消融 A1 阶段 (Range compensation only).
    """
    cfg = PreprocessConfig(
        enable_tvg=True, tvg_max_gain=3.0,
        tvg_alpha=2e-4, tvg_beta=2.0,
        tvg_norm_method='median',
        sensor_profile='generic',
    )
    out = preprocess_sss(img, cfg)
    return _min_max_normalize_uint8(out)


def tvg_truncated_bilateral(img: np.ndarray) -> np.ndarray:
    """Truncated-TVG + bilateral filter (NO CLAHE) — sequential ablation A2.

    顺序消融 A2 阶段: Range compensation + filtering.
    Config 匹配 MCM physd 的 TVG 和 bilateral 设置, 但 enable_enhance=False.
    用于隔离 CLAHE 对比度增强的增量贡献 (A3 - A2).

    Pipeline: Raw -> truncated TVG (g_max=3.0) -> bilateral filter -> normalize.
    对比 A1 (tvg_truncated): A2-A1 = 滤波增量贡献.
    对比 A3 (physd):          A3-A2 = 对比度增量贡献.
    """
    cfg = PreprocessConfig(
        enable_tvg=True, tvg_max_gain=3.0,
        tvg_alpha=2e-4, tvg_beta=2.0, tvg_norm_method='median',
        enable_water_column=False, enable_slant_correction=False,
        enable_despeckle=True, despeckle_method='bilateral',
        bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
        enable_enhance=False,  # NO CLAHE — key difference from physd
        sensor_profile='generic',
    )
    out = preprocess_sss(img, cfg)
    return _min_max_normalize_uint8(out)


def tvg_truncated_clahe(img: np.ndarray) -> np.ndarray:
    """Truncated-TVG + CLAHE (NO bilateral) — factorial cell C5 (R=1, F=0, C=1).

    完整 2^3 析因消融新增策略.
    保留 TVG 截断 + CLAHE 对比度增强, 跳过 bilateral 滤波, 用于估计:
      - CLAHE 在 TVG 已启用条件下的条件增量
      - TVG × CLAHE 二阶交互 (与 C3=clahe_only, A1=tvg_truncated, A3=physd 联合)

    Pipeline: Raw -> truncated TVG (g_max=3.0) -> CLAHE (clip=2.0, tile=8x8) -> normalize.
    操作顺序: R → F → C, F 缺失跳过, 不改变 R 与 C 的相对顺序.
    参数严格匹配 physd (TVG: g_max=3.0, alpha=2e-4, beta=2.0, norm=median;
      CLAHE: clipLimit=2.0, tile=(8,8)).
    """
    cfg = PreprocessConfig(
        enable_tvg=True, tvg_max_gain=3.0,
        tvg_alpha=2e-4, tvg_beta=2.0, tvg_norm_method='median',
        enable_water_column=False, enable_slant_correction=False,
        enable_despeckle=False,  # NO bilateral — key difference from A2/A3
        enable_enhance=True,  # CLAHE enabled
        sensor_profile='generic',
    )
    out = preprocess_sss(img, cfg)
    return _min_max_normalize_uint8(out)


def bilateral_clahe(img: np.ndarray) -> np.ndarray:
    """Bilateral + CLAHE (NO TVG) — factorial cell C6 (R=0, F=1, C=1).

    完整 2^3 析因消融新增策略.
    保留 bilateral 滤波 + CLAHE 对比度增强, 跳过 TVG 截断, 用于估计:
      - bilateral × CLAHE 二阶交互 (与 C2=bilateral_only, C3=clahe_only, A3=physd 联合)
      - CLAHE 在 bilateral 已启用条件下的条件增量

    Pipeline: Raw -> bilateral filter (d=9, sigma=75) -> CLAHE (clip=2.0, tile=8x8) -> normalize.
    操作顺序: R → F → C, R 缺失跳过, 不改变 F 与 C 的相对顺序.
    参数严格匹配 physd (bilateral: d=9, sigma_color=75, sigma_space=75;
      CLAHE: clipLimit=2.0, tile=(8,8)).
    """
    cfg = PreprocessConfig(
        enable_tvg=False,  # NO TVG — key difference from A1/A2/A3
        enable_water_column=False, enable_slant_correction=False,
        enable_despeckle=True, despeckle_method='bilateral',
        bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
        enable_enhance=True,  # CLAHE enabled
        sensor_profile='generic',
    )
    out = preprocess_sss(img, cfg)
    return _min_max_normalize_uint8(out)


def all_physd(img: np.ndarray, dataset: str) -> np.ndarray:
    """all-PhysD: metadata-conditional physical pipeline.

    Dataset-specific operations (frozen 2026-07-14):
      AI4:     normalization + bilateral + CLAHE (TVG 默认禁用, range axis 不可靠)
      MCM:     normalization + truncated TVG + bilateral + CLAHE
      SubPipe: water-column removal + slant-range + truncated TVG + bilateral + CLAHE
    """
    dataset = dataset.lower()
    if dataset == 'ai4shipwrecks':
        cfg = PreprocessConfig(
            enable_tvg=False,
            enable_water_column=False, enable_slant_correction=False,
            enable_despeckle=True, despeckle_method='bilateral',
            bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
            enable_enhance=True, sensor_profile='generic',
        )
    elif dataset == 'mcm':
        cfg = PreprocessConfig(
            enable_tvg=True, tvg_max_gain=3.0,
            tvg_alpha=2e-4, tvg_beta=2.0, tvg_norm_method='median',
            enable_water_column=False, enable_slant_correction=False,
            enable_despeckle=True, despeckle_method='bilateral',
            bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
            enable_enhance=True, sensor_profile='generic',
        )
    elif dataset == 'subpipe':
        cfg = PreprocessConfig(
            enable_tvg=True, tvg_max_gain=3.0,
            tvg_alpha=2e-4, tvg_beta=2.0, tvg_norm_method='median',
            enable_water_column=True,
            wc_fill_method='interpolate', wc_detection='adaptive',
            enable_slant_correction=True,
            enable_despeckle=True, despeckle_method='bilateral',
            bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
            enable_enhance=True, sensor_profile='klein3500',
        )
    else:
        raise ValueError(f"all_physd 不支持数据集: {dataset}")
    return preprocess_sss(img, cfg)


# tvg_only / tvg_truncated / tvg_truncated_bilateral / tvg_truncated_clahe / bilateral_clahe:
# 数据集无关 (同单操作策略接口)
# all_physd: 数据集相关 (需 dataset 参数)
_B456_DATASET_INDEPENDENT = {
    "tvg_only": tvg_only,
    "tvg_truncated": tvg_truncated,
    "tvg_truncated_bilateral": tvg_truncated_bilateral,
    "tvg_truncated_clahe": tvg_truncated_clahe,
    "bilateral_clahe": bilateral_clahe,
}


# AI4/MCM 的通用增强配置 (复用通用模块, 禁用 SubPipe 几何步骤)
_GENERIC_BASIC = PreprocessConfig(
    enable_tvg=True, tvg_max_gain=1e6,          # 无截断 TVG
    enable_slant_correction=False,              # SubPipe 几何假设, 禁用
    enable_water_column=False,                 # SubPipe 几何假设, 禁用
    enable_despeckle=True, despeckle_method='lee',
    sensor_profile='generic',                  # 非 klein3500
)

_GENERIC_PHYSD = PreprocessConfig(
    enable_tvg=True, tvg_max_gain=3.0,          # 截断 TVG
    tvg_alpha=2e-4, tvg_beta=2.0,
    tvg_norm_method='median',                   # v41.4 鲁棒归一化
    enable_water_column=False,                 # SubPipe 几何假设, 禁用
    enable_slant_correction=False,             # SubPipe 几何假设, 禁用
    enable_despeckle=True, despeckle_method='bilateral',
    bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
    enable_enhance=True,
    sensor_profile='generic',
)


def build_config(dataset: str, config_name: str) -> PreprocessConfig:
    """构建数据集+配置名对应的 PreprocessConfig.

    Args:
        dataset: 'subpipe' | 'ai4shipwrecks' | 'mcm'
        config_name: 'raw' | 'basic' | 'physd'
            (单操作策略 clahe_only/lee_only/bilateral_only/hist_eq 不走
             PreprocessConfig 流水线, 请用 build_preprocess_adapter)

    Returns:
        PreprocessConfig (可复用于 preprocess_sss)
    """
    dataset = dataset.lower()
    config_name = config_name.lower()
    dataset = DATASET_ALIASES.get(dataset, dataset)  # 归一化别名 (ai4 → ai4shipwrecks)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"不支持的数据集: {dataset} (支持: {SUPPORTED_DATASETS})")
    if config_name in SUPPORTED_SINGLE_OPS:
        raise ValueError(
            f"单操作策略 '{config_name}' 不走 PreprocessConfig 流水线; "
            f"请用 build_preprocess_adapter('{config_name}') 或 "
            f"preprocess_for_dataset(img, dataset, '{config_name}').")
    if config_name not in SUPPORTED_CONFIGS:
        raise ValueError(f"不支持的配置: {config_name} (支持: {SUPPORTED_STRATEGIES})")

    # raw: 任何数据集都是全关 (基线)
    if config_name == "raw":
        return PreprocessConfig(sensor_profile='generic')

    if dataset == "subpipe":
        # SubPipe: 沿用原配置 (含 WC/slant/klein3500), 向后兼容
        ablation = get_ablation_configs()
        key = "config_B_basic" if config_name == "basic" else "config_C_physd_final"
        return ablation[key]

    # ai4shipwrecks / mcm: 通用增强, 禁用 SubPipe 几何步骤
    return _GENERIC_BASIC if config_name == "basic" else _GENERIC_PHYSD


def build_preprocess_adapter(strategy_name: str,
                             dataset: Optional[str] = None) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """工厂函数: 根据策略名返回预处理可调用对象 f(img) -> img (uint8 2D).

    Paper1 §0.A.2 (Task #7): 支持 7 种策略, 用于单独消融:
      raw / basic / physd / clahe_only / lee_only / bilateral_only / hist_eq.

    Args:
        strategy_name: 策略名 (见 SUPPORTED_STRATEGIES)
        dataset: 'subpipe' | 'ai4shipwrecks' | 'mcm' | None
            - raw/basic/physd 必须提供 dataset (走 PreprocessConfig 流水线)
            - clahe_only/lee_only/bilateral_only/hist_eq 与数据集无关, dataset 可省略

    Returns:
        预处理函数 f(img)->img; raw 返回 None (表示无预处理, 与训练脚本约定一致)

    Raises:
        ValueError: 策略名未知, 或 raw/basic/physd 缺 dataset 参数
    """
    strategy_name = strategy_name.lower()
    if strategy_name not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"未知预处理策略: {strategy_name} (支持: {SUPPORTED_STRATEGIES})")

    # raw: 无预处理 (与 train_*_segformer.py build_preprocess_fn 约定一致)
    if strategy_name == "raw":
        return None

    # 单操作策略: 与数据集无关, 直接返回函数
    if strategy_name in _SINGLE_OP_FUNCS:
        return _SINGLE_OP_FUNCS[strategy_name]

    # V6 物理启发算法: 与数据集无关, 延迟导入后返回函数
    if strategy_name in V6_ALGO_OPS:
        return _get_v6_funcs()[strategy_name]

    # B456 数据集无关策略: tvg_only / tvg_truncated
    if strategy_name in _B456_DATASET_INDEPENDENT:
        return _B456_DATASET_INDEPENDENT[strategy_name]

    # raw/basic/physd/all_physd: 需要 dataset 参数
    if dataset is None:
        raise ValueError(
            f"策略 '{strategy_name}' 需要 dataset 参数 (subpipe/ai4shipwrecks/mcm)")
    dataset = dataset.lower()
    dataset = DATASET_ALIASES.get(dataset, dataset)  # 归一化别名 (ai4 → ai4shipwrecks)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"不支持的数据集: {dataset} (支持: {SUPPORTED_DATASETS})")

    def _fn(img: np.ndarray) -> np.ndarray:
        return preprocess_for_dataset(img, dataset=dataset, config_name=strategy_name)
    return _fn


def preprocess_for_dataset(img: np.ndarray,
                           dataset: str,
                           config_name: str) -> np.ndarray:
    """对图像应用数据集+配置对应的预处理.

    Args:
        img: 2D uint8 灰度图 (H, W), 或 3D (H, W, C) — 内部取首通道
        dataset: 'subpipe' | 'ai4shipwrecks' | 'mcm'
        config_name: 'raw' | 'basic' | 'physd' |
                    'clahe_only' | 'lee_only' | 'bilateral_only' | 'hist_eq'
            (单操作策略与数据集无关, 但仍接受 dataset 参数以保持调用接口统一)

    Returns:
        处理后的 2D uint8 灰度图 (H, W), 尺寸与输入一致
    """
    config_name = config_name.lower()
    # 单操作策略: 直接走对应函数 (与数据集无关)
    if config_name in _SINGLE_OP_FUNCS:
        return _SINGLE_OP_FUNCS[config_name](img)
    # V6 物理启发算法: 延迟导入后调用
    if config_name in V6_ALGO_OPS:
        return _get_v6_funcs()[config_name](img)
    # B456 数据集无关策略: tvg_only / tvg_truncated
    if config_name in _B456_DATASET_INDEPENDENT:
        return _B456_DATASET_INDEPENDENT[config_name](img)
    # B456 数据集相关策略: all_physd
    if config_name == "all_physd":
        return all_physd(img, dataset)
    # raw/basic/physd: 走原 PreprocessConfig 流水线 (向后兼容)
    cfg = build_config(dataset, config_name)
    return preprocess_sss(img, cfg)


def config_summary(dataset: str) -> Dict[str, dict]:
    """返回某数据集全部策略的摘要 (用于 config.json 记录).

    raw/basic/physd 走 PreprocessConfig 流水线 (含 enabled_steps/WC/slant/profile);
    单操作策略 (clahe_only/lee_only/bilateral_only/hist_eq) 与数据集无关,
    enabled_steps 标注其单一操作 + 引用.
    B456 顺序消融策略 (tvg_only/tvg_truncated/tvg_truncated_bilateral/all_physd)
    与数据集无关, enabled_steps 标注其顺序消融阶段.
    """
    out = {}
    # raw/basic/physd: PreprocessConfig 流水线摘要
    for name in SUPPORTED_CONFIGS:
        cfg = build_config(dataset, name)
        out[name] = {
            "enabled_steps": cfg.enabled_steps,
            "enable_water_column": cfg.enable_water_column,
            "enable_slant_correction": cfg.enable_slant_correction,
            "sensor_profile": cfg.sensor_profile,
        }
    # 单操作策略: 摘要标注单一操作与引用
    single_op_meta = {
        "clahe_only":     ("clahe",        "Zuiderveld 1994"),
        "lee_only":       ("lee_filter",   "Lee 1980"),
        "bilateral_only": ("bilateral",    "Tomasi 1998"),
        "hist_eq":        ("hist_eq",      "Gonzalez & Woods 2002"),
    }
    for name in SUPPORTED_SINGLE_OPS:
        step, ref = single_op_meta[name]
        out[name] = {
            "enabled_steps": [step],
            "enable_water_column": False,
            "enable_slant_correction": False,
            "sensor_profile": "generic",
            "reference": ref,
            "dataset_independent": True,
        }
    # B456 顺序消融策略: 数据集无关 (顺序消融链 A0/A1/A2/A3)
    b456_meta = {
        "tvg_only":               ("tvg",                  "A0 baseline TVG (no truncation)"),
        "tvg_truncated":          ("tvg_truncated",        "A1 Range: min(TVG, g_max=3.0)"),
        "tvg_truncated_bilateral": ("tvg_truncated+bilateral", "A2 Range+Filter: A1 + bilateral d=9"),
        "all_physd":              ("tvg_truncated+bilateral+clahe", "A3 Range+Filter+Contrast: A2 + CLAHE clip=2.0"),
    }
    for name in B456_OPS:
        step, ref = b456_meta[name]
        out[name] = {
            "enabled_steps": [step],
            "enable_water_column": False,
            "enable_slant_correction": False,
            "sensor_profile": "generic",
            "reference": ref,
            "dataset_independent": True,
            "ablation_stage": name,
        }
    # 完整 2^3 析因消融新增策略:
    #   C5 = tvg_truncated_clahe  (R=1, F=0, C=1)
    #   C6 = bilateral_clahe      (R=0, F=1, C=1)
    factorial_meta = {
        "tvg_truncated_clahe": ("tvg_truncated+clahe",   "C5 Range+Contrast: TVG + CLAHE (no bilateral)"),
        "bilateral_clahe":      ("bilateral+clahe",       "C6 Filter+Contrast: bilateral + CLAHE (no TVG)"),
    }
    for name in FACTORIAL_NEW_OPS:
        step, ref = factorial_meta[name]
        out[name] = {
            "enabled_steps": [step],
            "enable_water_column": False,
            "enable_slant_correction": False,
            "sensor_profile": "generic",
            "reference": ref,
            "dataset_independent": True,
            "factorial_design": "2x2x2",
            "factor_R": 1 if "tvg_truncated" in step else 0,
            "factor_F": 1 if "bilateral" in step else 0,
            "factor_C": 1 if "clahe" in step else 0,
            "operation_order": "tvg->bilateral->clahe",
        }
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="跨数据集预处理适配器自检")
    ap.add_argument("--dataset", default="ai4shipwrecks", choices=SUPPORTED_DATASETS)
    ap.add_argument("--image", default=None, help="测试图像路径 (默认仅打印摘要)")
    args = ap.parse_args()
    print(f"=== {args.dataset} 7 组策略摘要 (raw/basic/physd + 4 单操作) ===")
    print(json.dumps(config_summary(args.dataset), indent=2, ensure_ascii=False))

    if args.image:
        from PIL import Image as PILImage
        img = np.array(PILImage.open(args.image).convert("L"))
        print(f"\n=== 测试图像: {args.image} shape={img.shape} ===")
        for name in SUPPORTED_STRATEGIES:
            fn = build_preprocess_adapter(name, dataset=args.dataset)
            out = img if fn is None else fn(img)
            print(f"  {name:15s}: shape={out.shape} range=[{out.min()}, {out.max()}] "
                  f"mean={out.mean():.1f}")
