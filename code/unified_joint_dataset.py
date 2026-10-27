"""
Unified joint dataset wrappers for multi-dataset SegFormer training.

Wraps the per-dataset loaders (AI4Shipwrecks, SubPipe, MCM) into
a unified 8-class label space and provides a round-robin multi-dataset
iterator.

Label remapping:
    AI4Shipwrecks:  0→0 (C0 water_column_nadir), 1→1 (C1 shipwreck)  — no change
    SubPipe:        0→0 (C0),                    1→2 (C2 pipeline)   — remap 1→2
    MCM (bg0):      0→0 (C0),                    3→3 (C3 mine)       — no change

Valid class sets (for DMCE):
    ai4:       {0, 1}
    subpipe:   {0, 2}
    mcm:       {0, 3}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ai4shipwrecks_dataset import AI4ShipwrecksDataset
from subpipe_dataset import SubPipeDataset
from mcm_dataset import MCMDataset, load_year_split

UNIFIED_NUM_CLASSES = 8
UNIFIED_IGNORE_INDEX = 255

SAMPLER_TYPES = ("default", "class_balanced", "dataset_balanced")

DATASET_VALID_CLASSES = {
    "ai4": [0, 1],
    "subpipe": [0, 2],
    "mcm": [0, 3],
}

LABEL_REMAPPING = {
    "ai4": {},
    "subpipe": {1: 2},
    "mcm": {},
}

class UnifiedDatasetWrapper(Dataset):
    """Wraps a base dataset, remapping labels to unified 8-class IDs.

    The wrapper applies a label remapping table (e.g. SubPipe pipeline 1→2)
    so that all datasets share the same 8-class label space.

    RC 监督解耦 (Paper2 revision): box_edge_downweight 启用时, 额外返回
    per-pixel 权重图 "weight" (float32, H×W), 前景边界带降权 —
    实现 "uncertainty-weighted regional labeling" 基线:
    bbox 内部边界带 (不确定区域) 权重 = box_edge_downweight,
    其余区域权重 = 1.0。训练循环将其作为 reliability_weight 传入损失。
    """

    def __init__(self, base: Dataset, dataset_name: str,
                 remapping: Optional[Dict[int, int]] = None,
                 box_edge_downweight: Optional[float] = None,
                 box_edge_band: int = 8):
        self.base = base
        self.dataset_name = dataset_name
        self.remapping = remapping or {}
        self.box_edge_downweight = box_edge_downweight
        self.box_edge_band = box_edge_band

    def __len__(self):
        return len(self.base)

    def _compute_box_edge_weight(self, mask: torch.Tensor) -> torch.Tensor:
        """根据 (重映射后的) mask 计算边界带降权权重图.

        前景 (0 < mask < 255) 的边界带 (向内腐蚀 edge_band 像素) 权重为
        box_edge_downweight, 其余为 1.0。bbox 外 (255) 的像素本就被损失忽略,
        权重值无关紧要, 设为 1.0。
        """
        import numpy as _np
        import cv2 as _cv2
        m = mask.numpy()
        fg = ((m > 0) & (m != UNIFIED_IGNORE_INDEX)).astype(_np.uint8)
        weight = _np.ones(m.shape, dtype=_np.float32)
        if fg.any():
            # 核大小 = 2*band+1 (奇数), 使腐蚀每侧恰好移除 band 像素,
            # 即边界带宽度 = box_edge_band (每侧)。
            k = 2 * max(self.box_edge_band, 1) + 1
            kernel = _cv2.getStructuringElement(
                _cv2.MORPH_RECT, (k, k))
            core = _cv2.erode(fg, kernel)
            edge_band = fg & ~core
            weight[edge_band > 0] = float(self.box_edge_downweight)
        return torch.from_numpy(weight)

    def __getitem__(self, idx):
        sample = self.base[idx]
        mask = sample["mask"]
        if self.remapping:
            mask = mask.clone()
            for old, new in self.remapping.items():
                mask[mask == old] = new
            sample["mask"] = mask
        if self.box_edge_downweight is not None:
            sample["weight"] = self._compute_box_edge_weight(sample["mask"])
        return sample

def _build_class_balanced_sampler(
    dataset: Dataset,
    num_classes: int = UNIFIED_NUM_CLASSES,
    ignore_index: int = UNIFIED_IGNORE_INDEX,
    scan_max_samples: Optional[int] = None,
    seed: int = 42,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler using inverse-frequency class weights.

    Scans the (already remapped) wrapped dataset to compute per-class pixel
    frequencies, then assigns each sample a weight = mean(1/freq[c]) over the
    classes present in its mask. Samples with rare classes get higher weight.

    Lazy import: only loaded when sampler_type="class_balanced" is used, so
    the default path has no extra dependency.
    """
    from class_balanced_sampler import (
        compute_class_frequencies, compute_sample_weights)

    logging.info(f"  [class_balanced] scanning frequencies "
                 f"(max {scan_max_samples} samples)...")
    class_freq = compute_class_frequencies(
        dataset, num_classes=num_classes, ignore_index=ignore_index,
        max_samples=scan_max_samples,
    )
    logging.info(f"  [class_balanced] class_freq: {class_freq.tolist()}")

    weights = compute_sample_weights(
        dataset, class_freq, ignore_index=ignore_index,
        max_samples=scan_max_samples,
    )
    logging.info(
        f"  [class_balanced] weights: min={weights.min():.6f} "
        f"max={weights.max():.6f} mean={weights.mean():.6f}")

    num_samples = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights, num_samples=num_samples,
        replacement=True, generator=generator,
    )

class MultiDatasetLoader:
    """Round-robin iterator over multiple DataLoaders.

    Yields (dataset_name, batch) tuples. Each batch comes from a single
    dataset's DataLoader. Iteration continues until all dataloaders are
    exhausted. Shorter dataloaders finish first; longer ones continue
    until they too are exhausted.

    This ensures every sample in every dataset is seen each epoch,
    regardless of dataset size differences.
    """

    def __init__(self, dataloaders: Dict[str, DataLoader]):
        self.dataloaders = dataloaders
        self.dataset_names = list(dataloaders.keys())

    def __iter__(self):
        iters = {name: iter(dl) for name, dl in self.dataloaders.items()}
        active = set(self.dataset_names)
        while active:
            for name in self.dataset_names:
                if name not in active:
                    continue
                try:
                    batch = next(iters[name])
                    yield name, batch
                except StopIteration:
                    active.discard(name)

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders.values())

def create_ai4_dataset(data_dir: str, split: str, img_size: int,
                       augment: bool, batch_size: int, num_workers: int,
                       max_samples: Optional[int] = None,
                       sampler_type: str = "default",
                       sampler_seed: int = 42,
                       scan_max_samples: Optional[int] = None,
                       preprocess_fn=None,
                       degrade_to_box: bool = False) -> DataLoader:
    """Create AI4Shipwrecks dataloader wrapped in unified labels.

    degrade_to_box (RC 监督解耦): train split 时将精细掩码退化为 bbox 填充,
    模拟 SubPipe/MCM 的粗监督形式。
    """
    base = AI4ShipwrecksDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        augment=augment,
        max_samples=max_samples,
        preprocess_fn=preprocess_fn,
        degrade_to_box=degrade_to_box and (split == "train"),
    )
    wrapped = UnifiedDatasetWrapper(
        base, "ai4", remapping=LABEL_REMAPPING["ai4"])

    if sampler_type == "class_balanced" and augment:
        sampler = _build_class_balanced_sampler(
            wrapped, num_classes=UNIFIED_NUM_CLASSES,
            ignore_index=UNIFIED_IGNORE_INDEX,
            scan_max_samples=scan_max_samples, seed=sampler_seed,
        )
        return DataLoader(
            wrapped, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
    return DataLoader(
        wrapped, batch_size=batch_size,
        shuffle=augment, num_workers=num_workers,
        pin_memory=True, drop_last=augment,
    )

def create_subpipe_dataset(data_dir: str, split: str, img_size: int,
                            augment: bool, batch_size: int, num_workers: int,
                            chunk_split: Optional[Dict] = None,
                            sampler_type: str = "default",
                            sampler_seed: int = 42,
                            scan_max_samples: Optional[int] = None,
                            max_samples: Optional[int] = None,
                            preprocess_fn=None,
                            box_edge_downweight: Optional[float] = None,
                            box_edge_band: int = 8) -> DataLoader:
    """Create SubPipe dataloader wrapped in unified labels.

    box_edge_downweight (RC 监督解耦): train 时启用边界带降权
    (uncertainty-weighted regional labeling 基线)。
    """
    split_arg = "train" if split == "train" else "test"
    base = SubPipeDataset(
        data_dir=data_dir,
        frequency="hf",
        split=split_arg,
        img_size=img_size,
        augment=augment,
        chunk_split=chunk_split,
        preprocess_fn=preprocess_fn,
    )
    wrapped = UnifiedDatasetWrapper(
        base, "subpipe", remapping=LABEL_REMAPPING["subpipe"],
        box_edge_downweight=(box_edge_downweight if split == "train" else None),
        box_edge_band=box_edge_band,
    )

    if max_samples is not None and len(wrapped) > max_samples:
        from torch.utils.data import Subset
        wrapped = Subset(wrapped, list(range(max_samples)))

    if sampler_type == "class_balanced" and augment:
        sampler = _build_class_balanced_sampler(
            wrapped, num_classes=UNIFIED_NUM_CLASSES,
            ignore_index=UNIFIED_IGNORE_INDEX,
            scan_max_samples=scan_max_samples, seed=sampler_seed,
        )
        return DataLoader(
            wrapped, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
    return DataLoader(
        wrapped, batch_size=batch_size,
        shuffle=augment, num_workers=num_workers,
        pin_memory=True, drop_last=augment,
    )

def create_mcm_dataset(image_dir: str, mask_dir: str, years: List[int],
                        img_size: int, augment: bool, batch_size: int,
                        num_workers: int,
                        max_samples: Optional[int] = None,
                        sampler_type: str = "default",
                        sampler_seed: int = 42,
                        scan_max_samples: Optional[int] = None,
                        preprocess_fn=None,
                        box_edge_downweight: Optional[float] = None,
                        box_edge_band: int = 8) -> DataLoader:
    """Create MCM dataloader (already 8-class, no remapping needed).

    box_edge_downweight (RC 监督解耦): train 时启用 bbox 内部边界带降权
    (uncertainty-weighted regional labeling 基线)。
    """
    base = MCMDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        years=years,
        img_size=img_size,
        augment=augment,
        max_samples=max_samples,
        preprocess_fn=preprocess_fn,
    )
    wrapped = UnifiedDatasetWrapper(
        base, "mcm", remapping=LABEL_REMAPPING["mcm"],
        box_edge_downweight=(box_edge_downweight if augment else None),
        box_edge_band=box_edge_band,
    )

    if sampler_type == "class_balanced" and augment:
        sampler = _build_class_balanced_sampler(
            wrapped, num_classes=UNIFIED_NUM_CLASSES,
            ignore_index=UNIFIED_IGNORE_INDEX,
            scan_max_samples=scan_max_samples, seed=sampler_seed,
        )
        return DataLoader(
            wrapped, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
    return DataLoader(
        wrapped, batch_size=batch_size,
        shuffle=augment, num_workers=num_workers,
        pin_memory=True, drop_last=augment,
    )

def create_joint_loaders(
    ai4_dir: str,
    subpipe_dir: str,
    mcm_image_dir: str,
    mcm_mask_dir: str,
    mcm_split_file: str,
    subpipe_chunk_split: Optional[Dict] = None,
    img_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    sampler_type: str = "default",
    sampler_seed: int = 42,
    scan_max_samples: Optional[int] = None,
    preprocess_strategy: Optional[str] = None,
    ai4_degrade_to_box: bool = False,
    box_edge_downweight: Optional[float] = None,
    box_edge_band: int = 8,
) -> Tuple["MultiDatasetLoader", Dict[str, DataLoader], dict]:
    """Create joint train loader + per-dataset eval loaders.

    Args:
        ai4_dir: AI4Shipwrecks root (with train/ and test/ subdirs)
        subpipe_dir: SubPipe root (with DATA/Chunk0-4)
        mcm_image_dir: MCM_extracted root
        mcm_mask_dir: MCM_masks root (bg0 version)
        mcm_split_file: mcm_year_split_v2.json path
        subpipe_chunk_split: {"train": [1,2], "val": [], "test": [3], "discard": [0,4]}
        img_size, batch_size, num_workers: standard training params
        max_train_samples: limit per-dataset train samples (smoke test)
        sampler_type: "default" (round-robin), "class_balanced" (per-dataset
            inverse-freq WeightedRandomSampler), or "dataset_balanced"
            (domain-uniform sampling via DomainBalancedMultiLoader, gives
            each dataset equal sampling probability regardless of size)
        sampler_seed: seed for the WeightedRandomSampler
        scan_max_samples: cap on samples scanned for class frequency stats
            (None = scan full dataset; smaller speeds up cold start)
            Set to 0.01 to filter pure-sand tiles that cause class collapse.
            internal tile sampling. When "class_balanced", uses
            WeightedRandomSampler with inverse-frequency weights.
            sampler_type=="dataset_balanced", defaults to 5000 to prevent
            816K-tile explosion. Set explicitly for smoke tests.
        ai4_degrade_to_box (RC 监督解耦, Paper2 revision): True 时 AI4 train
            精细掩码退化为 bbox 填充, 模拟 SubPipe/MCM 粗监督。
        box_edge_downweight (RC 监督解耦): 非 None 时 SubPipe/MCM train 启用
            bbox 边界带降权 (uncertainty-weighted regional labeling 基线)。
        box_edge_band: 边界带宽度 (像素, 每侧, 512×512 空间, 默认 8)。
            注: MCM 目标极小 (bbox ~20-40px), 8px 带已覆盖大部分 bbox 内部。

    Returns:
        train_loader: MultiDatasetLoader or DomainBalancedMultiLoader
        eval_loaders: dict of per-dataset eval loaders
        info: dict with per-dataset sample counts and valid classes
    """
    if sampler_type not in SAMPLER_TYPES:
        raise ValueError(
            f"Unknown sampler_type: {sampler_type}. "
            f"Expected one of {SAMPLER_TYPES}")

    mcm_split = load_year_split(mcm_split_file)

    train_loaders = {}
    eval_loaders = {}
    info = {"datasets": {}, "num_classes": UNIFIED_NUM_CLASSES,
            "ignore_index": UNIFIED_IGNORE_INDEX,
            "sampler_type": sampler_type}

    max_train = max_train_samples

    # When dataset_balanced, cap each dataset at max_train or a sensible default
    if sampler_type == "dataset_balanced" and max_train is None:
        max_train = 3000  # default cap per dataset for domain-balanced mode

    # Build per-dataset preprocess callables from
    # preprocess_strategy. raw → None (no preprocessing). Single-op strategies
    # (clahe_only/lee_only/bilateral_only/hist_eq) are dataset-independent;
    # raw/basic/physd need per-dataset routing (build_preprocess_adapter builds
    # a closure capturing the dataset name).
    ai4_pre_fn = subpipe_pre_fn = mcm_pre_fn = None
    if preprocess_strategy:
        from sonar_preprocess_adapter import build_preprocess_adapter
        ai4_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                              dataset="ai4shipwrecks")
        subpipe_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                                  dataset="subpipe")
        mcm_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                              dataset="mcm")
        logging.info(f"  [preprocess_strategy] {preprocess_strategy} "
                     f"applied to ai4/subpipe/mcm")

    train_loaders["ai4"] = create_ai4_dataset(
        ai4_dir, "train", img_size, augment=True,
        batch_size=batch_size, num_workers=num_workers,
        max_samples=max_train, sampler_type=sampler_type,
        sampler_seed=sampler_seed, scan_max_samples=scan_max_samples,
        preprocess_fn=ai4_pre_fn,
        degrade_to_box=ai4_degrade_to_box)
    eval_loaders["ai4"] = create_ai4_dataset(
        ai4_dir, "test", img_size, augment=False,
        batch_size=batch_size, num_workers=num_workers,
        max_samples=max_eval_samples,
        preprocess_fn=ai4_pre_fn)
    info["datasets"]["ai4"] = {
        "train_samples": len(train_loaders["ai4"].dataset),
        "eval_samples": len(eval_loaders["ai4"].dataset),
        "valid_classes": DATASET_VALID_CLASSES["ai4"],
    }

    train_loaders["subpipe"] = create_subpipe_dataset(
        subpipe_dir, "train", img_size, augment=True,
        batch_size=batch_size, num_workers=num_workers,
        chunk_split=subpipe_chunk_split, sampler_type=sampler_type,
        sampler_seed=sampler_seed, scan_max_samples=scan_max_samples,
        max_samples=max_train,
        preprocess_fn=subpipe_pre_fn,
        box_edge_downweight=box_edge_downweight,
        box_edge_band=box_edge_band)
    eval_loaders["subpipe"] = create_subpipe_dataset(
        subpipe_dir, "test", img_size, augment=False,
        batch_size=batch_size, num_workers=num_workers,
        chunk_split=subpipe_chunk_split, max_samples=max_eval_samples,
        preprocess_fn=subpipe_pre_fn)
    info["datasets"]["subpipe"] = {
        "train_samples": len(train_loaders["subpipe"].dataset),
        "eval_samples": len(eval_loaders["subpipe"].dataset),
        "valid_classes": DATASET_VALID_CLASSES["subpipe"],
    }

    train_loaders["mcm"] = create_mcm_dataset(
        mcm_image_dir, mcm_mask_dir, mcm_split["train_years"],
        img_size, augment=True,
        batch_size=batch_size, num_workers=num_workers,
        max_samples=max_train, sampler_type=sampler_type,
        sampler_seed=sampler_seed, scan_max_samples=scan_max_samples,
        preprocess_fn=mcm_pre_fn,
        box_edge_downweight=box_edge_downweight,
        box_edge_band=box_edge_band)
    eval_loaders["mcm"] = create_mcm_dataset(
        mcm_image_dir, mcm_mask_dir, mcm_split["test_years"],
        img_size, augment=False,
        batch_size=batch_size, num_workers=num_workers,
        max_samples=max_eval_samples,
        preprocess_fn=mcm_pre_fn)
    info["datasets"]["mcm"] = {
        "train_samples": len(train_loaders["mcm"].dataset),
        "eval_samples": len(eval_loaders["mcm"].dataset),
        "valid_classes": DATASET_VALID_CLASSES["mcm"],
        "train_years": mcm_split["train_years"],
        "test_years": mcm_split["test_years"],
    }

    if sampler_type == "dataset_balanced":
        from domain_balanced_sampler import DomainBalancedMultiLoader

        dataset_sizes = {name: len(dl.dataset)
                         for name, dl in train_loaders.items()}
        # mode="uniform" => each dataset gets equal sampling probability,
        # regardless of size. This is what the user requested for the
        train_loader = DomainBalancedMultiLoader(
            train_loaders, dataset_sizes, mode="uniform", seed=sampler_seed)
        logging.info(
            f"[dataset_balanced] using DomainBalancedMultiLoader "
            f"(mode=uniform) with datasets={list(train_loaders.keys())} "
            f"sizes={dataset_sizes}")
    else:
        train_loader = MultiDatasetLoader(train_loaders)

    for name, d in info["datasets"].items():
        logging.info(
            f"  {name}: train={d['train_samples']} eval={d['eval_samples']} "
            f"valid_classes={d['valid_classes']}"
        )

    return train_loader, eval_loaders, info

# ===========================================================================
# v2 protocol validation loaders.
# ===========================================================================
# checkpoint selection must use an independent group-level
# validation set, NOT the test set. The v2 split files
# (experiments/splits/v2_protocol/*_split_v2.json) define group-level isolated
# train/val/test splits for all 4 datasets. This section builds per-dataset
# validation loaders from those split files so that training scripts can
# evaluate on val each epoch (for best-epoch selection) and evaluate on test
# only once at the end (avoiding test-set leakage into model selection).
#
# These helpers are ADDITIVE: they do not modify create_joint_loaders or any
# existing function. When --split_version=v1 (legacy), these helpers are not
# called and behavior is unchanged.

import json as _json
from torch.utils.data import Subset as _Subset

class FilenameFilterDataset(Dataset):
    """Filter a base dataset to only include samples whose filename is in
    the allowed set.

    Used to build v2 validation loaders for datasets (e.g. AI4) whose base
    constructor does not natively support an arbitrary file-list split.
    The base dataset must expose ``.samples`` as a list of dicts with a
    ``"filename"`` key (AI4ShipwrecksDataset, SubPipeDataset).
    """

    def __init__(self, base: Dataset, allowed_filenames):
        self.base = base
        allowed = set(allowed_filenames)
        # Build index mapping. base.samples is a list of dicts.
        base_samples = getattr(base, "samples", None)
        if base_samples is None:
            # Fallback: cannot filter, use all
            self.indices = list(range(len(base)))
        else:
            self.indices = [
                i for i, s in enumerate(base_samples)
                if s.get("filename") in allowed
            ]
        if not self.indices:
            logging.warning(
                f"FilenameFilterDataset: 0/{len(base) if base_samples is None else len(base_samples)} "
                f"samples matched the allowed filename set "
                f"({len(allowed)} filenames expected). Check that the split "
                f"file filenames match the dataset's actual filenames.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

def load_v2_split(split_dir, dataset_key):
    """Load a v2 protocol split JSON file.

    Args:
        split_dir: path to experiments/splits/v2_protocol/
        dataset_key: one of "ai4", "subpipe", "mcm"

    Returns:
        Parsed JSON dict with train_files/val_files/test_files lists.
    """
    split_dir = Path(split_dir)
    filename_map = {
        "ai4": "ai4_split_v2.json",
        "subpipe": "subpipe_split_v2.json",
        "mcm": "mcm_year_split_v2.json",
    }
    fname = filename_map.get(dataset_key)
    if fname is None:
        raise ValueError(f"Unknown dataset_key for v2 split: {dataset_key}")
    split_path = split_dir / fname
    if not split_path.exists():
        raise FileNotFoundError(
            f"v2 split file not found: {split_path}. "
            f"Run experiments/splits/v2_protocol/generate_v2_splits.py first.")
    with open(split_path) as f:
        return _json.load(f)

def create_v2_val_loaders(
    ai4_dir: str,
    subpipe_dir: str,
    mcm_image_dir: str,
    mcm_mask_dir: str,
    v2_split_dir: str,
    img_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    max_eval_samples: Optional[int] = None,
    preprocess_strategy: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """Build per-dataset VALIDATION loaders from v2_protocol split files.

    Validation sets are group-level isolated from train,
    used for checkpoint selection. Test sets are evaluated only once at end.

    Returns a dict of {dataset_name: val_DataLoader}. The loaders use the
    same UnifiedDatasetWrapper + label remapping as create_joint_loaders so
    the label space is consistent with training.

    For AI4: val_files are a subset of the train directory, so we build the
    full train dataset (use_val_split=False) and filter by filename.
    For SubPipe: val_groups define a chunk, so we use chunk_split with val.
    For MCM: val_years define the year list.
    """
    val_loaders: Dict[str, DataLoader] = {}

    # Preprocess callables (same logic as create_joint_loaders)
    ai4_pre_fn = subpipe_pre_fn = mcm_pre_fn = None
    if preprocess_strategy:
        from sonar_preprocess_adapter import build_preprocess_adapter
        ai4_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                              dataset="ai4shipwrecks")
        subpipe_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                                  dataset="subpipe")
        mcm_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                              dataset="mcm")

    # --- AI4 val ---
    ai4_split = load_v2_split(v2_split_dir, "ai4")
    ai4_val_files = ai4_split.get("val_files", [])
    if ai4_val_files:
        # Build AI4 with split="train", use_val_split=False (loads all 141
        # train images), then filter to only the v2 val filenames.
        base = AI4ShipwrecksDataset(
            data_dir=ai4_dir,
            split="train",
            img_size=img_size,
            augment=False,
            use_val_split=False,
            preprocess_fn=ai4_pre_fn,
        )
        filtered = FilenameFilterDataset(base, ai4_val_files)
        wrapped = UnifiedDatasetWrapper(
            filtered, "ai4", remapping=LABEL_REMAPPING["ai4"])
        if max_eval_samples is not None and len(wrapped) > max_eval_samples:
            wrapped = _Subset(wrapped, list(range(max_eval_samples)))
        val_loaders["ai4"] = DataLoader(
            wrapped, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False)
        logging.info(f"  [v2-val] ai4: {len(wrapped)} val samples "
                     f"(from {len(ai4_val_files)} val_files)")

    # --- SubPipe val ---
    subpipe_split = load_v2_split(v2_split_dir, "subpipe")
    subpipe_val_groups = subpipe_split.get("val_groups", [])
    if subpipe_val_groups:
        # Build SubPipe with chunk_split that puts val_groups in "val" key.
        # All other groups go to "discard" so they don't appear.
        all_groups = {0, 1, 2, 3, 4}
        val_set = set(subpipe_val_groups)
        discard = sorted(all_groups - val_set)
        v2_chunk_split = {
            "train": [],
            "val": subpipe_val_groups,
            "test": [],
            "discard": discard,
            "purge_files": subpipe_split.get("purge_files", []),
        }
        base = SubPipeDataset(
            data_dir=subpipe_dir,
            frequency="hf",
            split="val",
            img_size=img_size,
            augment=False,
            chunk_split=v2_chunk_split,
            preprocess_fn=subpipe_pre_fn,
        )
        wrapped = UnifiedDatasetWrapper(
            base, "subpipe", remapping=LABEL_REMAPPING["subpipe"])
        if max_eval_samples is not None and len(wrapped) > max_eval_samples:
            wrapped = _Subset(wrapped, list(range(max_eval_samples)))
        val_loaders["subpipe"] = DataLoader(
            wrapped, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False)
        logging.info(f"  [v2-val] subpipe: {len(wrapped)} val samples "
                     f"(chunks={subpipe_val_groups})")

    # --- MCM val ---
    mcm_split = load_v2_split(v2_split_dir, "mcm")
    mcm_val_years = mcm_split.get("val_years", [])
    if not mcm_val_years:
        # Fallback: derive from val_groups
        mcm_val_years = mcm_split.get("val_groups", [])
    if mcm_val_years:
        base = MCMDataset(
            image_dir=mcm_image_dir,
            mask_dir=mcm_mask_dir,
            years=mcm_val_years,
            img_size=img_size,
            augment=False,
            preprocess_fn=mcm_pre_fn,
        )
        wrapped = UnifiedDatasetWrapper(
            base, "mcm", remapping=LABEL_REMAPPING["mcm"])
        if max_eval_samples is not None and len(wrapped) > max_eval_samples:
            wrapped = _Subset(wrapped, list(range(max_eval_samples)))
        val_loaders["mcm"] = DataLoader(
            wrapped, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False)
        logging.info(f"  [v2-val] mcm: {len(wrapped)} val samples "
                     f"(years={mcm_val_years})")

    return val_loaders

# ---------------------------------------------------------------------------
# Corrected-protocol strict loaders (2026-09-11 train/val 重叠修复)
# ---------------------------------------------------------------------------
# 背景: 2026-09-11 复核发现 legacy 管线的 train loader 从未采用 v2 划分文件的
# train 侧 —— AI4 train=113 (数据集内部 legacy 28-val 剔除), SubPipe
# train=[Chunk1,Chunk2]=1080 (DEFAULT_SUBPIPE_CHUNK_SPLIT)。导致 SubPipe v2
# val (Chunk2) 100% 与 AI4 v2 val 20 张中 14 张实际参与训练 (checkpoint 选择
# 污染)。本节提供严格从 v2 划分 JSON 文件清单构建 train/val/test 的加载器 +
# Three-way non-overlap verification for the strict v2 loader.

# v2 划分清单的期望样本数 (audit 依据, 与 split JSON counts 字段一致)
V2_EXPECTED_COUNTS = {
    "ai4": {"train": 121, "val": 20, "test": 120},
    "subpipe": {"train": 540, "val": 537, "test": 189},
    "mcm": {"train": 558, "val": 564, "test": 48},
}

def verify_v2_split_isolation(v2_split_dir: str,
                              expected_counts: Optional[Dict] = None) -> Dict:
    """校验 v2 划分文件的隔离性与完整性 (corrected protocol 硬前置)。

    检查项 (每数据集):
      1. train/val/test 文件清单两两无交集 (样本级);
      2. group_key 组清单两两无交集 (site/chunk/year 组级);
      3. 各 split 文件数与期望清单一致 (防清单漂移)。

    任一项失败即 raise RuntimeError (fail-fast, 不允许带病训练)。
    返回逐项检查报告 dict (调用方写入 run 目录 split_verification.json)。
    """
    expected = expected_counts or V2_EXPECTED_COUNTS
    report: Dict = {"v2_split_dir": str(v2_split_dir), "checks": [],
                    "all_pass": True}

    def _check(name, ok, detail=""):
        report["checks"].append({"check": name, "pass": bool(ok),
                                 "detail": str(detail)})
        if not ok:
            report["all_pass"] = False

    for key, fname in (("ai4", "ai4_split_v2.json"),
                       ("subpipe", "subpipe_split_v2.json"),
                       ("mcm", "mcm_year_split_v2.json")):
        data = json.loads((Path(v2_split_dir) / fname).read_text())
        file_sets = {sp: set(data.get(f"{sp}_files", []))
                     for sp in ("train", "val", "test")}
        # 1. 样本级两两无交集
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter = file_sets[a] & file_sets[b]
            _check(f"{key}:{a}_cap_{b}==0", len(inter) == 0,
                   f"overlap={len(inter)} sample={sorted(inter)[:5]}")
        # 2. 组级两两无交集
        group_key = data.get("group_key", "?")
        groups = {sp: set(str(g) for g in data.get(f"{sp}_groups", []))
                  for sp in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter_g = groups[a] & groups[b]
            _check(f"{key}:group_{a}_cap_{b}==0 ({group_key})",
                   len(inter_g) == 0, f"overlap={sorted(inter_g)}")
        # 3. 文件数与期望清单一致
        exp = expected.get(key, {})
        for sp in ("train", "val", "test"):
            if sp in exp:
                _check(f"{key}:{sp}_count=={exp[sp]}",
                       len(file_sets[sp]) == exp[sp],
                       f"json has {len(file_sets[sp])}")

    if not report["all_pass"]:
        failed = [c["check"] for c in report["checks"] if not c["pass"]]
        raise RuntimeError(
            f"[strict_v2] split isolation verification FAILED: {failed}")

    # 校验通过后检查 item 级一致性记录
    report["expected_counts"] = expected
    return report

def _build_strict_split_loaders(base, allowed_files, name, remap_key,
                                batch_size, num_workers, shuffle,
                                max_samples=None):
    """按文件清单严格过滤 base 数据集 → UnifiedDatasetWrapper → DataLoader。

    硬一致性检查: 匹配到的样本集合必须与清单完全相等 (不多不少),
    防止 FilenameFilterDataset 静默丢样本 (loader↔split JSON 一致性)。
    """
    allowed = set(allowed_files)
    base_samples = getattr(base, "samples", None)
    if base_samples is None:
        raise RuntimeError(
            f"[strict_v2] {name}: base dataset 无 .samples, 无法清单过滤")
    matched = {s.get("filename") for s in base_samples} & allowed
    missing = allowed - matched
    if missing:
        raise RuntimeError(
            f"[strict_v2] {name}: {len(missing)} 个清单文件在数据集中不存在 "
            f"(loader↔JSON 不一致), 示例: {sorted(missing)[:5]}")
    ds = UnifiedDatasetWrapper(
        FilenameFilterDataset(base, allowed), remap_key,
        remapping=LABEL_REMAPPING[remap_key])
    if len(ds) != len(allowed):
        raise RuntimeError(
            f"[strict_v2] {name}: loader 大小 {len(ds)} != 清单数 "
            f"{len(allowed)} (loader↔JSON 不一致)")
    if max_samples is not None and max_samples < len(ds):
        from torch.utils.data import Subset as _Subset
        ds = _Subset(ds, list(range(max_samples)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=shuffle)

def create_v2_split_loaders(ai4_dir: str, subpipe_dir: str,
                            mcm_image_dir: str, mcm_mask_dir: str,
                            v2_split_dir: str, img_size: int = 512,
                            batch_size: int = 4, num_workers: int = 4,
                            max_train_samples: Optional[int] = None,
                            max_eval_samples: Optional[int] = None,
                            preprocess_strategy: Optional[str] = None,
                            sampler_type: str = "default",
                            sampler_seed: int = 42):
    """Corrected protocol: train/val/test 全部严格来自 v2 划分 JSON 文件清单。

    与 legacy create_joint_loaders 的唯一差异 = 划分组成 (单变量对照):
      - AI4: train=val_files 从 141 张 train 目录按清单过滤 (121/20),
        test 从 test 目录按清单过滤 (120); 不再使用数据集内部 legacy 28-val。
      - SubPipe: train=Chunk1 清单 (540), val=Chunk2 清单 (537),
        test=Chunk3 清单 (189); purge 边界帧由清单本身排除。
      - MCM: 年份划分 (v1==v2), train/val/test 按清单过滤 (558/564/48)。
    增强/预处理口径与 legacy 完全一致 (train augment=True, val/test False)。

    返回 (train_loader: MultiDatasetLoader, val_loaders, test_loaders, info);
    info["datasets"][name] 含 train/val/test 样本数与清单一致性摘要。
    """
    ai4_pre_fn = subpipe_pre_fn = mcm_pre_fn = None
    if preprocess_strategy:
        from sonar_preprocess_adapter import build_preprocess_adapter
        ai4_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                              dataset="ai4shipwrecks")
        subpipe_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                                  dataset="subpipe")
        mcm_pre_fn = build_preprocess_adapter(preprocess_strategy,
                                              dataset="mcm")

    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    test_loaders: Dict[str, DataLoader] = {}
    info: Dict = {"datasets": {}, "num_classes": UNIFIED_NUM_CLASSES,
                  "ignore_index": UNIFIED_IGNORE_INDEX,
                  "sampler_type": sampler_type, "strict_v2": True}

    # --- AI4 (train 目录: 141 张; test 目录: 120 张) ---
    ai4_split = load_v2_split(v2_split_dir, "ai4")
    ai4_train_base = AI4ShipwrecksDataset(
        data_dir=ai4_dir, split="train", img_size=img_size, augment=True,
        use_val_split=False,  # 绕过 legacy 28-val, 由 v2 清单接管划分
        preprocess_fn=ai4_pre_fn)
    ai4_test_base = AI4ShipwrecksDataset(
        data_dir=ai4_dir, split="test", img_size=img_size, augment=False,
        preprocess_fn=ai4_pre_fn)
    train_loaders["ai4"] = _build_strict_split_loaders(
        ai4_train_base, ai4_split["train_files"], "ai4:train", "ai4",
        batch_size, num_workers, shuffle=True, max_samples=max_train_samples)
    val_loaders["ai4"] = _build_strict_split_loaders(
        ai4_train_base, ai4_split["val_files"], "ai4:val", "ai4",
        batch_size, num_workers, shuffle=False, max_samples=max_eval_samples)
    test_loaders["ai4"] = _build_strict_split_loaders(
        ai4_test_base, ai4_split["test_files"], "ai4:test", "ai4",
        batch_size, num_workers, shuffle=False, max_samples=max_eval_samples)
    info["datasets"]["ai4"] = {
        "train_samples": len(train_loaders["ai4"].dataset),
        "val_samples": len(val_loaders["ai4"].dataset),
        "eval_samples": len(test_loaders["ai4"].dataset),
        "valid_classes": DATASET_VALID_CLASSES["ai4"],
        "split_source": "ai4_split_v2.json (strict file lists)",
    }
    logging.info(f"  [strict-v2] ai4: train={info['datasets']['ai4']['train_samples']} "
                 f"val={info['datasets']['ai4']['val_samples']} "
                 f"test={info['datasets']['ai4']['eval_samples']}")

    # --- SubPipe (Chunk1+2+3 联合收集, 按清单过滤) ---
    subpipe_split = load_v2_split(v2_split_dir, "subpipe")
    keep_all = {"train": [1, 2, 3], "val": [], "test": [], "discard": [0, 4]}
    subpipe_train_base = SubPipeDataset(
        data_dir=subpipe_dir, frequency="hf", split="train",
        img_size=img_size, augment=True, chunk_split=keep_all,
        preprocess_fn=subpipe_pre_fn)
    subpipe_eval_base = SubPipeDataset(
        data_dir=subpipe_dir, frequency="hf", split="val",
        img_size=img_size, augment=False,
        chunk_split={"train": [], "val": [1, 2, 3], "test": [],
                     "discard": [0, 4]},
        preprocess_fn=subpipe_pre_fn)
    train_loaders["subpipe"] = _build_strict_split_loaders(
        subpipe_train_base, subpipe_split["train_files"], "subpipe:train",
        "subpipe", batch_size, num_workers, shuffle=True,
        max_samples=max_train_samples)
    val_loaders["subpipe"] = _build_strict_split_loaders(
        subpipe_eval_base, subpipe_split["val_files"], "subpipe:val",
        "subpipe", batch_size, num_workers, shuffle=False,
        max_samples=max_eval_samples)
    test_loaders["subpipe"] = _build_strict_split_loaders(
        subpipe_eval_base, subpipe_split["test_files"], "subpipe:test",
        "subpipe", batch_size, num_workers, shuffle=False,
        max_samples=max_eval_samples)
    info["datasets"]["subpipe"] = {
        "train_samples": len(train_loaders["subpipe"].dataset),
        "val_samples": len(val_loaders["subpipe"].dataset),
        "eval_samples": len(test_loaders["subpipe"].dataset),
        "valid_classes": DATASET_VALID_CLASSES["subpipe"],
        "split_source": "subpipe_split_v2.json (strict file lists)",
    }
    logging.info(f"  [strict-v2] subpipe: "
                 f"train={info['datasets']['subpipe']['train_samples']} "
                 f"val={info['datasets']['subpipe']['val_samples']} "
                 f"test={info['datasets']['subpipe']['eval_samples']}")

    # --- MCM (年份划分, v1==v2, 按清单过滤) ---
    mcm_split = load_v2_split(v2_split_dir, "mcm")

    def _mcm_years(split_name: str) -> List[int]:
        # mcm_year_split_v2.json 不含 train_years 顶层键, 年份在 counts 下;
        # 缺失时回退为从文件名后缀解析 (xxx_<year>.jpg)。
        years = (mcm_split.get("counts", {}).get(split_name, {})
                 .get("years"))
        if years:
            return [int(y) for y in years]
        prefix = {"train": "train", "val": "val", "test": "test"}[split_name]
        files = mcm_split[f"{prefix}_files"]
        return sorted({int(Path(f).stem.rsplit("_", 1)[1]) for f in files})

    mcm_train_base = MCMDataset(
        image_dir=mcm_image_dir, mask_dir=mcm_mask_dir,
        years=_mcm_years("train"), img_size=img_size, augment=True,
        preprocess_fn=mcm_pre_fn)
    mcm_val_base = MCMDataset(
        image_dir=mcm_image_dir, mask_dir=mcm_mask_dir,
        years=_mcm_years("val"), img_size=img_size, augment=False,
        preprocess_fn=mcm_pre_fn)
    mcm_test_base = MCMDataset(
        image_dir=mcm_image_dir, mask_dir=mcm_mask_dir,
        years=_mcm_years("test"), img_size=img_size, augment=False,
        preprocess_fn=mcm_pre_fn)
    train_loaders["mcm"] = _build_strict_split_loaders(
        mcm_train_base, mcm_split["train_files"], "mcm:train", "mcm",
        batch_size, num_workers, shuffle=True, max_samples=max_train_samples)
    val_loaders["mcm"] = _build_strict_split_loaders(
        mcm_val_base, mcm_split["val_files"], "mcm:val", "mcm",
        batch_size, num_workers, shuffle=False, max_samples=max_eval_samples)
    test_loaders["mcm"] = _build_strict_split_loaders(
        mcm_test_base, mcm_split["test_files"], "mcm:test", "mcm",
        batch_size, num_workers, shuffle=False, max_samples=max_eval_samples)
    info["datasets"]["mcm"] = {
        "train_samples": len(train_loaders["mcm"].dataset),
        "val_samples": len(val_loaders["mcm"].dataset),
        "eval_samples": len(test_loaders["mcm"].dataset),
        "valid_classes": DATASET_VALID_CLASSES["mcm"],
        "train_years": _mcm_years("train"),
        "val_years": _mcm_years("val"),
        "test_years": _mcm_years("test"),
        "split_source": "mcm_year_split_v2.json (strict file lists)",
    }
    logging.info(f"  [strict-v2] mcm: train={info['datasets']['mcm']['train_samples']} "
                 f"val={info['datasets']['mcm']['val_samples']} "
                 f"test={info['datasets']['mcm']['eval_samples']}")

    if sampler_type == "dataset_balanced":
        # 2026-09-16 fix: strict 路径此前静默忽略 --sampler (bug 导致
        # c3a_*_balanced_sampler 10 个 run 实际为普通 round-robin)。
        # 与 legacy create_joint_loaders L590-611 相同的包装逻辑 (单变量保持)。
        from domain_balanced_sampler import DomainBalancedMultiLoader

        dataset_sizes = {name: len(dl.dataset)
                         for name, dl in train_loaders.items()}
        train_loader = DomainBalancedMultiLoader(
            train_loaders, dataset_sizes, mode="uniform", seed=sampler_seed)
        logging.info(
            f"[dataset_balanced] using DomainBalancedMultiLoader "
            f"(mode=uniform) with datasets={list(train_loaders.keys())} "
            f"sizes={dataset_sizes}")
    else:
        train_loader = MultiDatasetLoader(train_loaders)
    return train_loader, val_loaders, test_loaders, info
