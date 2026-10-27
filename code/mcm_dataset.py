#!/usr/bin/env python3
"""
tools/training/mcm_dataset.py

MCM (Mine CounterMeasures) 数据集加载器 — 用于 SegFormer 训练 (Route A6).

数据布局 (A5 转换后的 MCM_masks/<year>/<stem>.png):
  - 源图像: MCM_extracted/<year>/<stem>.jpg (RGB, 416x416 或 1024x1024 混合)
  - 弱 mask: MCM_masks/<year>/<stem>.png (单通道 uint8, 值 ∈ {3, 255})
    - 3 = C3 mine_like_contact (MILCO/NOMBO bbox 内部)
    - 255 = UNIFIED_IGNORE_INDEX (bbox 外, 训练时被忽略)
  - 866/1170 张为空 label (无目标), mask 全 255

年份划分 (mcm_year_split_v1.json, fallback 规则):
  train: 2010, 2015, 2017   (558 张, 165 有目标)
  val:   2018               (564 张, 112 有目标)
  test:  2021               (48 张, 27 有目标)

分类体系 (与 constants.py UNIFIED_CLASS_NAMES 一致):
  - 8 类统一语义空间 (C0-C7)
  - IGNORE_INDEX = 255
  - MCM 仅 C3 (mine_like_contact) 有正样本, 其余类别未标注 → 255

输出:
  - image: float32, (3, H, W), 归一化到 [0,1]
  - mask:  int64, (H, W), 值 ∈ {3, 255} (训练时被忽略的 255)

用法:
  from mcm_dataset import MCMDataSet, create_data_loaders
  train_loader, val_loader, test_loader, info = create_data_loaders(
      image_dir="datasets/MCM_extracted",
      mask_dir="datasets/MCM_masks",
      split_file="experiments/splits/mcm_year_split_v1.json",
  )
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# 8 类统一语义空间 (与 src/sonar_slam/sonar_slam/common/constants.py 一致)
NUM_CLASSES = 8
CLASS_NAMES = [
    "water_column_nadir", "shipwreck", "linear_infrastructure",
    "mine_like_contact", "rock_hard_substrate", "coarse_sediment",
    "fine_sediment", "vegetation_biogenic",
]
IGNORE_INDEX = 255
MCM_TARGET_CLASS_ID = 3  # C3 mine_like_contact


def _load_image(path: Path) -> np.ndarray:
    """读取 jpg/png 图像, 返回 uint8 灰度图 (2D)."""
    if HAS_PIL:
        try:
            img = PILImage.open(path)
            if img.mode != "L":
                img = img.convert("L")
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return arr
        except Exception:
            pass
    if HAS_CV2:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
    raise FileNotFoundError(f"无法读取图像: {path}")


def _load_mask(path: Path) -> np.ndarray:
    """读取单通道 mask PNG (PIL mode 'L'). 返回 uint8 (H, W)."""
    if HAS_PIL:
        try:
            img = PILImage.open(path)
            if img.mode != "L":
                img = img.convert("L")
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return arr
        except Exception:
            pass
    if HAS_CV2:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取 mask: {path}")
        return img
    raise FileNotFoundError(f"无法读取 mask (无 PIL/cv2): {path}")


class MCMDataset(Dataset):
    """MCM 声纳图像语义分割数据集 (8 类统一空间, 仅 C3 有正样本).

    年份 split 由 split_file 决定; image/mask 路径按 <year>/<stem> 配对.
    """

    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 years: List[int],
                 img_size: int = 512,
                 augment: bool = False,
                 max_samples: Optional[int] = None,
                 seed: int = 42,
                 preprocess_fn=None):
        """
        Args:
            image_dir: MCM_extracted 根目录 (含 <year>/*.jpg)
            mask_dir: MCM_masks 根目录 (含 <year>/*.png)
            years: 该 split 包含的年份列表 (如 [2010, 2015, 2017])
            img_size: 输出图像尺寸 (正方形)
            augment: 是否数据增强 (仅 train 生效)
            max_samples: 限制样本数 (smoke test 用, 按种子随机子采样)
            seed: 随机种子
            preprocess_fn: 可选, 2D uint8 → 2D uint8 (A1b adapter)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment
        self.seed = seed
        self.preprocess_fn = preprocess_fn

        self.samples = self._collect_samples(years)

        if max_samples is not None and max_samples < len(self.samples):
            rng = random.Random(seed)
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            indices = sorted(indices[:max_samples])
            self.samples = [self.samples[i] for i in indices]

    def _collect_samples(self, years: List[int]) -> List[Dict]:
        """收集 (image_path, mask_path) 对, 按年份 + 文件名排序."""
        samples = []
        for year in sorted(years):
            year_img_dir = self.image_dir / str(year)
            year_mask_dir = self.mask_dir / str(year)
            if not year_img_dir.exists():
                raise FileNotFoundError(f"年份目录不存在: {year_img_dir}")
            if not year_mask_dir.exists():
                raise FileNotFoundError(f"mask 年份目录不存在: {year_mask_dir}")
            for img_path in sorted(year_img_dir.glob("*.jpg")):
                mask_path = year_mask_dir / (img_path.stem + ".png")
                if not mask_path.exists():
                    continue  # 跳过无 mask 的样本
                samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "filename": img_path.name,
                    "year": year,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = _load_image(Path(sample["image_path"]))
        mask = _load_mask(Path(sample["mask_path"]))

        # 跨数据集预处理 (A1b adapter, 仅改像素值不改尺寸, 故 mask 无需同步)
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)

        # 数据增强 (resize 前应用, 保持空间一致)
        if self.augment:
            img, mask = self._augment(img, mask)

        # 尺寸对齐: 若 image 与 mask 尺寸不一致, 以 image 为准 resize mask
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # resize 到 img_size (image 双线性, mask 最近邻保留类别 id)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # 灰度图 → 3 通道 (SegFormer 需要 3 通道输入)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)  # (3, H, W)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
            img = img.transpose(2, 0, 1)
        elif img.ndim == 3:
            img = img.transpose(2, 0, 1)

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.int64)

        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "filename": sample["filename"],
            "year": sample.get("year", -1),
        }

    def _augment(self, img: np.ndarray, mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强: 随机 h/v 翻转 + 亮度/对比度 (image only)."""
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if random.random() < 0.3:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip(
                (img.astype(np.float32) - mean) * factor + mean, 0, 255
            ).astype(np.uint8)
        return img, mask


def load_year_split(split_file: str) -> Dict:
    """读取年份 split JSON. 返回 {train_years, val_years, test_years, ...}."""
    with open(split_file) as f:
        return json.load(f)


def create_data_loaders(image_dir: str,
                        mask_dir: str,
                        split_file: str,
                        img_size: int = 512,
                        batch_size: int = 4,
                        num_workers: int = 4,
                        max_train_samples: Optional[int] = None,
                        max_val_samples: Optional[int] = None,
                        max_test_samples: Optional[int] = None,
                        seed: int = 42,
                        preprocess_fn=None,
                        preprocess_name: str = "raw") -> Tuple:
    """创建 train/val/test 数据加载器.

    Args:
        image_dir: MCM_extracted 根目录
        mask_dir: MCM_masks 根目录 (A5 输出)
        split_file: mcm_year_split_v1.json 路径
        img_size: 图像尺寸
        batch_size: 批大小
        num_workers: DataLoader 线程数
        max_*_samples: 限制各 split 样本数 (smoke test 用)
        seed: 随机种子
        preprocess_fn: A1b adapter 函数
        preprocess_name: 预处理配置名 (用于 info 记录)

    Returns:
        (train_loader, val_loader, test_loader, dataset_info)
    """
    split = load_year_split(split_file)
    train_years = split["train_years"]
    val_years = split["val_years"]
    test_years = split["test_years"]

    train_ds = MCMDataset(
        image_dir=image_dir, mask_dir=mask_dir, years=train_years,
        img_size=img_size, augment=True,
        max_samples=max_train_samples, seed=seed,
        preprocess_fn=preprocess_fn,
    )
    val_ds = MCMDataset(
        image_dir=image_dir, mask_dir=mask_dir, years=val_years,
        img_size=img_size, augment=False,
        max_samples=max_val_samples, seed=seed,
        preprocess_fn=preprocess_fn,
    )
    test_ds = MCMDataset(
        image_dir=image_dir, mask_dir=mask_dir, years=test_years,
        img_size=img_size, augment=False,
        max_samples=max_test_samples, seed=seed,
        preprocess_fn=preprocess_fn,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    dataset_info = {
        "name": "mcm_mine",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
        "img_size": img_size,
        "preprocess": preprocess_name,
        "split_file": split_file,
        "split_version": split.get("version", "unknown"),
    }
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MCM dataset loader sanity check")
    ap.add_argument("--image_dir", default="datasets/MCM_extracted")
    ap.add_argument("--mask_dir", default="datasets/MCM_masks")
    ap.add_argument("--split_file",
                   default="mcm_year_split_v2.json")
    ap.add_argument("--img_size", type=int, default=512)
    args = ap.parse_args()

    if not Path(args.split_file).exists():
        print(f"split_file 不存在: {args.split_file}")
        print("请先生成 mcm_year_split_v1.json")
        raise SystemExit(1)

    train_loader, val_loader, test_loader, info = create_data_loaders(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        split_file=args.split_file,
        img_size=args.img_size,
        batch_size=2,
        num_workers=0,
    )
    print(f"Info: {info}")
    print()
    sample = train_loader.dataset[0]
    print(f"Sample 0: filename={sample['filename']} year={sample['year']}")
    print(f"  image shape={tuple(sample['image'].shape)} dtype={sample['image'].dtype} "
          f"min={sample['image'].min():.3f} max={sample['image'].max():.3f}")
    print(f"  mask  shape={tuple(sample['mask'].shape)} dtype={sample['mask'].dtype} "
          f"unique={torch.unique(sample['mask']).tolist()}")
