#!/usr/bin/env python3
"""
tools/training/ai4shipwrecks_dataset.py

AI4Shipwrecks 数据集加载器 — 用于 SegFormer 训练。

数据集结构:
  AI4Shipwrecks/
    train/
      images/  (141 张 PNG, grayscale, 宽=1728, 高可变)
      labels/  (141 张 PNG, mode=L, 二值 {0,1}: 0=背景, 1=沉船)
    test/
      images/  (120 张)
      labels/  (120 张)

与 SubPipe 的区别:
  - 标签是现成二值 mask (0/1),无需从 COCO bbox 生成伪标签
  - train/test 划分已由数据集预先定义,无需 chunk_split
  - 图像是 PNG grayscale (非 PBM)

分类体系 (与 SubPipe 一致的二分类):
  - 0: background
  - 1: target (shipwreck)

输出:
  - image: float32, (3, H, W), 归一化到 [0,1]
  - mask:  int64, (H, W), 值为 0 或 1

使用方式:
  from ai4shipwrecks_dataset import AI4ShipwrecksDataset, create_data_loaders
  # 旧行为 (向后兼容, train=141): 返回 3-tuple
  train_loader, test_loader, info = create_data_loaders(
      data_dir="datasets/AI4Shipwrecks",
      img_size=512,
      batch_size=4,
  )
  # 新行为 (修复 test 数据泄漏, train=113/val=28): 返回 4-tuple
  train_loader, val_loader, test_loader, info = create_data_loaders(
      data_dir="datasets/AI4Shipwrecks",
      img_size=512,
      batch_size=4,
      include_val=True,
  )
  # 或直接实例化:
  val_ds = AI4ShipwrecksDataset(data_dir, split="val", img_size=512)  # 28 样本
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from torch.utils.data import Dataset

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


# 二分类定义 (与 SubPipe 一致)
CLASS_NAMES = ["background", "target"]
NUM_CLASSES = 2
IGNORE_INDEX = 255

# 固定的 val 划分文件 (seed=42, 28 样本从 141 train 中划出)
# 用于修复 test-set 数据泄漏: checkpoint 选择基于 val_miou 而非 test_miou
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VAL_SPLIT_FILE = PROJECT_ROOT / "experiments" / "splits" / "ai4_val_split_v1.json"


def _load_val_split_filenames() -> set:
    """加载固定的 val 样本文件名集合 (从 ai4_val_split_v1.json).

    返回 val 样本的文件名 set, 用于在 train split 中排除这些样本,
    以及在 val split 中仅保留这些样本。

    若文件不存在则抛出 FileNotFoundError (调用方可设 use_val_split=False 回退)。
    """
    if not VAL_SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"找不到 val 划分文件: {VAL_SPLIT_FILE}. "
            f"请先生成该文件, 或设置 use_val_split=False 以使用旧 train/test 行为."
        )
    with open(VAL_SPLIT_FILE) as f:
        data = json.load(f)
    return set(data["val_filenames"])


def _load_image(path: Path) -> np.ndarray:
    """读取 PNG 图像, 返回 uint8 灰度图 (2D)."""
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


def _load_label(path: Path) -> np.ndarray:
    """读取 PNG 标签, 返回 uint8 二值 mask (0/1).

    AI4Shipwrecks 标签已是 0/1, 但做一次 binarize 防止边缘值.
    """
    if HAS_PIL:
        try:
            img = PILImage.open(path)
            if img.mode != "L":
                img = img.convert("L")
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
        except Exception:
            arr = None
    if arr is None and HAS_CV2:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取标签: {path}")
        arr = img
    elif arr is None:
        raise FileNotFoundError(f"无法读取标签 (无PIL/cv2): {path}")
    # 二值化: 任何非零像素 → 1
    arr = (arr > 0).astype(np.uint8)
    return arr


def _degrade_mask_to_box(label: np.ndarray) -> np.ndarray:
    """将精细二值掩码退化为外接矩形填充掩码 (RC 阶段监督解耦实验).

    取所有前景连通域的整体外接矩形 ( bounding box ), 将矩形区域全部填充为前景,
    模拟 SubPipe/MCM 从 COCO bbox 生成的粗监督形式。

    若无前景或 cv2 不可用, 原样返回。
    """
    if not HAS_CV2 or not (label > 0).any():
        return label
    contours, _ = cv2.findContours(
        label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return label
    # 所有连通域的整体外接矩形
    pts = np.concatenate([c.reshape(-1, 2) for c in contours])
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    box = np.zeros_like(label)
    box[y_min:y_max + 1, x_min:x_max + 1] = 1
    return box


class AI4ShipwrecksDataset(Dataset):
    """AI4Shipwrecks 声纳图像语义分割数据集。

    直接读取现成的 image + 二值 label 对, 无需伪标签生成。
    """

    def __init__(self,
                 data_dir: str = "datasets/AI4Shipwrecks",
                 split: str = "train",
                 img_size: int = 512,
                 augment: bool = False,
                 max_samples: Optional[int] = None,
                 seed: int = 42,
                 preprocess_fn=None,
                 use_val_split: bool = True,
                 degrade_to_box: bool = False):
        """
        Args:
            data_dir: AI4Shipwrecks 数据集根目录 (含 train/ 和 test/ 子目录)
            split: "train", "val" 或 "test"
                - "train": 训练集 (use_val_split=True 时为 113, 排除 val 样本)
                - "val":   验证集 (28 样本, 从 train 中固定划出, 修复 test 数据泄漏)
                - "test":  测试集 (120 样本, 不变)
            img_size: 输出图像尺寸 (正方形)
            augment: 是否数据增强 (仅 train 生效)
            max_samples: 若提供, 仅加载前 N 个样本 (用于 smoke test)
            seed: 随机种子 (用于可复现的 max_samples 子采样)
            preprocess_fn: 可选, 接收 2D uint8 灰度图返回 2D uint8 灰度图的函数,
                在 resize 前对全分辨率图像应用 (A1b 跨数据集预处理适配)
            use_val_split: 是否启用固定 val 划分 (从 141 train 中划出 28 作为 val)。
                - True (默认): split="train" 返回 113 (排除 val), split="val" 返回 28。
                  用于训练脚本基于 val_miou 选 checkpoint。
                - False: split="train" 返回 141 (旧行为, 向后兼容)。
                split="val" 时必须为 True。
            degrade_to_box: 是否将精细掩码退化为边界框填充矩形 (RC 阶段监督解耦实验)。
                - False (默认): 使用原始精细掩码 (真值细监督)。
                - True: 模拟 SubPipe/MCM 的 bbox 粗监督, 取目标外接矩形填充为前景。
                  仅用于训练集; val/test 保持精细掩码以评估真实性能。
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.seed = seed
        self.preprocess_fn = preprocess_fn
        self.use_val_split = use_val_split
        # RC 监督解耦: 仅 train split 退化, val/test 保持精细掩码
        self.degrade_to_box = degrade_to_box and (split == "train")

        if split not in ("train", "val", "test"):
            raise ValueError(f"split 必须是 'train', 'val' 或 'test', 当前: {split}")

        # val 样本物理上位于 train 目录, 从中筛选固定的 28 个
        source_split = "train" if split == "val" else split
        split_dir = self.data_dir / source_split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"找不到 {source_split} 目录: {split_dir}")

        self.samples = self._collect_samples(split_dir)

        # 应用固定 val 划分 (从 train 中划出 28 样本)
        if split == "val":
            if not use_val_split:
                raise ValueError("split='val' 需要 use_val_split=True")
            val_names = _load_val_split_filenames()
            self.samples = [s for s in self.samples if s["filename"] in val_names]
        elif split == "train" and use_val_split:
            val_names = _load_val_split_filenames()
            self.samples = [s for s in self.samples if s["filename"] not in val_names]

        if max_samples is not None and max_samples < len(self.samples):
            rng = random.Random(seed)
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            indices = sorted(indices[:max_samples])
            self.samples = [self.samples[i] for i in indices]

    def _collect_samples(self, split_dir: Path) -> List[Dict]:
        """收集 image-label 对。"""
        img_dir = split_dir / "images"
        lab_dir = split_dir / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"找不到图像目录: {img_dir}")

        samples = []
        img_files = sorted(img_dir.glob("*.png"))

        for img_path in img_files:
            label_path = lab_dir / img_path.name
            if not label_path.exists():
                continue  # 跳过没有标签的样本
            samples.append({
                "image_path": str(img_path),
                "label_path": str(label_path),
                "filename": img_path.name,
                "split": self.split,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = _load_image(Path(sample["image_path"]))
        label = _load_label(Path(sample["label_path"]))

        # RC 监督解耦: 将精细掩码退化为边界框填充矩形 (模拟 bbox 粗监督)
        # 在增强/resize 前应用, 保证与全分辨率标签语义一致
        if self.degrade_to_box:
            label = _degrade_mask_to_box(label)

        # 跨数据集预处理 (A1b adapter, 在 resize/augment 前对全分辨率图像应用)
        # 仅改像素值不改尺寸, 故 label 无需同步变换
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)

        # 数据增强 (在 resize 前应用, 保持空间一致性)
        if self.augment:
            img, label = self._augment(img, label)

        # 调整尺寸 (图像用双线性插值, 标签用最近邻)
        img = cv2.resize(img, (self.img_size, self.img_size),
                        interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.img_size, self.img_size),
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

        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(label.astype(np.int64)),
            "filename": sample["filename"],
        }

    def _augment(self, img: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强: 随机翻转 + 亮度/对比度。"""
        # 随机水平翻转
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            label = np.fliplr(label).copy()

        # 随机垂直翻转
        if random.random() < 0.3:
            img = np.flipud(img).copy()
            label = np.flipud(label).copy()

        # 随机亮度调整
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 随机对比度调整
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip(
                (img.astype(np.float32) - mean) * factor + mean,
                0, 255
            ).astype(np.uint8)

        return img, label


def create_data_loaders(data_dir: str = "datasets/AI4Shipwrecks",
                        img_size: int = 512,
                        batch_size: int = 4,
                        num_workers: int = 4,
                        max_train_samples: Optional[int] = None,
                        max_test_samples: Optional[int] = None,
                        seed: int = 42,
                        preprocess_fn=None,
                        preprocess_name: str = "raw",
                        include_val: bool = False):
    """创建训练/验证/测试数据加载器。

    Args:
        data_dir: AI4Shipwrecks 数据集根目录
        img_size: 图像尺寸
        batch_size: 批大小
        num_workers: 数据加载线程数
        max_train_samples: 限制训练样本数 (smoke test 用)
        max_test_samples: 限制测试样本数 (smoke test 用)
        seed: 随机种子
        preprocess_fn: 可选, 2D uint8 → 2D uint8 的预处理函数 (A1b adapter)
        preprocess_name: 预处理配置名 (raw/basic/physd), 用于 info 记录
        include_val: 是否创建独立 val_loader (从 141 train 中固定划出 28)。
            - False (默认): 返回 (train_loader, test_loader, info)，
              train_loader 含 141 样本 (旧行为, 向后兼容)。
            - True: 返回 (train_loader, val_loader, test_loader, info)，
              train_loader 含 113 样本 (排除 val), val_loader 含 28 样本。
              用于基于 val_miou 选 checkpoint, 修复 test 数据泄漏。

    Returns:
        include_val=False: (train_loader, test_loader, dataset_info)
        include_val=True:  (train_loader, val_loader, test_loader, dataset_info)
    """
    use_val = bool(include_val)
    train_dataset = AI4ShipwrecksDataset(
        data_dir=data_dir,
        split="train",
        img_size=img_size,
        augment=True,
        max_samples=max_train_samples,
        seed=seed,
        preprocess_fn=preprocess_fn,
        use_val_split=use_val,
    )

    test_dataset = AI4ShipwrecksDataset(
        data_dir=data_dir,
        split="test",
        img_size=img_size,
        augment=False,
        max_samples=max_test_samples,
        seed=seed,
        preprocess_fn=preprocess_fn,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dataset_info = {
        "name": "ai4shipwrecks",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "img_size": img_size,
        "split": "predefined_train_test",
        "preprocess": preprocess_name,
        "use_val_split": use_val,
    }

    if include_val:
        val_dataset = AI4ShipwrecksDataset(
            data_dir=data_dir,
            split="val",
            img_size=img_size,
            augment=False,
            seed=seed,
            preprocess_fn=preprocess_fn,
            use_val_split=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        dataset_info["val_samples"] = len(val_dataset)
        return train_loader, val_loader, test_loader, dataset_info

    return train_loader, test_loader, dataset_info


def generate_overlay_visualization(data_dir: str,
                                   output_dir: str,
                                   max_images: int = 100,
                                   img_size: int = 512,
                                   alpha: float = 0.5):
    """生成 image+mask overlay 可视化, 用于人工核查标签对齐。

    将目标区域 (label=1) 用红色半透明叠加在原图上。

    Args:
        data_dir: AI4Shipwrecks 数据集根目录
        output_dir: 可视化输出目录
        max_images: 最多生成多少张可视化
        img_size: 可视化尺寸
        alpha: 叠加透明度

    Returns:
        生成的可视化文件路径列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = AI4ShipwrecksDataset(
        data_dir=data_dir,
        split="train",
        img_size=img_size,
        augment=False,
    )

    n = min(max_images, len(dataset))
    out_files = []

    for i in range(n):
        sample = dataset[i]
        img = sample["image"].numpy()  # (3, H, W), float32 [0,1]
        mask = sample["mask"].numpy()   # (H, W), int64 {0,1}

        # 转回 HWC uint8
        img_hwc = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        if img_hwc.shape[2] == 1:
            img_hwc = np.concatenate([img_hwc, img_hwc, img_hwc], axis=2)

        # 红色 overlay 在目标区域
        overlay = img_hwc.copy()
        target_mask = (mask > 0)
        if target_mask.any():
            base = img_hwc[target_mask].astype(np.float32)
            blended_r = base[:, 0] * (1 - alpha) + 255 * alpha
            blended_g = base[:, 1] * (1 - alpha)
            blended_b = base[:, 2] * (1 - alpha)
            overlay[target_mask] = np.stack([
                np.clip(blended_r, 0, 255),
                np.clip(blended_g, 0, 255),
                np.clip(blended_b, 0, 255),
            ], axis=1).astype(np.uint8)

        # 拼接: 原图 | mask | overlay
        mask_rgb = np.stack([mask * 255, mask * 255, mask * 255], axis=2).astype(np.uint8)
        combined = np.concatenate([img_hwc, mask_rgb, overlay], axis=1)

        out_file = output_path / f"{sample['filename']}"
        cv2.imwrite(str(out_file), combined)
        out_files.append(str(out_file))

    return out_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI4Shipwrecks dataset loader check")
    parser.add_argument("--data_dir", default="datasets/AI4Shipwrecks",
                        help="AI4Shipwrecks dataset root")
    parser.add_argument("--viz_output", default=None,
                        help="If set, generate overlay visualizations to this dir")
    parser.add_argument("--viz_count", type=int, default=100,
                        help="Number of visualizations to generate")
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()

    train_ds = AI4ShipwrecksDataset(args.data_dir, split="train", img_size=args.img_size)
    test_ds = AI4ShipwrecksDataset(args.data_dir, split="test", img_size=args.img_size)
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    sample = train_ds[0]
    print(f"Sample 0: filename={sample['filename']}")
    print(f"  image shape={tuple(sample['image'].shape)} dtype={sample['image'].dtype} "
          f"min={sample['image'].min():.3f} max={sample['image'].max():.3f}")
    print(f"  mask  shape={tuple(sample['mask'].shape)} dtype={sample['mask'].dtype} "
          f"unique={torch.unique(sample['mask']).tolist()}")

    if args.viz_output:
        files = generate_overlay_visualization(
            args.data_dir, args.viz_output, max_images=args.viz_count,
            img_size=args.img_size)
        print(f"Generated {len(files)} visualizations in {args.viz_output}")
