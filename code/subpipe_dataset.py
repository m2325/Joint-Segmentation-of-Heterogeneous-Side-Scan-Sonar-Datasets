#!/usr/bin/env python3
"""
tools/training/subpipe_dataset.py

SubPipe数据集加载器 — 用于SegFormer训练。

支持两种模式：
  1. 原始模式：从SubPipe原始PBM图像+COCO标注加载，实时生成二值伪标签
  2. 预处理模式：从已生成的伪标签目录加载（配合subpipe_pseudo_label.py使用）

数据增强：
  - 随机翻转（水平/垂直）
  - 随机旋转（±10°）
  - 随机亮度/对比度
  - 随机裁剪

输出：
  - image: float32, (3, H, W), 归一化到[0,1]
  - mask: int64, (H, W), 值为0或1（二分类: background/pipeline）

分类体系：采用SubPipe原生二分类（不使用原5类）
  - 0: background  - 边界框外所有区域
  - 1: pipeline    - 边界框内所有区域

使用方式：
  from subpipe_dataset import SubPipeDataset
  dataset = SubPipeDataset(
      data_dir="/path/to/SubPipe/DATA/Chunk0",
      frequency="hf",
      split="train",
      img_size=512,
      augment=True,
  )
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

# 二分类定义（SubPipe原生分类体系）
CLASS_NAMES = ["background", "pipeline"]
NUM_CLASSES = 2
IGNORE_INDEX = 255


def load_pbm(pbm_path: Path) -> Optional[np.ndarray]:
    """读取PBM格式声纳图像，返回uint8灰度图(2D)。

    SubPipe的PBM文件实际为RGB格式存储，需转换为灰度。
    """
    if HAS_PIL:
        try:
            img = PILImage.open(pbm_path)
            if img.mode != "L":
                img = img.convert("L")
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return arr
        except Exception:
            pass
    if HAS_CV2:
        img = cv2.imread(str(pbm_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
    return None


class SubPipeDataset(Dataset):
    """SubPipe声纳图像语义分割数据集。

    模式1（原始）：从PBM+COCO加载，实时生成伪标签
    模式2（预处理）：从伪标签目录加载已生成的标签
    """

    def __init__(self,
                 data_dir: str,
                 frequency: str = "hf",
                 split: str = "train",
                 img_size: int = 512,
                 augment: bool = False,
                 pseudo_label_dir: str = None,
                 train_ratio: float = 0.8,
                 seed: int = 42,
                 preprocess_config=None,
                 chunk_split: Optional[Dict] = None,
                 split_file: Optional[str] = None,
                 preprocess_fn=None):
        """
        Args:
            data_dir: SubPipe数据目录
                      - Chunk目录: /path/to/Chunk0
                      - 数据集根: /path/to/SubPipe (含DATA/Chunk0-4)
            frequency: "hf" 或 "lf"
            split: "train" 或 "val"
            img_size: 输出图像尺寸
            augment: 是否数据增强
            pseudo_label_dir: 预生成的伪标签目录（模式2）
                              如果为None，使用模式1（实时生成）
            train_ratio: 训练集比例
            seed: 随机种子
            preprocess_config: 预处理配置(PreprocessConfig)，None表示不预处理
            chunk_split: 若提供,使用 chunk-level 划分而非随机划分。
                          格式: {"train": [1, 2], "val": [], "test": [3], "discard": [0, 4]}
                          整数对应 Chunk 编号 (Chunk1, Chunk2, ...)。
                          此时 `split` 参数选择返回哪个集合 ("train"/"val"/"test")。
                          "discard" 中的 chunk 完全排除。
            split_file: 若提供,将划分保存为 data_split.json (用于复现)。
            preprocess_fn: 可选, 2D uint8 → 2D uint8 的预处理函数. 优先级高于
                          preprocess_config; 若提供则忽略 preprocess_config.
        """
        self.data_dir = Path(data_dir)
        self.frequency = frequency
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.pseudo_label_dir = Path(pseudo_label_dir) if pseudo_label_dir else None
        self.preprocess_config = preprocess_config
        self.preprocess_fn = preprocess_fn
        self.chunk_split = chunk_split
        self.split_file = split_file
        self.train_ratio = train_ratio
        self.seed = seed

        # 收集所有图像-标签对
        self.samples = self._collect_samples()

        # 划分训练/验证/测试集
        if chunk_split is not None:
            # Chunk-level 划分 (防止 1Hz 连续帧数据泄漏)
            # 保存 split 时需要全部样本, 所以在过滤前保存
            if split_file is not None and split == "train":
                self._save_split_json(chunk_split)
            self.samples = self._apply_chunk_split(chunk_split, split)
        else:
            # 随机划分 (原始行为,向后兼容)
            rng = random.Random(seed)
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            n_train = int(len(indices) * train_ratio)

            if split == "train":
                indices = indices[:n_train]
            else:
                indices = indices[n_train:]

            self.samples = [self.samples[i] for i in indices]

        # 模式1：实时伪标签生成器
        if self.pseudo_label_dir is None:
            from subpipe_pseudo_label import generate_binary_label
            self._gen_label = generate_binary_label

    def _collect_samples(self) -> List[Dict]:
        """收集所有图像-标签样本。"""
        samples = []

        # 查找Chunk目录
        chunks = []
        if (self.data_dir / "DATA").exists():
            base = self.data_dir / "DATA"
        else:
            base = self.data_dir

        if base.name.startswith("Chunk"):
            chunks = [base]
        else:
            chunks = sorted(base.glob("Chunk*"))

        for chunk_dir in chunks:
            sss_dir = chunk_dir / f"SSS_{self.frequency.upper()}_images"
            img_dir = sss_dir / "Image"
            coco_path = sss_dir / "COCO_Annotation" / "coco_format.json"

            if not img_dir.exists():
                continue

            # 加载COCO标注（模式1需要）
            coco_anns = {}
            if self.pseudo_label_dir is None and coco_path.exists():
                with open(coco_path) as f:
                    coco = json.load(f)
                img_map = {img["file_name"]: img for img in coco.get("images", [])}
                for ann in coco.get("annotations", []):
                    img_id = ann["image_id"]
                    coco_anns.setdefault(img_id, []).append(ann["bbox"])
                # 文件名→标注
                self._coco_img_map = img_map
                self._coco_anns = coco_anns

            # 收集PBM文件
            pbm_files = sorted(img_dir.glob("*.pbm"))

            for pbm_path in pbm_files:
                sample = {
                    "image_path": str(pbm_path),
                    "chunk": chunk_dir.name,
                    "filename": pbm_path.name,
                }

                # 模式2：查找预生成的伪标签
                if self.pseudo_label_dir is not None:
                    label_path = self.pseudo_label_dir / (pbm_path.stem + "_label.png")
                    if label_path.exists():
                        sample["label_path"] = str(label_path)
                    else:
                        continue  # 跳过没有标签的样本

                # 模式1：查找COCO标注
                if self.pseudo_label_dir is None:
                    img_info = self._coco_img_map.get(pbm_path.name)
                    if img_info:
                        sample["bboxes"] = coco_anns.get(img_info["id"], [])
                    else:
                        sample["bboxes"] = []

                samples.append(sample)

        return samples

    def _apply_chunk_split(self, chunk_split: Dict, split: str) -> List[Dict]:
        """按 chunk-level 划分样本。防止 1Hz 连续帧跨 train/test 的数据泄漏。

        Args:
            chunk_split: {"train": [1], "val": [2], "test": [3], "discard": [0,4],
                          "purge_files": ["file1.pbm", ...] (optional)}
            split: "train" / "val" / "test" — 选择返回哪个集合

        Returns:
            属于指定 split 的样本列表
        """
        target_chunks = chunk_split.get(split, [])
        chunk_names = {f"Chunk{c}" for c in target_chunks}
        kept = [s for s in self.samples if s.get("chunk") in chunk_names]

        # Purge boundary frames
        purge_files = set(chunk_split.get("purge_files", []))
        if purge_files:
            kept = [s for s in kept if s["filename"] not in purge_files]

        return kept

    def _save_split_json(self, chunk_split: Dict) -> None:
        """将划分方案保存为 data_split.json (用于训练复现和论文可复现性)。

        保存到 self.split_file 路径。
        """
        from pathlib import Path as _P
        import json as _json
        out_path = _P(self.split_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "dataset": "subpipe",
            "split_method": "chunk_level",
            "train_chunks": chunk_split.get("train", []),
            "val_chunks": chunk_split.get("val", []),
            "test_chunks": chunk_split.get("test", []),
            "discard_chunks": chunk_split.get("discard", []),
            "purge_files": chunk_split.get("purge_files", []),
            "train_files": [s["filename"] for s in self.samples if s.get("chunk") in {f"Chunk{c}" for c in chunk_split.get("train", [])}],
            "val_files": [s["filename"] for s in self.samples if s.get("chunk") in {f"Chunk{c}" for c in chunk_split.get("val", [])}],
            "test_files": [s["filename"] for s in self.samples if s.get("chunk") in {f"Chunk{c}" for c in chunk_split.get("test", [])}],
            "discard_files": [s["filename"] for s in self.samples if s.get("chunk") in {f"Chunk{c}" for c in chunk_split.get("discard", [])}],
            "seed": self.seed,
            "split_generated_at": _P.now().isoformat() if hasattr(_P, 'now') else None,
        }
        # 使用 datetime (Path 没有 .now())
        from datetime import datetime
        record["split_generated_at"] = datetime.now().isoformat()
        with out_path.open("w", encoding="utf-8") as f:
            _json.dump(record, f, indent=2, ensure_ascii=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 读取图像
        img = load_pbm(Path(sample["image_path"]))
        if img is None:
            # 返回空白样本
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # 获取标签
        if self.pseudo_label_dir is not None:
            # 模式2：读取预生成标签
            label = cv2.imread(sample["label_path"], cv2.IMREAD_GRAYSCALE)
            if label is None:
                label = np.full_like(img, 0)  # 默认background
        else:
            # 模式1：实时生成二值伪标签
            from subpipe_pseudo_label import generate_binary_label
            label = generate_binary_label(img, sample.get("bboxes", []))

        # 应用预处理（仅对图像，不改变标签）
        # 优先级: preprocess_fn (callable, Paper1 §0.A.2 单操作策略) > preprocess_config
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
            # 预处理可能改变图像尺寸（如斜距校正），同步调整标签
            if img.shape[:2] != label.shape[:2]:
                label = cv2.resize(label, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        elif self.preprocess_config is not None:
            from subpipe_preprocess import preprocess_sss
            img = preprocess_sss(img, self.preprocess_config)
            # 预处理可能改变图像尺寸（如斜距校正），同步调整标签
            if img.shape[:2] != label.shape[:2]:
                label = cv2.resize(label, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # 数据增强
        if self.augment:
            img, label = self._augment(img, label)

        # 调整尺寸
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 转换为tensor
        # 图像：单通道→三通道（SegFormer需要3通道输入）
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)  # (3, H, W)
        else:
            img = img.transpose(2, 0, 1)

        img = img.astype(np.float32) / 255.0

        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(label.astype(np.int64)),
            "filename": sample["filename"],
        }

    def _augment(self, img: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强。"""
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
            img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        return img, label


def create_data_loaders(data_dir: str,
                        frequency: str = "hf",
                        img_size: int = 512,
                        batch_size: int = 8,
                        num_workers: int = 4,
                        pseudo_label_dir: str = None,
                        train_ratio: float = 0.8,
                        preprocess_config=None,
                        chunk_split: Optional[Dict] = None,
                        split_file: Optional[str] = None,
                        preprocess_fn=None):
    """创建训练和验证数据加载器。

    Args:
        data_dir: 数据集目录
        frequency: "hf" 或 "lf"
        img_size: 图像尺寸
        batch_size: 批大小
        num_workers: 数据加载线程数
        pseudo_label_dir: 伪标签目录（None则实时生成）
        train_ratio: 训练集比例
        preprocess_config: 预处理配置(PreprocessConfig)，None表示不预处理
        preprocess_fn: 可选 callable,
                       优先级高于 preprocess_config

    Returns:
        (train_loader, val_loader, dataset_info)
    """
    train_dataset = SubPipeDataset(
        data_dir=data_dir,
        frequency=frequency,
        split="train",
        img_size=img_size,
        augment=True,
        pseudo_label_dir=pseudo_label_dir,
        train_ratio=train_ratio,
        preprocess_config=preprocess_config,
        chunk_split=chunk_split,
        split_file=split_file,
        preprocess_fn=preprocess_fn,
    )

    # 当使用 chunk_split 时, 优先使用独立的 val chunks;
    # 仅当 val chunks 为空时回退到 test chunks (向后兼容旧 split 文件)
    if chunk_split is not None:
        val_split_name = "val" if chunk_split.get("val", []) else "test"
    else:
        val_split_name = "val"
    val_dataset = SubPipeDataset(
        data_dir=data_dir,
        frequency=frequency,
        split=val_split_name,
        img_size=img_size,
        augment=False,
        pseudo_label_dir=pseudo_label_dir,
        train_ratio=train_ratio,
        preprocess_config=preprocess_config,
        chunk_split=chunk_split,
        split_file=split_file,
        preprocess_fn=preprocess_fn,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    info = {
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "img_size": img_size,
    }

    return train_loader, val_loader, info


if __name__ == "__main__":
    # 测试
    import sys

    if len(sys.argv) < 2:
        print("Usage: python subpipe_dataset.py /path/to/SubPipe/DATA/Chunk0 [hf|lf]")
        sys.exit(1)

    data_dir = sys.argv[1]
    freq = sys.argv[2] if len(sys.argv) > 2 else "hf"

    dataset = SubPipeDataset(data_dir=data_dir, frequency=freq, split="train", augment=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {CLASS_NAMES}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Mask unique: {torch.unique(sample['mask']).tolist()}")
        print(f"Filename: {sample['filename']}")
