#!/usr/bin/env python3
"""
class_balanced_sampler.py

类别平衡采样器 — 作为 DMCE 的消融基线。

动机:
  DMCE 通过 logit masking 避免无效类别梯度, 但不改变采样频率。
  本采样器通过逆频率加权采样, 让少数类 (C1/C2/C3) 被更频繁地看到,
  作为"采样层平衡"基线, 与 DMCE 的"损失层 masking"形成对比。

消融矩阵 (readme2_paper2.md §二):
  | 采样           | 损失        | 说明                          |
  |----------------|-------------|-------------------------------|
  | Round-robin    | Naive CE    | 最弱基线                      |
  | Round-robin    | DMCE        | 本文主方法                    |
  | Class-balanced | Naive CE    | 采样层平衡 (本脚本)           |
  | Class-balanced | DMCE        | 双层平衡 (消融, 预期不显著提升)|
  | Domain-balanced| DMCE        | 域均衡采样 (domain_balanced_sampler.py) |

用法:
  from class_balanced_sampler import build_class_balanced_loader
  loader = build_class_balanced_loader(dataset, batch_size=4)
  for batch in loader:
      ...
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler


# ============================================================================
# 类别频率统计
# ============================================================================
def compute_class_frequencies(
    dataset, num_classes: int = 8, ignore_index: int = 255,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    """扫描整个数据集, 统计每个类别的像素频率。

    Args:
        dataset: 数据集对象, __getitem__ 返回 {"mask": (H, W) int64}
        num_classes: 类别数
        ignore_index: 忽略的标签值
        max_samples: 最多扫描多少样本 (None=全部)

    Returns:
        frequency: (num_classes,) 每个类别的像素占比 ∈ (0, 1]
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for i in range(n):
        sample = dataset[i]
        mask = sample["mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        mask = mask.astype(np.int64).ravel()
        valid = mask != ignore_index
        mask = mask[valid]
        mask = np.clip(mask, 0, num_classes - 1)
        counts += np.bincount(mask, minlength=num_classes)

    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes) / num_classes

    frequency = counts / total
    return torch.tensor(frequency, dtype=torch.float32)


def compute_sample_weights(
    dataset, class_freq: torch.Tensor, ignore_index: int = 255,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    """为每个样本计算采样权重 (基于该样本中少数类的占比)。

    权重 = mean( (1 / class_freq[c]) for c in sample_mask if c != ignore )

    少数类占比越高 → 权重越大 → 被采样概率越高。

    Args:
        dataset: 数据集对象
        class_freq: (num_classes,) 类别频率张量
        ignore_index: 忽略的标签值
        max_samples: 最多扫描多少样本

    Returns:
        weights: (len(dataset),) 每个样本的采样权重
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    inv_freq = 1.0 / (class_freq + 1e-8)
    weights = torch.zeros(n, dtype=torch.float32)

    for i in range(n):
        sample = dataset[i]
        mask = sample["mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        mask = mask.astype(np.int64).ravel()
        valid = mask != ignore_index
        mask = mask[valid]
        if len(mask) == 0:
            weights[i] = 1.0
            continue
        mask = np.clip(mask, 0, len(inv_freq) - 1)
        sample_inv_freq = inv_freq.numpy()[mask]
        weights[i] = float(np.mean(sample_inv_freq))

    # 归一化
    weights = weights / weights.sum()
    return weights


# ============================================================================
# 构建平衡 DataLoader
# ============================================================================
def build_class_balanced_loader(
    dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    num_samples: Optional[int] = None,
    num_classes: int = 8,
    ignore_index: int = 255,
    scan_max_samples: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """构建类别平衡 DataLoader。

    Args:
        dataset: 基础数据集 (返回 {"mask": (H, W) int64})
        batch_size: 批大小
        num_workers: 数据加载线程
        num_samples: 每轮采样数 (None=len(dataset))
        num_classes: 类别数
        ignore_index: 忽略标签
        scan_max_samples: 扫描频率时最多看多少样本 (加速)
        seed: 随机种子

    Returns:
        DataLoader (使用 WeightedRandomSampler)
    """
    print(f"[class_balanced] 扫描类别频率 (max {scan_max_samples} 样本)...")
    class_freq = compute_class_frequencies(
        dataset, num_classes=num_classes, ignore_index=ignore_index,
        max_samples=scan_max_samples,
    )
    print(f"[class_balanced] 类别频率: {class_freq.tolist()}")

    print(f"[class_balanced] 计算样本权重...")
    weights = compute_sample_weights(
        dataset, class_freq, ignore_index=ignore_index,
        max_samples=scan_max_samples,
    )
    print(f"[class_balanced] 权重范围: [{weights.min():.6f}, {weights.max():.6f}]")

    if num_samples is None:
        num_samples = len(dataset)

    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ============================================================================
# 自测
# ============================================================================
if __name__ == "__main__":
    # 模拟一个不均衡数据集: 90% 背景像素, 10% 前景
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, n=100, h=64, w=64):
            self.n = n
            self.h = h
            self.w = w

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            mask = np.zeros((self.h, self.w), dtype=np.int64)
            # 10% 的样本有前景, 90% 纯背景
            if idx % 10 == 0:
                mask[10:30, 10:30] = 1
            return {"mask": torch.from_numpy(mask), "image": torch.zeros(3, self.h, self.w)}

    ds = MockDataset(n=100)
    loader = build_class_balanced_loader(
        ds, batch_size=4, num_classes=2, scan_max_samples=50
    )

    # 统计一个 epoch 中前景样本的比例
    n_fg = 0
    n_total = 0
    for batch in loader:
        masks = batch["mask"]
        has_fg = (masks > 0).any(dim=(1, 2))
        n_fg += has_fg.sum().item()
        n_total += len(masks)

    fg_ratio = n_fg / n_total
    print(f"\n[class_balanced] 一个 epoch 前景样本比例: {fg_ratio:.2%}")
    print(f"  (原始数据集前景比例: 10%)")
    assert fg_ratio > 0.15, f"采样器似乎没起到平衡作用: {fg_ratio:.2%}"

    print("\n[ok] class_balanced_sampler.py 自测通过")
