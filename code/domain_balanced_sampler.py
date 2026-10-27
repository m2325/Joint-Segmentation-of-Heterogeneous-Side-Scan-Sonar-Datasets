#!/usr/bin/env python3
"""
paper2/domain_balanced_sampler.py

域均衡采样器 — 在多数据集联合训练中, 按域 (数据集) 而非类别进行平衡采样。

动机:
  当前 MultiDatasetLoader (unified_joint_dataset.py) 使用 round-robin,
  每个数据集轮流产出 batch。但 AI4 (141 train) / SubPipe (~2000+) / MCM (~200)
  大小差异悬殊, round-robin 会让小数据集 (AI4) 被过采样, 大数据集 (SubPipe)
  被欠采样。

  本采样器提供两种域均衡策略:
  1. uniform:   每个域等概率采样 (小数据集重复, 大数据集截断)
  2. sqrt_size: 按 sqrt(dataset_size) 加权 (折中方案)

消融对比 (readme2_paper2.md §二):
  | 采样           | 说明                                  |
  |----------------|---------------------------------------|
  | Round-robin    | 当前 MultiDatasetLoader               |
  | Domain-uniform | 本脚本 mode=uniform                   |
  | Domain-sqrt    | 本脚本 mode=sqrt                      |
  | Class-balanced | class_balanced_sampler.py             |

用法:
  from domain_balanced_sampler import build_domain_balanced_multi_loader
  loader = build_domain_balanced_multi_loader(
      datasets={"ai4": ds1, "subpipe": ds2, "mcm": ds3},
      batch_size=4,
      mode="sqrt",
  )
  for dataset_name, batch in loader:
      ...
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from unified_joint_dataset import MultiDatasetLoader
except ImportError:
    MultiDatasetLoader = None


# ============================================================================
# 域均衡采样权重
# ============================================================================
def compute_domain_weights(
    dataset_sizes: Dict[str, int],
    mode: str = "sqrt",
) -> Dict[str, float]:
    """计算每个域的采样权重。

    Args:
        dataset_sizes: {dataset_name: num_samples}
        mode: "uniform" (等权) 或 "sqrt" (sqrt(size) 加权)

    Returns:
        {dataset_name: weight} 归一化权重
    """
    if mode == "uniform":
        raw = {name: 1.0 for name in dataset_sizes}
    elif mode == "sqrt":
        raw = {name: math.sqrt(size) for name, size in dataset_sizes.items()}
    else:
        raise ValueError(f"Unknown mode: {mode}, expected 'uniform' or 'sqrt'")

    total = sum(raw.values())
    return {name: w / total for name, w in raw.items()}


# ============================================================================
# 域均衡多数据集 Loader
# ============================================================================
class DomainBalancedMultiLoader:
    """域均衡多数据集迭代器。

    每轮迭代按域权重选择一个数据集, 从中取一个 batch。
    较小数据集会被重复采样 (replacement=True), 较大数据集会被部分采样。

    与 MultiDatasetLoader (round-robin) 的区别:
      - round-robin: 每个数据集轮流, 不考虑大小差异
      - domain-balanced: 按域权重采样, 小数据集被更多关注
    """

    def __init__(
        self,
        dataloaders: Dict[str, DataLoader],
        dataset_sizes: Dict[str, int],
        mode: str = "sqrt",
        seed: int = 42,
    ):
        """
        Args:
            dataloaders: {dataset_name: DataLoader}
            dataset_sizes: {dataset_name: num_samples}
            mode: "uniform" 或 "sqrt"
            seed: 随机种子
        """
        self.dataloaders = dataloaders
        self.dataset_names = list(dataloaders.keys())
        self.domain_weights = compute_domain_weights(dataset_sizes, mode)
        self.mode = mode
        self._rng = torch.Generator().manual_seed(seed)

        # 每个域的迭代器
        self._iters: Dict[str, Iterator] = {}

        # 每轮总 batch 数 = max(dataset_batches) * 3 (确保大数据集被充分采样)
        max_batches = max(
            len(dl) for dl in dataloaders.values()
        )
        self._epoch_batches = max_batches * len(dataloaders)

    def __iter__(self) -> Iterator[Tuple[str, dict]]:
        # 重置所有迭代器
        self._iters = {name: iter(dl) for name, dl in self.dataloaders.items()}
        weights = torch.tensor(
            [self.domain_weights[name] for name in self.dataset_names]
        )

        for _ in range(self._epoch_batches):
            # 按域权重选择一个数据集
            idx = torch.multinomial(weights, 1, generator=self._rng).item()
            name = self.dataset_names[idx]

            try:
                batch = next(self._iters[name])
                yield name, batch
            except StopIteration:
                # 该数据集已遍历完, 重新开始 (replacement)
                self._iters[name] = iter(self.dataloaders[name])
                try:
                    batch = next(self._iters[name])
                    yield name, batch
                except StopIteration:
                    continue

    def __len__(self):
        return self._epoch_batches


# ============================================================================
# 构建函数
# ============================================================================
def build_domain_balanced_multi_loader(
    datasets: Dict[str, torch.utils.data.Dataset],
    batch_size: int = 4,
    num_workers: int = 4,
    mode: str = "sqrt",
    seed: int = 42,
) -> DomainBalancedMultiLoader:
    """构建域均衡多数据集 Loader。

    Args:
        datasets: {dataset_name: Dataset}
        batch_size: 每个域的 batch size
        num_workers: 数据加载线程
        mode: "uniform" 或 "sqrt"
        seed: 随机种子

    Returns:
        DomainBalancedMultiLoader
    """
    dataloaders = {}
    sizes = {}
    for name, ds in datasets.items():
        dataloaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        sizes[name] = len(ds)

    return DomainBalancedMultiLoader(dataloaders, sizes, mode=mode, seed=seed)


# ============================================================================
# 自测
# ============================================================================
if __name__ == "__main__":
    # 模拟三个大小不同的数据集
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, n, label):
            self.n = n
            self.label = label

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {"x": torch.tensor([self.label]), "idx": idx}

    datasets = {
        "ai4": MockDataset(100, 0),
        "subpipe": MockDataset(2000, 1),
        "mcm": MockDataset(200, 2),
    }

    print("=== 域权重 ===")
    for mode in ("uniform", "sqrt"):
        weights = compute_domain_weights(
            {k: len(v) for k, v in datasets.items()}, mode=mode
        )
        print(f"  {mode}: {weights}")

    print("\n=== 域均衡 Loader (sqrt) ===")
    loader = build_domain_balanced_multi_loader(
        datasets, batch_size=4, num_workers=0, mode="sqrt"
    )

    counts = {name: 0 for name in datasets}
    for name, batch in loader:
        counts[name] += 1

    print(f"  每个 epoch 各域 batch 数: {counts}")
    print(f"  比例: ai4:subpipe:mcm = "
          f"{counts['ai4']/sum(counts.values()):.2%}:"
          f"{counts['subpipe']/sum(counts.values()):.2%}:"
          f"{counts['mcm']/sum(counts.values()):.2%}")
    print(f"  (round-robin 应为 33.3%:33.3%:33.3%, sqrt 应接近 "
          f"sqrt(100):sqrt(2000):sqrt(200) 的比例)")

    print("\n[ok] domain_balanced_sampler.py 自测通过")
