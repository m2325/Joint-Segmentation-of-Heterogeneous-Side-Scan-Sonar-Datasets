#!/usr/bin/env python3
"""
tools/training/subpipe_pseudo_label.py

SubPipe COCO边界框 → 二值语义分割掩码生成器。

SubPipe数据集的SSS声纳图像只提供pipeline的目标检测边界框标注（COCO格式），
本脚本将边界框直接转换为二值语义分割掩码，忠实于SubPipe原生分类体系。

二分类体系（SubPipe原生）：
    0: background  - 边界框外所有区域
    1: pipeline    - 边界框内所有区域

使用方式：
  python subpipe_pseudo_label.py --chunk_dir /path/to/Chunk0 --frequency hf
  python subpipe_pseudo_label.py --dataset_dir /path/to/SubPipe --frequency hf --all_chunks
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# 二分类定义（SubPipe原生分类）
CLASS_BACKGROUND = 0
CLASS_PIPELINE = 1

NUM_CLASSES = 2
CLASS_NAMES = ["background", "pipeline"]

# 类别颜色（BGR，用于可视化）
CLASS_COLORS_BGR = {
    0: (0, 0, 0),        # background - 黑
    1: (0, 0, 255),      # pipeline - 红
}


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

    img = cv2.imread(str(pbm_path), cv2.IMREAD_GRAYSCALE)
    if img is not None and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def generate_binary_label(img: np.ndarray,
                          bboxes: List[List[float]]) -> np.ndarray:
    """从SSS图像+COCO边界框生成二值语义分割掩码。

    直接将边界框内区域标记为pipeline(1)，边界框外为background(0)。
    这是对SubPipe COCO目标检测标注最忠实的语义分割转换。

    Args:
        img: 灰度声纳图像 (H, W), uint8
        bboxes: COCO格式边界框列表 [[x, y, w, h], ...]

    Returns:
        二值标签图 (H, W), uint8, 值为0(background)或1(pipeline)
    """
    h, w = img.shape
    label = np.full((h, w), CLASS_BACKGROUND, dtype=np.uint8)

    # 将每个边界框内区域标记为pipeline
    for bbox in bboxes:
        bx, by, bw, bh = bbox
        x0 = max(0, int(bx))
        y0 = max(0, int(by))
        x1 = min(w, int(bx + bw))
        y1 = min(h, int(by + bh))

        if x1 > x0 and y1 > y0:
            label[y0:y1, x0:x1] = CLASS_PIPELINE

    return label


def process_chunk(chunk_dir: Path, frequency: str = "hf",
                  output_dir: Path = None,
                  visualize: bool = False) -> int:
    """处理单个Chunk，生成所有SSS图像的二值伪标签。

    Args:
        chunk_dir: Chunk目录路径
        frequency: "hf" 或 "lf"
        output_dir: 伪标签输出目录
        visualize: 是否保存可视化图像

    Returns:
        处理的图像数量
    """
    sss_dir = chunk_dir / f"SSS_{frequency.upper()}_images"
    img_dir = sss_dir / "Image"
    coco_path = sss_dir / "COCO_Annotation" / "coco_format.json"

    if not img_dir.exists():
        print(f"  [WARN] SSS图像目录不存在: {img_dir}")
        return 0

    # 加载COCO标注
    if not coco_path.exists():
        print(f"  [WARN] COCO标注不存在: {coco_path}")
        return 0

    with open(coco_path) as f:
        coco = json.load(f)

    # 建立图像ID→标注映射
    ann_by_img = {}
    for ann in coco.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann["bbox"])

    # 建立文件名→图像信息映射
    img_map = {img["file_name"]: img for img in coco.get("images", [])}

    # 输出目录
    if output_dir is None:
        output_dir = sss_dir / "PseudoLabels"
    output_dir.mkdir(parents=True, exist_ok=True)

    if visualize:
        vis_dir = output_dir / "visualization"
        vis_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    pbm_files = sorted(img_dir.glob("*.pbm"))

    for pbm_path in pbm_files:
        # 读取图像
        img = load_pbm(pbm_path)
        if img is None:
            continue

        # 获取该图像的边界框
        img_info = img_map.get(pbm_path.name)
        bboxes = []
        if img_info:
            bboxes = ann_by_img.get(img_info["id"], [])

        # 生成二值伪标签
        label = generate_binary_label(img, bboxes)

        # 保存伪标签（PNG，像素值为0或1）
        label_path = output_dir / (pbm_path.stem + "_label.png")
        cv2.imwrite(str(label_path), label)

        # 保存原始图像副本（用于训练时配对）
        img_copy_path = output_dir / (pbm_path.stem + ".png")
        cv2.imwrite(str(img_copy_path), img)

        # 可视化
        if visualize:
            vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            for cls_id, color in CLASS_COLORS_BGR.items():
                vis[label == cls_id] = color
            # 叠加边界框
            for bbox in bboxes:
                bx, by, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            vis_path = vis_dir / (pbm_path.stem + "_vis.png")
            cv2.imwrite(str(vis_path), vis)

        count += 1
        if count % 200 == 0:
            print(f"  进度: {count}/{len(pbm_files)}")

    print(f"  完成: {count} 张图像, 伪标签保存至 {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="SubPipe COCO边界框 → 二值语义分割掩码生成器"
    )
    parser.add_argument(
        "--chunk_dir", type=str, default=None,
        help="单个Chunk目录路径"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="数据集根目录（包含DATA/Chunk0-4）"
    )
    parser.add_argument(
        "--frequency", type=str, default="hf", choices=["hf", "lf"],
        help="声纳频率 (hf=高频, lf=低频)"
    )
    parser.add_argument(
        "--all_chunks", action="store_true",
        help="处理所有Chunk（需配合--dataset_dir）"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="伪标签输出目录"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="生成可视化图像"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SubPipe 二值伪标签生成器 (background/pipeline)")
    print(f"  频率: {args.frequency}")
    print("=" * 60)

    output_dir = Path(args.output_dir) if args.output_dir else None
    total = 0

    if args.all_chunks and args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        data_dir = dataset_dir / "DATA"
        base = data_dir if data_dir.exists() else dataset_dir

        for i in range(5):
            chunk_dir = base / f"Chunk{i}"
            if chunk_dir.exists():
                print(f"\n处理 Chunk{i}...")
                total += process_chunk(
                    chunk_dir, args.frequency, output_dir, args.visualize
                )
    elif args.chunk_dir:
        total = process_chunk(
            Path(args.chunk_dir), args.frequency, output_dir, args.visualize
        )
    else:
        parser.error("请指定 --chunk_dir 或 --dataset_dir --all_chunks")

    print(f"\n总计处理 {total} 张图像")


if __name__ == "__main__":
    main()
