#!/usr/bin/env python3
"""
tools/training/train_subpipe_segformer.py

基于SubPipe数据集训练SegFormer-B2声纳图像语义分割模型。

训练流程：
  1. 从SubPipe SSS图像+COCO标注生成二值伪标签（background/pipeline）
  2. 训练SegFormer-B2 (512x512, 2类)
  3. 导出ONNX + TensorRT引擎

分类体系：采用SubPipe原生二分类（不使用原5类）
  - 0: background  - 边界框外所有区域
  - 1: pipeline    - 边界框内所有区域

使用方式：
  # 训练
  python train_subpipe_segformer.py \
      --data_dir datasets/SubPipe/SubPipe \
      --frequency hf \
      --output_dir models/segformer_subpipe \
      --epochs 100 \
      --batch_size 8

  # 从预生成伪标签训练（更快）
  python train_subpipe_segformer.py \
      --pseudo_label_dir /path/to/PseudoLabels \
      --output_dir models/segformer_subpipe

  # 导出ONNX
  python train_subpipe_segformer.py --export_onnx \
      --checkpoint models/segformer_subpipe/best_model.pth

  # TensorRT优化
  python train_subpipe_segformer.py --export_trt \
      --onnx_model models/segformer_subpipe/segformer.onnx
"""

import os
import sys
import argparse
import time
import json
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src" / "sonar_slam"))
sys.path.insert(0, str(SCRIPT_DIR))

from subpipe_dataset import (
    SubPipeDataset,
    create_data_loaders,
    NUM_CLASSES,
    CLASS_NAMES,
    IGNORE_INDEX,
)
from sonar_preprocess_adapter import SUPPORTED_STRATEGIES


def setup_logging(output_dir: Path):
    """配置日志。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )


def build_segformer_model(num_classes: int = 2, pretrained: bool = False):
    """构建SegFormer-B2模型。

    使用项目自定义的SegFormerSonar实现（与5类训练相同的架构）。
    SegFormer-B2: depths=[3,4,6,3], embed_dims=[64,128,320,512]

    改进版：当环境变量USE_IMPROVED_MODEL=1时，使用ImprovedSegFormerSonar。
    PLA-SegFormer: 当环境变量USE_PLA=1时, 启用局域注意力高斯衰减调制. PLA_SIGMA_BASE 控制衰减尺度基准.
    """
    # 检查是否使用改进版模型
    use_improved = os.environ.get("USE_IMPROVED_MODEL", "0") == "1"
    use_coord_attn = os.environ.get("USE_COORD_ATTN", "0") == "1"
    use_nadir_aware = os.environ.get("USE_NADIR_AWARE", "0") == "1"
    # PLA-SegFormer
    use_pla = os.environ.get("USE_PLA", "0") == "1"
    pla_sigma_base = float(os.environ.get("PLA_SIGMA_BASE", "8.0"))

    if use_improved:
        from improved_segformer import build_improved_model
        model = build_improved_model(
            num_classes=num_classes,
            img_size=512,
            use_coord_attention=use_coord_attn,
            use_nadir_aware=use_nadir_aware,
        )
        logging.info(f"使用改进版SegFormer (coord_attn={use_coord_attn}, nadir_aware={use_nadir_aware})")
    else:
        from train_segformer import SegFormerSonar
        model = SegFormerSonar(
            num_classes=num_classes,
            use_pla=use_pla,
            pla_sigma_base=pla_sigma_base)
        if use_pla:
            logging.info(
                f"PLA-SegFormer 启用: sigma_base={pla_sigma_base} "
                )
        if pretrained:
            logging.info("注意: pretrained=True时需手动加载预训练权重(--pretrained)")
    return model


def build_simple_unet(num_classes: int = 2):
    """简易UNet模型（SegFormerSonar不可用时的回退方案）。"""
    from torchvision.models.segmentation import fcn_resnet50
    model = fcn_resnet50(pretrained=True, num_classes=num_classes)
    return model


def _build_torchvision_backbone(model_name: str, num_classes: int,
                                in_channels: int = 3, pretrained: bool = True):
    """Build a torchvision segmentation model with configurable input channels.

    Supports fcn_resnet50 (torchvision) and pspnet (segmentation_models_pytorch).
    For torchvision models with in_channels != 3, the first conv layer is
    replaced with a Conv2d(in_channels, 64, ...) and ImageNet pretrained
    weights are copied for the first 3 channels. For smp models, in_channels
    is passed directly to the constructor.
    """
    if model_name == "pspnet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "PSPNet requires segmentation_models_pytorch. "
                "Install: pip install segmentation-models-pytorch")
        model = smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
        )
        return model

    from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
    if model_name != "fcn":
        raise ValueError(f"Unknown torchvision backbone: {model_name!r}")
    if pretrained:
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    else:
        model = fcn_resnet50(pretrained=False, num_classes=num_classes)
    if in_channels != 3:
        old_conv = model.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
        model.backbone.conv1 = new_conv
    return model


def build_model_by_backbone(backbone: str, num_classes: int,
                            in_channels: int = 3, pretrained: bool = False):
    """Build a segmentation model by backbone name.

    Routes to the appropriate builder for `--backbone {segformer_b2, unet,
    deeplabv3plus, fcn, pspnet}`. Used by both train_subpipe_segformer.py and
    train_joint_dmce.py to support public-architecture baselines alongside
    the existing SegFormer-B2 path.

    Args:
        backbone: one of "segformer_b2", "unet", "deeplabv3plus", "fcn",
            "pspnet".
        num_classes: number of output classes (2 for SubPipe binary,
            8 for unified joint semantic space).
        in_channels: 3 (RGB triplicated grayscale, default) or 5 (SSPM).
            For segformer_b2 only in_channels=3 is supported here; the
            5-channel SegFormer path is handled separately via build_sspm_model.
        pretrained: load ImageNet-pretrained backbone weights (DeepLabV3+ /
            FCN / PSPNet only). SegFormer-B2 path always uses pretrained=False
            for backward compat.
    """
    if backbone == "segformer_b2":
        if in_channels != 3:
            raise ValueError(
                f"SegFormer-B2 requires in_channels=3 (got {in_channels}). "
                f"For 5-channel input use --use_sspm with segformer_b2 or "
                f"pick --backbone unet|deeplabv3plus|fcn|pspnet.")
        return build_segformer_model(num_classes=num_classes, pretrained=False)
    elif backbone == "unet":
        from unet_model import build_unet_model
        return build_unet_model(in_channels=in_channels, num_classes=num_classes)
    elif backbone == "deeplabv3plus":
        from deeplabv3plus import build_deeplabv3plus_model
        return build_deeplabv3plus_model(
            in_channels=in_channels, num_classes=num_classes,
            pretrained=pretrained)
    elif backbone in ("fcn", "pspnet"):
        return _build_torchvision_backbone(
            model_name=backbone, num_classes=num_classes,
            in_channels=in_channels, pretrained=True)
    else:
        raise ValueError(
            f"Unknown backbone: {backbone!r}. "
            f"Must be one of: segformer_b2, unet, deeplabv3plus, fcn, pspnet.")


class DiceLoss(nn.Module):
    """Dice Loss，适合类别不平衡的分割任务。"""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index)
        targets = targets.clone()
        targets[~valid] = 0

        # One-hot
        one_hot = F.one_hot(targets, num_classes=probs.shape[1])  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # 只计算有效区域
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        # 每类Dice
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice 组合损失。"""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5,
                 class_weights=None, ignore_index: int = 255):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


def compute_metrics(logits, targets, num_classes: int):
    """计算mIoU和各类IoU。"""
    preds = logits.argmax(dim=1)  # (B, H, W)
    valid = targets != IGNORE_INDEX

    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls) & valid
        gt_cls = (targets == cls) & valid
        union = pred_cls | gt_cls
        intersection = pred_cls & gt_cls
        if union.sum() == 0:
            ious.append(float("nan"))
        else:
            ious.append((intersection.sum().float() / union.sum().float()).item())

    miou = np.nanmean(ious)
    return miou, ious


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int):
    """训练一个epoch。"""
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # 前向
        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["out"]  # FCN格式

        loss = criterion(logits, masks)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 指标
        with torch.no_grad():
            miou, _ = compute_metrics(logits, masks, NUM_CLASSES)

        total_loss += loss.item()
        total_miou += miou
        n_batches += 1

        if batch_idx % 20 == 0:
            logging.info(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={loss.item():.4f} mIoU={miou:.4f}"
            )

    return total_loss / n_batches, total_miou / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证。返回 (loss, mIoU, per_class_IoU, pred_fg_ratio)。"""
    model.eval()
    total_loss = 0.0
    all_ious = []
    total_pred_fg = 0
    total_valid_pixels = 0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["out"]

        loss = criterion(logits, masks)
        miou, ious = compute_metrics(logits, masks, NUM_CLASSES)

        total_loss += loss.item()
        all_ious.append(ious)
        n_batches += 1

        # 预测前景像素比例
        preds = logits.argmax(dim=1)
        valid = masks != IGNORE_INDEX
        total_pred_fg += ((preds == 1) & valid).sum().item()
        total_valid_pixels += valid.sum().item()

    avg_loss = total_loss / n_batches
    avg_ious = np.nanmean(all_ious, axis=0)
    avg_miou = np.nanmean(avg_ious)
    pred_fg_ratio = total_pred_fg / max(total_valid_pixels, 1)

    return avg_loss, avg_miou, avg_ious, pred_fg_ratio


def train(args):
    """主训练函数。"""
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("=" * 60)
    logging.info("SubPipe SegFormer训练")
    logging.info(f"  设备: {device}")
    logging.info(f"  数据目录: {args.data_dir}")
    logging.info(f"  频率: {args.frequency}")
    logging.info(f"  输出目录: {output_dir}")
    logging.info(f"  类别: {CLASS_NAMES}")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  早停: {'禁用(固定epoch)' if args.no_early_stop else f'启用(patience={args.patience})'}")
    logging.info("=" * 60)

    # ── 预处理配置 ────────────────────────────────────────────────
    # --preprocess_strategy 支持 7 种策略
    #   (raw/basic/physd/clahe_only/lee_only/bilateral_only/hist_eq).
    #   优先级: --preprocess_strategy > --preprocess (旧版, 向后兼容).
    preprocess_config = None
    preprocess_fn = None
    if args.preprocess_strategy:
        from sonar_preprocess_adapter import build_preprocess_adapter
        preprocess_fn = build_preprocess_adapter(args.preprocess_strategy,
                                                 dataset="subpipe")
        # raw 返回 None (无预处理); 单操作/raw 都走 preprocess_fn 路径
        if preprocess_fn is not None:
            logging.info(f"预处理策略: {args.preprocess_strategy} "
                         )
        else:
            logging.info(f"预处理策略: {args.preprocess_strategy} (raw, 无预处理)")
    elif args.preprocess and args.preprocess != "none":
        from subpipe_preprocess import get_ablation_configs
        configs = get_ablation_configs()
        if args.preprocess in configs:
            preprocess_config = configs[args.preprocess]
            logging.info(f"预处理: {args.preprocess} (步骤: {preprocess_config.enabled_steps})")
        else:
            logging.warning(f"未知预处理配置: {args.preprocess}，使用无预处理")

    # ── 数据加载 ──────────────────────────────────────────────────
    chunk_split = None
    split_out_file = None
    if args.split_file:
        import json as _json
        with open(args.split_file) as f:
            split_data = _json.load(f)
        chunk_split = {
            "train": split_data.get("train_chunks", []),
            "val": split_data.get("val_chunks", []),
            "test": split_data.get("test_chunks", []),
            "discard": split_data.get("discard_chunks", []),
            "purge_files": split_data.get("purge_files", []),
        }
        if chunk_split.get("purge_files"):
            logging.info(f"边界 purge: {len(chunk_split['purge_files'])} frames "
                         )
        split_out_file = str(output_dir / "data_split_used.json")
        logging.info(f"使用 chunk_split (防数据泄漏): "
                     f"train={chunk_split['train']}, test={chunk_split['test']}, "
                     f"discard={chunk_split['discard']}")

    # 设置随机种子 (可复现性)
    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logging.info(f"随机种子: {args.seed}")

    train_loader, val_loader, info = create_data_loaders(
        data_dir=args.data_dir,
        frequency=args.frequency,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pseudo_label_dir=args.pseudo_label_dir,
        train_ratio=args.train_ratio,
        preprocess_config=preprocess_config,
        chunk_split=chunk_split,
        split_file=split_out_file,
        preprocess_fn=preprocess_fn,
    )
    logging.info(f"训练集: {info['num_train']} 样本")
    logging.info(f"验证集: {info['num_val']} 样本")

    # ── 独立测试集 ──
    # 当 chunk_split 有独立 val 和 test 时, 创建单独的 test_loader
    # 训练过程中只用 val_loader 做 checkpoint 选择, test_loader 仅用于最终评估
    test_loader = None
    if chunk_split is not None and chunk_split.get("val") and chunk_split.get("test"):
        from subpipe_dataset import SubPipeDataset as _SPD
        test_dataset = _SPD(
            data_dir=args.data_dir, frequency=args.frequency,
            split="test", img_size=args.img_size, augment=False,
            pseudo_label_dir=args.pseudo_label_dir,
            chunk_split=chunk_split, preprocess_fn=preprocess_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        info["num_test"] = len(test_dataset)
        logging.info(f"测试集 (独立): {len(test_dataset)} 样本 (chunks={chunk_split['test']})")
        logging.info("test_loader is used only for final evaluation, not for checkpoint selection")
    else:
        info["num_test"] = info["num_val"]  # backward compat: val=test when no independent val

    # ── 保存训练配置 (可复现性) ────────────────────────────────────
    import json as _json_mod
    config_record = {
        "data_dir": args.data_dir,
        "frequency": args.frequency,
        "pseudo_label_dir": args.pseudo_label_dir,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "no_early_stop": args.no_early_stop,
        "train_ratio": args.train_ratio,
        "preprocess": args.preprocess,
        "preprocess_strategy": args.preprocess_strategy,
        "split_file": args.split_file,
        "seed": args.seed,
        "chunk_split": chunk_split,
        "num_train": info["num_train"],
        "num_val": info["num_val"],
        "num_test": info.get("num_test", info["num_val"]),
        "checkpoint_selection_rule": "best_val_fg_iou" if chunk_split and chunk_split.get("val") else "best_val_miou",
        "has_independent_test": test_loader is not None,
    }
    with open(output_dir / "config.json", "w") as f:
        _json_mod.dump(config_record, f, indent=2, ensure_ascii=False)
    logging.info(f"训练配置已保存: {output_dir / 'config.json'}")

    # ── 模型 ──────────────────────────────────────────────────────
    model = build_model_by_backbone(
        backbone=args.backbone, num_classes=NUM_CLASSES,
        in_channels=3, pretrained=False)
    model = model.to(device)

    # ── 损失函数（类别权重：pipeline占比小，权重更高）─────────────
    # SubPipe二分类：pipeline边界框区域占比通常<10%，需要高权重
    class_weights = torch.tensor(
        [0.3, 3.0],  # background, pipeline
        dtype=torch.float32,
    ).to(device)

    criterion = CombinedLoss(
        ce_weight=0.5, dice_weight=0.5,
        class_weights=class_weights, ignore_index=IGNORE_INDEX,
    )

    # ── 优化器 ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── TensorBoard ───────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output_dir / "tensorboard"))
    except ImportError:
        writer = None

    # ── 训练循环 ──────────────────────────────────────────────────
    # 而非 mIoU (背景主导). 无独立 val 时保持原 mIoU 行为 (向后兼容).
    use_fg_iou_selection = (chunk_split is not None and bool(chunk_split.get("val")))
    best_miou_val = 0.0      # best val mIoU (for best_val_miou.pth)
    best_fg_iou = 0.0        # best val FG IoU (primary selection, best_val_fg_iou.pth)
    best_dice = 0.0          # best val FG Dice (for best_val_dice.pth)
    best_epoch = -1          # epoch of best primary metric
    patience_counter = 0
    selection_metric_name = "FG IoU" if use_fg_iou_selection else "mIoU"
    logging.info(f"Checkpoint 选择规则: best val {selection_metric_name} "
                 )
    logging.info(f"保存四类 checkpoint: best_val_fg_iou.pth, best_val_dice.pth, "
                 f"best_val_miou.pth, last.pth")

    # ── 每 epoch 验证指标 CSV ──
    import csv as _csv
    csv_path = output_dir / "epoch_metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = _csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "train_loss", "train_miou",
        "val_loss", "val_miou", "val_fg_iou", "val_bg_iou",
        "val_fg_dice", "val_bg_dice", "pred_fg_ratio",
    ])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_miou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_miou, val_ious, pred_fg_ratio = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        # FG IoU = pipeline IoU (val_ious[1]); BG IoU = background IoU (val_ious[0])
        val_fg_iou = float(val_ious[1]) if len(val_ious) > 1 and not np.isnan(val_ious[1]) else 0.0
        val_bg_iou = float(val_ious[0]) if len(val_ious) > 0 and not np.isnan(val_ious[0]) else 0.0
        # Dice = 2*IoU/(1+IoU) (monotonic, but saved separately for sensitivity analysis)
        val_fg_dice = 2 * val_fg_iou / (1 + val_fg_iou) if val_fg_iou > 0 else 0.0
        val_bg_dice = 2 * val_bg_iou / (1 + val_bg_iou) if val_bg_iou > 0 else 0.0

        elapsed = time.time() - t0
        logging.info(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train loss={train_loss:.4f} mIoU={train_miou:.4f} | "
            f"Val loss={val_loss:.4f} mIoU={val_miou:.4f} FG_IoU={val_fg_iou:.4f} "
            f"FG_Dice={val_fg_dice:.4f} pred_fg_ratio={pred_fg_ratio:.6f}"
        )

        # 写 CSV 行
        csv_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{train_miou:.6f}",
            f"{val_loss:.6f}", f"{val_miou:.6f}", f"{val_fg_iou:.6f}", f"{val_bg_iou:.6f}",
            f"{val_fg_dice:.6f}", f"{val_bg_dice:.6f}", f"{pred_fg_ratio:.6f}",
        ])
        csv_file.flush()

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("mIoU/train", train_miou, epoch)
            writer.add_scalar("mIoU/val", val_miou, epoch)
            writer.add_scalar("FG_IoU/val", val_fg_iou, epoch)
            writer.add_scalar("FG_Dice/val", val_fg_dice, epoch)
            writer.add_scalar("pred_fg_ratio/val", pred_fg_ratio, epoch)
            for name, iou in zip(CLASS_NAMES, val_ious):
                if not np.isnan(iou):
                    writer.add_scalar(f"IoU/val_{name}", iou, epoch)

        # ── 保存四类 checkpoint ──
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": val_miou,
            "val_fg_iou": val_fg_iou,
            "val_fg_dice": val_fg_dice,
            "val_bg_iou": val_bg_iou,
            "val_ious": val_ious,
            "pred_fg_ratio": pred_fg_ratio,
            "selection_metric": selection_metric_name,
        }

        # 1. best_val_fg_iou.pth (primary checkpoint for paper results)
        if val_fg_iou > best_fg_iou:
            best_fg_iou = val_fg_iou
            best_epoch = epoch
            torch.save(ckpt_data, output_dir / "best_val_fg_iou.pth")
            logging.info(f"  ★ best_val_fg_iou: {val_fg_iou:.4f} (epoch {epoch})")

        # 2. best_val_dice.pth (sensitivity analysis)
        if val_fg_dice > best_dice:
            best_dice = val_fg_dice
            torch.save(ckpt_data, output_dir / "best_val_dice.pth")

        # 3. best_val_miou.pth (sensitivity analysis)
        if val_miou > best_miou_val:
            best_miou_val = val_miou
            torch.save(ckpt_data, output_dir / "best_val_miou.pth")

        # 4. last.pth (overwrite every epoch)
        torch.save(ckpt_data, output_dir / "last.pth")

        # Early stopping based on primary selection metric
        current_metric = val_fg_iou if use_fg_iou_selection else val_miou
        best_metric = best_fg_iou if use_fg_iou_selection else best_miou_val
        if current_metric >= best_metric:
            patience_counter = 0
        else:
            patience_counter += 1
            if not args.no_early_stop and patience_counter >= args.patience:
                logging.info(f"  早停: {args.patience}个epoch无改善 (best {selection_metric_name}={best_metric:.4f})")
                break

        # 定期保存checkpoint (diagnostic)
        if epoch % 10 == 0:
            torch.save(ckpt_data, output_dir / f"checkpoint_epoch{epoch}.pth")

    csv_file.close()

    # backward compat: copy primary checkpoint → best_model.pth
    best_primary_path = output_dir / "best_val_fg_iou.pth" if use_fg_iou_selection else output_dir / "best_val_miou.pth"
    if best_primary_path.exists():
        import shutil
        shutil.copy2(best_primary_path, output_dir / "best_model.pth")

    # 保存最终模型信息
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "best_miou": best_miou_val,
        "best_fg_iou": best_fg_iou,
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric_name,
    }, output_dir / "final_model.pth")

    logging.info(f"\n训练完成! 最佳{selection_metric_name}="
                 f"{best_fg_iou if use_fg_iou_selection else best_miou_val:.4f} "
                 f"(epoch={best_epoch}, mIoU={best_miou_val:.4f})")
    logging.info(f"  best_val_fg_iou={best_fg_iou:.4f}, best_val_dice={best_dice:.4f}, best_val_miou={best_miou_val:.4f}")
    logging.info(f"  4 checkpoints: best_val_fg_iou.pth, best_val_dice.pth, best_val_miou.pth, last.pth")
    logging.info(f"  epoch_metrics.csv: {csv_path}")
    logging.info(f"模型保存至: {output_dir}")

    # ── 训练后 evaluate + 写 eval.json ─────────
    # The test set is evaluated only once after the checkpoint-selection rule is frozen.
    # 有独立 test_loader 时用 test_loader, 否则回退到 val_loader (向后兼容)
    try:
        best_ckpt_path = output_dir / "best_model.pth"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(
                best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            best_epoch_eval = int(best_ckpt.get("epoch", -1))
            logging.info(f"\n训练后评估: 加载 best_model.pth (epoch={best_epoch_eval}, "
                         f"selection={best_ckpt.get('selection_metric', 'mIoU')})")

            # 优先使用独立 test_loader; 无则回退到 val_loader
            eval_loader = test_loader if test_loader is not None else val_loader
            eval_loader_name = "test_loader (independent)" if test_loader is not None else "val_loader (fallback, no independent test)"
            logging.info(f"  评估使用: {eval_loader_name}")

            test_loss, test_miou, test_ious, test_pred_fg_ratio = validate(
                model, eval_loader, criterion, device)
            per_class_iou_dict = {}
            for name, iou in zip(CLASS_NAMES, test_ious):
                if not np.isnan(iou):
                    per_class_iou_dict[name] = float(iou)
            test_fg_iou = float(test_ious[1]) if len(test_ious) > 1 and not np.isnan(test_ious[1]) else 0.0
            test_fg_dice = 2 * test_fg_iou / (1 + test_fg_iou) if test_fg_iou > 0 else 0.0
            preprocess_tag = (
                args.preprocess_strategy
                if args.preprocess_strategy and args.preprocess_strategy != "none"
                else (args.preprocess if args.preprocess else "raw")
            )

            # 获取 git commit
            import subprocess as _sp
            try:
                git_commit = _sp.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_DIR),
                    stderr=_sp.DEVNULL).decode().strip()[:12]
            except Exception:
                git_commit = "unknown"

            eval_record = {
                "dataset": "subpipe",
                "checkpoint": str(best_ckpt_path),
                "checkpoint_files": {
                    "primary": "best_val_fg_iou.pth" if use_fg_iou_selection else "best_val_miou.pth",
                    "best_val_fg_iou": "best_val_fg_iou.pth",
                    "best_val_dice": "best_val_dice.pth",
                    "best_val_miou": "best_val_miou.pth",
                    "last": "last.pth",
                    "backward_compat": "best_model.pth (copy of primary)",
                },
                "best_epoch": best_epoch_eval,
                "best_miou": float(best_miou_val),
                "best_fg_iou": float(best_fg_iou),
                "best_dice": float(best_dice),
                "selection_metric": selection_metric_name,
                "per_class_iou": per_class_iou_dict,
                "target_class": {
                    "id": 1,
                    "name": "pipeline",
                },
                "test_metrics": {
                    "miou": float(test_miou),
                    "fg_iou": test_fg_iou,
                    "fg_dice": test_fg_dice,
                    "pred_fg_ratio": test_pred_fg_ratio,
                    "per_class_iou": per_class_iou_dict,
                    "note": f"evaluated on {eval_loader_name}",
                },
                "split_config": {
                    "split_file": args.split_file,
                    "train_chunks": chunk_split.get("train", []) if chunk_split else None,
                    "val_chunks": chunk_split.get("val", []) if chunk_split else None,
                    "test_chunks": chunk_split.get("test", []) if chunk_split else None,
                    "purge_files": chunk_split.get("purge_files", []) if chunk_split else [],
                    "val_equals_test": False if (chunk_split and chunk_split.get("val")) else True,
                    "has_independent_test": test_loader is not None,
                    "split_method": "chunk_split_independent_val_purged" if (args.split_file and chunk_split and chunk_split.get("purge_files")) else ("chunk_split_independent_val" if (args.split_file and test_loader is not None) else ("chunk_split" if args.split_file else "train_ratio")),
                },
                "n_train_samples": int(info["num_train"]),
                "n_val_samples": int(info["num_val"]),
                "n_test_samples": int(info.get("num_test", info["num_val"])),
                "img_size": args.img_size,
                "preprocess": preprocess_tag,
                "seed": args.seed,
                "git_commit": git_commit,
            }
            with open(output_dir / "eval.json", "w") as f:
                json.dump(eval_record, f, indent=2, ensure_ascii=False)
            logging.info(f"eval.json 已生成: {output_dir / 'eval.json'}")
            logging.info(f"  best_{selection_metric_name}={best_fg_iou if use_fg_iou_selection else best_miou_val:.4f}, "
                         f"test_miou={test_miou:.4f}, test_fg_iou={test_fg_iou:.4f}, "
                         f"test_fg_dice={test_fg_dice:.4f}, test_pred_fg_ratio={test_pred_fg_ratio:.6f}")
            logging.info(f"  per_class={per_class_iou_dict}")
            logging.info(f"  split: train_chunks={chunk_split.get('train') if chunk_split else 'N/A'}, "
                         f"val_chunks={chunk_split.get('val') if chunk_split else 'N/A'}, "
                         f"test_chunks={chunk_split.get('test') if chunk_split else 'N/A'}, "
                         f"purge={len(chunk_split.get('purge_files', [])) if chunk_split else 0} files")
        else:
            logging.warning(f"best_model.pth 不存在, 跳过 eval.json 生成")
    except Exception as e:
        logging.error(f"训练后 evaluate 失败: {e}", exc_info=True)

    if writer:
        writer.close()


def export_onnx(args):
    """导出ONNX模型。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = getattr(args, "backbone", "segformer_b2")
    model = build_model_by_backbone(
        backbone=backbone, num_classes=NUM_CLASSES,
        in_channels=3, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)

    onnx_path = args.onnx_model or str(Path(args.checkpoint).parent / "segformer.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"ONNX导出: {onnx_path}")


def export_trt(args):
    """TensorRT优化。"""
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    # FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用FP16")

    # 动态batch
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 512, 512), (8, 3, 512, 512), (16, 3, 512, 512))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("TensorRT引擎构建失败!")
        return

    trt_path = args.onnx_model.replace(".onnx", ".engine")
    with open(trt_path, "wb") as f:
        f.write(engine)
    print(f"TensorRT引擎: {trt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SubPipe SegFormer训练脚本"
    )
    # 训练参数
    parser.add_argument("--data_dir", type=str, required=False,
                        help="SubPipe数据目录")
    parser.add_argument("--frequency", type=str, default="hf",
                        choices=["hf", "lf"])
    parser.add_argument("--pseudo_label_dir", type=str, default=None,
                        help="预生成伪标签目录")
    parser.add_argument("--output_dir", type=str,
                        default="models/segformer_subpipe")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no_early_stop", action="store_true",
                        help="禁用早停，固定训练epochs轮（消融实验用，确保公平对比）")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_file", type=str, default=None,
                        help="chunk-level split JSON (W2 输出). "
                             "若提供, 使用 chunk_split 而非随机划分 (防数据泄漏)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (可复现性)")
    parser.add_argument("--preprocess", type=str, default="none",
                        help="预处理配置: none/baseline/+tvg/+tvg+homomorphic/"
                             "+tvg+homomorphic+wc/+tvg+homomorphic+wc+slant/"
                             "+tvg+homomorphic+wc+slant+denoise/full_pipeline")
    parser.add_argument("--preprocess_strategy", type=str, default=None,
                        choices=list(SUPPORTED_STRATEGIES),
                        help="预处理策略 (raw/basic/physd/tvg_only/tvg_truncated/"
                             "all_physd/clahe_only/lee_only/bilateral_only/hist_eq/...). "
                             "优先级高于 --preprocess")
    parser.add_argument("--backbone", type=str, default="segformer_b2",
                        choices=["segformer_b2", "unet", "deeplabv3plus"],
                        help="Backbone architecture: segformer_b2 (default, "
                             "backward compatible), unet (Ronneberger 2015), "
                             "deeplabv3plus (Chen 2018, torchvision ResNet50)")

    # 导出参数
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--export_trt", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--onnx_model", type=str, default=None)

    args = parser.parse_args()

    if args.export_trt:
        export_trt(args)
    elif args.export_onnx:
        if not args.checkpoint:
            parser.error("--export_onnx 需要 --checkpoint")
        export_onnx(args)
    else:
        if not args.data_dir:
            parser.error("训练需要 --data_dir")
        train(args)


if __name__ == "__main__":
    main()
