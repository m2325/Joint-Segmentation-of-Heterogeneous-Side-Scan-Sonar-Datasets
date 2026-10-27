#!/usr/bin/env python3
"""
tools/training/train_sspm_segformer.py

Train SegFormer-B2 with 5-channel SSPM input on side-scan sonar datasets.

Validates the SSPM-5ch representation (D.18/D.19 P1) against the existing
Raw / Basic / PhysD baselines. The SSPM channels (raw + range_norm + contrast
+ shadow + reliability) are generated from the raw grayscale and fused to 3
channels via a 3x3 conv + BN + GELU adapter before the SegFormer encoder.

Supports three datasets, each with its own class space:
  --dataset ai4     : 2-class {background, target}            (AI4Shipwrecks)
  --dataset subpipe : 2-class {background, pipeline}           (SubPipe chunk split)
  --dataset mcm     : 8-class unified, C3 target (mine)        (MCM year split)

Output files:
  - config.json     (full hyperparams + dataset + sspm config)
  - eval.json       (best epoch test metrics + per_class_iou)
  - train.log
  - best.pth / final.pth

Usage:
  # AI4 with SSPM-5ch (compare against exp_ai4shipwrecks_A_raw 0.6396)
  python tools/training/train_sspm_segformer.py \\
      --dataset ai4 --output_dir models/exp_ai4_sspm_s1 \\
      --epochs 100 --batch_size 4 --lr 1e-4 --seed 1 \\
      --device cuda --num_workers 4

  # SubPipe with SSPM-5ch (compare against exp_subpipe_A_raw 0.4986)
  python tools/training/train_sspm_segformer.py \\
      --dataset subpipe --output_dir models/exp_subpipe_sspm_s1 \\
      --epochs 100 --batch_size 4 --lr 1e-4 --seed 1 \\
      --device cuda --num_workers 4
"""

import os

import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from sspm_channels import generate_sspm_channels, SSPM_NUM_CHANNELS, SSPM_CHANNEL_NAMES  # noqa: E402
from sspm_segformer import build_sspm_model  # noqa: E402
from train_subpipe_segformer import (  # noqa: E402
    DiceLoss,
    CombinedLoss,
    compute_metrics,
)

# ---------- per-dataset config ----------

DATASET_CONFIG = {
    "ai4": {
        "num_classes": 2,
        "class_names": ["background", "target"],
        "target_class": 1,
        "ignore_index": 255,
    },
    "subpipe": {
        "num_classes": 2,
        "class_names": ["background", "pipeline"],
        "target_class": 1,
        "ignore_index": 255,
    },
    "mcm": {
        "num_classes": 8,
        "class_names": [
            "water_column_nadir", "shipwreck", "linear_infrastructure",
            "mine_like_contact", "rock_hard_substrate", "coarse_sediment",
            "fine_sediment", "vegetation_biogenic",
        ],
        "target_class": 3,
        "ignore_index": 255,
    },
}

class SSPMWrapper(Dataset):
    """Wrap a base dataset that returns {"image": (3,H,W), "mask": (H,W), ...}
    to return 5-channel SSPM tensors instead of triplicated 3-channel images.

    Channel 0 of the base image is the resized+preprocessed grayscale in [0,1];
    generate_sspm_channels reproduces all 5 SSPM channels from it.
    """

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img_3ch = item["image"]  # (3, H, W) float32 [0,1]
        raw_2d = img_3ch[0].cpu().numpy()  # (H, W)
        sspm = generate_sspm_channels(raw_2d)  # (5, H, W) float32
        item["image"] = torch.from_numpy(sspm)
        return item

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(output_dir / "train.log")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(fh); root.addHandler(sh)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_loaders(args, cfg):
    """Build train/test loaders for the selected dataset, wrapped with SSPM."""
    if args.dataset == "ai4":
        from ai4shipwrecks_dataset import create_data_loaders, AI4ShipwrecksDataset
        preprocess_fn = None  # SSPM derives from raw; no classical preprocessing
        train_loader, test_loader, info = create_data_loaders(
            data_dir=args.data_dir or "datasets/AI4Shipwrecks",
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preprocess_fn=preprocess_fn,
        )
    elif args.dataset == "subpipe":
        from subpipe_dataset import create_data_loaders
        chunk_split = {
            "train": [1, 2], "val": [3], "test": [3], "discard": [0, 4],
        }
        split_out_file = str(Path(args.output_dir) / "data_split_used.json")
        train_loader, test_loader, info = create_data_loaders(
            data_dir=args.data_dir or "datasets/SubPipe/SubPipe",
            frequency=args.frequency,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preprocess_config=None,  # raw
            chunk_split=chunk_split,
            split_file=split_out_file,
        )
    elif args.dataset == "mcm":
        from mcm_dataset import create_data_loaders
        split_file = str(PROJECT_DIR / "experiments" / "splits" / "mcm_year_split_v1.json")
        train_loader, _val_loader, test_loader, info = create_data_loaders(
            image_dir=args.data_dir or "datasets/MCM_extracted",
            mask_dir=args.mask_dir or "datasets/MCM_masks",
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_file=split_file,
            preprocess_fn=None,
        )
    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    # Wrap the underlying datasets with the SSPM 5-channel transform and
    # rebuild the DataLoaders (DataLoader forbids mutating .dataset post-init).
    train_ds = SSPMWrapper(train_loader.dataset)
    test_ds = SSPMWrapper(test_loader.dataset)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    return train_loader, test_loader, info

def build_class_weights(dataset_name, cfg, device):
    """Per-dataset class weights matching existing baselines."""
    nc = cfg["num_classes"]
    if dataset_name == "ai4":
        w = torch.tensor([0.3, 3.0], dtype=torch.float32)
    elif dataset_name == "subpipe":
        w = torch.tensor([0.3, 3.0], dtype=torch.float32)
    elif dataset_name == "mcm":
        w = torch.tensor([0.1] * 8, dtype=torch.float32)
        w[3] = 3.0
    else:
        w = torch.ones(nc, dtype=torch.float32)
    return w.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_classes):
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    n_batches = 0
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["out"]
        loss = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            miou, _ = compute_metrics(logits, masks, num_classes)
        total_loss += loss.item()
        total_miou += miou
        n_batches += 1
        if batch_idx % 20 == 0:
            logging.info(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={loss.item():.4f} mIoU={miou:.4f}"
            )
    n = max(n_batches, 1)
    return total_loss / n, total_miou / n

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, target_class, ignore_index):
    model.eval()
    total_loss = 0.0
    all_ious = []
    n_batches = 0
    tp = fp = fn = tn = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["out"]
        loss = criterion(logits, masks)
        miou, ious = compute_metrics(logits, masks, num_classes)
        preds = logits.argmax(dim=1)
        valid = masks != ignore_index
        tp += int(((preds == target_class) & (masks == target_class) & valid).sum().item())
        fp += int(((preds == target_class) & (masks != target_class) & valid).sum().item())
        fn += int(((preds != target_class) & (masks == target_class) & valid).sum().item())
        tn += int(((preds != target_class) & (masks != target_class) & valid).sum().item())
        total_loss += loss.item()
        all_ious.append(ious)
        n_batches += 1
    n = max(n_batches, 1)
    avg_loss = total_loss / n
    avg_ious = np.nanmean(all_ious, axis=0) if all_ious else np.array([float("nan")] * num_classes)
    avg_miou = float(np.nanmean(avg_ious))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou_target = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    target_metrics = {
        "target_class_id": int(target_class),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou_target),
        "accuracy": float(accuracy),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    return avg_loss, avg_miou, avg_ious, target_metrics

def train(args):
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = DATASET_CONFIG[args.dataset]
    args.num_classes = cfg["num_classes"]
    target_class = cfg["target_class"]
    ignore_index = cfg["ignore_index"]

    logging.info("=" * 60)
    logging.info(f"SSPM-5ch SegFormer Training — dataset={args.dataset}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Channels: {SSPM_CHANNEL_NAMES}")
    logging.info(f"  Classes: {cfg['num_classes']} ({cfg['class_names']})")
    logging.info(f"  Target class: {cfg['class_names'][target_class]} (id={target_class})")
    logging.info(f"  Epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} seed={args.seed}")
    logging.info("=" * 60)

    set_seed(args.seed)

    train_loader, test_loader, info = build_loaders(args, cfg)
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    logging.info(f"  train samples: {n_train}")
    logging.info(f"  test  samples: {n_test}")

    model = build_sspm_model(num_classes=cfg["num_classes"], input_channels=SSPM_NUM_CHANNELS)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  model params: {n_params/1e6:.2f}M")

    class_weights = build_class_weights(args.dataset, cfg, device)
    criterion = CombinedLoss(
        ce_weight=0.5, dice_weight=0.5,
        class_weights=class_weights, ignore_index=ignore_index,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    config_record = {
        "dataset": args.dataset,
        "input_channels": SSPM_NUM_CHANNELS,
        "sspm_channels": SSPM_CHANNEL_NAMES,
        "num_classes": cfg["num_classes"],
        "class_names": cfg["class_names"],
        "target_class": int(target_class),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "n_train": n_train,
        "n_test": n_test,
        "model_params": int(n_params),
        "adapter": "Conv2d(5,3,3,1,1)+BN+GELU",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2, ensure_ascii=False)
    logging.info(f"config saved: {output_dir / 'config.json'}")

    best_miou = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_miou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.num_classes
        )
        val_loss, val_miou, val_ious, val_target = evaluate(
            model, test_loader, criterion, device, cfg["num_classes"], target_class, ignore_index
        )
        scheduler.step()
        elapsed = time.time() - t0
        logging.info(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train loss={train_loss:.4f} mIoU={train_miou:.4f} | "
            f"Val loss={val_loss:.4f} mIoU={val_miou:.4f} "
            f"{cfg['class_names'][target_class]} IoU={val_target['iou']:.4f}"
        )
        for i, (name, iou) in enumerate(zip(cfg["class_names"], val_ious)):
            if not np.isnan(iou):
                logging.info(f"  {name}: IoU={iou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "val_target_iou": val_target["iou"],
            }, output_dir / "best.pth")
            logging.info(f"  ★ New best! val mIoU={val_miou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"  早停: {args.patience} 个 epoch 无改善")
                break

    # Final evaluation on test set with best checkpoint
    ckpt = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val mIoU={ckpt['val_miou']:.4f})")
    test_loss, test_miou, test_ious, test_target = evaluate(
        model, test_loader, criterion, device, cfg["num_classes"], target_class, ignore_index
    )
    logging.info(f"=== Final test evaluation ===")
    logging.info(f"Test mIoU={test_miou:.4f}  {cfg['class_names'][target_class]} IoU={test_target['iou']:.4f}")
    logging.info(f"训练完成! 最佳 mIoU={best_miou:.4f} (epoch {best_epoch})")
    logging.info(f"模型保存至: {output_dir}")

    eval_record = {
        "dataset": args.dataset,
        "checkpoint": str(output_dir / "best.pth"),
        "best_epoch": best_epoch,
        "best_miou": float(best_miou),
        "test_miou": float(test_miou),
        "per_class_iou": {
            name: (float(v) if not np.isnan(v) else None)
            for name, v in zip(cfg["class_names"], test_ious)
        },
        "target_metrics": test_target,
        "sspm_channels": SSPM_CHANNEL_NAMES,
    }
    with open(output_dir / "eval.json", "w") as f:
        json.dump(eval_record, f, indent=2, ensure_ascii=False)

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, output_dir / "final.pth")

def get_parser():
    p = argparse.ArgumentParser(description="SSPM-5ch SegFormer training")
    p.add_argument("--dataset", choices=["ai4", "subpipe", "mcm"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_dir", default=None, help="override dataset image dir")
    p.add_argument("--mask_dir", default=None, help="override mask dir (MCM only)")
    p.add_argument("--frequency", default="hf", choices=["hf", "lf"], help="SubPipe frequency")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    return p

if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
