"""
Joint multi-dataset SegFormer training with DMCE vs Naive CE.

Trains a single SegFormer-B2 (8-class) on AI4Shipwrecks + SubPipe + MCM
simultaneously. Compares:
  - Naive CE: standard cross-entropy, all 8 classes participate
  - DMCE: logit-masking CE, only per-dataset valid classes participate

Usage:
    # DMCE joint training
    python tools/training/train_joint_dmce.py \
        --loss_type dmce \
        --output_dir models/exp_joint_dmce_s1 \
        --epochs 100 --batch_size 4 --lr 1e-4 --seed 1

    # Naive CE baseline
    python tools/training/train_joint_dmce.py \
        --loss_type naive_ce \
        --output_dir models/exp_joint_naive_ce_s1 \
        --epochs 100 --batch_size 4 --lr 1e-4 --seed 1

    # Smoke test (small samples, few epochs)
    python tools/training/train_joint_dmce.py \
        --loss_type dmce \
        --output_dir /tmp/test_joint_dmce \
        --max_train_samples 10 --epochs 2 --no_early_stop
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
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dmce_loss import build_criterion
from unified_joint_dataset import (
    create_joint_loaders,
    create_v2_val_loaders,
    UNIFIED_NUM_CLASSES,
    UNIFIED_IGNORE_INDEX,
    DATASET_VALID_CLASSES,
)
from train_subpipe_segformer import build_segformer_model, build_model_by_backbone
from train_sspm_segformer import SSPMWrapper
from sspm_segformer import build_sspm_model

DEFAULT_AI4_DIR = "datasets/AI4Shipwrecks"
DEFAULT_SUBPIPE_DIR = "datasets/SubPipe/SubPipe"
DEFAULT_MCM_IMAGE_DIR = "datasets/MCM_extracted"
DEFAULT_MCM_MASK_DIR = "datasets/MCM_masks"
DEFAULT_MCM_SPLIT = str(REPO_ROOT / "splits" / "mcm_year_split_v2.json")

DEFAULT_SUBPIPE_CHUNK_SPLIT = {
    "train": [1, 2],
    "val": [],
    "test": [3],
    "discard": [0, 4],
}

# v2 protocol split directory (group-level isolated train/val/test).
DEFAULT_V2_SPLIT_DIR = str(REPO_ROOT / "splits"
                           / "v2_protocol")

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(output_dir / "train.log", mode="w")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(fh)
    root.addHandler(sh)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Frozen BatchNorm intervention
# ---------------------------------------------------------------------------
# Paper2 §6 documents that DMCE collapse involves invalid-class logits
# drifting under BatchNorm running stats + random init (DMCE puts invalid-class
# weights at zero gradient, so their BN running mean/var evolve freely from
# the initialization noise). Freezing BN is a diagnostic intervention to
# isolate whether BN stat drift is a contributing cause of invalid-class
# hijacking. This is an OPTIONAL flag (--freeze_bn); the default (False)
# preserves the exact training behavior of all prior experiments.

# BN layer name keywords heuristic for 'invalid_only' mode. BN modules whose
# qualified name contains any of these substrings are deemed "near the class
# logits" (i.e. in the segmentation head / decoder / classifier), where
# running stats most directly shape invalid-class logit magnitudes. This is
# a heuristic; per-architecture refinement may be needed.
_INVALID_ONLY_BN_KEYWORDS = (
    "classifier", "head", "decode", "output", "proj", "seg_head", "logits",
)

def apply_freeze_bn(model: torch.nn.Module, mode: str = "all",
                    log: bool = True) -> dict:
    """Freeze BatchNorm layers in ``model`` according to ``mode``.

    Intervention design:
        - momentum=0.0  : running_mean/running_var no longer update in train
          mode (the BN forward pass uses batch stats for normalization but
          leaves running stats unchanged). PyTorch BN running_mean/var are
          buffers (not Parameters), so requires_grad does not apply to them;
          setting momentum=0.0 is the mechanism that freezes them.
        - eval() lock   : force BN to use running stats for normalization
          (not batch stats), matching "BN frozen at pretrained/init values".
        - .train() override : monkey-patch so model.train() in the training
          loop does NOT switch the BN back to train mode. weight/bias
          (affine params) remain trainable — only running stats are locked.

    Args:
        model: the network whose BN layers should be frozen.
        mode:  "all"          — freeze every nn.BatchNorm1d / nn.BatchNorm2d.
               "invalid_only" — freeze only BN layers whose qualified name
                contains a head/classifier/decoder keyword (heuristic; see
                _INVALID_ONLY_BN_KEYWORDS). Targets the BN layers most
                responsible for invalid-class logit magnitudes.
        log:   if True, log a summary via logging.info.

    Returns:
        dict with keys {frozen, total_bn, mode} for inspection / unit tests.
    """
    import types

    if mode not in ("all", "invalid_only"):
        raise ValueError(
            f"freeze_bn_mode must be 'all' or 'invalid_only', got '{mode}'")

    total_bn = 0
    frozen = 0
    frozen_names = []
    for name, m in model.named_modules():
        if not isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            continue
        total_bn += 1
        if mode == "invalid_only":
            lname = name.lower()
            if not any(kw in lname for kw in _INVALID_ONLY_BN_KEYWORDS):
                continue
        # Freeze this BN module.
        m.momentum = 0.0
        m.eval()
        # Override .train() so subsequent model.train() calls leave this BN
        # in eval mode. weight/bias requires_grad is left untouched (still
        # trainable); only running stats are locked.
        _original_train = m.train

        def _frozen_train(self, mode=True, _orig=_original_train):
            # Always force eval mode regardless of `mode`.
            return _orig(False)

        m.train = types.MethodType(_frozen_train, m)
        frozen += 1
        frozen_names.append(name)

    if log:
        import logging as _lg
        _lg.getLogger(__name__).info(
            f"freeze_bn(mode={mode}): frozen {frozen}/{total_bn} "
            f"BatchNorm layers. names={frozen_names[:10]}"
            f"{' ...' if len(frozen_names) > 10 else ''}")
    return {"frozen": frozen, "total_bn": total_bn, "mode": mode,
            "frozen_names": frozen_names}

def compute_metrics_masked(logits, targets, valid_classes, ignore_index=255):
    """Compute mIoU and per-class IoU over valid classes only.

    Pixels predicted as classes outside valid_classes count as errors
    for the valid class they should have been (standard per-class IoU).
    """
    preds = logits.argmax(dim=1)
    valid = targets != ignore_index
    ious = []
    for cls in valid_classes:
        pred_cls = (preds == cls) & valid
        gt_cls = (targets == cls) & valid
        union = pred_cls | gt_cls
        intersection = pred_cls & gt_cls
        if union.sum() == 0:
            ious.append(float("nan"))
        else:
            ious.append(
                (intersection.sum().float() / union.sum().float()).item())
    miou = float(np.nanmean(ious)) if ious else 0.0
    return miou, ious

def apply_eval_logit_mask(logits, valid_classes, num_classes=8, mask_value=-1e9):
    """Mask invalid-class logits before argmax, mirroring DMCE training.

    For DMCE evaluation: set logits of classes NOT in valid_classes to -1e9
    so argmax can only pick from the valid set. This must match the masking
    applied inside DMCELoss during training, otherwise the eval mIoU is
    meaningless (model predicts arbitrary classes from the 8-class space).

    For Naive CE evaluation: do NOT call this (pass raw logits to
    compute_metrics_masked) — Naive CE has no notion of valid classes.
    """
    mask = torch.full((num_classes,), mask_value, device=logits.device, dtype=logits.dtype)
    for c in valid_classes:
        mask[c] = 0.0
    return logits + mask.view(1, -1, 1, 1)

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                    reliability_weighted=False,
                    aux_boundary_criterion=None, aux_boundary_weight=0.0):
    """Train one epoch over the multi-dataset round-robin loader.

    When reliability_weighted=True, extracts SSPM C5 (channel 4) from each
    batch and passes it as per-pixel loss weight. Requires --use_sspm.

    When aux_boundary_criterion is not None, the auxiliary Boundary Loss
    (Kervadec 2019) term is computed per-batch and added to the main loss:
        total_loss = main_loss + aux_boundary_weight * boundary_term
    The boundary term uses signed distance transforms of the GT masks.
    """
    model.train()
    total_loss = 0.0
    per_dataset_loss = {name: 0.0 for name in DATASET_VALID_CLASSES}
    per_dataset_count = {name: 0 for name in DATASET_VALID_CLASSES}
    n_batches = 0

    for dataset_name, batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["out"]

        if reliability_weighted:
            # SSPM 5-channel: index 4 is C5 reliability
            rel_w = images[:, 4].detach()
        else:
            rel_w = None
        # RC 监督解耦: 数据集返回的 bbox 边界带降权图 (若有)。
        # 与 SSPM reliability (若有) 相乘, 兼容两者同时启用的情形。
        if "weight" in batch:
            bw = batch["weight"].to(device)
            rel_w = bw if rel_w is None else rel_w * bw
        if rel_w is not None:
            loss = criterion(logits, masks, dataset_name,
                             reliability_weight=rel_w)
        else:
            loss = criterion(logits, masks, dataset_name)

        if aux_boundary_criterion is not None and aux_boundary_weight > 0.0:
            # boundary_term has its own (valid-mask based) weighting and does
            # not take reliability_weight (it operates on the softmax probs
            # directly, which already incorporate the SSPM-fed image).
            bnd = aux_boundary_criterion.boundary_term(
                logits, masks, dataset_name=dataset_name)
            loss = loss + aux_boundary_weight * bnd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        per_dataset_loss[dataset_name] += loss.item()
        per_dataset_count[dataset_name] += 1
        n_batches += 1

        if n_batches % 20 == 0:
            logging.info(
                f"  Epoch {epoch} [{n_batches}/{len(train_loader)}] "
                f"ds={dataset_name} loss={loss.item():.4f}"
            )

    n = max(n_batches, 1)
    avg_loss = total_loss / n
    per_ds_avg = {}
    for name in DATASET_VALID_CLASSES:
        c = max(per_dataset_count[name], 1)
        per_ds_avg[name] = per_dataset_loss[name] / c
    return avg_loss, per_ds_avg

@torch.no_grad()
def evaluate(model, loader, criterion, device, dataset_name, valid_classes,
             loss_type="dmce", reliability_weighted=False,
             full_space_eval=True, masked_eval=False):
    """Evaluate on a single dataset's loader.

    F5 fix (full-space inference): the primary evaluation protocol is
    full-space inference — no logit masking is applied, so the model can
    predict any of the 8 classes. When full_space_eval=True (default),
    raw logits are used for argmax.

    When masked_eval=True, also compute metrics with DMCE-style logit
    masking (invalid-class logits set to -1e9 before argmax) for
    comparison. This is the legacy evaluation mode.

    For DMCE loss_type: masked eval applies logit masking (legacy).
    For Naive CE: masked eval has no effect (no valid_classes masking).

    Args:
        full_space_eval: if True, compute full-space mIoU (no masking).
        masked_eval: if True, also compute masked mIoU (DMCE-style).

    Returns:
        If both modes: (avg_loss, full_miou, full_ious, masked_miou, masked_ious)
        If only full_space: (avg_loss, full_miou, full_ious)
        If only masked: (avg_loss, masked_miou, masked_ious)
    """
    model.eval()
    total_loss = 0.0
    all_full_ious = []
    all_masked_ious = []
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["out"]

        if reliability_weighted:
            rel_w = images[:, 4].detach()
            loss = criterion(logits, masks, dataset_name,
                             reliability_weight=rel_w)
        else:
            loss = criterion(logits, masks, dataset_name)

        # Full-space inference: raw logits, no masking (F5 primary protocol)
        if full_space_eval:
            fs_miou, fs_ious = compute_metrics_masked(
                logits, masks, valid_classes, UNIFIED_IGNORE_INDEX)
            all_full_ious.append(fs_ious)

        # Masked inference: DMCE-style logit masking (legacy / comparison)
        if masked_eval:
            masked_logits = logits
            if loss_type in ("dmce", "valid_masked_dmce",
                             "margin_reg_dmce", "soft_mask_dmce",
                             "classifier_suppression_dmce"):
                masked_logits = apply_eval_logit_mask(
                    logits, valid_classes, UNIFIED_NUM_CLASSES)
            m_miou, m_ious = compute_metrics_masked(
                masked_logits, masks, valid_classes, UNIFIED_IGNORE_INDEX)
            all_masked_ious.append(m_ious)

        total_loss += loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    avg_loss = total_loss / n

    # Aggregate full-space results
    full_miou = None
    full_ious_list = None
    if full_space_eval and all_full_ious:
        avg_full_ious = np.nanmean(all_full_ious, axis=0)
        full_miou = float(np.nanmean(avg_full_ious)) if len(avg_full_ious) > 0 else 0.0
        full_ious_list = avg_full_ious.tolist()
    elif full_space_eval:
        full_ious_list = [float("nan")] * len(valid_classes)
        full_miou = 0.0

    # Aggregate masked results
    masked_miou = None
    masked_ious_list = None
    if masked_eval and all_masked_ious:
        avg_masked_ious = np.nanmean(all_masked_ious, axis=0)
        masked_miou = float(np.nanmean(avg_masked_ious)) if len(avg_masked_ious) > 0 else 0.0
        masked_ious_list = avg_masked_ious.tolist()
    elif masked_eval:
        masked_ious_list = [float("nan")] * len(valid_classes)
        masked_miou = 0.0

    if full_space_eval and masked_eval:
        return avg_loss, full_miou, full_ious_list, masked_miou, masked_ious_list
    elif full_space_eval:
        return avg_loss, full_miou, full_ious_list
    else:
        return avg_loss, masked_miou, masked_ious_list

def build_class_weights(device, num_classes=8, bg_weight=0.1, fg_weight=3.0):
    """Class weighting: rare target classes get higher weight.

    bg_weight/fg_weight 由 CLI 传入 (审稿人大修敏感性实验用); 默认 0.1/3.0
    与 corrected-protocol 全部主表运行严格一致 (heuristic prior, 非调参)。
    """
    weights = [bg_weight] * num_classes
    weights[1] = fg_weight   # C1 shipwreck (rare in AI4)
    weights[2] = fg_weight   # C2 pipeline (rare in SubPipe)
    weights[3] = fg_weight   # C3 mine (very rare in MCM)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train(args):
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logging.info("=" * 60)
    logging.info(f"Joint DMCE Training — loss={args.loss_type} "
                 f"a8_scheme={args.a8_scheme} sampler={args.sampler}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Datasets: AI4 + SubPipe + MCM (8-class unified)")
    logging.info(f"  Epochs={args.epochs} batch_size={args.batch_size} "
                 f"lr={args.lr} seed={args.seed}")
    # log split version and checkpoint selection protocol
    logging.info(f"  Split version: {args.split_version} "
                 f"(v2=group-level val/test isolation, v1=legacy)")
    # log evaluation mode
    eval_mode_log = "full_space" if args.full_space_eval else "masked"
    if args.full_space_eval and args.masked_eval:
        eval_mode_log = "both (full_space primary + masked comparison)"
    logging.info(f"  Eval mode: {eval_mode_log}")
    if args.sampler == "class_balanced":
        logging.info(f"  Class-balanced sampler: seed={args.sampler_seed} "
                     f"scan_max_samples={args.scan_max_samples}")
    elif args.sampler == "dataset_balanced":
        logging.info(f"  Dataset-balanced sampler (DomainBalancedMultiLoader "
                     f"mode=uniform): each dataset gets equal sampling probability")
    logging.info(f"  Valid classes:")
    for name, classes in DATASET_VALID_CLASSES.items():
        logging.info(f"    {name}: {classes}")
    logging.info("=" * 60)

    set_seed(args.seed)

    subpipe_chunk_split = None
    if not args.no_subpipe_chunk_split:
        subpipe_chunk_split = DEFAULT_SUBPIPE_CHUNK_SPLIT

    if args.strict_v2_loader:
        # --- Corrected protocol: strict v2 split-list loading ---
        from unified_joint_dataset import (
            verify_v2_split_isolation, create_v2_split_loaders)
        logging.info("[STRICT-V2] Corrected protocol: isolation check + strict list loading")
        split_verification = verify_v2_split_isolation(DEFAULT_V2_SPLIT_DIR)
        train_loader, val_loaders_strict, eval_loaders, info = \
            create_v2_split_loaders(
                ai4_dir=args.ai4_dir,
                subpipe_dir=args.subpipe_dir,
                mcm_image_dir=args.mcm_image_dir,
                mcm_mask_dir=args.mcm_mask_dir,
                v2_split_dir=DEFAULT_V2_SPLIT_DIR,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                preprocess_strategy=args.preprocess_strategy,
                sampler_type=args.sampler,
                sampler_seed=args.sampler_seed)
        # split 校验报告持久化 (audit 证据)
        with open(output_dir / "split_verification.json", "w") as f:
            json.dump(split_verification, f, indent=2, ensure_ascii=False)
        logging.info("[STRICT-V2] split_verification.json 已保存 "
                     f"(all_pass={split_verification['all_pass']})")
    else:
        # --- Legacy loader (kept for reference; provisional pipeline) ---
        train_loader, eval_loaders, info = create_joint_loaders(
        ai4_dir=args.ai4_dir,
        subpipe_dir=args.subpipe_dir,
        mcm_image_dir=args.mcm_image_dir,
        mcm_mask_dir=args.mcm_mask_dir,
        mcm_split_file=args.mcm_split_file,
        subpipe_chunk_split=subpipe_chunk_split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        sampler_type=args.sampler,
        sampler_seed=args.sampler_seed,
        scan_max_samples=args.scan_max_samples,
        preprocess_strategy=args.preprocess_strategy,
        ai4_degrade_to_box=args.ai4_degrade_to_box,
        box_edge_downweight=args.box_edge_downweight,
        box_edge_band=args.box_edge_band,
    )

    held_out_dataset = args.held_out_dataset if args.held_out_dataset != "none" else None
    if held_out_dataset is not None:
        if held_out_dataset not in train_loader.dataloaders:
            raise ValueError(
                f"--held_out_dataset {held_out_dataset} not found in "
                f"train_loader (available: "
                f"{list(train_loader.dataloaders.keys())})")
        if held_out_dataset not in eval_loaders:
            raise ValueError(
                f"--held_out_dataset {held_out_dataset} not found in "
                f"eval_loaders (available: {list(eval_loaders.keys())})")
        removed_train_samples = info["datasets"][held_out_dataset]["train_samples"]
        del train_loader.dataloaders[held_out_dataset]
        train_loader.dataset_names = list(train_loader.dataloaders.keys())
        logging.info("=" * 60)
        logging.info(f"[HELD-OUT] Cross-domain held-out eval mode:")
        logging.info(f"  Held-out dataset: {held_out_dataset} "
                     f"(train_samples removed: {removed_train_samples})")
        logging.info(f"  Training datasets: {list(train_loader.dataloaders.keys())}")
        logging.info(f"  Eval datasets (including held-out for zero-shot): "
                     f"{list(eval_loaders.keys())}")
        logging.info(f"  Held-out eval will measure zero-shot cross-domain "
                     f"transfer to {held_out_dataset}.")
        logging.info("=" * 60)

    # --- SCI 补充阶段4: Oracle 单数据集训练 ---
    # 仅在指定数据集上训练 (其余从 train/val 移除), test 仍评估全部数据集,
    # 用于 Oracle 上界 (如 --train_only_dataset mcm = 仅 MCM 全监督).
    train_only_dataset = (args.train_only_dataset
                          if args.train_only_dataset != "none" else None)
    if train_only_dataset is not None and held_out_dataset is not None:
        raise SystemExit(
            "--train_only_dataset 与 --held_out_dataset 互斥 "
            "(Oracle 单数据集训练 vs LODO 留一协议)")
    if train_only_dataset is not None:
        if train_only_dataset not in train_loader.dataloaders:
            raise ValueError(
                f"--train_only_dataset {train_only_dataset} not found in "
                f"train_loader (available: "
                f"{list(train_loader.dataloaders.keys())})")
        removed = [n for n in list(train_loader.dataloaders.keys())
                   if n != train_only_dataset]
        for n in removed:
            del train_loader.dataloaders[n]
        train_loader.dataset_names = [train_only_dataset]
        logging.info("=" * 60)
        logging.info(f"[TRAIN-ONLY] Oracle single-dataset training mode:")
        logging.info(f"  Training dataset: {train_only_dataset} (removed: {removed})")
        logging.info(f"  Test datasets (all, for reference): "
                     f"{list(eval_loaders.keys())})")
        logging.info("=" * 60)

    total_train = sum(d["train_samples"] for n, d in info["datasets"].items()
                      if n != held_out_dataset) if held_out_dataset \
        else sum(d["train_samples"] for d in info["datasets"].values())
    logging.info(f"Total train samples (all datasets): {total_train}")
    logging.info(f"Total train batches per epoch: {len(train_loader)}")

    # --- F6 fix: build validation loaders for checkpoint selection ---
    # Checkpoint must be selected by
    # validation mIoU on an independent group-level val set, NOT by test
    # mIoU. When split_version="v2", eval_loaders from create_joint_loaders
    # are the TEST sets (unchanged from v1). We build separate val_loaders
    # from the v2_protocol split files for per-epoch checkpoint selection.
    # Test sets are evaluated ONLY ONCE at the end of training.
    # When split_version="v1" (legacy), val_loaders = eval_loaders (test
    # used for both checkpoint selection and final eval, as before).
    val_loaders = {}
    test_loaders = eval_loaders  # rename for clarity (test sets)
    if args.strict_v2_loader:
        # Corrected protocol: val 已严格来自 v2 val 清单 (与 train/test 同源
        # 校验过无重叠), 无需再建 F6 val loaders; checkpoint 选择口径 = 严格 val。
        val_loaders = val_loaders_strict
        logging.info("[STRICT-V2] val_loaders 来自 v2 清单 "
                     f"({ {k: len(v.dataset) for k, v in val_loaders.items()} })")
    elif args.split_version == "v2":
        logging.info(f"  Building v2 validation loaders from "
                     f"{DEFAULT_V2_SPLIT_DIR}")
        val_loaders = create_v2_val_loaders(
            ai4_dir=args.ai4_dir,
            subpipe_dir=args.subpipe_dir,
            mcm_image_dir=args.mcm_image_dir,
            mcm_mask_dir=args.mcm_mask_dir,
            v2_split_dir=DEFAULT_V2_SPLIT_DIR,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_eval_samples=args.max_eval_samples,
            preprocess_strategy=args.preprocess_strategy,
        )
        if not val_loaders:
            logging.warning(
                "  v2 val_loaders is empty! Falling back to v1 "
                "behavior (test set used for checkpoint selection).")
            val_loaders = test_loaders
    else:
        # v1 legacy: use test sets for checkpoint selection (old behavior)
        val_loaders = test_loaders

    # train_only 模式: checkpoint 选择只看目标数据集的 val mIoU
    if train_only_dataset is not None and isinstance(val_loaders, dict):
        removed_val = [k for k in list(val_loaders.keys())
                       if k != train_only_dataset]
        for k in removed_val:
            val_loaders.pop(k, None)
        if removed_val:
            logging.info(f"[TRAIN-ONLY] val_loaders 保留 {train_only_dataset} "
                         f"(checkpoint 选择口径), 移除 {removed_val}")

    if args.use_sspm:
        # Wrap each per-dataset's underlying dataset with SSPM 5-channel
        # transform. PyTorch DataLoader forbids in-place mutation of
        # .dataset after init (raises ValueError), so we rebuild each
        # DataLoader preserving its original kwargs. MultiDatasetLoader
        # holds dataloaders as a dict, so we replace entries in place.
        # SSPM only transforms images (3ch -> 5ch), not masks, so
        # class-frequency stats used by class-balanced samplers stay valid
        # if we preserve the sampler.
        import torch.utils.data as torch_data
        logging.info("Wrapping joint datasets with SSPM 5-channel representation")

        def _rebuild_with_sspm(dl):
            """Rebuild a DataLoader with its dataset wrapped by SSPM.

            Preserves batch_size / num_workers / pin_memory / drop_last /
            collate_fn / persistent_workers / prefetch_factor. For sampler:
            - WeightedRandomSampler / class-balanced sampler: reuse sampler
              (preserves class-frequency weights)
            - RandomSampler (shuffle=True train loader): use shuffle=True
            - SequentialSampler (eval loader): use shuffle=False
            """
            sspm_ds = SSPMWrapper(dl.dataset)
            sampler = dl.sampler
            common_kwargs = dict(
                batch_size=dl.batch_size,
                num_workers=dl.num_workers,
                pin_memory=dl.pin_memory,
                drop_last=dl.drop_last,
                collate_fn=dl.collate_fn,
                persistent_workers=getattr(dl, "persistent_workers", False),
            )
            if dl.num_workers > 0 and getattr(dl, "prefetch_factor", None) is not None:
                common_kwargs["prefetch_factor"] = dl.prefetch_factor

            if isinstance(sampler, torch_data.WeightedRandomSampler):
                # Class-balanced sampler: preserve it (SSPM doesn't change masks)
                common_kwargs["sampler"] = sampler
            elif isinstance(sampler, torch_data.RandomSampler):
                # Train loader created with shuffle=True
                common_kwargs["shuffle"] = True
            else:
                # Eval loader (SequentialSampler) — no shuffle
                common_kwargs["shuffle"] = False
            return DataLoader(sspm_ds, **common_kwargs)

        # Rebuild train loaders (MultiDatasetLoader.dataloaders dict)
        for name, dl in list(train_loader.dataloaders.items()):
            train_loader.dataloaders[name] = _rebuild_with_sspm(dl)
        # Rebuild test loaders (plain dict, renamed from eval_loaders)
        for name, dl in list(test_loaders.items()):
            test_loaders[name] = _rebuild_with_sspm(dl)
        # also wrap val_loaders if they were built (v2 mode) and
        # are distinct from test_loaders.
        if args.split_version == "v2" and val_loaders is not test_loaders:
            for name, dl in list(val_loaders.items()):
                val_loaders[name] = _rebuild_with_sspm(dl)
        logging.info("SSPM wrapping complete (5ch: raw + range_norm + contrast + shadow + reliability)")

    class_weights = build_class_weights(
        device, UNIFIED_NUM_CLASSES,
        bg_weight=args.bg_class_weight, fg_weight=args.fg_class_weight)
    # Parse Class-Balanced CE per-class samples (comma-separated) if given.
    cb_samples_tensor = None
    if args.loss_type == "class_balanced_ce" and args.cb_class_samples:
        try:
            cb_samples_tensor = torch.tensor(
                [float(x) for x in args.cb_class_samples.split(",")],
                dtype=torch.float32)
        except ValueError as e:
            raise SystemExit(f"--cb_class_samples parse error: {e}")
        if cb_samples_tensor.numel() != UNIFIED_NUM_CLASSES:
            raise SystemExit(
                f"--cb_class_samples must have {UNIFIED_NUM_CLASSES} entries, "
                f"got {cb_samples_tensor.numel()}")
    criterion_kwargs = dict(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        dice_smooth=args.dice_smooth,
        gen_dice_smooth=args.gen_dice_smooth,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_smooth=args.tversky_smooth,
        ohem_ratio=args.ohem_ratio,
        cb_beta=args.cb_beta,
        class_samples=cb_samples_tensor,
        boundary_alpha=args.aux_boundary_alpha,
        # When --loss_type boundary is used as a standalone main loss (no aux
        # flag), use combined mode (CE + boundary). When aux mode is on, the
        # main loss is something else and the BoundaryLoss(boundary_only=True)
        # is constructed separately below as aux_boundary_criterion.
        boundary_only=(not (args.loss_type == "boundary"
                            and not args.aux_boundary_loss)),
        # SBA-CE specific
        sba_img_width=args.sba_img_width,
        sba_sigma_water=args.sba_sigma_water,
        sba_gamma=args.sba_gamma,
        sba_foreground_classes=[
            int(c) for c in args.sba_foreground_classes.split(",")],
        # A8 invalid-class constraint correction
        a8_margin=args.a8_margin,
        a8_lambda=args.a8_lambda,
        a8_alpha=args.a8_alpha,
    )
    # A8 scheme overrides --loss_type when set to a non-'none' value.
    # The A8 loss classes inherit DMCE masking, so they are drop-in
    # replacements for --loss_type dmce with added invalid-class regularization.
    effective_loss_type = args.loss_type
    if args.a8_scheme == "margin_reg":
        effective_loss_type = "margin_reg_dmce"
        logging.info(f"  A8-1 Margin-Regularized DMCE: margin={args.a8_margin} "
                     f"lambda={args.a8_lambda}")
    elif args.a8_scheme == "soft_mask":
        effective_loss_type = "soft_mask_dmce"
        logging.info(f"  A8-2 Soft Mask DMCE: alpha={args.a8_alpha}")
    elif args.a8_scheme == "classifier_suppression":
        effective_loss_type = "classifier_suppression_dmce"
        logging.info(f"  A8-3 Classifier Suppression DMCE: lambda={args.a8_lambda}")
    criterion = build_criterion(
        loss_type=effective_loss_type,
        dataset_valid_classes=DATASET_VALID_CLASSES,
        ignore_index=UNIFIED_IGNORE_INDEX,
        num_classes=UNIFIED_NUM_CLASSES,
        class_weights=class_weights,
        **criterion_kwargs,
    )
    criterion = criterion.to(device)

    # Optional auxiliary Boundary Loss (Kervadec 2019). Added on top of the
    # main loss with weight args.aux_boundary_weight. Built as a separate
    # BoundaryLoss(boundary_only=True) instance so it contributes only the
    # distance-transform term; the main loss provides the region signal.
    aux_boundary_criterion = None
    if args.aux_boundary_loss:
        from dmce_loss import BoundaryLoss
        aux_boundary_criterion = BoundaryLoss(
            alpha=args.aux_boundary_alpha,
            ignore_index=UNIFIED_IGNORE_INDEX,
            num_classes=UNIFIED_NUM_CLASSES,
            boundary_only=True,
            dataset_valid_classes=DATASET_VALID_CLASSES,
        ).to(device)
        logging.info(
            f"  Auxiliary Boundary Loss enabled: weight="
            f"{args.aux_boundary_weight} alpha={args.aux_boundary_alpha} "
            f"(Kervadec 2019; uses scipy.ndimage.distance_transform_edt)")

    # PLA-SegFormer: 通过环境变量传递给 build_segformer_model
    if args.use_pla:
        if args.backbone != "segformer_b2":
            raise SystemExit(
                f"--use_pla 仅支持 --backbone segformer_b2 (当前: {args.backbone})")
        os.environ["USE_PLA"] = "1"
        os.environ["PLA_SIGMA_BASE"] = str(args.pla_sigma_local)
        logging.info(
            f"PLA-SegFormer 已启用 (USE_PLA=1, sigma_base={args.pla_sigma_local})")

    if args.use_sspm and args.backbone == "segformer_b2":
        # 5-channel SegFormer path: SSPM 5ch -> 3x3 conv adapter -> SegFormer-B2
        # (preserves the existing segformer_b2 + --use_sspm behavior unchanged)
        model = build_sspm_model(num_classes=UNIFIED_NUM_CLASSES).to(device)
        logging.info("Built SegFormerSonarSSPM (5ch input via 3x3 conv adapter)")
    else:
        in_ch = 5 if args.use_sspm else 3
        model = build_model_by_backbone(
            backbone=args.backbone,
            num_classes=UNIFIED_NUM_CLASSES,
            in_channels=in_ch,
            pretrained=args.pretrained,
        ).to(device)
        logging.info(
            f"Built backbone={args.backbone} in_channels={in_ch} "
            f"pretrained={args.pretrained}")

    # --- SCI 补充阶段4: Fine-tuned 跨域下界实验 ---
    # 从已有 checkpoint (best.pth 格式: dict 含 model_state_dict) 加载权重
    # 作为初始化 (非 resume: 不加载 optimizer/epoch, 从 Epoch 0 重新训练).
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.is_file():
            raise SystemExit(f"--init_from 文件不存在: {init_path}")
        ckpt = torch.load(str(init_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        logging.info("=" * 60)
        logging.info(f"[INIT-FROM] Weights initialized from: {init_path}")
        logging.info(f"  (fine-tuning mode: optimizer/epoch NOT restored)")
        logging.info("=" * 60)

    # Frozen BN intervention: applied AFTER
    # model creation and BEFORE the training loop. The .train() override in
    # apply_freeze_bn ensures model.train() inside train_one_epoch() does not
    # unfreeze the BN layers. Default (args.freeze_bn=False) skips this entirely.
    if args.freeze_bn:
        freeze_report = apply_freeze_bn(model, mode=args.freeze_bn_mode)
        # Persist the intervention in config.json for reproducibility audit.
        _freeze_record = {
            "freeze_bn": True,
            "freeze_bn_mode": args.freeze_bn_mode,
            "frozen_count": freeze_report["frozen"],
            "total_bn_count": freeze_report["total_bn"],
            "frozen_names": freeze_report["frozen_names"],
        }
    else:
        _freeze_record = {"freeze_bn": False, "freeze_bn_mode": None}

    # A8-3: Register classifier module so ClassifierSuppressionDMCELoss can
    # access the classifier weight for L2 regularization on invalid classes.
    # Called after model creation (the classifier head exists by this point).
    if args.a8_scheme == "classifier_suppression":
        from dmce_loss import find_classifier_module
        cls_mod = find_classifier_module(model, num_classes=UNIFIED_NUM_CLASSES)
        if cls_mod is not None and hasattr(criterion, "set_classifier_module"):
            criterion.set_classifier_module(cls_mod)
            logging.info(
                f"  A8-3 classifier module registered: "
                f"{type(cls_mod).__name__} weight shape={tuple(cls_mod.weight.shape)}")
        else:
            logging.warning(
                "  A8-3: could not locate classifier module in "
                f"{args.backbone}; ClassifierSuppressionDMCELoss will "
                "degrade to standard DMCE (no weight regularization).")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    config_record = {
        "experiment_type": "joint_dmce",
        "loss_type": args.loss_type,
        "use_sspm": args.use_sspm,
        "reliability_weighted_loss": args.reliability_weighted_loss,
        "held_out_dataset": held_out_dataset,
        # Protocol audit fields (recorded per run)
        "protocol": {
            "type": "joint3",
            "included_datasets": ["ai4", "subpipe", "mcm"],
            "excluded_datasets": [],
            "loader": ("strict_v2_split_lists" if args.strict_v2_loader
                       else "legacy_provisional"),
            "split_protocol": ("corrected_v2_split" if args.strict_v2_loader
                               else "legacy_loader_v2_val"),
            "expected_split_counts": ({"ai4": {"train": 121, "val": 20, "test": 120},
                                       "subpipe": {"train": 540, "val": 537, "test": 189},
                                       "mcm": {"train": 558, "val": 564, "test": 48}}
                                      if args.strict_v2_loader else None),
        },
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "freeze_bn": _freeze_record,
        "use_pla": args.use_pla,
        "pla_sigma_local": args.pla_sigma_local,
        "sba_gamma": args.sba_gamma,
        "sba_sigma_water": args.sba_sigma_water,
        "a8_scheme": args.a8_scheme,
        "a8_margin": args.a8_margin,
        "a8_lambda": args.a8_lambda,
        "a8_alpha": args.a8_alpha,
        "bg_class_weight": args.bg_class_weight,
        "fg_class_weight": args.fg_class_weight,
        "effective_loss_type": effective_loss_type,
        # RC 监督解耦
        "ai4_degrade_to_box": args.ai4_degrade_to_box,
        "box_edge_downweight": args.box_edge_downweight,
        "box_edge_band": args.box_edge_band,
        "datasets": list(DATASET_VALID_CLASSES.keys()),
        "dataset_valid_classes": DATASET_VALID_CLASSES,
        "num_classes": UNIFIED_NUM_CLASSES,
        "ignore_index": UNIFIED_IGNORE_INDEX,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "patience": args.patience,
        "subpipe_chunk_split": subpipe_chunk_split,
        "mcm_split_file": args.mcm_split_file,
        "class_weights": class_weights.tolist(),
        "sampler": {
            "type": args.sampler,
            "sampler_seed": args.sampler_seed,
            "scan_max_samples": args.scan_max_samples,
        },
        "dataset_info": info["datasets"],
        "total_train_samples": total_train,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2, ensure_ascii=False)
    logging.info("config.json saved")

    # Checkpoint filenames include backbone name to prevent accidental
    # overwriting of SegFormer-B2 weights when the same --output_dir is
    # reused across backbone runs (e.g. ablation sweeps). The "best" model
    # is still the canonical "best.pth" — but the backbone-tagged alias
    # "best_<backbone>.pth" is also saved as a redundant safety copy.
    best_ckpt_name = f"best_{args.backbone}.pth"
    final_ckpt_name = f"final_{args.backbone}.pth"

    best_joint_miou = -float("inf")
    best_epoch = 0
    best_metrics = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, per_ds_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            reliability_weighted=args.reliability_weighted_loss,
            aux_boundary_criterion=aux_boundary_criterion,
            aux_boundary_weight=args.aux_boundary_weight
            if args.aux_boundary_loss else 0.0)

        # evaluate on VALIDATION set only (not test) for checkpoint
        # selection. Test metrics are NOT logged here to avoid info leakage.
        # Test set (test_loaders) is evaluated ONLY ONCE at the end.
        # per-epoch eval uses full_space (primary) only for speed;
        # masked mode (if requested) is computed once at final eval.
        eval_results = {}
        joint_mious = []
        held_out_miou = None
        for name, loader in val_loaders.items():
            valid = DATASET_VALID_CLASSES[name]
            eloss, emiou, eious = evaluate(
                model, loader, criterion, device, name, valid,
                loss_type=effective_loss_type,
                reliability_weighted=args.reliability_weighted_loss,
                full_space_eval=args.full_space_eval, masked_eval=False)
            # rare_class_miou placeholder (kept for metric schema stability)
            rare_miou = None
            is_held_out = (held_out_dataset is not None and name == held_out_dataset)
            eval_results[name] = {
                "loss": eloss,
                "miou": emiou,
                "per_class_iou": eious,
                "rare_class_miou": rare_miou,
                "is_held_out": is_held_out,
            }
            if is_held_out:
                held_out_miou = emiou
            else:
                joint_mious.append(emiou)

        joint_miou = float(np.mean(joint_mious)) if joint_mious else 0.0
        scheduler.step()
        elapsed = time.time() - t0

        ds_summary_parts = []
        for n in eval_results:
            part = f"{n}: val_mIoU={eval_results[n]['miou']:.4f}"
            if eval_results[n].get("is_held_out"):
                part += " (held-out)"
            ds_summary_parts.append(part)
        ds_summary = " | ".join(ds_summary_parts)
        held_out_tag = f" | held-out {held_out_dataset} val_mIoU={held_out_miou:.4f}" \
            if held_out_miou is not None else ""
        logging.info(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train loss={train_loss:.4f} | Val mIoU={joint_miou:.4f}"
            f"{held_out_tag} [{ds_summary}]"
        )

        if joint_miou > best_joint_miou:
            best_joint_miou = joint_miou
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "val_joint_miou": joint_miou,
                "held_out_dataset": held_out_dataset,
                "held_out_miou_at_best": held_out_miou,
                "per_dataset_val": eval_results,
            }
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "joint_miou": joint_miou,
                "backbone": args.backbone,
            }, output_dir / "best.pth")
            # Backbone-tagged alias so SegFormer weights are never overwritten
            # by a U-Net / DeepLabV3+ run that reuses the same output_dir.
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "joint_miou": joint_miou,
                "backbone": args.backbone,
            }, output_dir / best_ckpt_name)
            logging.info(f"  ★ New best! Val mIoU={joint_miou:.4f}")
        else:
            patience_counter += 1
            if not args.no_early_stop and patience_counter >= args.patience:
                logging.info(f"  Early stop: {args.patience} epochs no improvement")
                break

    if best_metrics is None:
        best_epoch = epoch
        best_metrics = {
            "epoch": epoch,
            "val_joint_miou": 0.0,
            "per_dataset_val": eval_results,
        }
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "joint_miou": 0.0,
            "backbone": args.backbone,
        }, output_dir / "best.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "joint_miou": 0.0,
            "backbone": args.backbone,
        }, output_dir / best_ckpt_name)

    # --- Final TEST evaluation (best model, F6: evaluated once at end) ---
    # test set is evaluated ONLY ONCE here, after training is
    # complete (early stopping triggered or all epochs done). The best
    # checkpoint was selected by validation mIoU, NOT by test mIoU.
    # evaluate with both full_space (primary) and masked (if
    # --masked_eval is set) so both metrics are recorded in eval.json.
    logging.info("=== Final TEST evaluation (best model, evaluated once) ===")
    model.load_state_dict(
        torch.load(output_dir / "best.pth", map_location=device)["model_state_dict"])
    final_results = {}
    for name, loader in test_loaders.items():
        valid = DATASET_VALID_CLASSES[name]
        eval_ret = evaluate(
            model, loader, criterion, device, name, valid,
            loss_type=effective_loss_type,
            reliability_weighted=args.reliability_weighted_loss,
            full_space_eval=args.full_space_eval,
            masked_eval=args.masked_eval)
        # evaluate() returns different tuple lengths based on flags:
        #   both True: (loss, fs_miou, fs_ious, m_miou, m_ious)
        #   full_space only: (loss, fs_miou, fs_ious)
        #   masked only: (loss, m_miou, m_ious)
        if args.full_space_eval and args.masked_eval:
            floss, fs_miou, fs_ious, m_miou, m_ious = eval_ret
        elif args.full_space_eval:
            floss, fs_miou, fs_ious = eval_ret
            m_miou, m_ious = None, None
        else:  # masked only
            floss, m_miou, m_ious = eval_ret
            fs_miou, fs_ious = None, None
        # rare_class_miou placeholder (kept for metric schema stability)
        rare_miou = None
        result_entry = {
            "loss": floss,
            "full_space_miou": fs_miou,
            "full_space_per_class_iou": fs_ious,
            "rare_class_miou": rare_miou,
        }
        if args.masked_eval:
            result_entry["masked_miou"] = m_miou
            result_entry["masked_per_class_iou"] = m_ious
        # For backward compat: also store "miou" = primary metric
        # (full_space if available, else masked)
        primary_miou = fs_miou if fs_miou is not None else m_miou
        result_entry["miou"] = primary_miou
        result_entry["per_class_iou"] = primary_ious
        # Preserve is_held_out flag for downstream consumers
        is_held_out = (held_out_dataset is not None and name == held_out_dataset)
        result_entry["is_held_out"] = is_held_out
        final_results[name] = result_entry
        log_line = (f"  {name}: full_space_mIoU={fs_miou:.4f} "
                    f"per_class={[f'{v:.4f}' for v in (fs_ious or [])]}")
        if args.masked_eval and m_miou is not None:
            log_line += f" | masked_mIoU={m_miou:.4f}"
        if rare_miou is not None:
            log_line += f" rare(C467)={rare_miou:.4f}"
        if is_held_out:
            log_line += " (held-out)"
        logging.info(log_line)

    # Compute joint test mIoU (primary metric, excluding held-out)
    test_joint_mious = []
    for n, r in final_results.items():
        if r.get("is_held_out"):
            continue
        pm = r.get("full_space_miou")
        if pm is None:
            pm = r.get("masked_miou")
        if pm is not None:
            test_joint_mious.append(pm)
    test_joint_miou = float(np.mean(test_joint_mious)) if test_joint_mious else 0.0

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "num_classes": UNIFIED_NUM_CLASSES,
        "loss_type": args.loss_type,
        "best_miou": best_joint_miou,
        "backbone": args.backbone,
    }, output_dir / "final.pth")
    # Backbone-tagged alias (matches best_<backbone>.pth convention).
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "num_classes": UNIFIED_NUM_CLASSES,
        "loss_type": args.loss_type,
        "best_miou": best_joint_miou,
        "backbone": args.backbone,
    }, output_dir / final_ckpt_name)

    # --- Save eval.json ---
    # F5/F6: eval.json now records split_version, eval_mode, val_miou (for
    # checkpoint selection), and test_miou (evaluated once at end).
    eval_mode = "full_space" if args.full_space_eval else "masked"
    if args.full_space_eval and args.masked_eval:
        eval_mode = "both"
    eval_record = {
        "experiment_type": "joint_dmce",
        "loss_type": args.loss_type,
        "effective_loss_type": effective_loss_type,
        "a8_scheme": args.a8_scheme,
        "use_sspm": args.use_sspm,
        "reliability_weighted_loss": args.reliability_weighted_loss,
        "preprocess_strategy": args.preprocess_strategy,
        "held_out_dataset": held_out_dataset,
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        # F6: split version and checkpoint selection rule
        "split_version": args.split_version,
        "checkpoint_selection_rule": "best validation mIoU (test not used "
            "for model selection; test evaluated once at end)",
        # F5: evaluation mode (full_space primary, masked optional)
        "eval_mode": eval_mode,
        "best_epoch": int(best_epoch),
        "best_joint_miou": float(best_joint_miou),  # best VAL mIoU
        "val_miou": float(best_joint_miou),          # explicit val_miou
        "test_miou": float(test_joint_miou),         # test evaluated once at end
        "final_per_dataset": final_results,
        "best_epoch_metrics": best_metrics,
        "dataset_valid_classes": DATASET_VALID_CLASSES,
        "n_train_samples": {
            name: info["datasets"][name]["train_samples"]
            for name in info["datasets"]
        },
        "n_eval_samples": {
            name: info["datasets"][name]["eval_samples"]
            for name in info["datasets"] if name in test_loaders
        },
    }
    with open(output_dir / "eval.json", "w") as f:
        json.dump(eval_record, f, indent=2, ensure_ascii=False)

    logging.info(f"Done! Best VAL mIoU={best_joint_miou:.4f} "
                 f"(epoch {best_epoch}) | TEST mIoU={test_joint_miou:.4f}")
    logging.info(f"Models saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Joint DMCE vs Naive CE training (AI4+SubPipe+MCM)")
    parser.add_argument("--loss_type", type=str, required=True,
                        choices=["naive_ce", "valid_masked_ce", "dmce",
                                 "valid_masked_dmce", "focal", "dice",
                                 "tversky", "boundary", "ohem",
                                 "class_balanced_ce", "sba_ce",
                                 "generalized_dice", "lovasz", "lovasz_loss"],
                        help="Loss type: naive_ce (baseline), valid_masked_ce "
                             "(hard label mask only), dmce (proposed; soft logit "
                             "+ hard label mask), valid_masked_dmce (alias dmce), "
                             "focal (Lin 2017), dice (Milletari 2016), "
                             "tversky (Salehi 2017; alpha=0.3 beta=0.7 recall-"
                             "oriented generalization of Dice), "
                             "boundary (Kervadec 2019), ohem (Shrivastava 2016), "
                             "class_balanced_ce (Cui 2019, effective number), "
                             "sba_ce, "
                             "generalized_dice (Sudre 2017, SBA-CE 公开基线), "
                             "lovasz / lovasz_loss (Berman 2018 Lovasz-Softmax, "
                             "SBA-CE 公开基线; lovasz_loss exposes LovaszLoss name)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--ai4_dir", type=str, default=DEFAULT_AI4_DIR)
    parser.add_argument("--subpipe_dir", type=str, default=DEFAULT_SUBPIPE_DIR)
    parser.add_argument("--mcm_image_dir", type=str, default=DEFAULT_MCM_IMAGE_DIR)
    parser.add_argument("--mcm_mask_dir", type=str, default=DEFAULT_MCM_MASK_DIR)
    parser.add_argument("--mcm_split_file", type=str, default=DEFAULT_MCM_SPLIT)
    parser.add_argument("--use_sspm", action="store_true",
                        help="Use 5-channel SSPM input representation (raw + "
                             "range_norm + contrast + shadow + reliability) "
                             "via a 3x3 conv adapter before SegFormer encoder. "
                             "When set, wraps each per-dataset's images with "
                             "SSPMWrapper and uses build_sspm_model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--no_subpipe_chunk_split", action="store_true",
                        help="Use random split instead of chunk_split for SubPipe")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit per-dataset train samples (smoke test)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Limit per-dataset eval samples (smoke test)")
    parser.add_argument("--sampler", type=str, default="default",
                        choices=["default", "class_balanced", "dataset_balanced"],
                        help="Train sampler: 'default' (round-robin), "
                             "'class_balanced' (per-dataset inverse-freq "
                             "WeightedRandomSampler), or 'dataset_balanced' "
                             "(domain-uniform via DomainBalancedMultiLoader, "
                             "gives each dataset equal sampling probability).")
    parser.add_argument("--sampler_seed", type=int, default=42,
                        help="Seed for class_balanced WeightedRandomSampler")
    parser.add_argument("--scan_max_samples", type=int, default=None,
                        help="Cap samples scanned for class frequency stats "
                             "(class_balanced only; None = scan full dataset)")
    parser.add_argument("--train_only_dataset", type=str, default="none",
                        choices=["none", "ai4", "subpipe", "mcm"],
                        help="SCI stage-4 Oracle upper bound: train ONLY on "
                             "this dataset (others removed from train/val). "
                             "Test still evaluates all datasets. "
                             "Default none = normal joint training.")
    parser.add_argument("--init_from", type=str, default=None,
                        help="SCI stage-4 Fine-tuned lower bound: initialize "
                             "model weights from an existing best.pth "
                             "(model_state_dict only; optimizer/epoch NOT "
                             "restored, training restarts at Epoch 0).")
    parser.add_argument("--held_out_dataset", type=str, default="none",
                        choices=["none", "ai4", "subpipe", "mcm"],
                        help="Cross-domain held-out eval: train on the other "
                             "datasets, evaluate zero-shot transfer on the "
                             "held-out dataset each epoch. The held-out "
                             "dataset is removed from train_loader but kept "
                             "in eval_loaders.")
    parser.add_argument("--reliability_weighted_loss", action="store_true",
                        help="Use SSPM C5 reliability channel as per-pixel "
                             "loss weight. Requires --use_sspm. Pixels with "
                             "low reliability (flat/saturated regions) get "
                             "down-weighted in CE loss.")
    parser.add_argument("--backbone", type=str, default="segformer_b2",
                        choices=["segformer_b2", "unet", "deeplabv3plus",
                                 "fcn", "pspnet"],
                        help="Backbone architecture. segformer_b2 (default) "
                             "preserves existing behavior exactly. unet = "
                             "Ronneberger 2015 (in_channels supports 3 or 5). "
                             "deeplabv3plus = Chen 2018 via torchvision "
                             "(ResNet50 backbone; first conv replaced for "
                             "in_channels=1 or 5). fcn = Long 2015 via "
                             "torchvision fcn_resnet50. pspnet = Zhao 2017 via "
                             "torchvision pspnet_resnet50 (multi-scale context "
                             "baseline, recommended for Paper4 public backbone). "
                             "For --use_sspm with segformer_b2 the SSPM 3x3 "
                             "conv adapter path is kept unchanged; for other "
                             "backbones the 5-channel input goes directly.")
    parser.add_argument("--pretrained", action="store_true",
                        help="Load ImageNet-pretrained backbone weights "
                             "(DeepLabV3+ only). Default False for fair "
                             "comparison with SegFormer-B2 (which uses "
                             "pretrained=False). U-Net has no pretrained "
                             "weights; this flag is a no-op for it.")
    # ----- Frozen BN intervention -----
    parser.add_argument("--freeze_bn", action="store_true",
                        help="Freeze BatchNorm running stats as a diagnostic "
                             "intervention for invalid-class hijacking "
                             ". Sets momentum=0.0 + eval() lock + "
                             "overrides .train() so model.train() in the "
                             "training loop does NOT unfreeze BN. weight/bias "
                             "remain trainable. Default False preserves all "
                             "prior experiment behavior exactly.")
    parser.add_argument("--freeze_bn_mode", type=str, default="all",
                        choices=["all", "invalid_only"],
                        help="freeze_bn targeting: 'all' = freeze every "
                             "nn.BatchNorm1d/2d in the model; 'invalid_only' "
                             "= freeze only BN layers in the segmentation "
                             "head / decoder / classifier (heuristic by "
                             "module name keyword: classifier/head/decode/"
                             "output/proj). Only used when --freeze_bn is set.")
    # ----- Loss-specific hyperparams -----
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal Loss alpha (Lin 2017 default 0.25). "
                             "Only used when --loss_type focal.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma (Lin 2017 default 2.0). "
                             "Only used when --loss_type focal.")
    parser.add_argument("--dice_smooth", type=float, default=1.0,
                        help="Dice Loss smoothing constant (Milletari 2016 "
                             "default 1.0). Only used when --loss_type dice.")
    parser.add_argument("--tversky_alpha", type=float, default=0.3,
                        help="Tversky Loss alpha — FN weight (Salehi 2017 "
                             "default 0.3). With alpha=beta=0.5 reduces to "
                             "standard linear Dice. Only when --loss_type tversky.")
    parser.add_argument("--tversky_beta", type=float, default=0.7,
                        help="Tversky Loss beta — FP weight (Salehi 2017 "
                             "default 0.7). alpha<beta makes the loss recall-"
                             "oriented (FN penalized more). Only when --loss_type tversky.")
    parser.add_argument("--tversky_smooth", type=float, default=1.0,
                        help="Tversky Loss smoothing constant (default 1.0). "
                             "Only used when --loss_type tversky.")
    parser.add_argument("--gen_dice_smooth", type=float, default=1.0,
                        help="Generalized Dice smoothing constant (Sudre 2017 "
                             "default 1.0; added to per-class area^2 to stabilize "
                             "weight for tiny classes). Only used when "
                             "--loss_type generalized_dice.")
    parser.add_argument("--ohem_ratio", type=float, default=0.7,
                        help="OHEM keep ratio (Shrivastava 2016 default 0.7). "
                             "Keeps the top-ratio fraction of hardest pixels. "
                             "Only used when --loss_type ohem.")
    parser.add_argument("--cb_beta", type=float, default=0.9999,
                        help="Class-Balanced CE beta (Cui 2019 default 0.9999). "
                             "Effective number (1-beta)/(1-beta^n). Only used "
                             "when --loss_type class_balanced_ce. Requires "
                             "--cb_class_samples unless --cb_use_heuristic_weights.")
    parser.add_argument("--cb_class_samples", type=str, default=None,
                        help="Comma-separated per-class pixel counts for "
                             "Class-Balanced CE (Cui 2019). Example: "
                             "'1000000,5000,2000,500,10000,200000,3000,1000'. "
                             "Required for true effective-number weighting; "
                             "if omitted, falls back to the heuristic "
                             "build_class_weights() tensor (rare-class=3.0).")
    parser.add_argument("--aux_boundary_loss", action="store_true",
                        help="Add Boundary Loss (Kervadec 2019) as an "
                             "auxiliary term to the main loss. The boundary "
                             "term is computed from signed distance transforms "
                             "of the GT masks and added to the main loss with "
                             "weight --aux_boundary_weight. Requires scipy. "
                             "BoundaryLoss cannot be used as a standalone "
                             "main loss (it is region-less); use this flag "
                             "to combine it with --loss_type naive_ce/dmce/"
                             "focal/dice/ohem/class_balanced_ce.")
    parser.add_argument("--aux_boundary_weight", type=float, default=0.1,
                        help="Weight of the auxiliary boundary loss term "
                             "(default 0.1). Only used when --aux_boundary_loss.")
    parser.add_argument("--aux_boundary_alpha", type=float, default=0.5,
                        help="BoundaryLoss combined-mode alpha (default 0.5). "
                             "Only used when --loss_type boundary (combined "
                             "CE+boundary main loss mode).")
    # ----- SBA-CE specific hyperparams -----
    parser.add_argument("--sba_img_width", type=int, default=512,
                        help="SBA-CE: 图像宽度, 用于计算水柱中心 x_mid. "
                             "声纳瀑布图通常 512. Only used when --loss_type sba_ce.")
    parser.add_argument("--sba_sigma_water", type=float, default=50.0,
                        help="SBA-CE: 水柱高斯宽度 (像素). 控制水柱压制范围. "
                             "默认 50 (约图像宽度 10%%). Only when --loss_type sba_ce.")
    parser.add_argument("--sba_gamma", type=float, default=5.0,
                        help="SBA-CE: 水柱压制强度 γ. 越大水柱区前景类 logit "
                             "压制越强. 默认 5.0. Only when --loss_type sba_ce.")
    parser.add_argument("--sba_foreground_classes", type=str, default="1,2,3",
                        help="SBA-CE: 前景类列表 (逗号分隔), 这些类在水柱区被压制. "
                             "默认 '1,2,3' (shipwreck/pipeline/seafloor-objects). "
                             "Only when --loss_type sba_ce.")
    # ----- PLA-SegFormer switch -----
    parser.add_argument("--use_pla", action="store_true",
                        help="PLA-SegFormer: 在 EfficientSelfAttention 中加入"
                             "高斯衰减调制因子 (Locality-Constrained Attention). "
                             "通过 DistDecayMask 让自注意力重新局部化, 缓解全局"
                             "条纹噪声过凝聚.")
    parser.add_argument("--pla_sigma_local", type=float, default=8.0,
                        help="PLA-SegFormer: 局部高斯衰减尺度 (像素). 越小越"
                             "局域化. 默认 8.0 (约 patch size / 8). "
                             "Only used when --use_pla.")
    parser.add_argument("--preprocess_strategy", type=str, default=None,
                        choices=["raw", "basic", "physd",
                                 "clahe_only", "lee_only",
                                 "bilateral_only", "hist_eq",
                                 "piaonet", "taad_sspm", "srad"],
                        help="Paper1 §0.A.2 单操作消融策略 (Task #7): "
                             "applied uniformly to ai4/subpipe/mcm. "
                             "raw/basic/physd 走 PreprocessConfig 流水线; "
                             "clahe_only/lee_only/bilateral_only/hist_eq "
                             "为单操作策略. "
                             "V6 物理启发算法: "
                             "piaenet=PIAE-Net 物理强度锚定增强 (SRAD+残差); "
                             "taad_sspm=TAAD-SSPM 拓扑感知各向异性扩散 (Frangi+CED); "
                             "srad=裸 SRAD (PIAE 公开锚点消融, 对照链 Raw→SRAD→PIAE). "
                             "Default None = no preprocessing "
                             "(preserves current behavior).")
    # ----- F6 fix: split version for checkpoint selection protocol -----
    # Checkpoint must be selected by
    # validation mIoU on an independent group-level val set, NOT by test
    # mIoU. Test set is evaluated ONLY ONCE at the end of training.
    parser.add_argument("--split_version", type=str, default="v2",
                        choices=["v1", "v2"],
                        help="Split protocol version. 'v2' (default): "
                             "group-level isolated train/val/test, "
                             "checkpoint selected by val mIoU, test "
                             "evaluated once at end. 'v1' (legacy): test "
                             "set used for checkpoint selection (old "
                             "behavior, kept for backward compatibility).")
    # ----- Corrected protocol (2026-09-11 train/val 重叠修复) -----
    # legacy 管线 train loader 从未采用 v2 train 侧清单 (AI4 train=113 内部
    # 28-val 剔除; SubPipe train=[Chunk1,Chunk2]=1080), 导致 SubPipe v2 val
    # (Chunk2) 100% 与 AI4 v2 val 14/20 张实际参与训练 → checkpoint 选择污染。
    # --strict_v2_loader: train/val/test 全部严格来自 v2 划分 JSON 文件清单
    # (ai4 121/20/120, subpipe 540/537/189, mcm 558/564/48), 训练前强制
    # 三向无重叠 + loader↔JSON 一致性校验 (fail-fast)。
    parser.add_argument("--strict_v2_loader", action="store_true",
                        default=False,
                        help="Corrected protocol: build train/val/test "
                             "loaders strictly from v2 split JSON file "
                             "lists (single-variable change vs legacy: "
                             "split composition only). Enables isolation "
                             "verification before training.")
    # ----- F5 fix: full-space vs masked inference -----
    # The primary evaluation protocol is
    # full-space inference (no logit masking), so the model can predict any
    # of the 8 classes. Masked inference (DMCE-style -1e9 on invalid-class
    # logits) is kept as an optional comparison mode.
    parser.add_argument("--full_space_eval", action="store_true",
                        default=True,
                        help="Use full-space inference (no logit masking) "
                             "as the primary evaluation protocol. Default "
                             "True. When True, argmax over all 8 classes "
                             "without masking invalid classes.")
    parser.add_argument("--no_full_space_eval", dest="full_space_eval",
                        action="store_false",
                        help="Disable full-space inference (use masked "
                             "evaluation as primary).")
    parser.add_argument("--masked_eval", action="store_true",
                        default=False,
                        help="Also compute masked mIoU (DMCE-style logit "
                             "masking) for comparison. When set, eval.json "
                             "records both full_space_miou and masked_miou.")
    # ----- A8: Invalid-class constraint correction -----
    # Three candidate schemes that address DMCE invalid-class hijacking.
    # Used in combination with --loss_type dmce (the A8 schemes extend DMCE).
    # When --a8_scheme is 'none' (default), A8 is disabled and behavior is
    # identical to the standard --loss_type selection.
    parser.add_argument("--a8_scheme", type=str, default="none",
                        choices=["none", "margin_reg", "soft_mask",
                                 "classifier_suppression"],
                        help="A8 invalid-class constraint correction scheme "
                             ". 'none' = disabled "
                             "(default). 'margin_reg' = A8-1 Margin-Regularized "
                             "DMCE (hinge on logit gap). 'soft_mask' = A8-2 Soft "
                             "Mask DMCE (log(alpha) penalty on invalid logits). "
                             "'classifier_suppression' = A8-3 Classifier weight "
                             "L2 suppression. When set to a non-'none' value, "
                             "overrides --loss_type to use the corresponding A8 "
                             "loss class (which inherits DMCE masking).")
    parser.add_argument("--a8_margin", type=float, default=2.0,
                        help="A8-1 margin: required logit gap between max valid "
                             "and max invalid class logits (default 2.0). Only "
                             "used when --a8_scheme margin_reg.")
    parser.add_argument("--a8_lambda", type=float, default=0.1,
                        help="A8 regularization weight lambda (default 0.1). "
                             "Used by A8-1 (margin_reg, default 0.1) and A8-3 "
                             "(classifier_suppression, default 0.01). Only used "
                             "when --a8_scheme is margin_reg or "
                             "classifier_suppression.")
    parser.add_argument("--a8_alpha", type=float, default=0.1,
                        help="A8-2 soft mask alpha: down-weighting factor for "
                             "invalid class logits in log space (default 0.1). "
                             "0 = hard mask (= standard DMCE), 1 = no mask "
                             "(= NaiveCE). Only used when --a8_scheme soft_mask.")
    # ----- 大修敏感性实验 (Reviewer 1 comment 8): 类别权重参数化 -----
    parser.add_argument("--bg_class_weight", type=float, default=0.1,
                        help="Background (C0) class weight (default 0.1, "
                             "identical to all corrected-protocol main-table "
                             "runs; heuristic prior, not tuned).")
    parser.add_argument("--fg_class_weight", type=float, default=3.0,
                        help="Foreground (C1-C3) class weight (default 3.0, "
                             "identical to all corrected-protocol main-table "
                             "runs; heuristic prior, not tuned).")
    # ----- RC 监督解耦 -----
    parser.add_argument("--ai4_degrade_to_box", action="store_true",
                        help="RC 监督解耦: 将 AI4 train 精细掩码退化为 bbox "
                             "填充矩形, 模拟 SubPipe/MCM 的粗监督形式, 用于 "
                             "量化监督形式 (真值掩码 vs bbox) 对性能的贡献。 "
                             "val/test 保持精细掩码。")
    parser.add_argument("--box_edge_downweight", type=float, default=None,
                        help="RC 监督解耦: SubPipe/MCM train bbox 边界带降权 "
                             "系数 (uncertainty-weighted regional labeling "
                             "基线)。如 0.5 表示边界带像素损失权重减半。 "
                             "None = 不降权 (默认)。")
    parser.add_argument("--box_edge_band", type=int, default=8,
                        help="RC 监督解耦: bbox 边界带宽度 (像素, 每侧, "
                             "512×512 空间, 默认 8)。仅当 "
                             "--box_edge_downweight 设置时生效。")
    # ----- Per-epoch mechanism trajectory tracking -----
    # Appended instrumentation. Non-invasive:
    # when enabled, monkey-patches train_one_epoch / evaluate / model
    # builders via mechanism_tracker.install_mechanism_tracking() so that
    # NO existing training-loop code is modified. Disabled by default.
    parser.add_argument("--mechanism_track", action="store_true",
                        help="Enable per-epoch mechanism trajectory tracking "
                             ". Captures per-class classifier "
                             "weight norm/bias, valid/invalid logit stats, "
                             "ICAR, IPM, background ratio, classifier grad "
                             "norm, BN running stats, and per-domain "
                             "foreground IoU at every epoch end. Saves to "
                             "<output_dir>/<mechanism_track_dir>/"
                             "mechanism_trajectory.json. Non-invasive: "
                             "existing training logic is unmodified.")
    parser.add_argument("--mechanism_track_dir", type=str,
                        default="mechanism_trajectory",
                        help="Subdirectory under --output_dir where the "
                             "mechanism trajectory JSON and plots are "
                             "written. Default 'mechanism_trajectory'. "
                             "Only used when --mechanism_track is set.")
    args = parser.parse_args()

    # Per-epoch mechanism trajectory instrumentation. Non-invasive:
    # patches module-level symbols so existing train() logic is untouched.
    if getattr(args, "mechanism_track", False):
        setup_mechanism_tracker(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.reliability_weighted_loss and not args.use_sspm:
        print("[error] --reliability_weighted_loss requires --use_sspm "
              "(reliability comes from SSPM C5 channel)")
        raise SystemExit(1)

    if args.reliability_weighted_loss and args.loss_type in (
            "lovasz", "lovasz_loss"):
        print("[error] --reliability_weighted_loss is incompatible with "
              "--loss_type " + args.loss_type + " (LovaszSoftmaxLoss does not "
              "support per-pixel reliability weighting; SSPM reliability is "
              "already fed as input channel 4). Either drop "
              "--reliability_weighted_loss, or switch loss_type to "
              "'tversky' or 'dmce'.")
        raise SystemExit(1)

    if args.loss_type == "boundary" and not args.aux_boundary_loss:
        print("[warn] --loss_type boundary as a standalone main loss is "
              "discouraged: BoundaryLoss (Kervadec 2019) is region-less and "
              "typically combined with a region loss. Consider using "
              "--loss_type naive_ce/dmce/focal/dice/ohem/class_balanced_ce "
              "with --aux_boundary_loss instead. BoundaryLoss will be built "
              "in combined mode (alpha*CE + (1-alpha)*boundary_term) for "
              "compatibility; set --aux_boundary_alpha to tune the mix.")
        # Combined mode (boundary_only=False) so the loss is usable as main.
        # The build_criterion call below will receive boundary_only=False
        # only if --aux_boundary_loss is unset; otherwise the auxiliary
        # BoundaryLoss(boundary_only=True) is constructed separately.

    if args.loss_type != "class_balanced_ce" and args.cb_class_samples:
        print("[warn] --cb_class_samples is only used by "
              "--loss_type class_balanced_ce; ignoring.")

    train(args)

# ---------------------------------------------------------------------------
# Per-epoch mechanism trajectory instrumentation
# ---------------------------------------------------------------------------
# Appended below ``if __name__ == "__main__":`` so that the existing
# main() / train() code path is unmodified.  When --mechanism_track is set,
# setup_mechanism_tracker() imports the MechanismTracker module and
# monkey-patches the module-level ``train_one_epoch`` / ``evaluate`` /
# ``build_model_by_backbone`` / ``build_sspm_model`` / ``train`` symbols to
# install the tracker.  No existing function body is changed.

def setup_mechanism_tracker(args) -> None:
    """Install mechanism-tracking hooks on this module's functions.

    Called from :func:`main` when ``--mechanism_track`` is set.  Patches
    the module-level symbols so that, when ``train(args)`` later runs, the
    tracker captures per-epoch mechanism statistics without modifying any
    existing training-loop code.

    Args:
        args: argparse Namespace.  Must have ``mechanism_track`` (bool),
            ``mechanism_track_dir`` (str) and ``output_dir`` (str).
    """
    import sys as _sys
    try:
        from mechanism_tracker import install_mechanism_tracking
    except ImportError as e:
        print(f"[error] --mechanism_track enabled but mechanism_tracker "
              f"module could not be imported: {e}")
        raise SystemExit(1)

    # When run as a script the module is __main__; when imported it is
    # train_joint_dmce.  Patch the appropriate namespace.
    module_name = "__main__" if __name__ == "__main__" else __name__
    module_obj = _sys.modules[module_name]
    install_mechanism_tracking(module_obj.__dict__, args)
    print(f"[mechanism_track] enabled; trajectory will be saved to "
          f"{args.output_dir}/{args.mechanism_track_dir}/"
          f"mechanism_trajectory.json")

if __name__ == "__main__":
    main()
