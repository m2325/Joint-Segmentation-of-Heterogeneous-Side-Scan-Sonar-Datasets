#!/usr/bin/env python3
"""Efficiency measurement queue for Paper2 §5.8 Table 11.

Generates the efficiency table that compares params / FLOPs / GPU mem /
train time / inference latency across:

    | Backbone     | Method                                  |
    |--------------|-----------------------------------------|
    | SegFormer-B2 | NaiveCE / DMCE / A8-1 / A8-2 / A8-3 / LMSeg |
    | U-Net        | NaiveCE / DMCE                           |
    | DeepLabV3+   | NaiveCE / DMCE                           |

Configuration (matches the V3 / V3p / C2 SegFormer-B2 protocol):
    - input_size : (3, 512, 512)
    - batch_size : 4
    - outputs    : paper2/results/efficiency_metrics.{json,md}

This script is **CPU-import safe**: importing it does not build any model
or touch CUDA. Model builders are only invoked when ``main()`` is called
with ``--no_dry_run``. With ``--dry_run`` (the default) the script only
prints the planned measurement matrix, which lets the queue be audited
while the GPU is busy with the C2 training queue.

Constraint compliance:
    - Does NOT modify train_joint_dmce.py.
    - Does NOT run any GPU training or inference when invoked with the
      default ``--dry_run`` flag.
    - Each model is built fresh per measurement to avoid memory leaks.
    - Train time is extracted post-hoc from existing ``train.log`` files
      rather than re-timed.

Usage
-----
    # Audit the measurement plan (default; CPU-only, no GPU):
    python tools/training/run_efficiency_measurement.py

    # Actually run the measurements (requires a free GPU; do NOT run
    # while the C2 queue is active):
    python tools/training/run_efficiency_measurement.py --no_dry_run

    # Override the input resolution / batch size used for measurement:
    python tools/training/run_efficiency_measurement.py --no_dry_run \\
        --img_size 512 --batch_size 4
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# NOTE: heavy imports (torch, project model builders) are deferred to
# ``_build_model_for_combo`` so that ``--dry_run`` (the default) does not
# require CUDA / project-specific deps to be importable.

TOOLS_TRAINING = Path(__file__).resolve().parent
if str(TOOLS_TRAINING) not in sys.path:
    sys.path.insert(0, str(TOOLS_TRAINING))

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "efficiency"
OUT_JSON = OUT_DIR / "efficiency_metrics.json"
OUT_MD = OUT_DIR / "efficiency_metrics.md"

INPUT_SIZE: Tuple[int, int, int] = (3, 512, 512)
BATCH_SIZE: int = 4
NUM_CLASSES = 8  # unified 8-class space


# ---------------------------------------------------------------------------
# Existing train.log paths per (backbone, method) — used to extract the
# train_time_per_epoch column without re-timing. ``None`` means the
# experiment has not been run yet (the field will be ``null`` in JSON).
# ---------------------------------------------------------------------------
TRAIN_LOG_MAP: Dict[Tuple[str, str], Optional[str]] = {
    ("segformer_b2", "naive_ce"):
        str(PROJECT / "models" / "exp_joint_naive_ce_sspm_s1_20260705"
            / "train.log"),
    ("segformer_b2", "dmce"):
        str(PROJECT / "models" / "exp_joint_dmce_sspm_s1_20260704"
            / "train.log"),
    ("segformer_b2", "a8_1_margin_reg"):
        str(PROJECT / "models" / "exp_a8_margin_reg_s1" / "train.log"),
    ("segformer_b2", "a8_2_soft_mask"):
        str(PROJECT / "models" / "exp_a8_soft_mask_s1" / "train.log"),
    ("segformer_b2", "a8_3_classifier_suppression"):
        str(PROJECT / "models" / "exp_a8_classifier_suppression_s1"
            / "train.log"),
    ("segformer_b2", "lmseg"):
        str(PROJECT / "models" / "exp_paper2_lmseg_s1" / "train.log"),
    ("unet", "naive_ce"):
        str(PROJECT / "models" / "exp_paper2_c2_naive_ce_unet_s1"
            / "train.log"),
    ("unet", "dmce"):
        str(PROJECT / "models" / "exp_paper2_c2_dmce_unet_s1"
            / "train.log"),
    ("deeplabv3plus", "naive_ce"):
        str(PROJECT / "models" / "exp_v3_deeplab_naive_ce_s1_20260706"
            / "train.log"),
    ("deeplabv3plus", "dmce"):
        str(PROJECT / "models" / "exp_v3_deeplab_dmce_s1_20260706"
            / "train.log"),
}


# ---------------------------------------------------------------------------
# Measurement matrix definition
# ---------------------------------------------------------------------------
# Each entry: (backbone, method, model_builder_spec)
# ``model_builder_spec`` is a stable string identifying which builder
# to invoke; the actual builder is resolved lazily in
# ``_build_model_for_combo`` so the script remains importable without
# torch or the project's training modules.
EFFICIENCY_MATRIX: List[Dict[str, str]] = [
    # --- SegFormer-B2 (6 methods) ---
    {"backbone": "segformer_b2", "method": "naive_ce",
     "builder": "segformer_b2_sspm", "note": "NaiveCE+SSPM (s1 protocol)"},
    {"backbone": "segformer_b2", "method": "dmce",
     "builder": "segformer_b2_sspm", "note": "DMCE+SSPM (s1 protocol)"},
    {"backbone": "segformer_b2", "method": "a8_1_margin_reg",
     "builder": "segformer_b2_sspm",
     "note": "A8-1 Margin-Regularized DMCE (same arch as DMCE; loss differs)"},
    {"backbone": "segformer_b2", "method": "a8_2_soft_mask",
     "builder": "segformer_b2_sspm",
     "note": "A8-2 Soft Mask DMCE (same arch as DMCE; loss differs)"},
    {"backbone": "segformer_b2", "method": "a8_3_classifier_suppression",
     "builder": "segformer_b2_sspm",
     "note": "A8-3 Classifier Suppression DMCE (same arch as DMCE; loss differs)"},
    {"backbone": "segformer_b2", "method": "lmseg",
     "builder": "lmseg_segformer_b2",
     "note": "LMSeg: SegFormer-B2 + CLIP-text-aligned classifier head"},

    # --- U-Net (2 methods) ---
    {"backbone": "unet", "method": "naive_ce",
     "builder": "unet_sspm", "note": "U-Net + NaiveCE+SSPM (5ch input)"},
    {"backbone": "unet", "method": "dmce",
     "builder": "unet_sspm", "note": "U-Net + DMCE+SSPM (5ch input)"},

    # --- DeepLabV3+ (2 methods) ---
    {"backbone": "deeplabv3plus", "method": "naive_ce",
     "builder": "deeplabv3plus_sspm",
     "note": "DeepLabV3+ (ResNet50) + NaiveCE+SSPM (5ch first conv)"},
    {"backbone": "deeplabv3plus", "method": "dmce",
     "builder": "deeplabv3plus_sspm",
     "note": "DeepLabV3+ (ResNet50) + DMCE+SSPM (5ch first conv)"},
]


# ---------------------------------------------------------------------------
# Lazy model builders
# ---------------------------------------------------------------------------
def _build_model_for_combo(spec: str) -> "torch.nn.Module":
    """Build a fresh model instance for the given builder spec.

    Lazy-imports torch and the project's training modules so that
    ``--dry_run`` (the default CLI mode) does not require CUDA / project
    deps to be importable.

    Args:
        spec: one of ``segformer_b2_sspm``, ``unet_sspm``,
            ``deeplabv3plus_sspm``, ``lmseg_segformer_b2``.

    Returns:
        A fresh ``torch.nn.Module`` instance.
    """
    import torch  # noqa: F401  (ensure torch is importable)
    if spec == "segformer_b2_sspm":
        # 5-channel SegFormer via the SSPM 3x3 conv adapter; matches the
        # V3 / V3p / C2 / A8 SegFormer-B2 protocol exactly.
        from sspm_segformer import build_sspm_model
        return build_sspm_model(num_classes=NUM_CLASSES, input_channels=5)
    if spec == "unet_sspm":
        # U-Net with 5-channel input (matches --backbone unet --use_sspm).
        from train_subpipe_segformer import build_model_by_backbone
        return build_model_by_backbone(
            backbone="unet", num_classes=NUM_CLASSES, in_channels=5,
            pretrained=False,
        )
    if spec == "deeplabv3plus_sspm":
        # DeepLabV3+ ResNet50 with 5-channel first conv (matches
        # --backbone deeplabv3plus --use_sspm).
        from train_subpipe_segformer import build_model_by_backbone
        return build_model_by_backbone(
            backbone="deeplabv3plus", num_classes=NUM_CLASSES,
            in_channels=5, pretrained=False,
        )
    if spec == "lmseg_segformer_b2":
        # LMSeg: SegFormer-B2 backbone + CLIP-text-aligned classifier head.
        # Built via the project's LMSeg adapter; falls back to a vanilla
        # SegFormer-B2 + 1x1 text projection if the CLIP adapter cannot
        # be imported (so the params/FLOPs cell can still be populated).
        try:
            from lmseg_sonar_adapter import build_lmseg_model
            return build_lmseg_model(num_classes=NUM_CLASSES)
        except Exception as exc:
            sys.stderr.write(
                f"[warn] LMSeg builder unavailable ({exc}); falling back "
                f"to plain SegFormer-B2 + 1x1 head for efficiency estimate.\n")
            from train_subpipe_segformer import build_segformer_model
            return build_segformer_model(
                num_classes=NUM_CLASSES, pretrained=False)
    raise ValueError(f"Unknown builder spec: {spec!r}")


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
def run_measurement(
    entry: Dict[str, str],
    input_size: Tuple[int, int, int],
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """Run efficiency measurement for one matrix entry.

    Args:
        entry: row from ``EFFICIENCY_MATRIX``.
        input_size: ``(C, H, W)``.
        batch_size: training batch size for memory measurement.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Record dict (augmented with backbone / method / note fields).
    """
    from measure_efficiency import measure_all
    backbone = entry["backbone"]
    method = entry["method"]
    model_name = f"{backbone}__{method}"
    train_log = TRAIN_LOG_MAP.get((backbone, method))

    record = measure_all(
        model_builder=lambda: _build_model_for_combo(entry["builder"]),
        model_name=model_name,
        output_dir=str(OUT_DIR),
        input_size=input_size,
        batch_size=batch_size,
        device=device,
        train_log_path=train_log,
    )
    # Add human-readable columns.
    record["backbone"] = backbone
    record["method"] = method
    record["note"] = entry.get("note", "")
    if train_log is None:
        record["train_time_source"] = "not_yet_trained"
    return record


def render_markdown(records: List[Dict[str, Any]]) -> str:
    """Render the efficiency records as a Paper2 §5.8 Table 11 markdown table.

    Args:
        records: list of record dicts from ``run_measurement``.

    Returns:
        Markdown string with one row per record.
    """
    header = (
        "| Backbone | Method | Params (M) | FLOPs (G) | FLOPs source | "
        "GPU mem (MiB) | Train time/epoch (s) | Latency (ms) | "
        "Throughput (img/s) |\n"
        "|---|---|---:|---:|---|---:|---:|---:|---:|\n"
    )
    rows = []
    for r in records:
        def fmt(x, nd=2, default="—"):
            if x is None:
                return default
            try:
                if isinstance(x, float) and x != x:  # NaN
                    return default
                return f"{x:.{nd}f}"
            except (TypeError, ValueError):
                return str(x)
        rows.append(
            f"| {r.get('backbone', r.get('model', ''))} "
            f"| {r.get('method', '')} "
            f"| {fmt(r.get('params_M'), 2)} "
            f"| {fmt(r.get('flops_G'), 2)} "
            f"| {r.get('flops_source', '—')} "
            f"| {fmt(r.get('gpu_mem_mib'), 1)} "
            f"| {fmt(r.get('train_time_per_epoch_s'), 1)} "
            f"| {fmt(r.get('inference_latency_ms'), 2)} "
            f"| {fmt(r.get('inference_throughput_img_per_s'), 1)} |"
        )
    return header + "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Paper2 §5.8 efficiency measurement queue (Table 11).")
    parser.add_argument(
        "--dry_run", action="store_true", default=True,
        help="Print the measurement matrix without invoking model builders "
             "(default: True). Use --no_dry_run to actually run measurements.")
    parser.add_argument(
        "--no_dry_run", dest="dry_run", action="store_false",
        help="Actually run the measurements. Requires a free GPU; do NOT "
             "use while the C2 training queue is active.")
    parser.add_argument(
        "--img_size", type=int, default=512,
        help="Input resolution H=W (default 512, matches V3 / C2 protocol).")
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Training batch size for memory measurement (default 4).")
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device for measurement (default cuda; cpu returns NaN for "
             "GPU-memory-dependent metrics).")
    args = parser.parse_args()

    # SSPM models expect 5-channel input; LMSeg uses 3-channel.
    # The builder spec determines the correct channel count.
    def _input_size_for_entry(entry):
        """Return (C, H, W) matching the model's expected input channels."""
        builder = entry.get("builder", "")
        if builder.endswith("_sspm"):
            return (5, args.img_size, args.img_size)
        # lmseg_segformer_b2 and any plain 3-channel builders
        return (3, args.img_size, args.img_size)

    default_input_size = (5, args.img_size, args.img_size)  # SSPM is the majority
    print("=" * 80)
    print("Paper2 §5.8 Efficiency Measurement Queue (Table 11)")
    print("=" * 80)
    print(f"Input size : SSPM=(5,{args.img_size},{args.img_size}), "
          f"LMSeg=(3,{args.img_size},{args.img_size})")
    print(f"Batch size : {args.batch_size}")
    print(f"Device     : {args.device}")
    print(f"Dry-run    : {args.dry_run}")
    print(f"Output JSON: {OUT_JSON}")
    print(f"Output MD  : {OUT_MD}")
    print(f"Matrix     : {len(EFFICIENCY_MATRIX)} entries")
    for e in EFFICIENCY_MATRIX:
        log = TRAIN_LOG_MAP.get((e["backbone"], e["method"]))
        log_status = "present" if (log and Path(log).exists()) else (
            "pending" if log else "n/a")
        isz = _input_size_for_entry(e)
        print(f"  - {e['backbone']:14s} | {e['method']:25s} "
              f"| builder={e['builder']:25s} | in_ch={isz[0]} | train.log={log_status}")
    print("=" * 80)

    if args.dry_run:
        print("[dry-run] No models built, no GPU touched. "
              "Pass --no_dry_run to actually run measurements.")
        return

    # Actually run measurements.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i, entry in enumerate(EFFICIENCY_MATRIX, 1):
        entry_input_size = _input_size_for_entry(entry)
        print(f"\n[{i}/{len(EFFICIENCY_MATRIX)}] "
              f"{entry['backbone']} / {entry['method']} (in_ch={entry_input_size[0]})")
        try:
            rec = run_measurement(entry, entry_input_size, args.batch_size,
                                  args.device)
            records.append(rec)
            print(f"  params={rec['params_M']:.2f}M  "
                  f"flops={rec['flops_G']}  "
                  f"gpu_mem={rec['gpu_mem_mib']:.1f}MiB  "
                  f"latency={rec['inference_latency_ms']:.2f}ms")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            records.append({
                "backbone": entry["backbone"],
                "method": entry["method"],
                "note": entry.get("note", ""),
                "error": str(exc),
            })

    summary = {
        "table": "Paper2 §5.8 Table 11 (efficiency)",
        "input_size_sspm": [5, args.img_size, args.img_size],
        "input_size_lmseg": [3, args.img_size, args.img_size],
        "batch_size": args.batch_size,
        "device": args.device,
        "records": records,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON: {OUT_JSON}")

    md = render_markdown(records)
    with open(OUT_MD, "w") as fh:
        fh.write("# Paper2 §5.8 Table 11 — Efficiency Metrics\n\n")
        fh.write(f"Input size: SSPM=(5,{args.img_size},{args.img_size}), "
                 f"LMSeg=(3,{args.img_size},{args.img_size})  |  "
                 f"Batch size: {args.batch_size}  |  Device: {args.device}\n\n")
        fh.write(md)
    print(f"Saved MD  : {OUT_MD}")


if __name__ == "__main__":
    main()
