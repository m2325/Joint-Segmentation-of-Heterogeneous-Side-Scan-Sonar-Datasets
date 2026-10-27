#!/usr/bin/env python3
"""
paper4/大修/run_endtoend_timing.py

JMSE-4623206 大修端到端计时（审稿人1 意见11）：
表 8 现有延迟只含网络前向（论文 §4.8 已声明）。本脚本补充声学预处理
(PhysD) 与 SSPM 五通道构建耗时，报告两个层次：

  层次一（模型级, 与表 8 同协议）: 模型前向延迟/FPS（复测）
  层次二（端到端）: 图像读取 + PhysD 预处理 + resize512 + SSPM 构建
                   + 模型前向 → 端到端延迟与 FPS

训练链序（SSPMWrapper 核实）: raw 全分辨率 → PhysD → cv2 INTER_LINEAR
resize 512 → /255 → generate_sspm_channels(512×512) → 模型前向。
计时协议与表 8 一致: CPU-only, 5 warmup + 30 repeats, batch 1×5×512×512,
FPS = 1000/mean_ms。**正式数据须在表 8 同一设备 (Jetson Orin, 14 线程)
上运行**；本地 x86 结果仅作脚本冒烟验证。

用法:
  python paper4/大修/run_endtoend_timing.py
输出:
  paper4/results/efficiency/endtoend_timing.{json,md}
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

TOOLS_TRAINING = Path(__file__).resolve().parent
if str(TOOLS_TRAINING) not in sys.path:
    sys.path.insert(0, str(TOOLS_TRAINING))

from sonar_preprocess_adapter import preprocess_for_dataset  # noqa: E402
from sspm_channels import generate_sspm_channels  # noqa: E402
from sspm_segformer import build_sspm_model  # noqa: E402
from train_subpipe_segformer import build_model_by_backbone  # noqa: E402

NUM_CLASSES = 8
INPUT_SIZE = 512
WARMUP = 5
REPEATS = 30
N_IMAGES_PER_DATASET = 10
OUT_DIR = SLAM_ROOT / "paper4" / "results" / "efficiency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "ai4": ("ai4_split_v2.json",
            lambda f: SLAM_ROOT / "datasets" / "AI4Shipwrecks" / "test" / "images" / f),
    "subpipe": ("subpipe_split_v2.json",
                lambda f: SLAM_ROOT / "datasets" / "SubPipe" / "SubPipe" / "DATA" /
                "Chunk3" / "SSS_HF_images" / "Image" / f),
    "mcm": ("mcm_year_split_v2.json",
            lambda f: SLAM_ROOT / "datasets" / "MCM_extracted" / "2021" / f),
}
ADAPTER_NAME = {"ai4": "ai4shipwrecks", "subpipe": "subpipe", "mcm": "mcm"}


def test_images(dataset: str):
    split_file, path_fn = SPLITS[dataset]
    d = json.load(open(SLAM_ROOT / "experiments" / "splits" / "v2_protocol" / split_file))
    files = d["test_files"][:N_IMAGES_PER_DATASET]
    return [(f, path_fn(f)) for f in files]


def pipeline_stages(img_path: Path, dataset: str):
    """单张图像的逐阶段计时 (ms)。链序与训练/推理一致。"""
    t0 = time.perf_counter()
    img = PILImage.open(img_path)
    if img.mode != "L":
        img = img.convert("L")
    raw = np.array(img)
    t_read = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    physd = preprocess_for_dataset(raw, dataset=ADAPTER_NAME[dataset],
                                   config_name="physd")
    t_physd = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r = cv2.resize(physd, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    u = r.astype(np.float32) / 255.0
    t_resize = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sspm = generate_sspm_channels(u)  # (5, 512, 512)
    t_sspm = (time.perf_counter() - t0) * 1000

    return {"read": t_read, "physd": t_physd, "resize": t_resize,
            "sspm": t_sspm, "out": sspm}


def _device_name():
    """设备标识: /proc/device-tree/model (Tegra 系) + CPU part。
    注意: 表 8 的 s7 脚本曾把 CPU part 0xd83 (Cortex-A78AE) 硬编码标注为
    'Jetson Orin', 但实际测量设备是 Jetson AGX Thor (CPU 同为 A78AE)。"""
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip().rstrip("\x00")
    except Exception:
        return platform.processor() or "unknown"


def main():
    torch.set_num_threads(torch.get_num_threads())
    device = _device_name()
    print(f"[INFO] Device: {device} | CPU: {platform.processor()} "
          f"| arch: {platform.machine()} | threads: {torch.get_num_threads()}")
    print(f"[INFO] 输入 1x5x{INPUT_SIZE}x{INPUT_SIZE} | warmup {WARMUP} + "
          f"repeats {REPEATS} | 每数据集 {N_IMAGES_PER_DATASET} 张测试图")

    # ---- 阶段 1: 预处理链逐阶段计时 (三数据集) ----
    stage_stats = {}
    for ds in ("ai4", "subpipe", "mcm"):
        imgs = test_images(ds)
        recs = []
        for fname, p in imgs:
            if not p.exists():
                print(f"[WARN] 缺图: {p}")
                continue
            pipeline_stages(p, ds)  # 热身 1 次 (JIT/缓存)
            recs.append(pipeline_stages(p, ds))
        stages = {}
        for k in ("read", "physd", "resize", "sspm"):
            vals = [r[k] for r in recs]
            stages[k] = {"mean_ms": statistics.mean(vals),
                         "std_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        stages["preprocess_total"] = {
            "mean_ms": sum(s["mean_ms"] for s in stages.values()),
            "std_ms": float("nan")}
        stage_stats[ds] = {"n_images": len(recs), "stages": stages}
        m = stages
        print(f"[{ds}] n={len(recs)} | read={m['read']['mean_ms']:.1f} "
              f"physd={m['physd']['mean_ms']:.1f} resize={m['resize']['mean_ms']:.1f} "
              f"sspm={m['sspm']['mean_ms']:.1f} | 预处理合计 "
              f"{m['preprocess_total']['mean_ms']:.1f} ms")

    # ---- 阶段 2: 模型前向计时 (五架构, 表 8 协议复测) ----
    models = {
        "SegFormer-B2 (SSPM)": lambda: build_sspm_model(num_classes=NUM_CLASSES),
        "U-Net": lambda: build_model_by_backbone("unet", num_classes=NUM_CLASSES, in_channels=5, pretrained=False),
        "DeepLabV3+": lambda: build_model_by_backbone("deeplabv3plus", num_classes=NUM_CLASSES, in_channels=5, pretrained=False),
        "PSPNet": lambda: build_model_by_backbone("pspnet", num_classes=NUM_CLASSES, in_channels=5, pretrained=False),
        "FCN": lambda: build_model_by_backbone("fcn", num_classes=NUM_CLASSES, in_channels=5, pretrained=False),
    }
    torch.manual_seed(0)
    x = torch.randn(1, 5, INPUT_SIZE, INPUT_SIZE)
    model_stats = {}
    for name, builder in models.items():
        model = builder().eval()
        with torch.no_grad():
            for _ in range(WARMUP):
                model(x)
            lat = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                model(x)
                lat.append((time.perf_counter() - t0) * 1000)
        mm = statistics.mean(lat)
        model_stats[name] = {"latency_ms_mean": round(mm, 1),
                             "latency_ms_std": round(statistics.stdev(lat), 1),
                             "fps": round(1000.0 / mm, 2)}
        print(f"[{name}] 前向 {mm:.1f} ± {statistics.stdev(lat):.1f} ms | "
              f"FPS {1000/mm:.2f}")
        del model

    # ---- 阶段 3: 端到端汇总 ----
    # 端到端(arch, ds) = 预处理合计(ds) + 模型前向(arch)
    # 三数据集均值另给一列 (等权宏平均, 与论文 macro 口径一致)
    e2e = {}
    for arch, ms in model_stats.items():
        per_ds = {}
        for ds in ("ai4", "subpipe", "mcm"):
            pre = stage_stats[ds]["stages"]["preprocess_total"]["mean_ms"]
            total = pre + ms["latency_ms_mean"]
            per_ds[ds] = {"preprocess_ms": round(pre, 1),
                          "total_ms": round(total, 1),
                          "fps": round(1000.0 / total, 2)}
        macro_pre = statistics.mean(
            stage_stats[d]["stages"]["preprocess_total"]["mean_ms"] for d in per_ds)
        macro_total = macro_pre + ms["latency_ms_mean"]
        e2e[arch] = {"per_dataset": per_ds,
                     "macro": {"preprocess_ms": round(macro_pre, 1),
                               "total_ms": round(macro_total, 1),
                               "fps": round(1000.0 / macro_total, 2)}}

    result = {
        "measurement_conditions": {
            "device": device,
            "cpu": platform.processor(), "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
            "input": f"1x5x{INPUT_SIZE}x{INPUT_SIZE}",
            "warmup": WARMUP, "repeats": REPEATS,
            "n_images_per_dataset": N_IMAGES_PER_DATASET,
            "protocol_note": ("端到端 = 读取+PhysD+resize512+SSPM+模型前向; "
                              "与表 8 同设备 (Jetson AGX Thor, 表 8 误标为 "
                              "Orin) 运行, 为正式数据"),
        },
        "stage_stats": stage_stats,
        "model_forward": model_stats,
        "end_to_end": e2e,
    }
    with open(OUT_DIR / "endtoend_timing.json", "w") as f:
        json.dump(result, f, indent=1)

    # Markdown 摘要
    lines = ["# 端到端计时（大修 意见11）", "",
             "## 预处理链逐阶段 (ms, mean±std)", "",
             "| Dataset | n | Read | PhysD | Resize512 | SSPM | 预处理合计 |",
             "|---|---|---|---|---|---|---|"]
    for ds, s in stage_stats.items():
        st = s["stages"]
        lines.append(
            f"| {ds} | {s['n_images']} | "
            f"{st['read']['mean_ms']:.1f}±{st['read']['std_ms']:.1f} | "
            f"{st['physd']['mean_ms']:.1f}±{st['physd']['std_ms']:.1f} | "
            f"{st['resize']['mean_ms']:.1f}±{st['resize']['std_ms']:.1f} | "
            f"{st['sspm']['mean_ms']:.1f}±{st['sspm']['std_ms']:.1f} | "
            f"{st['preprocess_total']['mean_ms']:.1f} |")
    lines += ["", "## 端到端 (预处理宏平均 + 模型前向)", "",
              "| Architecture | Forward (ms) | E2E total (ms) | E2E FPS |", "|---|---|---|---|"]
    for arch, e in e2e.items():
        lines.append(f"| {arch} | {e['macro']['preprocess_ms'] + 0:.1f}→"
                     f"{model_stats[arch]['latency_ms_mean']} | "
                     f"{e['macro']['total_ms']} | {e['macro']['fps']} |")
    with open(OUT_DIR / "endtoend_timing.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVED] {OUT_DIR}/endtoend_timing.{{json,md}}")


if __name__ == "__main__":
    main()
