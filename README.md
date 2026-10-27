# Evaluating Dataset-Aware Masked Supervision for Joint Segmentation of Heterogeneous Side-Scan Sonar Datasets

This repository contains the source code, preprocessing scripts, dataset
partition files, and experimental configuration files for the paper:

> **Evaluating Dataset-Aware Masked Supervision for Joint Segmentation of
> Heterogeneous Side-Scan Sonar Datasets**
> Yang Liu<sup>1,2,\*</sup>, Yang Lin<sup>1</sup>, Chang Zou<sup>1,2</sup>,
> Haitao Gu<sup>1</sup>, Siquan Yu<sup>1</sup>
>
> <sup>1</sup> Shenyang Institute of Automation, Chinese Academy of Sciences,
> Shenyang 110169, China
> <sup>2</sup> University of Chinese Academy of Sciences, Beijing 100049, China
>
> \* Correspondence: liuyang21f@mails.ucas.ac.cn
>
> *Journal of Marine Science and Engineering* (JMSE), 2026. Submitted for
> open access publication under the terms and conditions of the Creative
> Commons Attribution (CC BY) license.

The study addresses joint semantic segmentation across three heterogeneous
public side-scan sonar datasets (AI4Shipwrecks, SubPipe, MCM) in a shared
four-class label space, and evaluates the trade-off of dataset-aware masked
cross-entropy (DMCE) — attenuating classes outside each dataset's
annotation scope — against unmasked cross-entropy. The repository includes:

- **Dataset-masked cross-entropy (DMCE)** with a soft logit-offset masking
  scheme (`α ∈ (0, 1)` attenuation for classes not supported by the source
  dataset) and its hard-mask counterpart;
- **SSPM**, a five-channel physically motivated input representation
  (intensity, range normalization, local contrast, shadow, reliability);
- **PhysD**, a dataset-conditioned physical preprocessing pipeline
  (TVG compensation, water-column removal, slant-range correction,
  bilateral despeckling, CLAHE);
- A **corrected group-level train/val/test split protocol (v2)** with
  verified train/val/test isolation for all three datasets;
- Cross-architecture comparisons (FCN, U-Net, PSPNet, DeepLabV3+,
  SegFormer-B2), sampler studies (ordinary / class-balanced /
  dataset-balanced), leave-one-dataset-out (LODO) evaluation, sensitivity
  studies (soft-mask attenuation `α`, foreground class weight `w_fg`),
  and on-device efficiency measurements.

## Repository Structure

```
.
├── code/                     # All source code
│   ├── train_joint_dmce.py   # Main joint-training entry point
│   ├── dmce_loss.py          # DMCE / naive CE / hard-mask losses
│   ├── unified_joint_dataset.py  # Unified multi-dataset loaders + samplers
│   ├── ai4shipwrecks_dataset.py  # AI4Shipwrecks dataset adapter
│   ├── subpipe_dataset.py        # SubPipe dataset adapter
│   ├── subpipe_pseudo_label.py   # SubPipe per-pixel mask rasterization
│   ├── mcm_dataset.py            # MCM dataset adapter
│   ├── sspm_channels.py      # SSPM five-channel construction
│   ├── sspm_segformer.py     # SSPM SegFormer wrapper (5ch -> encoder)
│   ├── train_sspm_segformer.py   # Single-dataset baselines (Raw/Basic/PhysD/SSPM)
│   ├── train_subpipe_segformer.py  # Model builders for all 5 architectures
│   ├── deeplabv3plus.py      # DeepLabV3+ implementation
│   ├── sonar_preprocess_adapter.py  # Dataset-conditioned PhysD routing
│   ├── subpipe_preprocess.py     # PhysD pipeline configuration
│   ├── improved_preprocess.py    # PhysD operators (TVG/despeckle/CLAHE/...)
│   ├── class_balanced_sampler.py     # Inverse-frequency pixel sampler
│   ├── domain_balanced_sampler.py    # Domain-balanced multi-loader
│   ├── run_efficiency_measurement.py # Model-forward timing protocol
│   ├── run_endtoend_timing.py        # Preprocessing-inclusive timing
│   └── sonar_slam/preprocessing/     # Low-level sonar operators
├── splits/                   # Group-level v2 partition files
│   ├── ai4_split_v2.json     # AI4: by shipwreck site (121/20/120)
│   ├── subpipe_split_v2.json # SubPipe: by chunk (540/537/189)
│   ├── mcm_year_split_v2.json# MCM: by year (558/564/48)
│   ├── split_manifest_v2.json# Split protocol manifest
│   └── generate_v2_splits.py # Split generation script
└── configs/                  # Experimental configuration files (one per
                              # representative run; recorded verbatim from
                              # the training-time config.json)
```

## Requirements

- Python >= 3.9
- PyTorch >= 1.13 (tested with 2.x, CUDA 11.x/12.x)
- See `requirements.txt` for the full dependency list.

```bash
pip install -r requirements.txt
```

## Datasets

The three datasets are publicly available and are **not** redistributed in
this repository. Download each dataset and place it under `datasets/`:

| Dataset  | Expected path                        | Source |
|----------|--------------------------------------|--------|
| AI4Shipwrecks | `datasets/AI4Shipwrecks`       | [AI4Shipwrecks project](https://github.com/ai4geo/ai4shipwrecks) |
| SubPipe  | `datasets/SubPipe/SubPipe`           | Álvarez-Tuñón et al., Zenodo [10.5281/zenodo.10808161](https://doi.org/10.5281/zenodo.10808161) |
| MCM      | `datasets/MCM_extracted`, `datasets/MCM_masks` | See paper for the public source |

Run all commands from the repository root so that the relative default
paths resolve correctly.

## Usage

### 1. Joint training (main protocol)

Soft-mask DMCE with SSPM five-channel input, strict v2 splits
(SegFormer-B2, from scratch, 100 epochs, early stopping patience 20):

```bash
python code/train_joint_dmce.py \
    --output_dir runs/soft_dmce_s1 \
    --loss_type dmce --use_sspm --a8_scheme soft_mask --a8_alpha 0.1 \
    --strict_v2_loader \
    --epochs 100 --patience 20 --batch_size 4 --lr 1e-4 \
    --weight_decay 1e-4 --img_size 512 --seed 1
```

Variants used in the paper:

| Experiment | Key arguments |
|---|---|
| Naive joint CE | `--loss_type naive_ce --use_sspm` |
| Hard-masked DMCE | `--loss_type dmce --use_sspm` (hard masking) |
| Dataset-balanced sampling | append `--sampler dataset_balanced` |
| Class-balanced sampling | append `--sampler class_balanced` |
| LODO (leave-one-dataset-out) | append `--held_out_dataset {ai4\|subpipe\|mcm}` |
| Cross-architecture | append `--backbone {fcn\|unet\|pspnet\|deeplabv3plus}` |
| Attenuation sensitivity | `--a8_alpha {0.5\|0.01}` |
| Foreground-weight sensitivity | `--fg_class_weight {1.0\|6.0}` |

Each run writes `train.log`, `config.json`, `best.pth`, and `eval.json`
(per-dataset and macro mIoU, per-class IoU, full-space evaluation) to the
output directory.

### 2. Single-dataset baselines

```bash
python code/train_sspm_segformer.py --dataset ai4 \
    --representation sspm --output_dir runs/ai4_sspm_s1
```

### 3. Regenerating the data splits

The group-level v2 partitions can be regenerated from the raw datasets:

```bash
python splits/generate_v2_splits.py
```

### 4. Efficiency measurement

```bash
# Model-forward latency (CPU, 1x5x512x512, 5 warm-up + 30 timed)
python code/run_efficiency_measurement.py

# Preprocessing-inclusive (input-image-to-logit) timing
python code/run_endtoend_timing.py
```

## Configuration Files

`configs/` contains one representative `config.json` per experiment family
(joint3 naive CE / soft DMCE / hard DMCE, dataset-balanced, LODO,
cross-architecture, and the two sensitivity studies). These files are
recorded verbatim by the training script at run time and document every
hyperparameter, protocol field, and dataset statistic of the corresponding
experiment in the paper.

## Citation

If you use this code, please cite:

```bibtex
@article{liu2026evaluating,
  title   = {Evaluating Dataset-Aware Masked Supervision for Joint
             Segmentation of Heterogeneous Side-Scan Sonar Datasets},
  author  = {Liu, Yang and Lin, Yang and Zou, Chang and Gu, Haitao and
             Yu, Siquan},
  journal = {Journal of Marine Science and Engineering},
  year    = {2026},
  note    = {Received: 27 September 2026; Revised: 23 October 2026;
             Accepted: 24 October 2026}
}
```

## License

Copyright © 2026 by the authors. This repository is distributed under the
terms of the Creative Commons Attribution 4.0 International License (CC BY
4.0), consistent with the open-access license of the associated paper —
see `LICENSE` for details.
