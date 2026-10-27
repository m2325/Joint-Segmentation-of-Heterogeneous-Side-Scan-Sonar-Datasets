"""
DMCE (Dynamic Mask Cross-Entropy) Loss for multi-dataset joint training.

Each dataset defines its own valid class set. Invalid class logits are masked
to a large negative value before softmax, preventing gradient flow to classes
the dataset doesn't label. This avoids the "unlabeled classes treated as
background" penalty that arises in naive cross-dataset training.

Example:
    AI4 valid = {0, 1}   → only C0 (water_column_nadir), C1 (shipwreck) participate
    SubPipe valid = {0, 2} → only C0, C2 (pipeline) participate
    MCM valid = {0, 3}   → only C0, C3 (mine_like_contact) participate

When training on AI4, the model isn't penalized for predicting rock (C4) on a
non-shipwreck pixel — C4 is masked out of the softmax entirely, so no gradient
flows to or from it. This lets the model learn rare classes without
unlearning it from AI4.

Public baseline losses:
    - NaiveCELoss              : standard CE (baseline)
    - DMCELoss                 : Dynamic Mask CE (proposed)
    - FocalLoss                : Lin 2017 ICCV         (ref: lin2017focal)
    - DiceLoss                 : Milletari 2016 3DV   (ref: milletari2016vnet)
    - TverskyLoss              : Salehi 2017           (ref: salehi2017tversky)
    - BoundaryLoss             : Kervadec 2019 MIDL   (ref: kervadec2019boundary)
    - OHEMWrapper              : Shrivastava 2016 CVPR(ref: shrivastava2016training)
    - ClassBalancedCELoss      : Cui 2019 CVPR        (ref: cui2019class)
    - GeneralizedDiceLoss      : Sudre 2017 MIDL      (ref: sudre2017generalised)
    - LovaszSoftmaxLoss        : Berman 2018 CVPR     (ref: berman2018lovasz)
    - LovaszLoss               : alias for LovaszSoftmaxLoss (Berman 2018)

All public losses accept an optional `dataset_valid_classes` constructor
arg; when supplied and `dataset_name` is passed to forward(), they apply the
same DMCE-style logit/label masking as DMCELoss. Without it they behave as
plain (non-masked) versions, drop-in replacements for NaiveCELoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


def _weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    per_pixel_weight: torch.Tensor,
    ignore_index: int,
    class_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-pixel weighted cross-entropy.

    Standard F.cross_entropy supports per-class weights but not per-pixel
    weights. For SSPM reliability-weighted loss, we need per-pixel weights
    (high where the pixel is acoustically reliable, low where flat/saturated).

    Implementation:
        loss = -sum( per_pixel_weight * class_weight[gold] * log_softmax[gold] )
                / sum( per_pixel_weight * (label != ignore_index) )

    Args:
        logits: (B, C, H, W)
        labels: (B, H, W) long
        per_pixel_weight: (B, H, W) in [0, 1]; pixels with weight 0 contribute 0
        ignore_index: label value to skip (set weight 0 there)
        class_weights: optional (C,) tensor; per-class multiplier
        eps: numerical floor
    """
    valid_mask = (labels != ignore_index)
    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0  # placeholder; will be masked out

    log_probs = F.log_softmax(logits, dim=1)  # (B, C, H, W)
    gathered = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)  # (B, H, W)

    if class_weights is not None:
        cw = class_weights.gather(0, safe_labels.view(-1)).view_as(safe_labels)
    else:
        cw = torch.ones_like(safe_labels, dtype=logits.dtype)

    pixel_weight = per_pixel_weight * valid_mask.to(logits.dtype)
    weighted = -gathered * cw * pixel_weight
    denom = pixel_weight.sum().clamp(min=eps)
    return weighted.sum() / denom


# ---------------------------------------------------------------------------
# DMCE-style masking helpers (shared by all public baseline losses)
# ---------------------------------------------------------------------------

def _build_dmce_buffers(dataset_valid_classes: Dict[str, List[int]],
                        num_classes: int,
                        mask_value: float = -1e9):
    """Precompute per-dataset logit mask and valid-label mask tensors.

    Returns (logit_masks, valid_labels) dicts keyed by dataset name.
    Designed to be stored as instance attributes (not registered buffers,
    so losses remain picklable without re-registration). Caller is expected
    to .to(device) them lazily via the helper below.
    """
    logit_masks: Dict[str, torch.Tensor] = {}
    valid_labels: Dict[str, torch.Tensor] = {}
    for name, classes in dataset_valid_classes.items():
        lm = torch.full((num_classes,), mask_value)
        for c in classes:
            lm[c] = 0.0
        logit_masks[name] = lm
        v = torch.zeros(num_classes, dtype=torch.bool)
        for c in classes:
            v[c] = True
        valid_labels[name] = v
    return logit_masks, valid_labels


def _apply_logit_mask(logits: torch.Tensor,
                      dataset_name: Optional[str],
                      logit_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Add per-dataset logit mask to logits. No-op if dataset not registered."""
    if dataset_name is None or dataset_name not in logit_masks:
        return logits
    lm = logit_masks[dataset_name].to(logits.device, logits.dtype)
    return logits + lm.view(1, -1, 1, 1)


def _apply_label_mask(labels: torch.Tensor,
                      dataset_name: Optional[str],
                      valid_labels: Dict[str, torch.Tensor],
                      num_classes: int,
                      ignore_index: int) -> torch.Tensor:
    """Set labels of classes not in the dataset's valid set to ignore_index.

    Pixels already at ignore_index (or out of range) are left untouched.
    Returns a cloned tensor; the input is not modified.
    """
    if dataset_name is None or dataset_name not in valid_labels:
        return labels
    v = valid_labels[dataset_name].to(labels.device)
    safe = labels.clone()
    in_range = (labels >= 0) & (labels < num_classes)
    clamped = labels.clamp(0, num_classes - 1)
    invalid = in_range & ~v[clamped]
    safe[invalid] = ignore_index
    return safe


class NaiveCELoss(nn.Module):
    """Standard Cross-Entropy with ignore_index. Baseline for DMCE comparison.

    All 8 classes participate in the softmax. Labels set to ignore_index (255)
    are skipped. The model is penalized for predicting any wrong class, including
    classes that the dataset doesn't label (e.g., predicting C4 rock on an AI4
    non-shipwreck pixel labeled C0).
    """

    def __init__(self, ignore_index: int = 255,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if reliability_weight is not None:
            return _weighted_ce(
                logits, labels, reliability_weight,
                ignore_index=self.ignore_index,
                class_weights=self.class_weights,
            )
        return F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


class ValidMaskedCELoss(nn.Module):
    """Cross-Entropy with hard label-validity masking (no logit masking).

    For each dataset, labels outside the valid class set are set to
    ignore_index before computing CE. Unlike DMCE, all 8 class logits
    participate in the softmax — there is no logit masking. This isolates
    the contribution of hard label masking from DMCE's soft logit masking.

    Ablation role:
        NaiveCE        : no masking (baseline)
        valid_masked_ce: hard label mask only  (this class)
        DMCE           : soft logit mask + hard label mask

    The valid-label masks are precomputed per dataset at init time and
    stored as registered buffers for automatic device transfer.
    """

    def __init__(self, dataset_valid_classes: Dict[str, List[int]],
                 ignore_index: int = 255, num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.dataset_valid_classes = dataset_valid_classes
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.class_weights = class_weights

        for name, classes in dataset_valid_classes.items():
            valid_label = torch.zeros(num_classes, dtype=torch.bool)
            for c in classes:
                valid_label[c] = True
            self.register_buffer(f"_valid_label_{name}", valid_label)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not hasattr(self, f"_valid_label_{dataset_name}"):
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Registered: {list(self.dataset_valid_classes.keys())}")

        valid_label = getattr(self, f"_valid_label_{dataset_name}")

        safe_labels = labels.clone()
        in_range = (labels >= 0) & (labels < self.num_classes)
        clamped = labels.clamp(0, self.num_classes - 1)
        invalid = in_range & ~valid_label[clamped]
        safe_labels[invalid] = self.ignore_index

        return F.cross_entropy(
            logits, safe_labels,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


class DMCELoss(nn.Module):
    """Dynamic Mask Cross-Entropy Loss.

    For each dataset, only valid class logits participate in the softmax.
    Invalid class logits are set to -1e9, giving them ~0 probability after
    softmax. Labels not in the valid set are set to ignore_index before
    computing CE, preventing inf loss from -log(0).

    The logit and valid-label masks are precomputed per dataset at init time
    and stored as registered buffers for automatic device transfer.

    Optional per-pixel reliability_weight: when provided (B, H, W) tensor,
    the loss becomes per-pixel weighted CE (used by --reliability_weighted_loss).
    """

    MASK_VALUE = -1e9

    def __init__(self, dataset_valid_classes: Dict[str, List[int]],
                 ignore_index: int = 255, num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            dataset_valid_classes: {dataset_name: [valid_class_ids]}
                e.g. {"ai4": [0, 1], "subpipe": [0, 2], "mcm": [0, 3]}
            ignore_index: label value to skip in loss (default 255)
            num_classes: total number of classes in the unified space
            class_weights: optional per-class weighting tensor
        """
        super().__init__()
        self.dataset_valid_classes = dataset_valid_classes
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.class_weights = class_weights

        for name, classes in dataset_valid_classes.items():
            logit_mask = torch.full((num_classes,), self.MASK_VALUE)
            for c in classes:
                logit_mask[c] = 0.0
            self.register_buffer(f"_logit_mask_{name}", logit_mask)

            valid_label = torch.zeros(num_classes, dtype=torch.bool)
            for c in classes:
                valid_label[c] = True
            self.register_buffer(f"_valid_label_{name}", valid_label)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not hasattr(self, f"_logit_mask_{dataset_name}"):
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Registered: {list(self.dataset_valid_classes.keys())}"
            )

        logit_mask = getattr(self, f"_logit_mask_{dataset_name}")
        valid_label = getattr(self, f"_valid_label_{dataset_name}")

        masked_logits = logits + logit_mask.view(1, -1, 1, 1)

        safe_labels = labels.clone()
        in_range = (labels >= 0) & (labels < self.num_classes)
        clamped = labels.clamp(0, self.num_classes - 1)
        invalid = in_range & ~valid_label[clamped]
        safe_labels[invalid] = self.ignore_index

        if reliability_weight is not None:
            return _weighted_ce(
                masked_logits, safe_labels, reliability_weight,
                ignore_index=self.ignore_index,
                class_weights=self.class_weights,
            )
        return F.cross_entropy(
            masked_logits, safe_labels,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


# ===========================================================================
# Public baseline losses
# ===========================================================================
# All classes below accept an optional `dataset_valid_classes` constructor
# arg. When supplied AND forward() receives a registered `dataset_name`,
# the loss applies the same DMCE-style logit/label masking as DMCELoss.
# Without dataset_valid_classes, they behave as plain non-masked losses
# (drop-in NaiveCE replacements for ablation).
# ===========================================================================


class FocalLoss(nn.Module):
    """Focal Loss (Lin 2017, ICCV; ref: lin2017focal).

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Multi-class: softmax over C, pick p_t = prob of true class.
    Down-weights easy examples (high p_t) so the model focuses on hard ones.
    Mirrors RetinaNet hyper-parameters alpha=0.25, gamma=2.0.

    With gamma=0 and alpha=1 the loss reduces to standard CE (modulo the
    alpha multiplier). With dataset_valid_classes supplied, applies DMCE
    logit/label masking identical to DMCELoss.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 ignore_index: int = 255, num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None,
                 dataset_valid_classes: Optional[Dict[str, List[int]]] = None,
                 eps: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.dataset_valid_classes = dataset_valid_classes or {}
        self.eps = float(eps)
        self._logit_masks, self._valid_labels = _build_dmce_buffers(
            self.dataset_valid_classes, num_classes, DMCELoss.MASK_VALUE)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        masked_logits = _apply_logit_mask(logits, dataset_name, self._logit_masks)
        safe_labels = _apply_label_mask(
            labels, dataset_name, self._valid_labels,
            self.num_classes, self.ignore_index)

        valid_mask = (safe_labels != self.ignore_index)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0  # empty: 0 with grad linkage

        sl = safe_labels.clone()
        sl[~valid_mask] = 0
        log_probs = F.log_softmax(masked_logits, dim=1)
        gathered_log = log_probs.gather(1, sl.unsqueeze(1)).squeeze(1)
        p_t = gathered_log.exp()

        focal_weight = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        per_pixel = -self.alpha * focal_weight * gathered_log

        if self.class_weights is not None:
            cw = self.class_weights.gather(0, sl.view(-1)).view_as(sl)
            per_pixel = per_pixel * cw.to(per_pixel.dtype)

        pixel_weight = valid_mask.to(logits.dtype)
        if reliability_weight is not None:
            pixel_weight = pixel_weight * reliability_weight
        per_pixel = per_pixel * pixel_weight
        denom = pixel_weight.sum().clamp(min=self.eps)
        return per_pixel.sum() / denom


class DiceLoss(nn.Module):
    """Dice Loss (Milletari 2016, 3DV; ref: milletari2016vnet).

    DL = 1 - (2 * sum(p_i * g_i) + smooth) / (sum(p_i^2) + sum(g_i^2) + smooth)

    Multi-class: convert labels to one-hot, average per-class Dice.
    `smooth` (default 1.0) avoids division by zero on empty classes.
    Pixels with label == ignore_index are excluded from both numerator
    and denominator (mask is applied to probs and one-hot).

    With dataset_valid_classes supplied, applies DMCE logit/label masking.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255,
                 num_classes: int = 8,
                 dataset_valid_classes: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dataset_valid_classes = dataset_valid_classes or {}
        self._logit_masks, self._valid_labels = _build_dmce_buffers(
            self.dataset_valid_classes, num_classes, DMCELoss.MASK_VALUE)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        masked_logits = _apply_logit_mask(logits, dataset_name, self._logit_masks)
        safe_labels = _apply_label_mask(
            labels, dataset_name, self._valid_labels,
            self.num_classes, self.ignore_index)

        valid_mask = (safe_labels != self.ignore_index)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0

        sl = safe_labels.clone()
        sl[~valid_mask] = 0
        probs = F.softmax(masked_logits, dim=1)  # (B, C, H, W)
        onehot = F.one_hot(sl, self.num_classes).permute(0, 3, 1, 2).float()

        mask_f = valid_mask.to(logits.dtype).unsqueeze(1)  # (B, 1, H, W)
        if reliability_weight is not None:
            mask_f = mask_f * reliability_weight.unsqueeze(1)

        probs = probs * mask_f
        onehot = onehot * mask_f

        dims = (0, 2, 3)  # batch + spatial
        inter = (probs * onehot).sum(dim=dims)
        union = (probs * probs + onehot * onehot).sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss (Salehi 2017; ref: salehi2017tversky).

    Paper: "Tversky loss function for image segmentation using 3D fully
    convolutional deep networks" (arXiv:1706.05721).

    Generalization of Dice loss that decouples the penalty for false
    negatives (FN) and false positives (FP):
        TI_c = (TP_c + smooth) / (TP_c + alpha*FN_c + beta*FP_c + smooth)
        loss = 1 - mean_c(TI_c)
    where TP_c = sum(p_c * g_c), FN_c = sum((1-p_c) * g_c),
          FP_c = sum(p_c * (1-g_c)) over valid pixels.

    With alpha=beta=0.5 the Tversky index reduces to the STANDARD (linear)
    Dice coefficient 2*TP/(|GT|+|Pred|). Note this differs slightly from
    the DiceLoss class in this file, which uses the Milletari V-Net squared
    variant 2*TP/(sum(p^2)+sum(g^2)); the two Dice forms agree at the
    extrema (perfect/empty) but differ in gradient magnitude otherwise.
    The default alpha=0.3, beta=0.7 (Salehi 2017) weighs FN more heavily
    than FP, making the loss recall-oriented — appropriate for small
    foreground objects (e.g. SSS shipwreck/pipeline/mine pixels < 1%)
    where missing a target is costlier than a false alarm.

    Multi-class: convert labels to one-hot, average per-class Tversky
    index over classes present in the batch (empty classes contribute
    nothing because TP=FN=0 -> TI=1 -> loss contribution 0). Pixels with
    label == ignore_index are excluded from TP/FN/FP (mask applied to
    probs and one-hot).

    With dataset_valid_classes supplied, applies DMCE logit/label masking
    identical to DiceLoss, so it can be a drop-in baseline under joint
    training.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0, ignore_index: int = 255,
                 num_classes: int = 8,
                 dataset_valid_classes: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        if alpha < 0.0 or beta < 0.0:
            raise ValueError(
                f"alpha and beta must be non-negative, got alpha={alpha} "
                f"beta={beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dataset_valid_classes = dataset_valid_classes or {}
        self._logit_masks, self._valid_labels = _build_dmce_buffers(
            self.dataset_valid_classes, num_classes, DMCELoss.MASK_VALUE)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        masked_logits = _apply_logit_mask(logits, dataset_name, self._logit_masks)
        safe_labels = _apply_label_mask(
            labels, dataset_name, self._valid_labels,
            self.num_classes, self.ignore_index)

        valid_mask = (safe_labels != self.ignore_index)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0

        sl = safe_labels.clone()
        sl[~valid_mask] = 0
        probs = F.softmax(masked_logits, dim=1)  # (B, C, H, W)
        onehot = F.one_hot(sl, self.num_classes).permute(0, 3, 1, 2).float()

        mask_f = valid_mask.to(logits.dtype).unsqueeze(1)  # (B, 1, H, W)
        if reliability_weight is not None:
            mask_f = mask_f * reliability_weight.unsqueeze(1)

        probs = probs * mask_f
        onehot = onehot * mask_f

        dims = (0, 2, 3)  # batch + spatial -> (C,)
        tp = (probs * onehot).sum(dim=dims)                       # TP_c
        fn = ((1.0 - probs) * onehot).sum(dim=dims)              # FN_c
        fp = (probs * (1.0 - onehot)).sum(dim=dims)              # FP_c
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth)
        # Only average over classes present in the GT (onehot area > 0);
        # absent classes get TP=FN=0 -> TI=1 -> loss 0, which would bias
        # the mean upward. Mirrors 'present' mode in DiceLoss/Lovasz.
        present = (onehot.sum(dim=dims) > 0)
        if not bool(present.any()):
            return logits.sum() * 0.0
        tversky = tversky[present]
        return 1.0 - tversky.mean()


class BoundaryLoss(nn.Module):
    """Boundary Loss (Kervadec 2019, MIDL; ref: kervadec2019boundary).

    Boundary term = sum_c <p_c, phi_c> / sum(p)
    where phi_c is the signed distance transform of class c's mask
    (positive outside, negative inside). Encourages softmax probabilities
    to align with the distance field so the predicted boundary moves
    toward the ground truth.

    Two usage modes:
      * Auxiliary (recommended): `boundary_only=True` (default). Returns
        just the boundary term. The training loop adds it to a main region
        loss (CE/Dice/Focal) with a tunable weight, matching the paper's
        alpha schedule (alpha 0 -> 1 over training; caller controls schedule).
      * Combined: `boundary_only=False`. Returns `alpha*CE + (1-alpha)*bnd`
        so it can be used as a standalone main loss. Default alpha=0.5.

    Notes:
      - Requires scipy for distance_transform_edt. Falls back to a 0
        boundary term (with grad linkage) if scipy is unavailable.
      - For empty masks (class absent), phi_c is left as zero so that class
        contributes nothing.
      - Distance transform is computed on CPU per-batch; correct but slow
        for large images. Adequate for paper-2 ablation runs.
      - With dataset_valid_classes supplied, applies DMCE logit/label
        masking to the boundary term (CE in combined mode always uses
        plain NaiveCE — pass DMCELoss as `ce_loss` to override).
    """

    def __init__(self, alpha: float = 0.5, ignore_index: int = 255,
                 num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None,
                 boundary_only: bool = True,
                 dataset_valid_classes: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        self.alpha = float(alpha)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.boundary_only = bool(boundary_only)
        self.dataset_valid_classes = dataset_valid_classes or {}
        self._logit_masks, self._valid_labels = _build_dmce_buffers(
            self.dataset_valid_classes, num_classes, DMCELoss.MASK_VALUE)
        # CE used only in combined mode (boundary_only=False)
        self.ce = NaiveCELoss(ignore_index=ignore_index,
                              class_weights=class_weights)

    def _compute_distance_maps(self, labels: torch.Tensor) -> torch.Tensor:
        # labels: (B, H, W) on device; returns (B, C, H, W) on device
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            return torch.zeros((labels.shape[0], self.num_classes,
                                *labels.shape[1:]),
                               device=labels.device, dtype=torch.float32)
        import numpy as np

        B, H, W = labels.shape
        device = labels.device
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        dist_maps = np.zeros((B, self.num_classes, H, W), dtype=np.float32)

        for b in range(B):
            for c in range(self.num_classes):
                fg = (labels_np[b] == c).astype(np.uint8)
                if fg.sum() == 0:
                    continue
                # Signed distance: positive outside, negative inside.
                # Minimising <p, phi> drives probability mass inside the GT
                # region (where phi<0), shrinking false positives outside.
                dist_outside = distance_transform_edt(1 - fg)
                dist_inside = distance_transform_edt(fg)
                dist_maps[b, c] = (dist_outside - dist_inside).astype(np.float32)

        return torch.from_numpy(dist_maps).to(device)

    def boundary_term(self, logits: torch.Tensor, labels: torch.Tensor,
                      dataset_name: str = None) -> torch.Tensor:
        """Compute the pure boundary term (no CE). Suitable for aux use."""
        masked_logits = _apply_logit_mask(logits, dataset_name, self._logit_masks)
        safe_labels = _apply_label_mask(
            labels, dataset_name, self._valid_labels,
            self.num_classes, self.ignore_index)

        valid_mask = (safe_labels != self.ignore_index)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0

        phi = self._compute_distance_maps(safe_labels)  # (B, C, H, W)
        probs = F.softmax(masked_logits, dim=1)

        mask_f = valid_mask.to(logits.dtype).unsqueeze(1)
        weighted = (probs * phi * mask_f).sum(dim=(1, 2, 3))  # (B,)
        denom = (probs * mask_f).sum(dim=(1, 2, 3)).clamp(min=1e-6)
        bnd = (weighted / denom).mean()
        return bnd

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.boundary_only:
            # Ignore reliability_weight in pure-boundary mode; the boundary
            # term has its own (valid-mask based) weighting.
            return self.boundary_term(logits, labels, dataset_name)
        # Combined mode: alpha * CE + (1-alpha) * boundary_term
        ce_loss = self.ce(logits, labels, dataset_name, reliability_weight)
        bnd = self.boundary_term(logits, labels, dataset_name)
        return self.alpha * ce_loss + (1.0 - self.alpha) * bnd


class OHEMWrapper(nn.Module):
    """OHEM (Shrivastava 2016, CVPR; ref: shrivastava2016training).

    Wraps a base pixel-wise loss and keeps only the hardest `ohem_ratio`
    fraction of valid pixels per batch (top-k by per-pixel CE). All other
    pixels are excluded from the gradient.

    The base loss is NaiveCELoss by default (no class weights). Pass
    `base_loss=DMCELoss(...)` to wrap a DMCE loss, or supply class_weights
    for a weighted NaiveCE. With `ohem_ratio=1.0` the wrapper reproduces
    the base loss exactly.

    When dataset_valid_classes is supplied (and no explicit base_loss),
    a DMCELoss is constructed internally so masking is preserved.
    """

    def __init__(self, base_loss: Optional[nn.Module] = None,
                 ohem_ratio: float = 0.7, ignore_index: int = 255,
                 num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None,
                 dataset_valid_classes: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        if not (0.0 < ohem_ratio <= 1.0):
            raise ValueError(f"ohem_ratio must be in (0, 1], got {ohem_ratio}")
        if base_loss is None:
            if dataset_valid_classes:
                base_loss = DMCELoss(
                    dataset_valid_classes=dataset_valid_classes,
                    ignore_index=ignore_index,
                    num_classes=num_classes,
                    class_weights=class_weights)
            else:
                base_loss = NaiveCELoss(ignore_index=ignore_index,
                                        class_weights=class_weights)
        self.base_loss = base_loss
        self.ohem_ratio = float(ohem_ratio)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        # Cache class_weights for the per-pixel CE computation in forward.
        self.class_weights = getattr(self.base_loss, "class_weights", None)
        self.dataset_valid_classes = dataset_valid_classes or {}
        # For DMCE-style masking when base_loss is NaiveCE (no internal masking)
        self._logit_masks, self._valid_labels = _build_dmce_buffers(
            self.dataset_valid_classes, num_classes, DMCELoss.MASK_VALUE)

    def _per_pixel_ce(self, logits: torch.Tensor, labels: torch.Tensor,
                     dataset_name: Optional[str],
                     reliability_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-pixel CE with DMCE masking if applicable. Returns (B, H, W)."""
        masked_logits = _apply_logit_mask(logits, dataset_name, self._logit_masks)
        safe_labels = _apply_label_mask(
            labels, dataset_name, self._valid_labels,
            self.num_classes, self.ignore_index)
        valid_mask = (safe_labels != self.ignore_index)
        sl = safe_labels.clone()
        sl[~valid_mask] = 0
        per_pixel = F.cross_entropy(
            masked_logits, sl,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W); invalid pixels already give 0 by F.cross_entropy
        per_pixel = per_pixel * valid_mask.to(per_pixel.dtype)
        if reliability_weight is not None:
            per_pixel = per_pixel * reliability_weight
        return per_pixel, valid_mask

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        per_pixel, valid_mask = self._per_pixel_ce(
            logits, labels, dataset_name, reliability_weight)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return logits.sum() * 0.0
        k = max(int(valid_count * self.ohem_ratio), 1)
        flat = per_pixel.view(-1)
        k = min(k, flat.numel())
        topk_loss, _ = torch.topk(flat, k=k, sorted=False)
        return topk_loss.mean()


# Backward-compat alias: previous version of this file used the name OHEMLoss.
OHEMLoss = OHEMWrapper


class ClassBalancedCELoss(nn.Module):
    """Class-Balanced CE Loss (Cui 2019, CVPR; ref: cui2019class).

    Uses the effective-number formula
        w_c = (1 - beta) / (1 - beta^n_c)
    where n_c is the sample count of class c (here interpreted as pixel
    counts). With beta -> 0 this reduces to uniform weighting; beta -> 1
    reduces to plain inverse-frequency. Typical beta values: 0.9, 0.99,
    0.999, 0.9999 (paper recommends 0.9999 for ImageNet-LT).

    Note: the existing `class_balanced` sampler in
    `class_balanced_sampler.py` uses plain inverse-frequency
    (1/freq), which is the beta -> 1 limit. This loss applies the same
    idea at the *loss* layer with the proper effective-number form, which
    is the form evaluated in Cui 2019.

    If `class_samples` is None, falls back to the provided `class_weights`
    tensor (so it can be plugged in alongside NaiveCE in the existing
    build_class_weights() heuristic path).

    With `dataset_valid_classes` supplied, wraps a DMCELoss internally so
    DMCE-style masking is preserved; otherwise wraps NaiveCELoss.
    """

    def __init__(self, class_samples: Optional[torch.Tensor] = None,
                 beta: float = 0.9999, num_classes: int = 8,
                 ignore_index: int = 255,
                 class_weights: Optional[torch.Tensor] = None,
                 dataset_valid_classes: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = float(beta)
        self.num_classes = num_classes
        self.dataset_valid_classes = dataset_valid_classes or {}

        if class_samples is not None:
            class_samples = torch.as_tensor(class_samples, dtype=torch.float32)
            n_c = class_samples.clamp(min=1e-6)
            cb_weights = (1.0 - beta) / (1.0 - beta ** n_c)
            cb_weights = cb_weights / cb_weights.mean().clamp(min=1e-6)
            self.cb_weights = nn.Parameter(cb_weights, requires_grad=False)
        elif class_weights is not None:
            self.cb_weights = nn.Parameter(
                class_weights.clone().to(torch.float32), requires_grad=False)
        else:
            # Uniform weights (no class balancing). Still returns a valid
            # CE/DMCE loss so the loss layer is usable for ablation.
            self.cb_weights = nn.Parameter(
                torch.ones(num_classes, dtype=torch.float32),
                requires_grad=False)

        if self.dataset_valid_classes:
            self._inner = DMCELoss(
                dataset_valid_classes=self.dataset_valid_classes,
                ignore_index=ignore_index,
                num_classes=num_classes,
                class_weights=self.cb_weights)
        else:
            self._inner = NaiveCELoss(
                ignore_index=ignore_index,
                class_weights=self.cb_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._inner(logits, labels, dataset_name=dataset_name,
                          reliability_weight=reliability_weight)


class SBACELoss(nn.Module):
    """Spatial Boundary-Aware Cross-Entropy (SBA-CE).

    DMCE 在开放世界部署崩溃的根因是
    logit 截断缺几何边界约束。SBA-CE 在 DMCE logit masking 基础上,
    加入物理空间坐标先验: 侧扫声纳水柱区 (nadir) 在瀑布图中心列,
    该区域的非水柱类 logit 被物理性地压制。

    公式: ẑ_c(x,y) = z_c(x,y) - γ·P_water(x)·W_c
        P_water(x) = exp(-(x - x_mid)² / (2σ²))  (高斯水柱概率面)
        W_c = 类别敏感度阻尼 (前景类=1, 背景类=0)
        γ = 总体压制强度 (默认 5.0)

    训练时: 在 DMCE masked_logits 基础上再减去 γ·P_water·W_c
    推理时: 配合 U-Gate 后处理 (见 tools/training/u_gate_inference.py)

    Args:
        dataset_valid_classes: 同 DMCELoss
        img_width: 输入图像宽度 (用于计算 x_mid), 默认 512
        sigma_water: 水柱高斯宽度 (像素), 默认 50
        gamma: 水柱压制强度, 默认 5.0
        foreground_classes: 前景类列表 (这些类在水柱区被压制), 默认 [1,2,3]
    """

    def __init__(self, dataset_valid_classes: Dict[str, List[int]],
                 ignore_index: int = 255, num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None,
                 img_width: int = 512, sigma_water: float = 50.0,
                 gamma: float = 5.0,
                 foreground_classes: Optional[List[int]] = None):
        super().__init__()
        self.dataset_valid_classes = dataset_valid_classes
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.img_width = img_width
        self.sigma_water = sigma_water
        self.gamma = gamma
        self.foreground_classes = foreground_classes or [1, 2, 3]

        # 预计算 P_water(x) — 高斯水柱概率面 (1D, 沿宽度方向)
        x_mid = (img_width - 1) / 2.0
        x_coords = torch.arange(img_width, dtype=torch.float32)
        p_water_1d = torch.exp(-((x_coords - x_mid) ** 2) /
                               (2.0 * sigma_water ** 2))
        self.register_buffer("_p_water_1d", p_water_1d)

        # W_c — 类别敏感度阻尼 (前景类=1, 背景/预留类=0)
        w_c = torch.zeros(num_classes, dtype=torch.float32)
        for c in self.foreground_classes:
            w_c[c] = 1.0
        self.register_buffer("_w_c", w_c)

        # 复用 DMCE 的 per-dataset logit/label masking buffers
        for name, classes in dataset_valid_classes.items():
            logit_mask = torch.full((num_classes,), DMCELoss.MASK_VALUE)
            for c in classes:
                logit_mask[c] = 0.0
            self.register_buffer(f"_logit_mask_{name}", logit_mask)

            valid_label = torch.zeros(num_classes, dtype=torch.bool)
            for c in classes:
                valid_label[c] = True
            self.register_buffer(f"_valid_label_{name}", valid_label)

    def _spatial_bias(self, height: int, width: int,
                      device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """计算空间偏置 (B=1, C, H, W) = γ·P_water(x)·W_c.

        如果 width != img_width, 线性插值 P_water 到实际宽度。
        """
        if width != self.img_width:
            # 插值 P_water 到实际宽度
            p_water = self._p_water_1d.unsqueeze(0).unsqueeze(0)  # (1,1,W)
            p_water = torch.nn.functional.interpolate(
                p_water, size=(width,), mode='linear', align_corners=False)
            p_water = p_water.squeeze(0).squeeze(0)  # (W,)
        else:
            p_water = self._p_water_1d.to(device, dtype)

        # (W,) -> (1, C, 1, W) 广播
        # bias[c, :, x] = gamma * p_water[x] * w_c[c]
        w_c = self._w_c.to(device, dtype)
        bias = self.gamma * p_water.unsqueeze(0) * w_c.unsqueeze(1)  # (C, W)
        return bias.unsqueeze(0).unsqueeze(2)  # (1, C, 1, W)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not hasattr(self, f"_logit_mask_{dataset_name}"):
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Registered: {list(self.dataset_valid_classes.keys())}")

        logit_mask = getattr(self, f"_logit_mask_{dataset_name}")
        valid_label = getattr(self, f"_valid_label_{dataset_name}")

        # DMCE logit masking
        masked_logits = logits + logit_mask.view(1, -1, 1, 1)

        # SBA 空间偏置: 在水柱区压制前景类 logit
        _, _, h, w = logits.shape
        spatial_bias = self._spatial_bias(h, w, logits.device, logits.dtype)
        masked_logits = masked_logits - spatial_bias

        # DMCE label masking
        safe_labels = labels.clone()
        in_range = (labels >= 0) & (labels < self.num_classes)
        clamped = labels.clamp(0, self.num_classes - 1)
        invalid = in_range & ~valid_label[clamped]
        safe_labels[invalid] = self.ignore_index

        return F.cross_entropy(
            masked_logits, safe_labels,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


# ---------------------------------------------------------------------------
# Public baseline losses added 2026-07-08 per V6_EXPERIMENT_PLAN §12 P0
# (SBA-CE comparison baselines, ref user second-round review)
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t sorted errors.

    Standard implementation from Berman 2018 (github.com/bermanmaxim/LovaszSoftmax).
    Given gt_sorted (binary, sorted by error descending), returns the per-rank
    gradient used to weight the sorted errors.

    Args:
        gt_sorted: (N,) float, binary class labels sorted by error descending.
    Returns:
        (N,) gradient tensor.
    """
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-12)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss (Sudre 2017, MIDL; ref: sudre2017generalised).

    GDL = 1 - 2 * sum_c(w_c * inter_c) / sum_c(w_c * union_c)
        where w_c = 1 / (sum(g_c))^2  (per-class weight inversely
        proportional to class area squared).

    Difference from standard DiceLoss:
        - Standard Dice: each class contributes equally to the final mean.
        - Generalized Dice: each class is weighted by 1/area^2, so minority
          classes (small area) dominate the gradient — suitable for highly
          imbalanced segmentation (e.g. SSS where C0 water > 90%).
        - Empty classes (sum(g_c) == 0) contribute weight 0, skipped.

    With dataset_valid_classes supplied, applies DMCE logit/label masking
    (same as DiceLoss), so it can be a drop-in baseline under joint training.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255,
                 num_classes: int = 8,
                 dataset_valid_classes: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dataset_valid_classes = dataset_valid_classes or {}
        self._logit_masks, self._valid_labels = _build_dmce_buffers(
            self.dataset_valid_classes, num_classes, DMCELoss.MASK_VALUE)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        masked_logits = _apply_logit_mask(logits, dataset_name, self._logit_masks)
        safe_labels = _apply_label_mask(
            labels, dataset_name, self._valid_labels,
            self.num_classes, self.ignore_index)

        valid_mask = (safe_labels != self.ignore_index)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0

        sl = safe_labels.clone()
        sl[~valid_mask] = 0
        probs = F.softmax(masked_logits, dim=1)  # (B, C, H, W)
        onehot = F.one_hot(sl, self.num_classes).permute(0, 3, 1, 2).float()

        mask_f = valid_mask.to(logits.dtype).unsqueeze(1)  # (B, 1, H, W)
        if reliability_weight is not None:
            mask_f = mask_f * reliability_weight.unsqueeze(1)

        probs = probs * mask_f
        onehot = onehot * mask_f

        dims = (0, 2, 3)  # batch + spatial -> (C,)
        g = onehot.sum(dim=dims)                      # per-class GT area
        inter = (probs * onehot).sum(dim=dims)        # per-class intersection
        union = probs.sum(dim=dims) + g               # per-class union
        # weight = 1 / (area^2 + smooth), zeroed for empty classes.
        # smooth here stabilizes the weight for tiny classes (not the final
        # denominator); the final denom uses a tiny eps to avoid dominating
        # the weighted sum (which is O(1/area) in magnitude).
        nonempty = g > 0
        w = torch.where(nonempty, 1.0 / (g * g + self.smooth),
                        torch.zeros_like(g))
        numerator = 2.0 * (w * inter).sum()
        denom = (w * union).sum().clamp(min=1e-8)
        gdl = 1.0 - numerator / denom
        return gdl


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax Loss (Berman 2018, CVPR; ref: berman2018lovasz).

    Paper: "The Lovasz-Softmax loss: A tractable surrogate for the
    optimization of the intersection-over-union in neural networks".

    The Lovasz extension provides a convex surrogate of the Jaccard (IoU)
    loss, directly optimizing mIoU rather than per-pixel CE. Compared to CE:
        - More robust to class imbalance (no per-pixel weighting needed).
        - Directly targets the mIoU evaluation metric.

    Implementation follows the author's public code
    (github.com/bermanmaxim/LovaszSoftmax), adapted to:
        - Accept DMCE logit/label masking (dataset_valid_classes).
        - Skip classes absent from the current batch ('present' mode).
        - Skip ignore_index pixels.

    For each class c present in the batch:
        fg = (labels == c)            binary foreground
        pc = softmax_prob[:, c]       per-pixel probability of class c
        error = (1 - pc) if fg else pc   (in [0, 1])
        Sort errors descending, compute Lovasz gradient from sorted fg,
        loss_c = dot(relu(error_sorted), grad)
    Final loss = mean over present classes.
    """

    def __init__(self, ignore_index: int = 255, num_classes: int = 8,
                 classes_to_avg: Optional[List[int]] = None,
                 dataset_valid_classes: Optional[Dict[str, List[int]]] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.classes_to_avg = classes_to_avg  # None = all classes
        self.dataset_valid_classes = dataset_valid_classes or {}
        self._logit_masks, self._valid_labels = _build_dmce_buffers(
            self.dataset_valid_classes, num_classes, DMCELoss.MASK_VALUE)

    def _lovasz_hinge_flat(self, prob_flat: torch.Tensor,
                           fg_flat: torch.Tensor) -> torch.Tensor:
        """Binary Lovasz hinge for one class.

        Args:
            prob_flat: (N,) softmax probability of class c being foreground.
            fg_flat: (N,) float, 1.0 if pixel is class c else 0.0.
        Returns:
            scalar loss for this class.
        """
        if prob_flat.numel() == 0:
            return prob_flat.sum() * 0.0
        signs = 2.0 * fg_flat - 1.0  # +1 if fg, -1 if bg
        errors = (1.0 - prob_flat * signs).clamp(min=0.0)
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = fg_flat[perm]
        grad = _lovasz_grad(gt_sorted)
        loss = (errors_sorted * grad).sum()
        return loss

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if reliability_weight is not None:
            raise ValueError(
                "LovaszSoftmaxLoss does not support reliability_weight. "
                "The Lovasz surrogate operates on class-level sorted errors and "
                "cannot be weighted per-pixel without altering its surrogate "
                "semantics. SSPM reliability is already incorporated via the "
                "input image channel (see train_joint_dmce.py "
                "reliability_weighted=True passes images[:, 4] into the model "
                "as channel 4); per-pixel re-weighting is redundant here. "
                "To use reliability_weight, switch loss_type to 'tversky' or "
                "'dmce' (which support per-pixel weighting). "
                "If you intended to use Lovasz with reliability_weight, remove "
                "--reliability_weighted_loss or change loss_type."
            )
        masked_logits = _apply_logit_mask(logits, dataset_name, self._logit_masks)
        safe_labels = _apply_label_mask(
            labels, dataset_name, self._valid_labels,
            self.num_classes, self.ignore_index)

        valid_mask = (safe_labels != self.ignore_index)
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0

        probs = F.softmax(masked_logits, dim=1)  # (B, C, H, W)
        # 展平到 (N, C) 和 (N,)
        B, C, H, W = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        labels_flat = safe_labels.reshape(-1)                  # (N,)
        valid_flat = valid_mask.reshape(-1)
        # 仅保留 valid 像素
        probs_flat = probs_flat[valid_flat]
        labels_flat = labels_flat[valid_flat]
        if probs_flat.numel() == 0:
            return logits.sum() * 0.0

        classes = self.classes_to_avg if self.classes_to_avg is not None \
            else list(range(C))
        losses = []
        for c in classes:
            fg = (labels_flat == c).float()
            if fg.sum() == 0:
                continue  # 'present' mode: skip absent classes
            pc = probs_flat[:, c]
            losses.append(self._lovasz_hinge_flat(pc, fg))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


# Public alias: the task spec names the
# Berman 2018 loss `LovaszLoss`. The implementation above (`LovaszSoftmaxLoss`)
# is the multi-class Lovasz hinge applied to softmax probabilities — exactly
# the "Lovasz hinge loss for multi-class" requested. Expose both names so
# downstream code can import either; behaviour is identical. Mirrors the
# `OHEMLoss = OHEMWrapper` backward-compat alias pattern.
LovaszLoss = LovaszSoftmaxLoss


def build_criterion(loss_type: str, dataset_valid_classes: Dict[str, List[int]],
                     ignore_index: int = 255, num_classes: int = 8,
                     class_weights: Optional[torch.Tensor] = None,
                     **kwargs) -> nn.Module:
    """Factory: build loss by type.

    Args:
        loss_type: one of
            - "naive_ce"          : standard CE (baseline)
            - "valid_masked_ce"   : CE with hard label-validity masking (no logit mask)
            - "dmce"              : Dynamic Mask CE (proposed; soft logit mask + hard label mask)
            - "valid_masked_dmce" : alias for dmce (same dual-mask behavior)
            - "focal"             : Focal Loss (Lin 2017)
            - "dice"              : Dice Loss (Milletari 2016)
            - "tversky"           : Tversky Loss (Salehi 2017; alpha=0.3 beta=0.7)
            - "boundary"          : Boundary Loss (Kervadec 2019)
            - "ohem"              : OHEM wrapper (Shrivastava 2016)
            - "class_balanced_ce" : Class-Balanced CE (Cui 2019)
            - "sba_ce"            : Spatial Boundary-Aware CE; DMCE masking + 物理
                                   空间水柱 logit 偏置
            - "generalized_dice" : Generalized Dice Loss (Sudre 2017, §12 P0)
            - "lovasz"           : Lovasz-Softmax Loss (Berman 2018, §12 P0)
            - "lovasz_loss"      : alias for "lovasz" (exposes the LovaszLoss name)
            - "margin_reg_dmce"           : A8-1 Margin-Regularized DMCE
            - "soft_mask_dmce"            : A8-2 Soft Mask DMCE
            - "classifier_suppression_dmce": A8-3 Classifier Suppression DMCE
        dataset_valid_classes: required for dmce; passed to all losses that
            support DMCE-style masking (focal/dice/tversky/boundary/ohem/
            class_balanced_ce/sba_ce/generalized_dice/lovasz). Ignored for naive_ce.
        ignore_index, num_classes, class_weights: shared params
        **kwargs: loss-specific overrides
            focal_alpha, focal_gamma : Focal hyperparams (default 0.25 / 2.0)
            dice_smooth              : Dice smoothing (default 1.0)
            tversky_alpha, tversky_beta : Tversky FN/FP weights (default 0.3 / 0.7)
            tversky_smooth           : Tversky smoothing (default 1.0)
            gen_dice_smooth          : Generalized Dice smoothing (default 1.0)
            ohem_ratio               : OHEM keep ratio (default 0.7)
            cb_beta, class_samples   : Class-Balanced hyperparams
                (default beta=0.9999; class_samples=None -> uniform)
            boundary_alpha           : Boundary combined-mode alpha (default 0.5)
            boundary_only            : if True, BoundaryLoss returns just the
                boundary term (auxiliary use, default True)
            sba_img_width            : SBA-CE 图像宽度 (default 512)
            sba_sigma_water          : SBA-CE 水柱高斯宽度 (default 50.0)
            sba_gamma                : SBA-CE 水柱压制强度 (default 5.0)
            sba_foreground_classes   : SBA-CE 前景类列表 (default [1,2,3])
            lovasz_classes           : Lovasz 类列表 (None = all present)
            a8_margin                : A8-1 margin gap (default 2.0)
            a8_lambda                : A8-1/A8-3 regularization weight (default 0.1)
            a8_alpha                 : A8-2 soft mask alpha (default 0.1)

    Returns:
        nn.Module criterion with signature
        (logits, labels, dataset_name=None, reliability_weight=None) -> loss

    Note:
        LovaszSoftmaxLoss does NOT support reliability_weight (it operates on
        class-level sorted errors, not per-pixel). Calling it with
        reliability_weight will raise ValueError at forward time. If you need
        reliability weighting, use 'tversky' or 'dmce' loss_type instead, or
        remove --reliability_weighted_loss when using Lovasz.
    """
    if loss_type == "naive_ce":
        return NaiveCELoss(ignore_index=ignore_index,
                           class_weights=class_weights)
    elif loss_type == "valid_masked_ce":
        return ValidMaskedCELoss(dataset_valid_classes=dataset_valid_classes,
                                 ignore_index=ignore_index,
                                 num_classes=num_classes,
                                 class_weights=class_weights)
    elif loss_type == "valid_masked_dmce" or loss_type == "dmce":
        return DMCELoss(dataset_valid_classes=dataset_valid_classes,
                        ignore_index=ignore_index,
                        num_classes=num_classes,
                        class_weights=class_weights)
    elif loss_type == "focal":
        return FocalLoss(alpha=kwargs.get("focal_alpha", 0.25),
                         gamma=kwargs.get("focal_gamma", 2.0),
                         ignore_index=ignore_index,
                         num_classes=num_classes,
                         class_weights=class_weights,
                         dataset_valid_classes=dataset_valid_classes)
    elif loss_type == "dice":
        return DiceLoss(smooth=kwargs.get("dice_smooth", 1.0),
                        ignore_index=ignore_index,
                        num_classes=num_classes,
                        dataset_valid_classes=dataset_valid_classes)
    elif loss_type == "tversky":
        # Tversky Loss (Salehi 2017) — generalization of Dice with separate
        # FN (alpha) / FP (beta) weighting; alpha=0.3 beta=0.7 default is
        # recall-oriented.
        return TverskyLoss(alpha=kwargs.get("tversky_alpha", 0.3),
                           beta=kwargs.get("tversky_beta", 0.7),
                           smooth=kwargs.get("tversky_smooth", 1.0),
                           ignore_index=ignore_index,
                           num_classes=num_classes,
                           dataset_valid_classes=dataset_valid_classes)
    elif loss_type == "boundary":
        return BoundaryLoss(alpha=kwargs.get("boundary_alpha", 0.5),
                            ignore_index=ignore_index,
                            num_classes=num_classes,
                            class_weights=class_weights,
                            boundary_only=kwargs.get("boundary_only", True),
                            dataset_valid_classes=dataset_valid_classes)
    elif loss_type == "ohem":
        return OHEMWrapper(
            ohem_ratio=kwargs.get("ohem_ratio", 0.7),
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights,
            dataset_valid_classes=dataset_valid_classes)
    elif loss_type == "class_balanced_ce":
        return ClassBalancedCELoss(
            class_samples=kwargs.get("class_samples"),
            beta=kwargs.get("cb_beta", 0.9999),
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
            dataset_valid_classes=dataset_valid_classes)
    elif loss_type == "sba_ce":
        # SBA-CE: DMCE logit masking + 物理空间水柱偏置
        return SBACELoss(
            dataset_valid_classes=dataset_valid_classes,
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights,
            img_width=kwargs.get("sba_img_width", 512),
            sigma_water=kwargs.get("sba_sigma_water", 50.0),
            gamma=kwargs.get("sba_gamma", 5.0),
            foreground_classes=kwargs.get("sba_foreground_classes", [1, 2, 3]))
    elif loss_type == "generalized_dice":
        # Generalized Dice Loss (Sudre 2017) — SBA-CE 公开基线 (V6_EXPERIMENT_PLAN §12 P0)
        return GeneralizedDiceLoss(
            smooth=kwargs.get("gen_dice_smooth", 1.0),
            ignore_index=ignore_index,
            num_classes=num_classes,
            dataset_valid_classes=dataset_valid_classes)
    elif loss_type == "lovasz" or loss_type == "lovasz_loss":
        # Lovasz-Softmax Loss (Berman 2018) — SBA-CE 公开基线 (V6_EXPERIMENT_PLAN §12 P0)
        # "lovasz_loss" is an alias that exposes the LovaszLoss class name.
        return LovaszSoftmaxLoss(
            ignore_index=ignore_index,
            num_classes=num_classes,
            classes_to_avg=kwargs.get("lovasz_classes"),
            dataset_valid_classes=dataset_valid_classes)
    elif loss_type == "margin_reg_dmce":
        # A8-1: Margin/Energy Regularized DMCE
        return MarginRegularizedDMCELoss(
            dataset_valid_classes=dataset_valid_classes,
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights,
            margin=kwargs.get("a8_margin", 2.0),
            lambda_margin=kwargs.get("a8_lambda", 0.1))
    elif loss_type == "soft_mask_dmce":
        # A8-2: Soft Mask DMCE
        return SoftMaskDMCELoss(
            dataset_valid_classes=dataset_valid_classes,
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights,
            alpha=kwargs.get("a8_alpha", 0.1))
    elif loss_type == "classifier_suppression_dmce":
        # A8-3: Dataset-Aware Classifier Suppression DMCE
        return ClassifierSuppressionDMCELoss(
            dataset_valid_classes=dataset_valid_classes,
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights,
            lambda_suppress=kwargs.get("a8_lambda", 0.01))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ===========================================================================
# A8: Invalid-class constraint correction methods
# ===========================================================================
# The three classes below address the "invalid-class hijacking" failure mode
# diagnosed in Paper2 §6 (DMCE diagnostic). Standard DMCE hard-masks invalid
# class logits to -1e9, which completely cuts gradient flow to those classes.
# This can cause the classifier weights for invalid classes to drift unchecked
# (no gradient signal = no correction), leading to logit hijacking at deployment
# when the mask is removed.
#
# A8 provides three correction strategies (multi-scheme ablation):
#   A8-1 (MarginRegularizedDMCELoss)      : hinge margin on logit gap
#   A8-2 (SoftMaskDMCELoss)               : soft log-space mask (log(α) penalty)
#   A8-3 (ClassifierSuppressionDMCELoss)  : L2 regularization on invalid weights
#
# All three:
#   - Inherit DMCELoss masking logic via super().__init__
#   - Have identical forward signature to DMCELoss
#   - Handle reliability_weight=None gracefully (no ValueError, unlike Lovasz)
#   - Reduce to standard DMCE when their hyper-parameter is zeroed
# ===========================================================================


class MarginRegularizedDMCELoss(DMCELoss):
    """A8-1: Margin/Energy Regularized DMCE Loss.

    Instead of hard-masking invalid class logits to -1e9 (which cuts all
    gradient flow), A8-1 keeps the standard DMCE CE loss but adds a hinge
    margin regularizer on the RAW (unmasked) logits. The regularizer
    penalizes cases where the maximum invalid-class logit exceeds the
    maximum valid-class logit by more than ``-margin``.

    Formula:
        loss = DMCE_CE(masked_logits, labels)
               + λ * mean_b( relu( max(z_invalid) - max(z_valid) + margin ) )

    where ``z_invalid`` / ``z_valid`` are the raw logits restricted to
    invalid / valid classes, and the max is taken over the spatial dims
    and the class dim per batch sample.

    When ``lambda_margin=0``: reduces to standard DMCE (no regularization).
    When ``margin=0``: penalizes any invalid logit exceeding any valid logit.

    Args:
        dataset_valid_classes: per-dataset valid class dict (same as DMCELoss)
        ignore_index, num_classes, class_weights: same as DMCELoss
        margin: required logit gap (max_valid - max_invalid >= margin).
            Default 2.0.
        lambda_margin: regularization weight. Default 0.1.
    """

    def __init__(self, dataset_valid_classes: Dict[str, List[int]],
                 ignore_index: int = 255, num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None,
                 margin: float = 2.0, lambda_margin: float = 0.1):
        super().__init__(
            dataset_valid_classes=dataset_valid_classes,
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights)
        self.margin = float(margin)
        self.lambda_margin = float(lambda_margin)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- Standard DMCE CE loss (with hard logit/label masking) ---
        if dataset_name is not None and hasattr(self, f"_logit_mask_{dataset_name}"):
            ce_loss = super().forward(logits, labels, dataset_name,
                                      reliability_weight)
        else:
            # Fallback: plain CE (no masking) when dataset_name is unknown/None
            if reliability_weight is not None:
                ce_loss = _weighted_ce(
                    logits, labels, reliability_weight,
                    ignore_index=self.ignore_index,
                    class_weights=self.class_weights)
            else:
                ce_loss = F.cross_entropy(
                    logits, labels, weight=self.class_weights,
                    ignore_index=self.ignore_index)

        # --- Margin regularization on RAW logits ---
        if self.lambda_margin == 0.0:
            return ce_loss
        if dataset_name is None or not hasattr(self, f"_valid_label_{dataset_name}"):
            return ce_loss  # no valid/invalid distinction available

        valid_label = getattr(self, f"_valid_label_{dataset_name}")
        invalid_mask = ~valid_label  # (num_classes,) bool
        if not bool(invalid_mask.any()):
            return ce_loss  # all classes valid -> no invalid logits to constrain

        # RAW logits (before DMCE masking): (B, C, H, W)
        z_invalid = logits[:, invalid_mask]   # (B, num_invalid, H, W)
        z_valid = logits[:, valid_label]      # (B, num_valid, H, W)

        # Per-sample max over (class, H, W)
        max_invalid = z_invalid.amax(dim=(1, 2, 3))  # (B,)
        max_valid = z_valid.amax(dim=(1, 2, 3))      # (B,)

        # Hinge: active when max_invalid > max_valid - margin
        hinge = torch.relu(max_invalid - max_valid + self.margin)  # (B,)
        margin_loss = self.lambda_margin * hinge.mean()

        return ce_loss + margin_loss


class SoftMaskDMCELoss(DMCELoss):
    """A8-2: Soft Mask DMCE Loss.

    Replaces DMCE's hard logit mask (invalid logits set to -1e9) with a
    *soft* log-space mask: invalid-class logits receive an additive penalty
    of ``log(α)`` (a negative value), reducing their softmax probability
    without completely cutting gradient flow.

    Implementation (additive soft mask in logit space):
        soft_mask[c] = 0.0          if c in valid_classes
        soft_mask[c] = log(α)       if c in invalid_classes  (α > 0)
        soft_mask[c] = -1e9         if α == 0  (hard mask = standard DMCE)
        masked_logits = logits + soft_mask
        loss = CE(masked_logits, safe_labels)

    Mathematical note: adding ``log(α)`` to a logit is equivalent to
    multiplying that class's softmax probability by α (before renormalization).
    This is the standard "soft mask" in logit space and matches the A8-2
    concept of "down-weighting invalid classes by factor α".

    When ``alpha=0``: reduces to standard DMCE (hard mask -1e9).
    When ``alpha=1``: no masking (log(1)=0), reduces to NaiveCE.

    Args:
        dataset_valid_classes: per-dataset valid class dict (same as DMCELoss)
        ignore_index, num_classes, class_weights: same as DMCELoss
        alpha: soft mask strength for invalid classes in (0, 1).
            0 = hard mask (standard DMCE), 1 = no mask (NaiveCE). Default 0.1.
    """

    def __init__(self, dataset_valid_classes: Dict[str, List[int]],
                 ignore_index: int = 255, num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None,
                 alpha: float = 0.1):
        super().__init__(
            dataset_valid_classes=dataset_valid_classes,
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights)
        self.alpha = float(alpha)

    def _build_soft_mask(self, dataset_name: str,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build per-class additive soft mask: 0 for valid, log(α) for invalid."""
        valid_label = getattr(self, f"_valid_label_{dataset_name}")
        soft_mask = torch.zeros(self.num_classes, device=device, dtype=dtype)
        if self.alpha <= 0.0:
            # Hard mask (standard DMCE): invalid -> -1e9
            soft_mask[~valid_label] = DMCELoss.MASK_VALUE
        else:
            # Soft mask: invalid -> log(α)  (negative penalty)
            soft_mask[~valid_label] = float(torch.log(torch.tensor(self.alpha)))
        return soft_mask

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dataset_name is None or not hasattr(self, f"_valid_label_{dataset_name}"):
            # Fallback: plain CE (no masking)
            if reliability_weight is not None:
                return _weighted_ce(
                    logits, labels, reliability_weight,
                    ignore_index=self.ignore_index,
                    class_weights=self.class_weights)
            return F.cross_entropy(
                logits, labels, weight=self.class_weights,
                ignore_index=self.ignore_index)

        # --- Soft logit masking ---
        soft_mask = self._build_soft_mask(dataset_name, logits.device, logits.dtype)
        masked_logits = logits + soft_mask.view(1, -1, 1, 1)

        # --- Label masking (same as DMCE: invalid labels -> ignore_index) ---
        valid_label = getattr(self, f"_valid_label_{dataset_name}")
        safe_labels = labels.clone()
        in_range = (labels >= 0) & (labels < self.num_classes)
        clamped = labels.clamp(0, self.num_classes - 1)
        invalid = in_range & ~valid_label[clamped]
        safe_labels[invalid] = self.ignore_index

        # --- CE with optional reliability weighting ---
        if reliability_weight is not None:
            return _weighted_ce(
                masked_logits, safe_labels, reliability_weight,
                ignore_index=self.ignore_index,
                class_weights=self.class_weights)
        return F.cross_entropy(
            masked_logits, safe_labels,
            weight=self.class_weights,
            ignore_index=self.ignore_index)


def find_classifier_module(model: nn.Module,
                           num_classes: int = 8) -> Optional[nn.Module]:
    """Locate the final classifier module in a segmentation model.

    Tries common attribute paths across the supported backbones:
      - SegFormer (SSPM): model.decoder.seg_head  (Conv2d, weight shape (C, D, 1, 1))
      - DeepLabV3+ / FCN / PSPNet (torchvision): model.classifier[-1]
      - U-Net: model.classifier

    Returns the first module whose ``weight`` has out_channels == num_classes,
    or None if no classifier is found (caller should handle gracefully).
    """
    candidates = []
    # SegFormer decoder seg_head
    decoder = getattr(model, "decoder", None) or getattr(model, "decode_head", None)
    if decoder is not None:
        candidates.append(getattr(decoder, "seg_head", None))
        candidates.append(getattr(decoder, "classifier", None))
    # Direct classifier attribute (U-Net, torchvision DeepLabV3+ wrapper)
    cls = getattr(model, "classifier", None)
    if cls is not None:
        if isinstance(cls, nn.Sequential):
            candidates.append(cls[-1] if len(cls) > 0 else None)
        else:
            candidates.append(cls)
    candidates.append(getattr(model, "head", None))

    for mod in candidates:
        if mod is None:
            continue
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        if w.shape[0] == num_classes:
            return mod
    return None


class ClassifierSuppressionDMCELoss(DMCELoss):
    """A8-3: Dataset-Aware Classifier Suppression DMCE Loss.

    Instead of operating on logits (A8-1/A8-2), A8-3 regularizes the
    classifier *weights* directly: the L2 norm of invalid-class classifier
    weights is penalized, preventing them from growing unchecked (which is
    the root cause of logit hijacking when DMCE mask is removed).

    Formula:
        loss = DMCE_CE(masked_logits, labels)
               + λ * ||W_invalid||_2^2

    where ``W_invalid`` is the weight matrix of the classifier head for
    classes not in the current dataset's valid set.

    The classifier module must be registered via ``set_classifier_module()``
    after the model is built (the training loop calls this). If no module
    is registered, the loss degrades to standard DMCE (no regularization).

    When ``lambda_suppress=0``: reduces to standard DMCE.

    Args:
        dataset_valid_classes: per-dataset valid class dict (same as DMCELoss)
        ignore_index, num_classes, class_weights: same as DMCELoss
        lambda_suppress: L2 regularization weight for invalid classifier
            weights. Default 0.01.
    """

    def __init__(self, dataset_valid_classes: Dict[str, List[int]],
                 ignore_index: int = 255, num_classes: int = 8,
                 class_weights: Optional[torch.Tensor] = None,
                 lambda_suppress: float = 0.01):
        super().__init__(
            dataset_valid_classes=dataset_valid_classes,
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights)
        self.lambda_suppress = float(lambda_suppress)
        self._classifier_module: Optional[nn.Module] = None

    def set_classifier_module(self, module: nn.Module) -> None:
        """Register the classifier head module so forward can access its weight.

        Called by the training loop after the model is built:
            criterion.set_classifier_module(find_classifier_module(model))
        """
        self._classifier_module = module

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                dataset_name: str = None,
                reliability_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # --- Standard DMCE CE loss ---
        if dataset_name is not None and hasattr(self, f"_logit_mask_{dataset_name}"):
            ce_loss = super().forward(logits, labels, dataset_name,
                                      reliability_weight)
        else:
            if reliability_weight is not None:
                ce_loss = _weighted_ce(
                    logits, labels, reliability_weight,
                    ignore_index=self.ignore_index,
                    class_weights=self.class_weights)
            else:
                ce_loss = F.cross_entropy(
                    logits, labels, weight=self.class_weights,
                    ignore_index=self.ignore_index)

        # --- Classifier weight suppression ---
        if self.lambda_suppress == 0.0:
            return ce_loss
        if self._classifier_module is None:
            return ce_loss  # no classifier registered -> degrade to DMCE
        if dataset_name is None or not hasattr(self, f"_valid_label_{dataset_name}"):
            return ce_loss

        W = self._classifier_module.weight  # (num_classes, D, *) or (num_classes, D)
        if W.shape[0] != self.num_classes:
            return ce_loss  # shape mismatch -> skip regularization

        valid_label = getattr(self, f"_valid_label_{dataset_name}")
        invalid_mask = ~valid_label  # (num_classes,) bool
        if not bool(invalid_mask.any()):
            return ce_loss

        W_invalid = W[invalid_mask]  # (num_invalid, D, *)
        reg_loss = self.lambda_suppress * (W_invalid ** 2).sum()

        return ce_loss + reg_loss
