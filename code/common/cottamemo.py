"""CoTTA and MEMO test-time adaptation methods (comparison baselines for UCDPA).

MEMO: Marginal Entropy Minimization with Augmentations.
  - For each batch, generate A augmentations (tensor-level ops, no PIL).
  - Forward all augmentations, compute marginal prediction (mean softmax).
  - Loss = entropy(marginal) = -sum(p * log p).
  - Update LayerNorm affine params only.

CoTTA: Continual Test-Time Adaptation (simplified for ViT/LayerNorm).
  - Maintain a teacher = EMA copy of student (momentum 0.999), tracked via
    a teacher_state_dict of LayerNorm params (param-swap for teacher forward).
  - Pseudo-label from teacher on weak (identity) augmentation.
  - Strong augmentation: random flip + random resized crop + color jitter.
  - Loss = CE(pseudo_label, softmax(logits_strong))
          + aug_strength * entropy(softmax(logits_strong)).
  - EMA update teacher LayerNorm from student.
  - Stochastic restore: with prob p_restore, restore student LayerNorm to source.

Both methods update only LayerNorm affine parameters and keep the model in eval mode.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .come_utils import (
    softmax_entropy,
    softmax_entropy_from_probs,
    come_opinion,
    come_entropy,
    safe_log,
)
from .ucdpa import compute_mutual_information


# ---------------------------------------------------------------------------
# Tensor-level augmentations (no PIL)
# ---------------------------------------------------------------------------
def random_horizontal_flip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Random horizontal flip along width dimension. x: [B, C, H, W]."""
    if torch.rand(1).item() < p:
        return torch.flip(x, dims=[3])
    return x


def random_resized_crop(
    x: torch.Tensor,
    scale: Tuple[float, float] = (0.7, 1.0),
    ratio: Tuple[float, float] = (0.8, 1.2),
    size: int = 224,
) -> torch.Tensor:
    """Random resized crop on a tensor batch. x: [B, C, H, W] -> [B, C, size, size]."""
    B, C, H, W = x.shape
    s = torch.empty(1).uniform_(scale[0], scale[1]).item()
    # Sample ratio uniformly in log space for symmetry around 1.
    r = torch.empty(1).uniform_(math.log(ratio[0]), math.log(ratio[1])).exp().item()
    target_area = s * H * W
    crop_h = int(round(math.sqrt(target_area / r)))
    crop_w = int(round(math.sqrt(target_area * r)))
    crop_h = min(max(crop_h, 1), H)
    crop_w = min(max(crop_w, 1), W)
    top = torch.randint(0, H - crop_h + 1, (1,)).item() if H > crop_h else 0
    left = torch.randint(0, W - crop_w + 1, (1,)).item() if W > crop_w else 0
    x_cropped = x[:, :, top:top + crop_h, left:left + crop_w]
    x_resized = F.interpolate(
        x_cropped, size=(size, size), mode="bilinear", align_corners=False
    )
    return x_resized


def color_jitter(
    x: torch.Tensor,
    brightness: Tuple[float, float] = (0.8, 1.2),
    contrast: Tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """Color jitter on a normalized tensor. x: [B, C, H, W].

    brightness: scale all values by a random factor in [0.8, 1.2].
    contrast: scale around the per-sample spatial mean by a random factor.
    """
    b = torch.empty(1).uniform_(brightness[0], brightness[1]).item()
    x = x * b
    c = torch.empty(1).uniform_(contrast[0], contrast[1]).item()
    mean = x.mean(dim=[2, 3], keepdim=True)
    x = (x - mean) * c + mean
    return x


def augment(x: torch.Tensor) -> torch.Tensor:
    """Apply random horizontal flip + random resized crop + color jitter."""
    x = random_horizontal_flip(x)
    x = random_resized_crop(x)
    x = color_jitter(x)
    return x


# ---------------------------------------------------------------------------
# MEMO state
# ---------------------------------------------------------------------------
class MEMOState:
    """State container for MEMO. Only tracks the number of augmentations."""

    def __init__(self, num_augmentations: int = 4):
        self.num_augmentations = num_augmentations


# ---------------------------------------------------------------------------
# CoTTA state
# ---------------------------------------------------------------------------
class CoTTAState:
    """State container for CoTTA.

    Tracks:
      - teacher_state_dict: dict {param_index -> tensor} of teacher LayerNorm
        params (initialized to source values, EMA-updated from student).
      - source_state_dict: dict {param_index -> tensor} of source LayerNorm
        params (for stochastic restore).
      - restore_prob: probability of restoring student to source per step.

    The teacher forward is implemented via param-swap: temporarily copy teacher
    LayerNorm params into the model, forward, then restore student params.
    """

    def __init__(
        self,
        model,
        params: List[torch.Tensor],
        source_ckpt_path: str,
        teacher_momentum: float = 0.999,
        restore_prob: float = 0.01,
        device: str = "cuda",
    ):
        self.params = params  # reference to student's trainable LayerNorm params
        self.teacher_momentum = teacher_momentum
        self.restore_prob = restore_prob
        self.device = device
        # teacher_state_dict: copy of source LayerNorm params (will be EMA-updated)
        self.teacher_state_dict: Dict[int, torch.Tensor] = {
            i: p.data.clone().detach() for i, p in enumerate(params)
        }
        # source_state_dict: for stochastic restore
        self.source_state_dict: Dict[int, torch.Tensor] = {
            i: p.data.clone().detach() for i, p in enumerate(params)
        }

    @torch.no_grad()
    def teacher_forward(self, model, x: torch.Tensor) -> torch.Tensor:
        """Forward x through the teacher by swapping LayerNorm params."""
        student_backup = {i: p.data.clone() for i, p in enumerate(self.params)}
        for i, p in enumerate(self.params):
            p.data.copy_(self.teacher_state_dict[i])
        logits = model(x)
        for i, p in enumerate(self.params):
            p.data.copy_(student_backup[i])
        return logits

    @torch.no_grad()
    def update_teacher(self, params: Optional[List[torch.Tensor]] = None):
        """EMA update: teacher = momentum * teacher + (1-momentum) * student."""
        params = params if params is not None else self.params
        for i, p in enumerate(params):
            self.teacher_state_dict[i].mul_(self.teacher_momentum).add_(
                (1.0 - self.teacher_momentum) * p.data.detach()
            )

    @torch.no_grad()
    def maybe_restore(self, params: Optional[List[torch.Tensor]] = None):
        """Stochastic restore: with prob restore_prob, restore student to source."""
        params = params if params is not None else self.params
        if torch.rand(1).item() < self.restore_prob:
            for i, p in enumerate(params):
                p.data.copy_(self.source_state_dict[i])


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------
def _build_diag(
    logits: torch.Tensor,
    cfg,
    pseudo_labels: Optional[torch.Tensor] = None,
    pl_loss_per_sample: Optional[torch.Tensor] = None,
    objective_per_sample: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Build the standard diag dict from pre-update student logits.

    Computes the same diagnostic fields used by the runner for all methods:
    pmax, entropy, uncertainty_u, mutual_info, ucb_score, come_loss, pl_loss,
    pseudo_label, class_balance_weight, L_cal, L_fpa, L_objective.
    """
    probs = F.softmax(logits, dim=1)
    pmax = probs.max(dim=1).values
    ent = softmax_entropy_from_probs(probs, eps=cfg.eps)
    _, u, _, _, _ = come_opinion(
        logits, p_norm=cfg.come_p_norm, tau=cfg.come_tau,
        evidence_mode=cfg.come_evidence_mode, eps=cfg.eps,
    )
    mi = compute_mutual_information(
        logits, p_norm=cfg.come_p_norm, tau=cfg.come_tau,
        evidence_mode=cfg.come_evidence_mode, eps=cfg.eps,
    )
    if pseudo_labels is None:
        pseudo_labels = probs.argmax(dim=1).detach()
    if pl_loss_per_sample is None:
        pl_loss_per_sample = torch.zeros_like(pmax)
    if objective_per_sample is None:
        objective_per_sample = ent.detach()
    diag: Dict[str, torch.Tensor] = {
        "pmax": pmax.detach(),
        "entropy": ent.detach(),
        "uncertainty_u": u.detach(),
        "mutual_info": mi,
        "ucb_score": torch.zeros_like(pmax),
        "come_loss": ent.detach(),
        "pl_loss": pl_loss_per_sample.detach() if torch.is_tensor(pl_loss_per_sample) else pl_loss_per_sample,
        "pseudo_label": pseudo_labels,
        "class_balance_weight": torch.ones_like(pmax),
        "L_cal": torch.zeros_like(pmax),
        "L_fpa": torch.zeros_like(pmax),
        "L_objective": objective_per_sample.detach() if torch.is_tensor(objective_per_sample) else objective_per_sample,
    }
    return diag


# ---------------------------------------------------------------------------
# MEMO loss
# ---------------------------------------------------------------------------
def _compute_memo_loss(
    model, x: torch.Tensor, cfg, state: MEMOState, device: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """MEMO: marginal entropy minimization with augmentations.

    Returns (loss, sample_loss, diag, logits_orig).
    """
    # Pre-update student predictions on original x (for metrics, no grad)
    with torch.no_grad():
        logits_orig = model(x)

    # Generate A augmentations
    A = state.num_augmentations
    aug_list = []
    for _ in range(A):
        x_aug = augment(x)
        aug_list.append(x_aug)
    x_augs = torch.cat(aug_list, dim=0)  # [A*B, C, H, W]

    # Forward all augmentations (with grad)
    logits_aug = model(x_augs)  # [A*B, K]
    B = x.shape[0]
    K = logits_aug.shape[1]
    # Reshape to [B, A, K]
    logits_aug = logits_aug.view(A, B, K).permute(1, 0, 2).contiguous()

    # Marginal prediction = mean of softmax over A dimension
    probs_aug = F.softmax(logits_aug, dim=2)  # [B, A, K]
    marginal = probs_aug.mean(dim=1)  # [B, K]

    # Loss = entropy(marginal) = -sum(p * log p)
    loss = -(marginal * safe_log(marginal, cfg.eps)).sum(dim=1).mean()

    # Per-sample objective loss (for logging)
    sample_loss = -(marginal * safe_log(marginal, cfg.eps)).sum(dim=1).detach()

    # Diagnostics from original predictions
    diag = _build_diag(logits_orig, cfg, objective_per_sample=sample_loss)

    return loss, sample_loss, diag, logits_orig


# ---------------------------------------------------------------------------
# CoTTA loss
# ---------------------------------------------------------------------------
def _compute_cotta_loss(
    model, x: torch.Tensor, cfg, state: CoTTAState, device: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """CoTTA: consistency + diversity with teacher pseudo-labels.

    Returns (loss, sample_loss, diag, logits_orig).
    """
    # Pre-update student predictions on original x (weak aug = identity, for metrics)
    with torch.no_grad():
        logits_orig = model(x)

    # Pseudo-label from teacher
    with torch.no_grad():
        teacher_logits = state.teacher_forward(model, x)
        pseudo_labels = teacher_logits.argmax(dim=1).detach()

    # Strong augmentation
    x_strong = augment(x)

    # Student forward on strong augmentation (with grad)
    logits_strong = model(x_strong)
    probs_strong = F.softmax(logits_strong, dim=1)

    # Loss = CE(pseudo_label, softmax(logits_strong))
    #        + aug_strength * entropy(softmax(logits_strong))
    ce_loss = F.cross_entropy(logits_strong, pseudo_labels)
    ent_loss = -(probs_strong * safe_log(probs_strong, cfg.eps)).sum(dim=1).mean()
    aug_strength = getattr(cfg, "cotta_aug_strength", 0.5)
    loss = ce_loss + aug_strength * ent_loss

    # Per-sample losses (for logging)
    ce_per_sample = F.cross_entropy(logits_strong, pseudo_labels, reduction="none").detach()
    ent_per_sample = -(probs_strong * safe_log(probs_strong, cfg.eps)).sum(dim=1).detach()
    sample_loss = ce_per_sample + aug_strength * ent_per_sample

    # Diagnostics from original predictions
    diag = _build_diag(
        logits_orig, cfg,
        pseudo_labels=pseudo_labels,
        pl_loss_per_sample=ce_per_sample,
        objective_per_sample=sample_loss,
    )

    return loss, sample_loss, diag, logits_orig


# ---------------------------------------------------------------------------
# Loss dispatch
# ---------------------------------------------------------------------------
def compute_cotta_memo_loss(
    model, x: torch.Tensor, cfg, method: str, state, device: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Compute CoTTA or MEMO loss.

    The model forward happens INSIDE this function (MEMO does multiple forward
    passes with augmentations; CoTTA needs teacher forward). The returned
    logits are the pre-update student predictions on the original
    (non-augmented) x, used by the runner for pre-update prediction metrics.

    Returns:
        - loss: scalar loss for backprop.
        - sample_loss: per-sample objective loss (detached, for logging).
        - diag: dict of detached per-sample diagnostics.
        - logits: pre-update student predictions on original x (detached).
    """
    if method == "memo":
        return _compute_memo_loss(model, x, cfg, state, device)
    if method == "cotta":
        return _compute_cotta_loss(model, x, cfg, state, device)
    raise ValueError(f"Unknown method={method} for compute_cotta_memo_loss")
