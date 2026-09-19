"""UCDPA: Uncertainty-Calibrated Dual-Path Adaptation for Test-Time Adaptation.

Implements the proposed UCDPA method on top of COME conservative entropy:

  1. UCB (Uncertainty-Confidence Balance) score: a per-sample balance score b_i
     in [0,1] that routes each sample between the conservative-entropy path
     (uncertain/low-confidence samples) and the pseudo-label cross-entropy path
     (confident/certain samples).
  2. Dual-path loss: L = mean( (1-b_i) * w_cb_i * H_come_i + b_i * w_cb_i * CE_i )
     where w_cb is the class-balanced inverse-frequency weight and CE_i is the
     pseudo-label cross-entropy.
  3. Calibration regularizer: L_cal = | mean(confidence) - mean(1-uncertainty) |.
  4. Feature prototype alignment (optional): contrastive pull of features toward
     their pseudo-label class prototype (EMA bank).
  5. Class-balanced reweighting via a running EMA of pseudo-label class counts.

Also provides:
  - UCDPAConfig dataclass with all UCDPA hyperparameters.
  - A unified MethodConfig for the 7 comparison methods (source, tent, eata,
    sar, come, tentcome, ucdpa).
  - compute_method_loss_and_diag dispatch for the non-EATA/SAR methods.
  - compute_mutual_information for the Dirichlet-based MI diagnostic.

All balance scores, class-balance weights, and reliability diagnostics are
detached (no gradients through the routing/weighting). The come_loss and
pseudo-label CE carry gradients through the logits.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .come_utils import (
    softmax_entropy,
    softmax_entropy_from_probs,
    come_opinion,
    come_entropy,
    safe_log,
)


# ---------------------------------------------------------------------------
# Method registry (7 comparison methods)
# ---------------------------------------------------------------------------
METHODS = [
    "source",
    "tent",
    "eata",
    "sar",
    "come",
    "tentcome",
    "ucdpa",
    "cotta",
    "memo",
    "poem",
    "foa",
]

METHOD_DISPLAY = {
    "source": "Source-only",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "come": "COME",
    "tentcome": "Tent-COME",
    "ucdpa": "UCDPA",
    "cotta": "CoTTA",
    "memo": "MEMO",
    "poem": "POEM",
    "foa": "FOA",
}

# Methods that perform online parameter updates.
# FOA is derivative-free (adapts an input prompt, not model weights) but is
# still an adapting method in the sense that it changes predictions online.
ADAPTING_METHODS = {"tent", "eata", "sar", "come", "tentcome", "ucdpa", "cotta", "memo", "poem", "foa"}

# Default learning rates per method (Adam optimizer for ViT LayerNorm affine).
# FOA does not use the optimizer (derivative-free), but a value is kept for registry consistency.
DEFAULT_LR = {
    "source": 0.0,
    "tent": 1e-4,
    "eata": 1e-4,
    "sar": 1e-4,
    "come": 1e-4,
    "tentcome": 1e-4,
    "ucdpa": 2e-4,
    "cotta": 1e-4,
    "memo": 1e-4,
    "poem": 2e-4,
    "foa": 1e-3,
}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
@dataclass
class UCDPAConfig:
    """Hyperparameters for the UCDPA method."""
    method: str = "ucdpa"
    lr: float = 2e-4
    tau_c: float = 0.7        # confidence threshold
    tau_u: float = 0.3        # uncertainty threshold
    T_c: float = 0.1          # confidence temperature
    T_u: float = 0.1          # uncertainty temperature
    lambda_cal: float = 0.02  # calibration regularizer weight
    lambda_fpa: float = 0.0   # feature prototype alignment weight (disabled)
    lambda_come: float = 0.03 # COME loss weight (for tentcome mixing)
    come_p_norm: int = 2
    come_tau: float = 1.0
    come_evidence_mode: str = "exp_relu"
    class_balance_ema: float = 0.99   # EMA for class counts
    prototype_momentum: float = 0.9   # prototype EMA
    prototype_temperature: float = 0.1
    eps: float = 1e-12

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MethodConfig:
    """Unified config for all 7 comparison methods.

    The UCDPA-specific fields are only used when method == 'ucdpa'. The
    come_* fields are shared by come/tentcome/ucdpa. The eata_*/sar_* fields
    are used by eata/sar.
    """
    method: str = "tent"
    # Optimizer
    lr: float = 1e-4
    optimizer: str = "adam"
    weight_decay: float = 0.0
    # COME shared
    come_p_norm: float = 2.0
    come_tau: float = 1.0
    come_evidence_mode: str = "exp_relu"
    lambda_come: float = 0.03   # tentcome mixing weight
    eps: float = 1e-12
    # UCDPA
    tau_c: float = 0.7
    tau_u: float = 0.3
    T_c: float = 0.1
    T_u: float = 0.1
    lambda_cal: float = 0.02
    lambda_fpa: float = 0.0
    class_balance_ema: float = 0.99
    # Label-shift safeguards (W2: prevents pathological w_cb inflation under
    # extreme label shift). When a class is absent for an extended period,
    # its EMA count decays toward zero and would inflate w_cb unboundedly.
    class_balance_count_floor: float = 0.01  # EMA floor on per-class counts
    class_balance_w_clip_min: float = 0.5    # lower clip on w_cb
    class_balance_w_clip_max: float = 2.0   # upper clip on w_cb
    prototype_momentum: float = 0.9
    prototype_temperature: float = 0.1
    # EATA
    eata_eps_scale: float = 0.4
    eata_fisher_lambda: float = 1.0
    eata_alpha_f: float = 0.9
    eata_alpha_w: float = 0.9
    # SAR
    sar_eps_scale: float = 0.25
    sar_rho: float = 0.05
    # CoTTA
    cotta_teacher_momentum: float = 0.999
    cotta_restore_prob: float = 0.01
    cotta_aug_strength: float = 0.5
    # MEMO
    memo_num_augmentations: int = 4
    memo_aug_scale: float = 0.3
    # POEM
    poem_entropy_threshold: float = 0.5   # multiplier for ln(C): threshold = 0.5 * ln(1000)
    poem_margin: float = 0.5              # margin above threshold for potentially-reliable samples
    # FOA
    foa_num_candidates: int = 5           # N candidate prompts sampled per batch
    foa_prompt_size: int = 224            # NOT used directly; prompt is 3 x prompt_size x prompt_size
    # Ablation flags
    disable_ucb: bool = False            # ablation: fixed b=0.5 routing
    disable_class_balance: bool = False  # ablation: w_cb=1.0 for all
    disable_scale_norm: bool = False     # ablation: skip pl_loss scale normalization

    def as_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# UCB (Uncertainty-Confidence Balance) score
# ---------------------------------------------------------------------------
def compute_ucb_score(
    confidence: torch.Tensor,
    uncertainty: torch.Tensor,
    tau_c: float = 0.7,
    tau_u: float = 0.3,
    T_c: float = 0.1,
    T_u: float = 0.1,
) -> torch.Tensor:
    """Compute the per-sample UCB balance score b_i in [0,1].

    b_i -> 1: sample is confident AND certain  -> use pseudo-label path.
    b_i -> 0: sample is uncertain OR low-conf  -> use conservative entropy path.

    Args:
        confidence: [B] max softmax probability (detached).
        uncertainty: [B] Dirichlet uncertainty u = K / S (detached).
        tau_c: confidence threshold.
        tau_u: uncertainty threshold.
        T_c: confidence temperature (sharpness of the sigmoid).
        T_u: uncertainty temperature.

    Returns:
        b: [B] balance score in [0,1].
    """
    b = torch.sigmoid((confidence - tau_c) / T_c) * torch.sigmoid((1.0 - uncertainty - tau_u) / T_u)
    return b


# ---------------------------------------------------------------------------
# Mutual information (Dirichlet-based diagnostic)
# ---------------------------------------------------------------------------
def compute_mutual_information(
    logits: torch.Tensor,
    p_norm: float = 2.0,
    tau: float = 1.0,
    evidence_mode: str = "exp_relu",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-sample mutual information for the Dirichlet opinion.

    MI = H(E[p]) - E[H(p)] = H(conservative_prob) - H(belief), where
    conservative_prob_k = belief_k + u/K (the Dirichlet mean) and
    H(belief) is the belief entropy.

    Higher MI => lower epistemic uncertainty (model is more confident in its
    belief structure). MI is small when the Dirichlet is flat (high u).

    Returns:
        mi: [B] per-sample mutual information (detached).
    """
    belief, uncertainty, alpha, strength, z = come_opinion(
        logits, p_norm=p_norm, tau=tau, evidence_mode=evidence_mode, eps=eps
    )
    k = float(logits.shape[1])
    # Conservative / mean predictive probability: b_k + u/K.
    u = uncertainty.unsqueeze(1)  # [B,1]
    conservative_prob = belief + u / k  # [B,K]
    conservative_prob = conservative_prob / conservative_prob.sum(dim=1, keepdim=True).clamp_min(eps)
    # H(E[p]) = H(conservative_prob)
    h_ep = -(conservative_prob * safe_log(conservative_prob, eps)).sum(dim=1)
    # E[H(p)] ~= H(belief)
    h_belief = -(belief * safe_log(belief, eps)).sum(dim=1)
    mi = (h_ep - h_belief).detach()
    return mi


# ---------------------------------------------------------------------------
# Class-balance tracker (running EMA of pseudo-label class counts)
# ---------------------------------------------------------------------------
class ClassBalanceTracker:
    """Running EMA of per-class pseudo-label counts.

    The weight for class c is w_cb(c) = 1 / (count[c] + 1), normalized so the
    mean weight is 1.0. This down-weights over-predicted classes and up-weights
    rare classes, preventing class collapse during self-training.

    Safeguards (W2: label-shift robustness):
      - `count_floor`: per-class EMA count is clamped to >= count_floor,
        preventing unbounded w_cb inflation when a class is absent from
        the stream for an extended period.
      - `w_clip_min`, `w_clip_max`: clip the final per-sample weight to
        `[w_clip_min, w_clip_max]` so no single class can dominate the loss.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        ema: float = 0.99,
        device: str = "cuda",
        count_floor: float = 0.01,
        w_clip_min: float = 0.5,
        w_clip_max: float = 2.0,
    ):
        self.num_classes = num_classes
        self.ema = ema
        self.count_floor = count_floor
        self.w_clip_min = w_clip_min
        self.w_clip_max = w_clip_max
        # Initialize to ones so early weights are ~uniform and division is safe.
        self.counts = torch.ones(num_classes, device=device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, pseudo_labels: torch.Tensor):
        """Update EMA class counts from a batch of pseudo-labels.

        count_c <- ema * count_c + (1-ema) * batch_count_c
        Then clamp to >= count_floor (label-shift safeguard).
        """
        batch_counts = torch.zeros_like(self.counts)
        batch_counts.scatter_add_(
            0, pseudo_labels.detach().long(), torch.ones_like(pseudo_labels, dtype=torch.float32)
        )
        self.counts.mul_(self.ema).add_((1.0 - self.ema) * batch_counts)
        # Safeguard: floor counts so absent classes don't inflate w_cb.
        self.counts.clamp_(min=self.count_floor)

    @torch.no_grad()
    def weights(self, pseudo_labels: torch.Tensor) -> torch.Tensor:
        """Return normalized inverse-frequency weights for the given labels.

        Applies two safeguards:
          1. EMA count floor (in `update`) — prevents division-by-near-zero.
          2. w_cb clip to [w_clip_min, w_clip_max] — bounds per-sample weight.
        """
        w = 1.0 / (self.counts.detach() + 1.0)
        w = w / w.mean().clamp_min(1e-12)
        # Safeguard: clip per-class weight to prevent any single class from
        # dominating the loss under extreme label shift.
        w = w.clamp(min=self.w_clip_min, max=self.w_clip_max)
        return w[pseudo_labels.detach().long()]


# ---------------------------------------------------------------------------
# Prototype bank (EMA feature prototypes for prototype alignment)
# ---------------------------------------------------------------------------
class PrototypeBank:
    """EMA bank of per-class feature prototypes.

    prototypes[c] = momentum * prototypes[c] + (1-momentum) * mean(features[label==c]).
    Initialized to zeros; updated only for classes present in the batch.
    """

    def __init__(self, num_classes: int = 1000, feat_dim: int = 768, momentum: float = 0.9, device: str = "cuda"):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device, dtype=torch.float32)
        self.initialized = torch.zeros(num_classes, device=device, dtype=torch.bool)

    @torch.no_grad()
    def update(self, features: torch.Tensor, pseudo_labels: torch.Tensor):
        """Update prototypes with EMA of per-class mean features."""
        features = features.detach()
        labels = pseudo_labels.detach().long()
        for c in labels.unique():
            mask = labels == c
            if mask.sum() == 0:
                continue
            mean_feat = features[mask].mean(dim=0)
            ci = int(c.item())
            if not bool(self.initialized[ci].item()):
                self.prototypes[ci] = mean_feat
                self.initialized[ci] = True
            else:
                self.prototypes[ci].mul_(self.momentum).add_((1.0 - self.momentum) * mean_feat)


def prototype_contrastive_loss(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Contrastive loss pulling features toward their pseudo-label prototype.

    Uses cosine similarity between features and prototypes as logits, then
    cross-entropy with the pseudo-labels. Prototypes are detached.

    Args:
        features: [B, D] feature embeddings (with grad).
        pseudo_labels: [B] detached class indices.
        prototypes: [K, D] prototype bank (detached).
        temperature: softmax temperature.

    Returns:
        Scalar loss (with grad through features).
    """
    features = F.normalize(features, dim=1)
    proto = F.normalize(prototypes.detach(), dim=1)
    sim = features @ proto.t() / max(temperature, eps)  # [B, K]
    return F.cross_entropy(sim, pseudo_labels.detach().long())


# ---------------------------------------------------------------------------
# UCDPA dual-path loss
# ---------------------------------------------------------------------------
def compute_ucdpa_loss_and_diag(
    logits: torch.Tensor,
    cfg: MethodConfig,
    state: "UCDPAState",
    features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the UCDPA dual-path loss and per-sample diagnostics.

    Returns:
        - total_loss: scalar loss for backprop.
        - sample_loss: per-sample objective loss (detached, for logging).
        - diag: dict of detached per-sample diagnostics.
    """
    probs = F.softmax(logits, dim=1)
    pmax = probs.max(dim=1).values
    ent = softmax_entropy_from_probs(probs, eps=cfg.eps)

    # COME opinion (belief, uncertainty, alpha, strength)
    belief, uncertainty, alpha, strength, _ = come_opinion(
        logits, p_norm=cfg.come_p_norm, tau=cfg.come_tau,
        evidence_mode=cfg.come_evidence_mode, eps=cfg.eps,
    )
    k = float(logits.shape[1])

    # Path 1: COME conservative entropy (per-sample, with grad)
    come_loss = come_entropy(
        logits, p_norm=cfg.come_p_norm, tau=cfg.come_tau,
        evidence_mode=cfg.come_evidence_mode, eps=cfg.eps,
    )

    # Path 2: pseudo-label cross-entropy (per-sample, with grad through logits)
    pseudo_labels = probs.argmax(dim=1).detach()
    pl_loss = F.cross_entropy(logits, pseudo_labels, reduction="none")

    # UCB balance score (detached)
    confidence_detached = pmax.detach()
    uncertainty_detached = uncertainty.detach()
    if cfg.disable_ucb:
        # Ablation: fixed 0.5 routing (no adaptive UCB)
        b = torch.full_like(pmax, 0.5)
    else:
        b = compute_ucb_score(
            confidence_detached, uncertainty_detached,
            tau_c=cfg.tau_c, tau_u=cfg.tau_u, T_c=cfg.T_c, T_u=cfg.T_u,
        ).detach()

    # Class-balanced reweighting (detached)
    if cfg.disable_class_balance:
        w_cb = torch.ones_like(pmax)
    else:
        state.class_balance.update(pseudo_labels)
        w_cb = state.class_balance.weights(pseudo_labels).detach()

    # Scale-normalized dual-path loss: normalize pl_loss to the same scale as
    # come_loss so that the pseudo-label path contributes a comparable gradient
    # magnitude. Without this, come_loss (entropy, ~1.4) dominates pl_loss
    # (-log pmax, ~0.27) and UCDPA degenerates to COME on accuracy.
    come_mean = come_loss.detach().mean().clamp_min(cfg.eps)
    pl_mean = pl_loss.detach().mean().clamp_min(cfg.eps)
    pl_scale = (come_mean / pl_mean).detach()
    if cfg.disable_scale_norm:
        pl_loss_scaled = pl_loss
    else:
        pl_loss_scaled = pl_loss * pl_scale

    # Combined dual-path loss (with grad through come_loss and pl_loss_scaled)
    total_path = (1.0 - b) * w_cb * come_loss + b * w_cb * pl_loss_scaled
    total = total_path.mean()

    # Calibration regularizer: | mean(confidence) - mean(1-uncertainty) |
    # confidence and uncertainty carry grad through logits (regularizer).
    L_cal = (pmax.mean() - (1.0 - uncertainty).mean()).abs()
    total = total + cfg.lambda_cal * L_cal

    # Feature prototype alignment (optional)
    l_fpa_val = torch.tensor(0.0, device=logits.device)
    if features is not None and cfg.lambda_fpa > 0.0:
        # Update prototype bank (detached)
        state.prototypes.update(features, pseudo_labels)
        # Compute contrastive loss only over classes with initialized prototypes.
        init_mask = state.prototypes.initialized
        if init_mask.any():
            # Restrict logits to initialized classes and remap labels.
            valid_idx = torch.where(init_mask)[0]
            valid_set = set(valid_idx.tolist())
            sample_valid = torch.tensor(
                [int(l.item()) in valid_set for l in pseudo_labels],
                device=logits.device, dtype=torch.bool,
            )
            if sample_valid.any():
                feats_sel = features[sample_valid]
                labels_sel = pseudo_labels[sample_valid]
                proto_sel = state.prototypes.prototypes[valid_idx]
                # Remap labels to the valid_idx ordering for cross_entropy.
                remap = torch.full((state.num_classes,), -1, device=logits.device, dtype=torch.long)
                remap[valid_idx] = torch.arange(len(valid_idx), device=logits.device)
                labels_remapped = remap[labels_sel]
                l_fpa = prototype_contrastive_loss(
                    feats_sel, labels_remapped, proto_sel,
                    temperature=cfg.prototype_temperature, eps=cfg.eps,
                )
                total = total + cfg.lambda_fpa * l_fpa
                l_fpa_val = l_fpa.detach()

    # Mutual information (diagnostic)
    mi = compute_mutual_information(
        logits, p_norm=cfg.come_p_norm, tau=cfg.come_tau,
        evidence_mode=cfg.come_evidence_mode, eps=cfg.eps,
    )

    diag: Dict[str, torch.Tensor] = {
        "pmax": pmax.detach(),
        "entropy": ent.detach(),
        "uncertainty_u": uncertainty.detach(),
        "mutual_info": mi,
        "ucb_score": b,
        "come_loss": come_loss.detach(),
        "pl_loss": pl_loss_scaled.detach(),
        "pseudo_label": pseudo_labels,
        "class_balance_weight": w_cb,
        "L_cal": L_cal.detach().expand_as(pmax),
        "L_fpa": l_fpa_val.expand_as(pmax),
        "L_objective": total_path.detach(),
    }
    # Per-sample objective loss for logging
    sample_loss = ((1.0 - b) * come_loss + b * pl_loss_scaled).detach()

    return total, sample_loss, diag


class UCDPAState:
    """State container for UCDPA: class-balance tracker + prototype bank."""

    def __init__(self, num_classes: int = 1000, feat_dim: int = 768, cfg: Optional[MethodConfig] = None, device: str = "cuda"):
        cfg = cfg or MethodConfig(method="ucdpa")
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.class_balance = ClassBalanceTracker(
            num_classes=num_classes,
            ema=cfg.class_balance_ema,
            device=device,
            count_floor=cfg.class_balance_count_floor,
            w_clip_min=cfg.class_balance_w_clip_min,
            w_clip_max=cfg.class_balance_w_clip_max,
        )
        self.prototypes = PrototypeBank(
            num_classes=num_classes, feat_dim=feat_dim, momentum=cfg.prototype_momentum, device=device,
        )

    def reset(self):
        self.class_balance.counts.fill_(1.0)
        self.prototypes.prototypes.zero_()
        self.prototypes.initialized.fill_(False)


# ---------------------------------------------------------------------------
# Loss dispatch for the non-EATA/SAR methods (source, tent, come, tentcome, ucdpa)
# ---------------------------------------------------------------------------
def compute_method_loss_and_diag(
    logits: torch.Tensor,
    cfg: MethodConfig,
    state: Optional[UCDPAState] = None,
    features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Return (total_loss_for_backprop, per_sample_objective_loss, diag).

    For source (non-adapting), total_loss is a zero tensor. For ucdpa, the
    full dual-path loss is computed. For tent/come/tentcome, the corresponding
    per-sample entropy loss is mean-reduced.
    """
    method = cfg.method
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
    # Default UCB score = 0 (non-UCDPA methods do not route).
    b_zero = torch.zeros_like(pmax)

    diag: Dict[str, torch.Tensor] = {
        "pmax": pmax.detach(),
        "entropy": ent.detach(),
        "uncertainty_u": u.detach(),
        "mutual_info": mi,
        "ucb_score": b_zero,
        "come_loss": ent.detach(),
        "pl_loss": torch.zeros_like(pmax),
        "pseudo_label": probs.argmax(dim=1).detach(),
        "class_balance_weight": torch.ones_like(pmax),
        "L_cal": torch.zeros_like(pmax),
        "L_fpa": torch.zeros_like(pmax),
        "L_objective": ent.detach(),
    }

    if method == "source":
        total = logits.sum() * 0.0
        sample_loss = ent.detach()
        return total, sample_loss, diag

    if method == "tent":
        sample_loss = ent
        total = sample_loss.mean()
        diag["L_objective"] = sample_loss.detach()
        return total, sample_loss, diag

    if method == "come":
        come = come_entropy(
            logits, p_norm=cfg.come_p_norm, tau=cfg.come_tau,
            evidence_mode=cfg.come_evidence_mode, eps=cfg.eps,
        )
        sample_loss = come
        total = sample_loss.mean()
        diag["come_loss"] = come.detach()
        diag["L_objective"] = come.detach()
        return total, sample_loss, diag

    if method == "tentcome":
        come = come_entropy(
            logits, p_norm=cfg.come_p_norm, tau=cfg.come_tau,
            evidence_mode=cfg.come_evidence_mode, eps=cfg.eps,
        )
        sample_loss = (1.0 - cfg.lambda_come) * ent + cfg.lambda_come * come
        total = sample_loss.mean()
        diag["come_loss"] = come.detach()
        diag["L_objective"] = sample_loss.detach()
        return total, sample_loss, diag

    if method == "ucdpa":
        if state is None:
            state = UCDPAState(num_classes=logits.shape[1], feat_dim=768, cfg=cfg, device=str(logits.device))
        return compute_ucdpa_loss_and_diag(logits, cfg, state, features=features)

    raise ValueError(f"Unknown method={method} for compute_method_loss_and_diag")
