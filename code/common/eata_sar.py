"""EATA and SAR test-time adaptation methods (comparison baselines for UCDPA).

EATA: entropy minimization on reliable samples (entropy < eps_e) + Fisher
anti-forgetting regularization + weight averaging (EMA of parameters).
  - eps_e = eata_eps_scale * ln(num_classes)  (default 0.4 * ln(1000) ~ 2.77)
  - Fisher EMA: fisher_i <- alpha_f * fisher_i + (1-alpha_f) * grad_i^2
  - Weight averaging: param_star_i <- alpha_w * param_star_i + (1-alpha_w) * param_i
  - L_fisher = lambda_fisher * sum_i fisher_i * (param_i - param_star_i)^2
  - Total loss = reliable_sample_loss + lambda_fisher * L_fisher

SAR: sharpness-aware minimization (SAM) with reliability sample selection.
  - eps_sar = sar_eps_scale * ln(num_classes)  (default 0.25 * ln(1000) ~ 1.73)
  - SAM two-step:
      1. Compute L1 on reliable samples, get g1
      2. Perturb: param_eps = param + rho * g1 / (||g1|| + eps)
      3. Compute L2 on perturbed params, get g2
      4. Revert params to original
      5. Update with g2: param -= lr * g2

Both methods update only LayerNorm affine parameters (configured by the runner)
and keep the model in eval mode. Reliability filtering uses detached entropy.
Labels are never used.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

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
from .ucdpa import MethodConfig, compute_mutual_information


class EATAState:
    """State for EATA: Fisher EMA and weight-averaging EMA.

    Attributes:
        fisher: dict mapping param index -> EMA of squared gradients (alpha_f decay)
        param_star: dict mapping param index -> EMA of param values (alpha_w decay)
    """

    def __init__(self):
        self.fisher: Dict[int, torch.Tensor] = {}
        self.param_star: Dict[int, torch.Tensor] = {}

    def init(self, params: List[torch.Tensor]):
        for i, p in enumerate(params):
            self.fisher[i] = torch.zeros_like(p.data)
            self.param_star[i] = p.data.clone().detach()

    def update_fisher(self, params: List[torch.Tensor], grads: List[torch.Tensor], alpha_f: float):
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(params, grads)):
                if g is None:
                    continue
                self.fisher[i].mul_(alpha_f).add_((1.0 - alpha_f) * g.detach().pow(2))

    def compute_fisher_loss(self, params: List[torch.Tensor]) -> torch.Tensor:
        loss = params[0].new_zeros(())
        for i, p in enumerate(params):
            w = self.fisher[i].detach()
            star = self.param_star[i].detach()
            loss = loss + (w * (p - star).pow(2)).sum()
        return loss

    def update_weight_star(self, params: List[torch.Tensor], alpha_w: float):
        with torch.no_grad():
            for i, p in enumerate(params):
                self.param_star[i].mul_(alpha_w).add_((1.0 - alpha_w) * p.data.detach())


class SARState:
    """State for SAR: SAM perturbation helper."""

    def __init__(self, rho: float = 0.05):
        self.rho = rho

    def sam_perturb(self, params: List[torch.Tensor], grads: List[torch.Tensor], rho: float = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        rho = rho if rho is not None else self.rho
        originals: List[torch.Tensor] = []
        with torch.no_grad():
            sq_sum = params[0].new_zeros(())
            for g in grads:
                if g is not None:
                    sq_sum = sq_sum + g.detach().pow(2).sum()
            g_norm = torch.sqrt(sq_sum).clamp_min(1e-12)
            for p, g in zip(params, grads):
                originals.append(p.data.clone())
                if g is not None:
                    p.data.add_(rho * g.detach() / g_norm)
        return originals, g_norm

    def sam_restore(self, params: List[torch.Tensor], originals: List[torch.Tensor]):
        with torch.no_grad():
            for p, o in zip(params, originals):
                p.data.copy_(o)


def compute_eata_sar_loss(
    logits: torch.Tensor,
    cfg: MethodConfig,
    method: str,
    state,
    params: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], int]:
    """Compute EATA/SAR loss on reliable samples.

    Returns:
        - total_loss: scalar loss for backprop (reliable sample loss + fisher
          reg for EATA). Zero tensor if no reliable samples.
        - sample_loss_full: per-sample objective loss for ALL samples (detached,
          for diagnostics).
        - diag_dict: full-batch diagnostics.
        - n_reliable: number of reliable samples surviving the entropy filter.
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

    num_classes = logits.shape[1]
    log_nc = float(np.log(num_classes))

    if method.startswith("eata"):
        eps_thresh = cfg.eata_eps_scale * log_nc
    else:  # sar
        eps_thresh = cfg.sar_eps_scale * log_nc

    reliable_mask = ent.detach() < eps_thresh
    n_reliable = int(reliable_mask.sum().item())

    diag: Dict[str, torch.Tensor] = {
        "pmax": pmax.detach(),
        "entropy": ent.detach(),
        "uncertainty_u": u.detach(),
        "mutual_info": mi,
        "ucb_score": torch.zeros_like(pmax),
        "come_loss": ent.detach(),
        "pl_loss": torch.zeros_like(pmax),
        "pseudo_label": probs.argmax(dim=1).detach(),
        "class_balance_weight": torch.ones_like(pmax),
        "L_cal": torch.zeros_like(pmax),
        "L_fpa": torch.zeros_like(pmax),
        "L_objective": ent.detach(),
    }

    if n_reliable == 0:
        total = logits.sum() * 0.0
        sample_loss_full = ent.detach()
        return total, sample_loss_full, diag, 0

    # Plain entropy objective on reliable samples
    sample_loss_full = ent  # has grad
    reliable_loss = sample_loss_full[reliable_mask].mean()
    diag["L_objective"] = ent.detach()

    total = reliable_loss

    # EATA: add Fisher anti-forgetting regularization
    if method.startswith("eata"):
        fisher_loss = state.compute_fisher_loss(params)
        total = total + cfg.eata_fisher_lambda * fisher_loss

    return total, sample_loss_full, diag, n_reliable
