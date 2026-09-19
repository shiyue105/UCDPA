"""COME / Tent losses and subjective-logic uncertainty utilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def softmax_entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = torch.clamp(probs, min=eps)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)
    return -(probs * safe_log(probs, eps)).sum(dim=1)


def softmax_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return softmax_entropy_from_probs(F.softmax(logits, dim=1), eps=eps)


def normalize_logits_for_come(
    logits: torch.Tensor,
    p_norm: float = 2.0,
    tau: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """COME-style detached logit-norm normalization.

    z = logits / ||logits||_p * stopgrad(||logits||_p) * tau.
    """
    if p_norm is None or p_norm <= 0:
        return logits * tau
    norm = torch.norm(logits, p=p_norm, dim=1, keepdim=True).clamp_min(eps)
    return logits / norm * norm.detach() * tau


def come_opinion(
    logits: torch.Tensor,
    p_norm: float = 2.0,
    tau: float = 1.0,
    evidence_mode: str = "exp_relu",
    eps: float = 1e-12,
):
    """Return belief, uncertainty, Dirichlet alpha, strength, and normalized logits.

    Default mode follows the COME-style evidence transform:
      alpha_k = exp(ReLU(z_k)), evidence_k = alpha_k - 1, S = sum alpha,
      belief_k = evidence_k / S, uncertainty = K / S.

    The returned uncertainty is the core diagnostic used by every method in this
    project, including Tent, so all ablations are compared with the same U metric.
    """
    z = normalize_logits_for_come(logits, p_norm=p_norm, tau=tau, eps=eps)
    k = z.shape[1]

    if evidence_mode == "exp_relu":
        alpha = torch.exp(F.relu(z)).clamp_min(1.0 + eps)
        evidence = alpha - 1.0
    elif evidence_mode == "softplus":
        evidence = F.softplus(z)
        alpha = evidence + 1.0
    elif evidence_mode == "exp_minus_one":
        evidence = torch.exp(torch.clamp(z, min=0.0)).clamp_min(1.0) - 1.0
        alpha = evidence + 1.0
    else:
        raise ValueError(f"Unknown evidence_mode={evidence_mode}")

    strength = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    belief = evidence / strength
    uncertainty = (float(k) / strength).squeeze(1)
    return belief, uncertainty, alpha, strength.squeeze(1), z


def come_entropy(
    logits: torch.Tensor,
    p_norm: float = 2.0,
    tau: float = 1.0,
    evidence_mode: str = "exp_relu",
    eps: float = 1e-12,
) -> torch.Tensor:
    belief, uncertainty, _, _, _ = come_opinion(
        logits, p_norm=p_norm, tau=tau, evidence_mode=evidence_mode, eps=eps
    )
    belief_entropy = -(belief * safe_log(belief, eps)).sum(dim=1)
    uncertainty_entropy = -(uncertainty * safe_log(uncertainty, eps))
    return belief_entropy + uncertainty_entropy


def tent_come_loss(
    logits: torch.Tensor,
    lambda_mix: float = 0.03,
    p_norm: float = 2.0,
    tau: float = 1.0,
    evidence_mode: str = "exp_relu",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-sample Tent-COME loss: (1-lambda)*Tent entropy + lambda*COME entropy."""
    tent = softmax_entropy(logits, eps=eps)
    come = come_entropy(logits, p_norm=p_norm, tau=tau, evidence_mode=evidence_mode, eps=eps)
    return (1.0 - lambda_mix) * tent + lambda_mix * come


def get_sample_loss(
    logits: torch.Tensor,
    base_loss: str = "tent_come",
    lambda_mix: float = 0.03,
    p_norm: float = 2.0,
    tau: float = 1.0,
    evidence_mode: str = "exp_relu",
    eps: float = 1e-12,
) -> torch.Tensor:
    if base_loss == "tent":
        return softmax_entropy(logits, eps=eps)
    if base_loss == "come":
        return come_entropy(logits, p_norm=p_norm, tau=tau, evidence_mode=evidence_mode, eps=eps)
    if base_loss == "tent_come":
        return tent_come_loss(
            logits,
            lambda_mix=lambda_mix,
            p_norm=p_norm,
            tau=tau,
            evidence_mode=evidence_mode,
            eps=eps,
        )
    raise ValueError(f"Unknown base_loss={base_loss}")
