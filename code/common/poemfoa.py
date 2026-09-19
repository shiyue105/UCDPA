"""POEM and FOA test-time adaptation methods (comparison baselines for UCDPA).

POEM (Explore unexplored reliable samples):
  - Entropy minimization with two-tier sample selection.
  - Reliable samples: entropy < threshold (threshold = mult * ln(C)).
  - Potentially reliable samples: threshold <= entropy < threshold + margin.
  - Loss = mean(entropy[selected]). Updates LayerNorm affine only.
  - EMA of the loss for degradation monitoring (no reset in this simplified
    port; the source repo resets when EMA < 0.2 which is a very specific
    DEYO artifact and is intentionally omitted for a faithful-but-fair
    ViT-B/16 comparison).

FOA (Forward-Optimization Adaptation):
  - Derivative-free adaptation via input-prompt optimization (CMA-ES-like).
  - Maintains a prompt distribution N(mean, std) over a low-resolution
    prompt tensor (3 x prompt_size x prompt_size), upsampled and added to x.
  - For each batch: sample N candidates, forward each, compute fitness =
    mean(entropy). Select the best candidate and EMA-update the distribution.
  - Prediction uses the current best prompt (pre-update, online protocol).
  - No backpropagation through model weights.

Both methods keep the model in eval mode. POEM updates LayerNorm affine
parameters (configured by the runner). FOA does NOT update model weights.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

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
from .cottamemo import _build_diag


# ---------------------------------------------------------------------------
# POEM state
# ---------------------------------------------------------------------------
class POEMState:
    """State container for POEM.

    Attributes:
        entropy_threshold: samples with entropy below this are 'reliable'.
            Default = 0.5 * ln(num_classes).
        margin: samples with entropy in [threshold, threshold + margin) are
            'potentially reliable' and also used for adaptation.
        ema_loss: EMA of the adaptation loss (for diagnostics / monitoring).
    """

    def __init__(
        self,
        entropy_threshold_mult: float = 0.5,
        margin: float = 0.5,
        num_classes: int = 1000,
        device: str = "cuda",
    ):
        self.entropy_threshold = entropy_threshold_mult * float(math.log(num_classes))
        self.margin = margin
        self.ema_loss: Optional[float] = None
        self.device = device

    def reset(self):
        self.ema_loss = None


# ---------------------------------------------------------------------------
# FOA state
# ---------------------------------------------------------------------------
class FOAState:
    """State container for FOA (derivative-free prompt optimization).

    Maintains a Gaussian distribution over an input prompt. The prompt is a
    tensor of shape [3, prompt_size, prompt_size]. If prompt_size < input
    resolution, it is bilinearly upsampled before being added to x.

    Attributes:
        prompt_size: spatial size of the prompt. Default 224 (full resolution,
            matching the task spec R^{1x3x224x224}). Smaller values (e.g. 8)
            are more tractable for CMA-ES-style search but less expressive.
        num_candidates: N candidate prompts sampled per batch.
        prompt_mean: mean of the prompt distribution [3*prompt_size^2].
        prompt_std: std of the prompt distribution (scalar, annealed).
        best_prompt: the best-scoring prompt found so far.
        best_loss: fitness of best_prompt.
        prompt_scale: scale factor applied to the prompt before adding to x
            (keeps the perturbation small relative to input magnitude).
        ema_decay: EMA decay for updating prompt_mean toward best candidate.
        std_decay: per-step multiplicative decay of prompt_std (exploration
            annealing).
    """

    def __init__(
        self,
        prompt_size: int = 224,
        num_candidates: int = 5,
        device: str = "cuda",
        prompt_scale: float = 0.01,
        ema_decay: float = 0.9,
        std_decay: float = 0.999,
        init_std: float = 0.01,
    ):
        self.prompt_size = prompt_size
        self.num_candidates = num_candidates
        self.prompt_dim = 3 * prompt_size * prompt_size
        self.prompt_scale = prompt_scale
        self.ema_decay = ema_decay
        self.std_decay = std_decay
        self.prompt_mean = torch.zeros(self.prompt_dim, device=device)
        self.prompt_std = init_std
        self.best_prompt = torch.zeros(self.prompt_dim, device=device)
        self.best_loss = float("inf")
        self.device = device

    def reset(self):
        self.prompt_mean.zero_()
        self.prompt_std = 0.01
        self.best_prompt.zero_()
        self.best_loss = float("inf")

    def reshape_prompt(self, prompt_flat: torch.Tensor, target_size: int = 224) -> torch.Tensor:
        """Reshape prompt to [1,3,P,P] and upsample to [1,3,target_size,target_size] if needed."""
        p = prompt_flat.view(1, 3, self.prompt_size, self.prompt_size)
        if self.prompt_size != target_size:
            p = F.interpolate(p, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return p


# ---------------------------------------------------------------------------
# POEM loss
# ---------------------------------------------------------------------------
def _compute_poem_loss(
    model, x: torch.Tensor, cfg, state: POEMState, device: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """POEM: entropy minimization on reliable + potentially-reliable samples.

    Returns (loss, sample_loss, diag, logits_orig).
    logits_orig is the pre-update forward (detached) used by the runner for
    prediction metrics. Uses a single forward pass (with grad) for both the
    loss and the pre-update predictions (no model update has happened yet).
    """
    # Single forward pass with grad (pre-update = this forward, since no
    # optimizer.step has been called yet for this batch).
    logits = model(x)
    logits_orig = logits.detach()

    entropys = softmax_entropy(logits)  # [B]

    threshold = state.entropy_threshold
    margin = state.margin

    # Two-tier sample selection (detached masks).
    ent_detached = entropys.detach()
    reliable_mask = ent_detached < threshold
    pot_reliable_mask = (ent_detached >= threshold) & (ent_detached < threshold + margin)
    selected_mask = reliable_mask | pot_reliable_mask
    n_selected = int(selected_mask.sum().item())

    diag = _build_diag(logits_orig, cfg, objective_per_sample=ent_detached)

    if n_selected == 0:
        # No sample passes the filter; return zero loss (no update).
        loss = logits.sum() * 0.0
        sample_loss = ent_detached
        return loss, sample_loss, diag, logits_orig

    # Entropy minimization on selected samples.
    selected_entropys = entropys[selected_mask]
    loss = selected_entropys.mean()

    # EMA of the loss (diagnostics).
    with torch.no_grad():
        lv = float(loss.item())
        if math.isfinite(lv):
            if state.ema_loss is None:
                state.ema_loss = lv
            else:
                state.ema_loss = 0.9 * state.ema_loss + 0.1 * lv

    sample_loss = ent_detached
    diag["L_objective"] = ent_detached
    return loss, sample_loss, diag, logits_orig


# ---------------------------------------------------------------------------
# FOA loss
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_prompt(model, x: torch.Tensor, prompt_flat: torch.Tensor,
                     state: FOAState) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward x + prompt and return (logits, mean_entropy)."""
    prompt_2d = state.reshape_prompt(prompt_flat, target_size=x.shape[-1])
    x_pert = x + state.prompt_scale * prompt_2d
    logits = model(x_pert)
    ent = softmax_entropy(logits)
    return logits, ent.mean()


def _compute_foa_loss(
    model, x: torch.Tensor, cfg, state: FOAState, device: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """FOA: derivative-free input-prompt optimization.

    Samples N candidate prompts, evaluates each via forward pass, selects the
    best (lowest mean entropy), and EMA-updates the distribution mean.

    Returns (loss, sample_loss, diag, logits_orig) where loss is a zero tensor
    (FOA does not backprop through model weights) and logits_orig is the
    pre-update prediction using the current best prompt.
    """
    # Pre-update prediction using current best prompt (online protocol).
    with torch.no_grad():
        logits_orig, _ = _evaluate_prompt(model, x, state.best_prompt, state)

    # Sample N candidates from N(prompt_mean, prompt_std).
    candidates = []
    for _ in range(state.num_candidates):
        noise = torch.randn_like(state.prompt_mean) * state.prompt_std
        candidates.append(state.prompt_mean + noise)
    # Always include the current best and the current mean as candidates.
    candidates.append(state.best_prompt.clone())
    candidates.append(state.prompt_mean.clone())

    # Evaluate all candidates.
    best_loss = state.best_loss
    best_cand = None
    for cand in candidates:
        _, ent_mean = _evaluate_prompt(model, x, cand, state)
        lv = float(ent_mean.item())
        if math.isfinite(lv) and lv < best_loss:
            best_loss = lv
            best_cand = cand

    # Update prompt distribution.
    if best_cand is not None:
        state.best_prompt = best_cand.clone()
        state.prompt_mean = state.ema_decay * state.prompt_mean + (1.0 - state.ema_decay) * best_cand
        state.best_loss = best_loss
    # Anneal exploration std.
    state.prompt_std *= state.std_decay

    # Build diagnostics from the pre-update prediction.
    diag = _build_diag(logits_orig, cfg)
    sample_loss = diag["L_objective"]

    # Zero loss: FOA is derivative-free; runner must skip backward/step for FOA.
    loss = logits_orig.sum() * 0.0
    return loss, sample_loss, diag, logits_orig


# ---------------------------------------------------------------------------
# Loss dispatch
# ---------------------------------------------------------------------------
def compute_poem_foa_loss(
    model, x: torch.Tensor, cfg, method: str, state, device: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Compute POEM or FOA loss.

    The model forward happens INSIDE this function. POEM does a single
    forward pass (with grad) for entropy minimization. FOA does N+2
    forward passes (no grad) for derivative-free prompt evaluation.

    Returns:
        - loss: scalar loss for backprop (POEM). Zero tensor for FOA
          (derivative-free; runner skips backward/step).
        - sample_loss: per-sample objective loss (detached, for logging).
        - diag: dict of detached per-sample diagnostics.
        - logits: pre-update student predictions on original x (detached).
    """
    if method == "poem":
        return _compute_poem_loss(model, x, cfg, state, device)
    if method == "foa":
        return _compute_foa_loss(model, x, cfg, state, device)
    raise ValueError(f"Unknown method={method} for compute_poem_foa_loss")
