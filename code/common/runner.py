"""Online TTA runner for UCDPA experiments on ImageNet-C ViT-B/16.

Protocol: evaluate (predict) on each batch with the current model, then perform
one online update using the same unlabeled batch. Per-sample diagnostics and the
full metric suite are computed from the pre-update predictions.

Adaptation: LayerNorm affine parameters only (ViT-B/16). Optimizer: Adam.
Methods: source, tent, eata, sar, come, tentcome, ucdpa.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .ucdpa import (
    MethodConfig,
    UCDPAState,
    METHODS,
    METHOD_DISPLAY,
    ADAPTING_METHODS,
    DEFAULT_LR,
    compute_method_loss_and_diag,
)
from .eata_sar import EATAState, SARState, compute_eata_sar_loss
from .come_utils import come_opinion, softmax_entropy
from .data_utils import make_loader, discover_imagenet_c_combos, load_imagenet_class_mapping
from .model_utils import (
    load_local_pretrained_model,
    save_source_checkpoint,
    load_source_checkpoint,
    configure_model_for_tta,
    make_optimizer,
    quick_forward_check,
)
from .metrics import compute_all_metrics


PROJECT_ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
SOURCE_CKPT = PROJECT_ROOT / "checkpoints" / "vit_base_patch16_224_source.pth"
RESNET_SOURCE_CKPT = PROJECT_ROOT / "checkpoints" / "resnet50_source.pth"
CIFAR_RESNET_SOURCE_CKPT = PROJECT_ROOT / "checkpoints" / "cifar_resnet18_source.pth"
VIT_LOCAL_DIR = "/data/share/models/timm/vit_base_patch16_224"
VIT_FEAT_DIM = 768
RESNET_FEAT_DIM = 2048
CIFAR_RESNET_FEAT_DIM = 512

# Backbone registry: name -> (source_ckpt, feat_dim, num_classes, dataset_name)
BACKBONES = {
    "vit_base_patch16_224": (SOURCE_CKPT, VIT_FEAT_DIM, 1000, "imagenet-c"),
    "resnet50": (RESNET_SOURCE_CKPT, RESNET_FEAT_DIM, 1000, "imagenet-c"),
    "cifar_resnet18": (CIFAR_RESNET_SOURCE_CKPT, CIFAR_RESNET_FEAT_DIM, 100, "cifar100-c"),
}


def set_seed(seed: int):
    """Set all random seeds and enable deterministic cuDNN for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_source_checkpoint(device: str, backbone: str = "vit_base_patch16_224") -> str:
    """Ensure a clean source checkpoint exists; create it from local pretrained weights if missing."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone={backbone}. Supported: {list(BACKBONES.keys())}")
    ckpt_path, _feat_dim, _nc, _ds = BACKBONES[backbone]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists():
        return str(ckpt_path)
    model, local_weight = load_local_pretrained_model(
        backbone, device=device,
        vit_local_dir=VIT_LOCAL_DIR,
        vit_model_name=backbone,
    )
    quick_forward_check(model, device=device, backbone=backbone)
    save_source_checkpoint(
        model, str(ckpt_path), backbone,
        meta={"local_weight": local_weight, "created_by": "ucdpa_runner"},
        overwrite=False, num_classes=_nc,
    )
    del model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(ckpt_path)


def prepare_model(device: str, backbone: str = "vit_base_patch16_224"):
    """Load source checkpoint and configure trainable affine params for TTA."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone={backbone}. Supported: {list(BACKBONES.keys())}")
    ckpt_path, _feat_dim, _nc, _ds = BACKBONES[backbone]
    model = load_source_checkpoint(backbone, str(ckpt_path), device=device, vit_model_name=backbone)
    quick_forward_check(model, device=device, backbone=backbone)
    params, trainable_count = configure_model_for_tta(model, backbone)
    # ViT: eval mode (dropout off), LayerNorm affine trainable.
    # ResNet/CIFAR-ResNet: train mode (BN uses batch stats), BN affine trainable.
    if backbone in ("resnet50", "cifar_resnet18"):
        model.train()
    else:
        model.eval()
    return model, params, trainable_count


def extract_features(model, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Extract pooled features [B, D] from a timm ViT model.

    Returns None if the model does not expose forward_features / forward_head
    (in which case prototype alignment is skipped).
    """
    try:
        feats = model.forward_features(x)
        pooled = model.forward_head(feats, pre_logits=True)
        return pooled
    except Exception:
        return None


def _save_samples(path: Path, rows: List[Dict]):
    """Save per-sample rows to parquet (zstd) with a gzipped CSV fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        clean = []
        keys = set()
        for r in rows:
            keys.update(r.keys())
        for r in rows:
            rr = {}
            for k in keys:
                v = r.get(k, None)
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                elif v is None:
                    v = float("nan")
                rr[k] = v
            clean.append(rr)
        table = pa.Table.from_pylist(clean)
        pq.write_table(table, str(path), compression="zstd")
    except Exception:
        import csv
        fieldnames: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(str(path) + ".gz", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run_one_combo(
    cfg: MethodConfig,
    seed: int,
    corruption: str,
    severity: int,
    batch_size: int,
    data_root: str,
    output_dir: Path,
    max_batches: int = -1,
    num_workers: int = 8,
    device: str = "cuda",
    imagenet_mapping: Optional[Dict] = None,
    config_id: str = "",
    save_samples: bool = True,
    backbone: str = "vit_base_patch16_224",
) -> Dict:
    """Run one (method, seed, corruption, severity) online TTA stream.

    Returns the per-run summary dict. Also writes:
      - results/summaries/<fname>.csv   (summary metrics)
      - results/summaries/<fname>.completed.flag
      - results/samples/<fname>.parquet (per-sample diagnostics)
      - results/summaries/<fname>_config.json
    """
    set_seed(seed)
    method = cfg.method
    method_display = METHOD_DISPLAY.get(method, method)
    adapting = method in ADAPTING_METHODS

    model, params, trainable_count = prepare_model(device, backbone=backbone)
    # Adam optimizer with per-method learning rate.
    lr = cfg.lr if cfg.lr > 0 else DEFAULT_LR.get(method, 1e-4)
    optimizer = make_optimizer(params, lr=lr, optimizer=cfg.optimizer, weight_decay=cfg.weight_decay) if adapting else None

    # EATA / SAR state initialization
    eata_state = None
    sar_state = None
    use_eata_sar = method in ("eata", "sar")
    if use_eata_sar:
        if method == "eata":
            eata_state = EATAState()
            eata_state.init(params)
        else:
            sar_state = SARState(rho=cfg.sar_rho)

    # UCDPA state
    ucdpa_state = None
    if method == "ucdpa":
        _feat_dim, _num_classes = BACKBONES[backbone][1], BACKBONES[backbone][2]
        ucdpa_state = UCDPAState(num_classes=_num_classes, feat_dim=_feat_dim, cfg=cfg, device=device)

    # CoTTA/MEMO state
    cotta_memo_state = None
    if method in ("cotta", "memo"):
        from .cottamemo import CoTTAState, MEMOState
        source_ckpt_path, _feat_dim, _num_classes, _ds = BACKBONES[backbone]
        if method == "cotta":
            cotta_memo_state = CoTTAState(
                model=model, params=params, source_ckpt_path=str(source_ckpt_path),
                teacher_momentum=cfg.cotta_teacher_momentum,
                restore_prob=cfg.cotta_restore_prob,
                device=device,
            )
        else:
            cotta_memo_state = MEMOState(num_augmentations=cfg.memo_num_augmentations)

    # POEM/FOA state
    poemfoa_state = None
    if method in ("poem", "foa"):
        from .poemfoa import POEMState, FOAState
        _feat_dim, _num_classes = BACKBONES[backbone][1], BACKBONES[backbone][2]
        if method == "poem":
            poemfoa_state = POEMState(
                entropy_threshold_mult=cfg.poem_entropy_threshold,
                margin=cfg.poem_margin,
                num_classes=_num_classes,
                device=device,
            )
        else:  # foa
            poemfoa_state = FOAState(
                prompt_size=cfg.foa_prompt_size,
                num_candidates=cfg.foa_num_candidates,
                device=device,
            )

    # Select dataset and transform based on backbone.
    _dataset_name = BACKBONES[backbone][3]
    if _dataset_name == "cifar100-c":
        from .data_utils import cifar100_transform
        loader = make_loader(
            data_root, "cifar100-c", batch_size=batch_size, num_workers=num_workers,
            corruption=corruption, severity=severity, imagenet_mapping=None,
            shuffle=False, max_samples=None, transform_override=cifar100_transform(),
        )
    else:
        loader = make_loader(
            data_root, "imagenet-c", batch_size=batch_size, num_workers=num_workers,
            corruption=corruption, severity=severity, imagenet_mapping=imagenet_mapping,
            shuffle=False, max_samples=None,
        )

    sample_rows: List[Dict] = []
    start = time.time()
    optimizer_steps = 0
    peak_mem = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x, y, _is_ood, _paths = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.eval()
        features = None
        if use_eata_sar:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            state = eata_state if eata_state is not None else sar_state
            loss, sample_loss, diag, n_reliable = compute_eata_sar_loss(
                logits, cfg, method, state, params
            )
            if n_reliable > 0 and torch.isfinite(loss):
                if eata_state is not None:
                    # EATA: backward -> update fisher -> step -> update weight star
                    loss.backward()
                    grads = [p.grad for p in params]
                    eata_state.update_fisher(params, grads, cfg.eata_alpha_f)
                    optimizer.step()
                    optimizer_steps += 1
                    eata_state.update_weight_star(params, cfg.eata_alpha_w)
                else:
                    # SAR: SAM two-step (perturb, second forward-backward, revert, step)
                    loss.backward()
                    grads1 = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                    originals, _ = sar_state.sam_perturb(params, grads1, cfg.sar_rho)
                    optimizer.zero_grad(set_to_none=True)
                    logits2 = model(x)
                    loss2, _, _, n_reliable2 = compute_eata_sar_loss(
                        logits2, cfg, method, sar_state, params
                    )
                    if n_reliable2 > 0 and torch.isfinite(loss2):
                        loss2.backward()
                        sar_state.sam_restore(params, originals)
                        optimizer.step()
                        optimizer_steps += 1
                    else:
                        sar_state.sam_restore(params, originals)
            optimizer.zero_grad(set_to_none=True)
            batch_loss_val = float(loss.detach().cpu().item())
        elif method in ("cotta", "memo"):
            from .cottamemo import compute_cotta_memo_loss
            optimizer.zero_grad(set_to_none=True)
            loss, sample_loss, diag, logits = compute_cotta_memo_loss(
                model, x, cfg, method, cotta_memo_state, device,
            )
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                if method == "cotta":
                    cotta_memo_state.update_teacher(params)
                    cotta_memo_state.maybe_restore(params)
            optimizer.zero_grad(set_to_none=True)
            batch_loss_val = float(loss.detach().cpu().item())
        elif method in ("poem", "foa"):
            from .poemfoa import compute_poem_foa_loss
            if method == "poem":
                optimizer.zero_grad(set_to_none=True)
                loss, sample_loss, diag, logits = compute_poem_foa_loss(
                    model, x, cfg, method, poemfoa_state, device,
                )
                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
            else:  # foa - derivative-free, no backward/step needed
                loss, sample_loss, diag, logits = compute_poem_foa_loss(
                    model, x, cfg, method, poemfoa_state, device,
                )
            batch_loss_val = float(loss.detach().cpu().item())
        elif adapting:
            optimizer.zero_grad(set_to_none=True)
            if method == "ucdpa":
                # Extract features for prototype alignment
                features = extract_features(model, x)
                logits = model.head(features) if features is not None else model(x)
            else:
                logits = model(x)
            loss, sample_loss, diag = compute_method_loss_and_diag(logits, cfg, state=ucdpa_state, features=features)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            batch_loss_val = float(loss.detach().cpu().item())
        else:
            with torch.no_grad():
                logits = model(x)
            loss, sample_loss, diag = compute_method_loss_and_diag(logits, cfg)
            batch_loss_val = 0.0

        # Pre-update predictions (logits computed before optimizer.step).
        probs = F.softmax(logits.detach(), dim=1)
        conf, pred = probs.max(dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).detach()
        top2_prob = top2[:, 1].detach()

        n = logits.shape[0]
        for i in range(n):
            yi = int(y[i].item())
            pi = int(pred[i].item())
            valid = 0 <= yi < probs.shape[1]
            row = {
                "method": method,
                "method_display": method_display,
                "config_id": config_id,
                "seed": seed,
                "corruption": corruption,
                "severity": severity,
                "batch_index": batch_idx,
                "sample_index": int(batch_idx * batch_size + i),
                "true_label": yi if valid else -1,
                "pred_label": pi,
                "correct": int(valid and pi == yi),
                "pmax": float(conf[i].cpu().item()),
                "top2_prob": float(top2_prob[i].cpu().item()),
                "margin": float(margin[i].cpu().item()),
                "entropy": float(diag["entropy"][i].cpu().item()),
                "uncertainty_u": float(diag["uncertainty_u"][i].cpu().item()),
                "mutual_info": float(diag["mutual_info"][i].cpu().item()),
                "ucb_score": float(diag["ucb_score"][i].cpu().item()),
                "come_loss": float(diag["come_loss"][i].cpu().item()),
                "pl_loss": float(diag["pl_loss"][i].cpu().item()),
                "pseudo_label": int(diag["pseudo_label"][i].cpu().item()),
                "class_balance_weight": float(diag["class_balance_weight"][i].cpu().item()),
                "L_cal": float(diag["L_cal"][i].cpu().item()),
                "L_fpa": float(diag["L_fpa"][i].cpu().item()),
                "L_objective": float(diag["L_objective"][i].cpu().item()) if torch.isfinite(diag["L_objective"][i]).any() else batch_loss_val,
            }
            sample_rows.append(row)

        if device != "cpu" and torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024 ** 2))

    elapsed = time.time() - start

    # Compute full metric suite
    def col(name):
        return np.array([r[name] for r in sample_rows], dtype=float)

    correct = np.array([r["correct"] for r in sample_rows], dtype=int)
    pred = np.array([r["pred_label"] for r in sample_rows], dtype=int)
    metrics = compute_all_metrics(
        correct=correct, pred=pred,
        pmax=col("pmax"), top2_prob=col("top2_prob"), margin=col("margin"),
        entropy=col("entropy"), uncertainty=col("uncertainty_u"),
        mutual_info=col("mutual_info"), ucb_score=col("ucb_score"),
        come_loss=col("come_loss"),
    )

    summary = {
        "method": method,
        "method_display": method_display,
        "config_id": config_id,
        "seed": seed,
        "corruption": corruption,
        "severity": severity,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "trainable_parameters": trainable_count,
        "runtime_seconds": round(elapsed, 2),
        "peak_gpu_memory_mb": round(peak_mem, 1),
        "optimizer_steps": optimizer_steps,
        "lr": lr,
        "optimizer": cfg.optimizer,
        **metrics,
        **{f"cfg_{k}": v for k, v in cfg.as_dict().items() if k != "method"},
    }

    # Save outputs
    out_base = output_dir
    out_base.mkdir(parents=True, exist_ok=True)
    safe_corr = corruption.replace("/", "_")
    fname = f"{config_id}__{method}__seed{seed}__{safe_corr}__sev{severity}"
    _write_summary_csv(out_base / "summaries" / f"{fname}.csv", summary)
    (out_base / "summaries" / f"{fname}.completed.flag").write_text("completed\n")
    if save_samples:
        _save_samples(out_base / "samples" / f"{fname}.parquet", sample_rows)
    with open(out_base / "summaries" / f"{fname}_config.json", "w") as f:
        json.dump({**cfg.as_dict(), "config_id": config_id, "seed": seed,
                   "corruption": corruption, "severity": severity,
                   "batch_size": batch_size, "max_batches": max_batches}, f, indent=2, default=str)

    print(
        f"[DONE] {config_id} {method_display} seed{seed} {corruption} sev{severity} "
        f"acc={metrics['top1_accuracy']:.2f} ece={metrics['ece_15bins_percent']:.2f} "
        f"mi={metrics['mean_mutual_info']:.4f} hce90={metrics['hce_rate@0.9']:.4f} "
        f"aurc={metrics['aurc']:.4f} eaurc={metrics['e_aurc']:.4f} "
        f"auroc_comb={metrics['auroc_combined']:.4f} ({elapsed:.0f}s)",
        flush=True,
    )
    return summary


def _write_summary_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
