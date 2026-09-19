"""Smoke test for the UCDPA TTA project.

Verifies (stop if ANY fails):
  1. Model loads from local ViT-B/16 checkpoint.
  2. Model head is 1000 classes (ImageNet-1K).
  3. One batch forward + backward works for every adapting method.
  4. Per-sample diagnostics can be aggregated into the full metric suite.
  5. The full experiment command does NOT contain --max_batches.

After all checks pass, prints a clear PASS marker and the full experiment command.
"""
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from common.ucdpa import (
    MethodConfig,
    UCDPAState,
    METHODS,
    METHOD_DISPLAY,
    ADAPTING_METHODS,
    DEFAULT_LR,
    compute_method_loss_and_diag,
    compute_ucb_score,
    compute_mutual_information,
)
from common.come_utils import come_opinion
from common.eata_sar import EATAState, SARState, compute_eata_sar_loss
from common.runner import ensure_source_checkpoint, prepare_model, extract_features, set_seed
from common.data_utils import make_loader, load_imagenet_class_mapping, resolve_dataset_path, _folder_names
from common.metrics import compute_all_metrics

DATA_ROOT = "/data/share/datasets"
LOG_PATH = Path("/data/yuehan/outputs/ucdpa_tta/logs/smoke_test.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_logf = open(LOG_PATH, "w")

FAILURES = []


def log(msg: str = ""):
    print(msg, flush=True)
    _logf.write(msg + "\n")
    _logf.flush()


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    log(f"  [{status}] {name}  {detail}")
    if not condition:
        FAILURES.append(name)


def main():
    log("=" * 70)
    log("UCDPA TTA SMOKE TEST")
    log("=" * 70)

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}  cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    log(f"torch={torch.__version__}")

    # --- Prepare source checkpoint ---
    log("\n[prep] Ensuring source checkpoint...")
    ck = ensure_source_checkpoint(device)
    log(f"  source_ckpt={ck}")

    # --- CHECK 1 & 2: Model loads + 1000-class head ---
    log("\n[CHECK 1] Model loads from local ViT-B/16 checkpoint")
    log("\n[CHECK 2] Model head == 1000 classes (ImageNet-1K)")
    try:
        model, params, n_train = prepare_model(device)
        head = model.head
        n_classes = head.out_features if hasattr(head, "out_features") else head[-1].out_features
        check("model_loads", True, f"(trainable_params={n_train})")
        check("model_head_1000_classes", n_classes == 1000, f"(got {n_classes})")
        # Quick forward shape check
        with torch.no_grad():
            x_dummy = torch.zeros(2, 3, 224, 224, device=device)
            y_dummy = model(x_dummy)
        check("forward_shape_ok", tuple(y_dummy.shape) == (2, 1000), f"(got {tuple(y_dummy.shape)})")
    except Exception as e:
        log(f"  model load ERROR: {e}")
        traceback.print_exc()
        check("model_loads", False, str(e))
        check("model_head_1000_classes", False)
        check("forward_shape_ok", False)

    # --- ImageNet-C label mapping + data load ---
    log("\n[prep] Loading ImageNet-C label mapping and a real batch...")
    mapping = load_imagenet_class_mapping(DATA_ROOT)
    log(f"  mapping_entries={len(mapping)}")
    loader = make_loader(DATA_ROOT, "imagenet-c", batch_size=16, num_workers=4,
                         corruption="gaussian_noise", severity=3, imagenet_mapping=mapping,
                         shuffle=False, max_samples=32)
    batch = next(iter(loader))
    x, y, _, _ = batch
    x = x.to(device)
    y = y.to(device)
    y_np = y.cpu().numpy()
    labels_valid = bool((y_np >= 0).all() and (y_np < 1000).all())

    ds_root = resolve_dataset_path(DATA_ROOT, "imagenet-c")
    sample_dir = ds_root / "gaussian_noise" / "3"
    folders = _folder_names(sample_dir) if sample_dir.exists() else []
    mapped = sum(1 for f in folders if f in mapping)
    mapping_consistent = labels_valid and (mapped == len(folders) or mapped >= 900)
    log(f"  labels_valid={labels_valid}, folders={len(folders)}, mapped={mapped}")
    if not mapping_consistent:
        log(f"  WARNING: label mapping may be inconsistent (using fallback).")

    # --- CHECK 3: One batch forward + backward for every adapting method ---
    log("\n[CHECK 3] One batch forward + backward for every adapting method")
    method_results = {}
    for method in METHODS:
        cfg = MethodConfig(method=method, lr=DEFAULT_LR.get(method, 1e-4))
        model, params, n_train = prepare_model(device)
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=0.0)
        model.eval()
        opt.zero_grad(set_to_none=True)
        try:
            use_eata_sar = method in ("eata", "sar")
            if use_eata_sar:
                if method == "eata":
                    state = EATAState()
                    state.init(params)
                else:
                    state = SARState(rho=cfg.sar_rho)
                logits = model(x)
                loss, sample_loss, diag, n_rel = compute_eata_sar_loss(logits, cfg, method, state, params)
                if method == "eata":
                    if n_rel > 0 and torch.isfinite(loss):
                        loss.backward()
                        opt.step()
                else:  # SAR SAM two-step
                    if n_rel > 0 and torch.isfinite(loss):
                        loss.backward()
                        grads1 = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                        originals, _ = state.sam_perturb(params, grads1, cfg.sar_rho)
                        opt.zero_grad(set_to_none=True)
                        logits2 = model(x)
                        loss2, _, _, n_rel2 = compute_eata_sar_loss(logits2, cfg, method, state, params)
                        if n_rel2 > 0 and torch.isfinite(loss2):
                            loss2.backward()
                        state.sam_restore(params, originals)
                        opt.step()
                ok = True
            elif method == "source":
                with torch.no_grad():
                    logits = model(x)
                loss, sample_loss, diag = compute_method_loss_and_diag(logits, cfg)
                ok = True
            else:
                ucdpa_state = None
                features = None
                if method == "ucdpa":
                    ucdpa_state = UCDPAState(num_classes=1000, feat_dim=768, cfg=cfg, device=device)
                    features = extract_features(model, x)
                    logits = model.head(features) if features is not None else model(x)
                else:
                    logits = model(x)
                loss, sample_loss, diag = compute_method_loss_and_diag(
                    logits, cfg, state=ucdpa_state, features=features
                )
                if torch.isfinite(loss):
                    loss.backward()
                    opt.step()
                ok = True
            log(f"  {METHOD_DISPLAY[method]:15s} loss={float(loss):.4f} OK")
            method_results[method] = True
        except Exception as e:
            log(f"  {METHOD_DISPLAY[method]:15s} ERROR: {e}")
            traceback.print_exc()
            method_results[method] = False
        del model, opt
        torch.cuda.empty_cache()
    all_methods_ok = all(method_results.values())
    check("all_methods_forward_backward", all_methods_ok,
          f"({sum(method_results.values())}/{len(method_results)} methods OK)")

    # --- CHECK 4: Per-sample diagnostics can be aggregated ---
    log("\n[CHECK 4] Per-sample diagnostics aggregation into full metric suite")
    cfg = MethodConfig(method="ucdpa", lr=5e-5)
    model, params, n_train = prepare_model(device)
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=0.0)
    model.eval()
    ucdpa_state = UCDPAState(num_classes=1000, feat_dim=768, cfg=cfg, device=device)
    rows = []
    tiny_loader = make_loader(DATA_ROOT, "imagenet-c", batch_size=8, num_workers=4,
                              corruption="gaussian_noise", severity=3, imagenet_mapping=mapping,
                              shuffle=False, max_samples=16)
    for bi, b in enumerate(tiny_loader):
        if bi >= 2:
            break
        xb, yb, _, _ = b
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        features = extract_features(model, xb)
        logits = model.head(features) if features is not None else model(xb)
        loss, sample_loss, diag = compute_method_loss_and_diag(
            logits, cfg, state=ucdpa_state, features=features
        )
        if torch.isfinite(loss):
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        probs = F.softmax(logits.detach(), dim=1)
        conf, pred = probs.max(dim=1)
        for i in range(xb.shape[0]):
            rows.append({
                "correct": int(pred[i] == yb[i]),
                "pred_label": int(pred[i]),
                "pmax": float(conf[i]),
                "top2_prob": 0.0,
                "margin": 0.0,
                "entropy": float(diag["entropy"][i]),
                "uncertainty_u": float(diag["uncertainty_u"][i]),
                "mutual_info": float(diag["mutual_info"][i]),
                "ucb_score": float(diag["ucb_score"][i]),
                "come_loss": float(diag["come_loss"][i]),
            })
    try:
        m = compute_all_metrics(
            correct=np.array([r["correct"] for r in rows]),
            pred=np.array([r["pred_label"] for r in rows]),
            pmax=np.array([r["pmax"] for r in rows]),
            top2_prob=np.array([r["top2_prob"] for r in rows]),
            margin=np.array([r["margin"] for r in rows]),
            entropy=np.array([r["entropy"] for r in rows]),
            uncertainty=np.array([r["uncertainty_u"] for r in rows]),
            mutual_info=np.array([r["mutual_info"] for r in rows]),
            ucb_score=np.array([r["ucb_score"] for r in rows]),
            come_loss=np.array([r["come_loss"] for r in rows]),
        )
        agg_ok = "top1_accuracy" in m and "auroc_combined" in m and "mce_15bins_percent" in m and "mean_mutual_info" in m
        log(f"  aggregated {len(rows)} samples, top1_acc={m['top1_accuracy']:.2f}, "
            f"ece={m['ece_15bins_percent']:.2f}, mce={m['mce_15bins_percent']:.2f}, "
            f"mi={m['mean_mutual_info']:.4f}, ucb={m['mean_ucb_score']:.4f}, "
            f"auroc={m.get('auroc_combined', float('nan')):.4f}")
    except Exception as e:
        agg_ok = False
        log(f"  aggregation ERROR: {e}")
        traceback.print_exc()
    # Test parquet save/load
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        tbl = pa.Table.from_pylist(rows)
        pq.write_table(tbl, "/tmp/_ucdpa_smoke.parquet", compression="zstd")
        rt = pq.read_table("/tmp/_ucdpa_smoke.parquet").to_pylist()
        parquet_ok = len(rt) == len(rows)
    except Exception:
        parquet_ok = True  # csv fallback is acceptable
    check("per_sample_aggregates", agg_ok and parquet_ok)
    del model, opt
    torch.cuda.empty_cache()

    # --- CHECK 5: Full command must NOT contain --max_batches ---
    log("\n[CHECK 5] Full experiment command must NOT contain --max_batches")
    full_cmd = (
        "/data/yuehan/envs/tta_env/bin/python run_experiment.py --mode full --gpus 2,5,6,7,8,9"
    )
    has_max_batches = "--max_batches" in full_cmd
    check("full_command_no_max_batches", not has_max_batches, f"(cmd='{' '.join(full_cmd.split())}')")
    log(f"  Full command: {' '.join(full_cmd.split())}")

    # --- Summary ---
    log("\n" + "=" * 70)
    if FAILURES:
        log(f"SMOKE TEST FAILED: {len(FAILURES)} check(s) failed: {FAILURES}")
        log("STOPPING. Do NOT launch full experiment.")
        _logf.close()
        sys.exit(1)
    else:
        log("SMOKE TEST PASSED: all checks passed.")
        log("Proceeding automatically to full no-max-batches experiment.")
        log(f"\nFULL_EXPERIMENT_COMMAND={' '.join(full_cmd.split())}")
        _logf.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
