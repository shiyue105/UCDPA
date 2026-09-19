"""Build work-item lists for supplementary experiments requested by AAAI reviewers.

Generates 4 item categories:
  1. ucdpa_noscalenorm: scale-norm ablation (95 runs, seed 0, all 19x5 combos)
  2. ucdpa_sensitivity_{tau_c,tau_u,T_c,T_u}: 4 one-at-a-time sweeps
     (5 values each x 3 corruptions x 3 severities x seed 0 = 45 runs each)
  3. ucdpa_batchsize_{1,8,32,64}: batch-size sensitivity
     (4 batch sizes x 19 corruptions x severity 3 x seed 0 = 76 runs each)

All items are split round-robin across 6 GPUs and written as JSON files
named _supp_gpu{g}.json in the project root, suitable for passing to
run_experiment.py --worker --items_file.

Usage:
  /data/yuehan/envs/tta_env/bin/python build_supplementary_items.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ucdpa import MethodConfig, DEFAULT_LR
from common.data_utils import discover_imagenet_c_combos

PROJECT_ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
DATA_ROOT = "/data/share/datasets"
GPUS = ["2", "5", "6", "7", "8", "9"]

# Representative corruptions for sensitivity sweep (one per family)
SENSITIVITY_CORRUPTIONS = ["gaussian_noise", "snow", "zoom_blur", "contrast", "spatter"]
SENSITIVITY_SEVERITIES = [1, 3, 5]

# Sweep grids (one-at-a-time)
DEFAULT = dict(tau_c=0.7, tau_u=0.3, T_c=0.1, T_u=0.1)
SWEEPS = {
    "tauc":  [0.3, 0.5, 0.7, 0.8, 0.9],
    "tauu":  [0.1, 0.2, 0.3, 0.4, 0.5],
    "Tc":    [0.05, 0.1, 0.2, 0.3, 0.5],
    "Tu":    [0.05, 0.1, 0.2, 0.3, 0.5],
}

BATCH_SIZES = [1, 8, 32, 64]


def _ucdpa_cfg(**overrides) -> MethodConfig:
    """Build a UCDPA MethodConfig with optional overrides."""
    kw = dict(
        method="ucdpa",
        lr=DEFAULT_LR["ucdpa"],
        lambda_cal=0.02,
        lambda_fpa=0.0,
    )
    kw.update(overrides)
    return MethodConfig(**kw)


def _item(cid: str, cfg: MethodConfig, seed: int, corr: str, sev: int,
          batch_size: int = 128, max_batches: int = -1, phase: str = "supp") -> Dict:
    return {
        "config_id": cid,
        "method": cfg.method,
        "cfg": cfg.as_dict(),
        "seed": seed,
        "corruption": corr,
        "severity": sev,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "phase": phase,
    }


def build_scale_norm_ablation() -> List[Dict]:
    """Scale-norm ablation: 19 corruptions x 5 severities x seed 0 = 95 runs."""
    items = []
    combos = discover_imagenet_c_combos(DATA_ROOT, corruption="all", severity="all")
    for (corr, sev, _p) in combos:
        cfg = _ucdpa_cfg(disable_scale_norm=True)
        items.append(_item("ucdpa_noscalenorm", cfg, seed=0, corr=corr, sev=sev))
    return items


def build_sensitivity_sweep() -> List[Dict]:
    """One-at-a-time sensitivity sweep on tau_c, tau_u, T_c, T_u.

    For each hyperparameter, sweep 5 values, over 5 corruptions x 3 severities x seed 0.
    Plus the default config at each (corr, sev) for reference (so 5*5*3 + 5*3 = 90 each).
    Actually: 5 values x 5 corr x 3 sev = 75 runs per hyperparameter, 4 hyperparams = 300.
    The default value is included in the 5 values, so no separate baseline needed.
    """
    items = []
    for sweep_name, values in SWEEPS.items():
        param_key = {"tauc": "tau_c", "tauu": "tau_u",
                     "Tc": "T_c", "Tu": "T_u"}[sweep_name]
        for val in values:
            for corr in SENSITIVITY_CORRUPTIONS:
                for sev in SENSITIVITY_SEVERITIES:
                    cfg = _ucdpa_cfg(**{param_key: val})
                    cid = f"ucdpa_sens_{sweep_name}_{val}"
                    items.append(_item(cid, cfg, seed=0, corr=corr, sev=sev))
    return items


def build_batch_size_sweep() -> List[Dict]:
    """Batch-size sensitivity: 4 batch sizes x 19 corruptions x severity 3 x seed 0."""
    items = []
    combos = discover_imagenet_c_combos(DATA_ROOT, corruption="all", severity="all")
    sev3_combos = [(c, s, p) for (c, s, p) in combos if s == 3]
    for bs in BATCH_SIZES:
        for (corr, sev, _p) in sev3_combos:
            cfg = _ucdpa_cfg()
            cid = f"ucdpa_bs{bs}"
            items.append(_item(cid, cfg, seed=0, corr=corr, sev=sev, batch_size=bs))
    return items


def chunk(items: List[Dict], n: int) -> List[List[Dict]]:
    chunks = [[] for _ in range(n)]
    for i, it in enumerate(items):
        chunks[i % n].append(it)
    return chunks


def main():
    all_items: List[Dict] = []
    all_items.extend(build_scale_norm_ablation())
    all_items.extend(build_sensitivity_sweep())
    all_items.extend(build_batch_size_sweep())

    print(f"Total supplementary items: {len(all_items)}")
    print(f"  scale_norm ablation: {len(build_scale_norm_ablation())}")
    print(f"  sensitivity sweep:   {len(build_sensitivity_sweep())}")
    print(f"  batch size sweep:    {len(build_batch_size_sweep())}")

    chunks = chunk(all_items, len(GPUS))
    for i, ck in enumerate(chunks):
        cf = PROJECT_ROOT / f"_supp_gpu{GPUS[i]}.json"
        cf.write_text(json.dumps(ck))
        print(f"  GPU {GPUS[i]}: {len(ck)} items -> {cf.name}")

    print("\nTo launch:")
    for g in GPUS:
        print(f"  CUDA_VISIBLE_DEVICES={g} /data/yuehan/envs/tta_env/bin/python "
              f"run_experiment.py --worker --items_file _supp_gpu{g}.json --gpu {g} &")


if __name__ == "__main__":
    main()
