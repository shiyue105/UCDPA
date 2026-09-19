"""Build ResNet-50 backbone experiment items for cross-architecture validation.

Runs 7 methods × 19 corruptions × severity 3 × seed 0 = 133 runs on ResNet-50.
This validates that UCDPA's gains transfer to a CNN backbone with BatchNorm
adaptation, addressing reviewer concern about single-backbone evaluation.

Usage:
  /data/yuehan/envs/tta_env/bin/python build_resnet50_items.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ucdpa import MethodConfig, DEFAULT_LR
from common.data_utils import discover_imagenet_c_combos
from run_experiment import method_configs

PROJECT_ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
DATA_ROOT = "/data/share/datasets"
GPUS = ["2", "3", "5", "6", "7", "8", "9"]

RESNET_SEVERITY = 3
RESNET_SEED = 0
RESNET_BATCH_SIZE = 128


def build_resnet50_items() -> List[Dict]:
    """Build 133 items: 7 methods × 19 corruptions × severity 3 × seed 0."""
    combos = discover_imagenet_c_combos(DATA_ROOT, corruption="all", severity="all")
    sev3 = [(c, s, p) for (c, s, p) in combos if s == RESNET_SEVERITY]
    cfgs = method_configs()
    items: List[Dict] = []
    for (corr, sev, _p) in sev3:
        for cid, cfg in cfgs.items():
            items.append({
                "config_id": f"{cid}_resnet50",
                "method": cfg.method,
                "cfg": cfg.as_dict(),
                "seed": RESNET_SEED,
                "corruption": corr,
                "severity": sev,
                "batch_size": RESNET_BATCH_SIZE,
                "max_batches": -1,
                "phase": "resnet50",
                "backbone": "resnet50",
            })
    return items


def chunk(items: List[Dict], n: int) -> List[List[Dict]]:
    chunks = [[] for _ in range(n)]
    for i, it in enumerate(items):
        chunks[i % n].append(it)
    return chunks


def main():
    items = build_resnet50_items()
    print(f"Total ResNet-50 items: {len(items)} (7 methods × 19 corruptions × sev {RESNET_SEVERITY} × seed {RESNET_SEED})")

    chunks = chunk(items, len(GPUS))
    for i, ck in enumerate(chunks):
        cf = PROJECT_ROOT / f"_resnet50_gpu{GPUS[i]}.json"
        cf.write_text(json.dumps(ck))
        print(f"  GPU {GPUS[i]}: {len(ck)} items -> {cf.name}")


if __name__ == "__main__":
    main()
