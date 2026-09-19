"""Build CIFAR-100-C experiment items for cross-dataset validation.

Runs 7 methods × 19 corruptions × severity 3 × seed 0 = 133 runs on CIFAR-100-C
with ResNet-18 backbone (BN affine adaptation).

This validates that UCDPA's gains transfer to:
  - A different dataset (CIFAR-100-C vs ImageNet-C)
  - A different backbone (ResNet-18 vs ViT-B/16)
  - A different adaptation locus (BatchNorm vs LayerNorm)

Usage:
  /data/yuehan/envs/tta_env/bin/python build_cifar100_items.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ucdpa import MethodConfig, DEFAULT_LR
from common.data_utils import discover_cifar100_c_combos
from run_experiment import method_configs

PROJECT_ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
DATA_ROOT = "/data/share/datasets"
GPUS = ["2", "5", "6", "7", "8", "9"]

CIFAR_SEVERITY = 3
CIFAR_SEED = 0
CIFAR_BATCH_SIZE = 128


def build_cifar100_items() -> List[Dict]:
    """Build 133 items: 7 methods × 19 corruptions × severity 3 × seed 0."""
    combos = discover_cifar100_c_combos(DATA_ROOT, severity=CIFAR_SEVERITY)
    cfgs = method_configs()
    items: List[Dict] = []
    for (corr, sev, _p) in combos:
        for cid, cfg in cfgs.items():
            items.append({
                "config_id": f"{cid}_cifar100",
                "method": cfg.method,
                "cfg": cfg.as_dict(),
                "seed": CIFAR_SEED,
                "corruption": corr,
                "severity": sev,
                "batch_size": CIFAR_BATCH_SIZE,
                "max_batches": -1,
                "phase": "cifar100",
                "backbone": "cifar_resnet18",
            })
    return items


def chunk(items: List[Dict], n: int) -> List[List[Dict]]:
    chunks = [[] for _ in range(n)]
    for i, it in enumerate(items):
        chunks[i % n].append(it)
    return chunks


def main():
    items = build_cifar100_items()
    print(f"Total CIFAR-100-C items: {len(items)} "
          f"(7 methods × 19 corruptions × sev {CIFAR_SEVERITY} × seed {CIFAR_SEED})")

    chunks = chunk(items, len(GPUS))
    for i, ck in enumerate(chunks):
        cf = PROJECT_ROOT / f"_cifar100_gpu{GPUS[i]}.json"
        cf.write_text(json.dumps(ck))
        print(f"  GPU {GPUS[i]}: {len(ck)} items -> {cf.name}")


if __name__ == "__main__":
    main()
