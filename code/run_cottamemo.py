"""Launch CoTTA and MEMO TTA baselines on ImageNet-C (ViT-B/16).

Builds work items for:
  - cotta: 19 corruptions x severity 3 x seed 0, batch_size=128, full stream
  - memo:  19 corruptions x severity 3 x seed 0, batch_size=64,  full stream

Splits across GPUs 8 (CoTTA) and 9 (MEMO), launches workers in parallel,
and waits for completion.

Usage:
  cd /data/yuehan/outputs/ucdpa_tta/code && python run_cottamemo.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure the code directory is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ucdpa import MethodConfig, DEFAULT_LR
from common.data_utils import discover_imagenet_c_combos

PROJECT_ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
DATA_ROOT = "/data/share/datasets"
CODE_DIR = Path(__file__).resolve().parent

SEVERITY = 3
SEED = 0
COTTA_BATCH_SIZE = 128
MEMO_BATCH_SIZE = 64
MAX_BATCHES = -1  # full 50000-image stream

# CoTTA on GPU 8, MEMO on GPU 9 (parallel).
GPU_COTTA = "8"
GPU_MEMO = "9"


def _cotta_cfg() -> MethodConfig:
    return MethodConfig(method="cotta", lr=DEFAULT_LR["cotta"], optimizer="adam")


def _memo_cfg() -> MethodConfig:
    return MethodConfig(method="memo", lr=DEFAULT_LR["memo"], optimizer="adam")


def build_items() -> List[Dict]:
    """Build 38 items: cotta (19) + memo (19) at severity 3, seed 0."""
    combos = discover_imagenet_c_combos(DATA_ROOT, corruption="all", severity="all")
    sev3 = [(c, s, p) for (c, s, p) in combos if s == SEVERITY]
    items: List[Dict] = []
    cotta_cfg = _cotta_cfg()
    memo_cfg = _memo_cfg()
    for (corr, sev, _p) in sev3:
        items.append({
            "config_id": "cotta",
            "method": "cotta",
            "cfg": cotta_cfg.as_dict(),
            "seed": SEED,
            "corruption": corr,
            "severity": sev,
            "batch_size": COTTA_BATCH_SIZE,
            "max_batches": MAX_BATCHES,
            "phase": "cottamemo",
            "backbone": "vit_base_patch16_224",
        })
        items.append({
            "config_id": "memo",
            "method": "memo",
            "cfg": memo_cfg.as_dict(),
            "seed": SEED,
            "corruption": corr,
            "severity": sev,
            "batch_size": MEMO_BATCH_SIZE,
            "max_batches": MAX_BATCHES,
            "phase": "cottamemo",
            "backbone": "vit_base_patch16_224",
        })
    return items


def _flag_name(it: Dict) -> str:
    return (f"{it['config_id']}__{it['method']}__seed{it['seed']}__"
            f"{it['corruption']}__sev{it['severity']}.completed.flag")


def launch(items: List[Dict]):
    """Launch one worker per GPU (CoTTA on GPU 8, MEMO on GPU 9)."""
    cotta_items = [it for it in items if it["method"] == "cotta"]
    memo_items = [it for it in items if it["method"] == "memo"]

    chunks = {
        GPU_COTTA: cotta_items,
        GPU_MEMO: memo_items,
    }

    chunk_files = {}
    for gpu, ck in chunks.items():
        cf = PROJECT_ROOT / f"_cottamemo_gpu{gpu}.json"
        cf.write_text(json.dumps(ck))
        chunk_files[gpu] = cf

    env_base = {
        "HF_HOME": "/data/yuehan/cache/huggingface",
        "TORCH_HOME": "/data/yuehan/cache/torch",
        "TIMM_HOME": "/data/yuehan/cache/timm",
        "XDG_CACHE_HOME": "/data/yuehan/cache",
        "MPLCONFIGDIR": "/data/yuehan/cache/matplotlib",
        "PYTHONPATH": str(CODE_DIR),
        "PATH": os.environ.get("PATH", ""),
    }

    procs = []
    for gpu, ck in chunks.items():
        if not ck:
            continue
        env = dict(env_base)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            "/data/yuehan/envs/tta_env/bin/python",
            str(CODE_DIR / "run_experiment.py"),
            "--worker", "--items_file", str(chunk_files[gpu]), "--gpu", str(gpu),
        ]
        log_path = PROJECT_ROOT / "logs" / f"_cottamemo_gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=str(CODE_DIR))
        procs.append((p, gpu, logf))
        print(f"Launched worker on gpu={gpu} with {len(ck)} items (pid={p.pid}, log={log_path})", flush=True)

    rcs = []
    for p, gpu, logf in procs:
        rc = p.wait()
        logf.close()
        print(f"Worker gpu={gpu} exited rc={rc}", flush=True)
        rcs.append(rc)
    return rcs


def main():
    items = build_items()
    n_cotta = sum(1 for it in items if it["method"] == "cotta")
    n_memo = sum(1 for it in items if it["method"] == "memo")
    print(f"CoTTA/MEMO experiment: {len(items)} items "
          f"(cotta={n_cotta} on GPU {GPU_COTTA}, memo={n_memo} on GPU {GPU_MEMO})")
    print(f"Severity={SEVERITY}, seed={SEED}, max_batches={MAX_BATCHES}")
    print(f"  cotta batch_size={COTTA_BATCH_SIZE}, memo batch_size={MEMO_BATCH_SIZE}")

    # Skip already-completed items.
    out_dir = PROJECT_ROOT / "results" / "summaries"
    pending = []
    n_skip = 0
    for it in items:
        flag = out_dir / _flag_name(it)
        if flag.exists():
            n_skip += 1
        else:
            pending.append(it)
    print(f"  already completed: {n_skip}, pending: {len(pending)}")

    if not pending:
        print("All items already completed. Nothing to do.")
        return

    t0 = time.time()
    launch(pending)
    elapsed = time.time() - t0
    print(f"\nAll workers finished in {elapsed:.0f}s.")
    print(f"Results: /data/yuehan/outputs/ucdpa_tta/results/summaries/")


if __name__ == "__main__":
    main()
