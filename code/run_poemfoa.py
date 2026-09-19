"""Launch POEM and FOA TTA baselines on ImageNet-C (ViT-B/16).

Builds work items for:
  - poem: 19 corruptions x severity 3 x seed 0, batch_size=128, full stream
  - foa:  19 corruptions x severity 3 x seed 0, batch_size=128, full stream

Splits across GPUs 8 (POEM) and 9 (FOA), launches workers in parallel,
and waits for completion.

Usage:
  cd /data/yuehan/outputs/ucdpa_tta/code && python run_poemfoa.py
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
BATCH_SIZE = 128
MAX_BATCHES = -1  # full 50000-image stream

# POEM on GPU 8, FOA on GPU 9 (parallel).
GPU_POEM = "8"
GPU_FOA = "9"


def _poem_cfg() -> MethodConfig:
    return MethodConfig(
        method="poem",
        lr=DEFAULT_LR["poem"],
        optimizer="adam",
        poem_entropy_threshold=0.5,
        poem_margin=0.5,
    )


def _foa_cfg() -> MethodConfig:
    return MethodConfig(
        method="foa",
        lr=DEFAULT_LR["foa"],
        optimizer="adam",
        foa_num_candidates=5,
        foa_prompt_size=224,
    )


def build_items() -> List[Dict]:
    """Build 38 items: poem (19) + foa (19) at severity 3, seed 0."""
    combos = discover_imagenet_c_combos(DATA_ROOT, corruption="all", severity="all")
    sev3 = [(c, s, p) for (c, s, p) in combos if s == SEVERITY]
    items: List[Dict] = []
    poem_cfg = _poem_cfg()
    foa_cfg = _foa_cfg()
    for (corr, sev, _p) in sev3:
        items.append({
            "config_id": "poem",
            "method": "poem",
            "cfg": poem_cfg.as_dict(),
            "seed": SEED,
            "corruption": corr,
            "severity": sev,
            "batch_size": BATCH_SIZE,
            "max_batches": MAX_BATCHES,
            "phase": "poemfoa",
            "backbone": "vit_base_patch16_224",
        })
        items.append({
            "config_id": "foa",
            "method": "foa",
            "cfg": foa_cfg.as_dict(),
            "seed": SEED,
            "corruption": corr,
            "severity": sev,
            "batch_size": BATCH_SIZE,
            "max_batches": MAX_BATCHES,
            "phase": "poemfoa",
            "backbone": "vit_base_patch16_224",
        })
    return items


def _flag_name(it: Dict) -> str:
    return (f"{it['config_id']}__{it['method']}__seed{it['seed']}__"
            f"{it['corruption']}__sev{it['severity']}.completed.flag")


def launch(items: List[Dict]):
    """Launch one worker per GPU (POEM on GPU 8, FOA on GPU 9)."""
    poem_items = [it for it in items if it["method"] == "poem"]
    foa_items = [it for it in items if it["method"] == "foa"]

    chunks = {
        GPU_POEM: poem_items,
        GPU_FOA: foa_items,
    }

    chunk_files = {}
    for gpu, ck in chunks.items():
        cf = PROJECT_ROOT / f"_poemfoa_gpu{gpu}.json"
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
        log_path = PROJECT_ROOT / "logs" / f"_poemfoa_gpu{gpu}.log"
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
    n_poem = sum(1 for it in items if it["method"] == "poem")
    n_foa = sum(1 for it in items if it["method"] == "foa")
    print(f"POEM/FOA experiment: {len(items)} items "
          f"(poem={n_poem} on GPU {GPU_POEM}, foa={n_foa} on GPU {GPU_FOA})")
    print(f"Severity={SEVERITY}, seed={SEED}, max_batches={MAX_BATCHES}")
    print(f"  batch_size={BATCH_SIZE}")

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
