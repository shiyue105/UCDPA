"""CLI orchestrator for UCDPA Test-Time Adaptation experiments on ImageNet-C.

Builds the full experiment grid:
  7 methods x 19 corruptions x 5 severities x 3 seeds = 1995 runs

Methods: source, tent, eata, sar, come, tentcome, ucdpa.

Usage:
  # Full experiment (1995 runs) across 6 GPUs:
  python run_experiment.py --mode full --gpus 2,5,6,7,8,9

  # Smoke / quick test (2 batches per run):
  python run_experiment.py --mode smoke --gpus 2

  # Worker mode (launched internally, one per GPU):
  python run_experiment.py --worker --items_file /tmp/items.json --gpu 2

The full experiment MUST NOT use --max_batches (full 50000-image streams).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure the code directory is on sys.path for both launcher and worker modes.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ucdpa import MethodConfig, METHODS, METHOD_DISPLAY, DEFAULT_LR
from common.data_utils import discover_imagenet_c_combos, load_imagenet_class_mapping

PROJECT_ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
DATA_ROOT = "/data/share/datasets"

FULL_SEVERITIES = [1, 2, 3, 4, 5]
FULL_SEEDS = [0, 1, 2]
FULL_BATCH_SIZE = 128

SMOKE_MAX_BATCHES = 2
SMOKE_BATCH_SIZE = 32
SMOKE_CORRUPTIONS = ["gaussian_noise"]
SMOKE_SEVERITIES = [3]


def base_cfg(method: str, **kw) -> MethodConfig:
    """Build a MethodConfig with the default learning rate for the method."""
    defaults = {"lr": DEFAULT_LR.get(method, 1e-4), "optimizer": "adam"}
    defaults.update(kw)
    return MethodConfig(method=method, **defaults)


def method_configs() -> Dict[str, MethodConfig]:
    """Return the 7 comparison method configs keyed by config_id."""
    cfgs: Dict[str, MethodConfig] = {}
    cfgs["source"] = base_cfg("source")
    cfgs["tent"] = base_cfg("tent")
    cfgs["eata"] = base_cfg("eata")
    cfgs["sar"] = base_cfg("sar")
    cfgs["come"] = base_cfg("come")
    cfgs["tentcome"] = base_cfg("tentcome", lambda_come=0.03)
    cfgs["ucdpa"] = base_cfg("ucdpa", lr=2e-4, lambda_cal=0.02, lambda_fpa=0.0)
    cfgs["cotta"] = base_cfg("cotta", lr=1e-4)
    cfgs["memo"] = base_cfg("memo", lr=1e-4)
    cfgs["poem"] = base_cfg("poem", lr=2e-4, poem_entropy_threshold=0.5, poem_margin=0.5)
    cfgs["foa"] = base_cfg("foa", lr=1e-3, foa_num_candidates=5, foa_prompt_size=224)
    return cfgs


def build_full_items(max_batches: int = -1) -> List[Dict]:
    """Build all 1995 work items: 7 methods x 19 corruptions x 5 severities x 3 seeds."""
    combos = discover_imagenet_c_combos(DATA_ROOT, corruption="all", severity="all")
    cfgs = method_configs()
    items: List[Dict] = []
    for seed in FULL_SEEDS:
        for (corr, sev, _p) in combos:
            for cid, cfg in cfgs.items():
                items.append({
                    "config_id": cid,
                    "method": cfg.method,
                    "cfg": cfg.as_dict(),
                    "seed": seed,
                    "corruption": corr,
                    "severity": sev,
                    "batch_size": FULL_BATCH_SIZE,
                    "max_batches": max_batches,
                    "phase": "full",
                })
    return items


def build_smoke_items() -> List[Dict]:
    """Build a small smoke-test item set (7 methods x 1 corruption x 1 severity x 1 seed)."""
    cfgs = method_configs()
    items: List[Dict] = []
    for cid, cfg in cfgs.items():
        for corr in SMOKE_CORRUPTIONS:
            for sev in SMOKE_SEVERITIES:
                items.append({
                    "config_id": cid,
                    "method": cfg.method,
                    "cfg": cfg.as_dict(),
                    "seed": 0,
                    "corruption": corr,
                    "severity": sev,
                    "batch_size": SMOKE_BATCH_SIZE,
                    "max_batches": SMOKE_MAX_BATCHES,
                    "phase": "smoke",
                })
    return items


def chunk(items: List[Dict], n: int) -> List[List[Dict]]:
    """Round-robin distribute items into n chunks for balanced GPU load."""
    chunks = [[] for _ in range(n)]
    for i, it in enumerate(items):
        chunks[i % n].append(it)
    return chunks


def _flag_name(it: Dict) -> str:
    return f"{it['config_id']}__{it['method']}__seed{it['seed']}__{it['corruption']}__sev{it['severity']}.completed.flag"


def torch_cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def run_worker(items_file: str, gpu: str):
    """Run a chunk of work items sequentially on the current (already-pinned) GPU."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    from common.runner import run_one_combo, ensure_source_checkpoint

    device = "cuda" if torch_cuda_available() else "cpu"
    items = json.loads(Path(items_file).read_text())
    phase = items[0].get("phase", "full")
    # Allow per-item backbone override; default to ViT-B/16 for backward compat.
    last_backbone = "vit_base_patch16_224"
    ensure_source_checkpoint(device, backbone=last_backbone)
    mapping = load_imagenet_class_mapping(DATA_ROOT)

    out_dir = PROJECT_ROOT / "results"
    n_done = 0
    n_skip = 0
    n_err = 0
    for idx, it in enumerate(items):
        flag = out_dir / "summaries" / _flag_name(it)
        if flag.exists():
            print(f"[skip] {flag.name} already completed", flush=True)
            n_skip += 1
            continue
        cfg = MethodConfig(**it["cfg"])
        backbone = it.get("backbone", "vit_base_patch16_224")
        if backbone != last_backbone:
            ensure_source_checkpoint(device, backbone=backbone)
            last_backbone = backbone
        try:
            run_one_combo(
                cfg=cfg, seed=it["seed"], corruption=it["corruption"], severity=it["severity"],
                batch_size=it["batch_size"], data_root=DATA_ROOT, output_dir=out_dir,
                max_batches=it["max_batches"], num_workers=6, device=device,
                imagenet_mapping=mapping, config_id=it["config_id"], save_samples=True,
                backbone=backbone,
            )
            n_done += 1
        except Exception as e:
            import traceback
            print(f"[ERROR] {it['config_id']} {it['corruption']} sev{it['severity']}: {e}", flush=True)
            traceback.print_exc()
            n_err += 1
    print(f"[worker gpu={gpu}] done={n_done} skip={n_skip} err={n_err} total={len(items)}", flush=True)


def launch(items: List[Dict], gpus: List[str]):
    """Launch one worker subprocess per GPU, distributing items round-robin."""
    code_dir = Path(__file__).resolve().parent
    chunks = chunk(items, len(gpus))
    chunk_files = []
    for i, ck in enumerate(chunks):
        cf = PROJECT_ROOT / f"_items_gpu{gpus[i]}.json"
        cf.write_text(json.dumps(ck))
        chunk_files.append(cf)

    env_base = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HOME": "/data/yuehan/cache/huggingface",
        "TORCH_HOME": "/data/yuehan/cache/torch",
        "TIMM_HOME": "/data/yuehan/cache/timm",
        "XDG_CACHE_HOME": "/data/yuehan/cache",
        "MPLCONFIGDIR": "/data/yuehan/cache/matplotlib",
        "PYTHONPATH": str(code_dir),
        "PATH": os.environ.get("PATH", ""),
    }
    procs = []
    for i, gpu in enumerate(gpus):
        if not chunks[i]:
            continue
        env = dict(env_base)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            "/data/yuehan/envs/tta_env/bin/python",
            str(code_dir / "run_experiment.py"),
            "--worker", "--items_file", str(chunk_files[i]), "--gpu", str(gpu),
        ]
        log_path = PROJECT_ROOT / "logs" / f"_worker_gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=str(code_dir))
        procs.append((p, gpu, logf))
        print(f"Launched worker on gpu={gpu} with {len(chunks[i])} items (pid={p.pid}, log={log_path})")

    # Wait for all workers
    rcs = []
    for p, gpu, logf in procs:
        rc = p.wait()
        logf.close()
        print(f"Worker gpu={gpu} exited rc={rc}")
        rcs.append(rc)
    return rcs


def main():
    parser = argparse.ArgumentParser(description="UCDPA TTA experiment launcher")
    parser.add_argument("--mode", choices=["full", "smoke", "list"], default="full",
                        help="Experiment mode: full (1995 runs), smoke (quick test), list (print count).")
    parser.add_argument("--gpus", type=str, default="2,5,6,7,8,9",
                        help="Comma-separated GPU IDs to use (e.g. 2,5,6,7,8,9).")
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="Max batches per run (-1 = whole 50000-image stream). Default -1. "
                             "The full experiment MUST NOT set this.")
    # Worker mode
    parser.add_argument("--worker", action="store_true", help="Run as a single-GPU worker.")
    parser.add_argument("--items_file", type=str, default="", help="Path to items JSON for worker mode.")
    parser.add_argument("--gpu", type=str, default="", help="GPU ID for worker mode.")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.items_file, args.gpu)
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if args.mode == "list":
        items = build_full_items()
        print(f"Full experiment: {len(items)} items (7 methods x 19 corruptions x 5 severities x 3 seeds)")
        print(f"Methods: {list(method_configs().keys())}")
        return

    if args.mode == "smoke":
        items = build_smoke_items()
        print(f"Smoke test: {len(items)} items across GPUs {gpus}")
        launch(items, gpus)
        print("Smoke test complete. See /data/yuehan/outputs/ucdpa_tta/results/summaries/")
        return

    # Full mode
    if args.max_batches > 0:
        print("WARNING: --max_batches is set for full mode. The full experiment should use "
              "complete 50000-image streams. Remove --max_batches for the production run.",
              file=sys.stderr)
    items = build_full_items(max_batches=args.max_batches)
    print(f"Full experiment: {len(items)} items across GPUs {gpus} (max_batches={args.max_batches})")
    launch(items, gpus)
    print("Full experiment complete. See /data/yuehan/outputs/ucdpa_tta/results/summaries/")
    print("Next: python aggregate.py && python make_figures.py")


if __name__ == "__main__":
    main()
