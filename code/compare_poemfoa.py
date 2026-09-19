#!/usr/bin/env python
"""Compute POEM/FOA/UCDPA/COME comparison table.

Run after FOA experiment completes. Reads CSV summaries and prints
mean accuracy over 19 corruptions (seed=0, sev=3) plus per-corruption.

Usage:
  cd /data/yuehan/outputs/ucdpa_tta/results/summaries && \
  /data/yuehan/envs/tta_env/bin/python \
  /data/yuehan/outputs/ucdpa_tta/code/compare_poemfoa.py
"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

SUMMARY_DIR = Path("/data/yuehan/outputs/ucdpa_tta/results/summaries")

# UCDPA / COME reference means (seed=0, sev=3, 19 corruptions)
# Populated from prior runs in this project.
UCDPA_MEAN = 70.83
COME_MEAN = 68.10


def collect(prefix: str):
    rows = []
    for f in sorted(glob.glob(str(SUMMARY_DIR / f"{prefix}__*.csv"))):
        with open(f) as fh:
            r = next(csv.DictReader(fh))
        base = os.path.basename(f).replace(".csv", "").split("__")
        corr = base[3]
        rows.append(
            (corr, float(r["accuracy"]),
             float(r["ece_15bins_percent"]), float(r["aurc"]))
        )
    return rows


def main():
    poem = collect("poem__poem")
    foa = collect("foa__foa")

    print("=" * 70)
    print(f"POEM ({len(poem)}/19 corruptions)")
    print("=" * 70)
    print(f"{'corruption':20s} {'acc':>8s} {'ece':>8s} {'aurc':>8s}")
    for c, a, e, u in poem:
        print(f"{c:20s} {a:8.2f} {e:8.2f} {u:8.4f}")
    if poem:
        m = sum(a for _, a, _, _ in poem) / len(poem)
        print(f"{'MEAN':20s} {m:8.2f}")

    print()
    print("=" * 70)
    print(f"FOA ({len(foa)}/19 corruptions)")
    print("=" * 70)
    print(f"{'corruption':20s} {'acc':>8s} {'ece':>8s} {'aurc':>8s}")
    for c, a, e, u in foa:
        print(f"{c:20s} {a:8.2f} {e:8.2f} {u:8.4f}")
    if foa:
        m = sum(a for _, a, _, _ in foa) / len(foa)
        label = "MEAN" if len(foa) == 19 else "PARTIAL MEAN"
        print(f"{label:20s} {m:8.2f}  ({len(foa)}/19)")

    print()
    print("=" * 70)
    print("COMPARISON (seed=0, sev=3)")
    print("=" * 70)
    print(f"{'method':10s} {'mean acc':>10s}")
    if poem:
        m = sum(a for _, a, _, _ in poem) / len(poem)
        print(f"{'POEM':10s} {m:10.2f}")
    if foa:
        m = sum(a for _, a, _, _ in foa) / len(foa)
        tag = "" if len(foa) == 19 else f"  (partial, {len(foa)}/19)"
        print(f"{'FOA':10s} {m:10.2f}{tag}")
    print(f"{'UCDPA':10s} {UCDPA_MEAN:10.2f}")
    print(f"{'COME':10s} {COME_MEAN:10.2f}")

    # FOA completion check
    flag_count = len(glob.glob(str(SUMMARY_DIR / "foa__*.completed.flag")))
    print(f"\nFOA .completed.flag files: {flag_count}/19")


if __name__ == "__main__":
    main()
