"""Paired bootstrap confidence intervals and significance tests for UCDPA.

For each metric and each baseline method (COME, Tent-COME, Tent, EATA, SAR,
Source), compute the paired delta (UCDPA - baseline) using
(corruption, severity, seed) as the pairing unit, then:

  * Bootstrap CI (10000 resamples) for the mean delta -> 2.5% / 97.5%.
  * Paired t-test p-value.
  * Wilcoxon signed-rank p-value.

Metrics covered:
  top1_accuracy, ece_15bins_percent, mce_15bins_percent,
  mean_predictive_entropy, mean_mutual_info, auroc_combined.

Results are written to:
  results/bootstrap_ci_summary.csv

Usage:
    python3 bootstrap_ci.py
    python3 bootstrap_ci.py --seed 42 --resamples 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
SUMMARY_CSV = ROOT / "results" / "summaries" / "full_summary.csv"
OUTPUT_CSV = ROOT / "results" / "bootstrap_ci_summary.csv"

UCDPA_ID = "ucdpa"
BASELINES = ["come", "tentcome", "tent", "eata", "sar", "source"]

METRICS = [
    "top1_accuracy",
    "ece_15bins_percent",
    "mce_15bins_percent",
    "mean_predictive_entropy",
    "mean_mutual_info",
    "auroc_combined",
]

# Pairing keys (the resampling unit).
PAIR_KEYS = ["corruption", "severity", "seed"]

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 0


def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {SUMMARY_CSV}\n"
            "Run aggregate.py first (or wait for experiments to finish)."
        )
    df = pd.read_csv(SUMMARY_CSV)
    print(f"[INFO] Loaded {len(df)} rows from {SUMMARY_CSV}")
    print(f"[INFO] Methods: {sorted(df['config_id'].unique().tolist())}")
    return df


def paired_values(df: pd.DataFrame, metric: str, a_id: str, b_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired metric values for methods a_id and b_id, matched on PAIR_KEYS.

    Rows with NaN in the metric are dropped from the pairing.
    """
    # Ensure the pairing keys exist.
    for k in PAIR_KEYS:
        if k not in df.columns:
            raise KeyError(f"Required pairing column '{k}' not in summary CSV.")
    if metric not in df.columns:
        raise KeyError(f"Metric column '{metric}' not in summary CSV.")

    da = df[df["config_id"] == a_id].set_index(PAIR_KEYS)[metric]
    db = df[df["config_id"] == b_id].set_index(PAIR_KEYS)[metric]
    da = pd.to_numeric(da, errors="coerce")
    db = pd.to_numeric(db, errors="coerce")

    common = da.index.intersection(db.index)
    va = da.loc[common].to_numpy(dtype=float)
    vb = db.loc[common].to_numpy(dtype=float)
    mask = ~(np.isnan(va) | np.isnan(vb))
    return va[mask], vb[mask]


def bootstrap_delta_ci(va: np.ndarray, vb: np.ndarray, n_resamples: int,
                       rng: np.random.Generator) -> Tuple[float, float, float, float, float]:
    """Paired bootstrap of mean(va - vb).

    Returns (mean_delta, ci_low, ci_high, mean_a, mean_b).
    """
    delta = va - vb
    n = len(delta)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = delta[idx].mean()

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi), float(va.mean()), float(vb.mean())


def paired_tests(va: np.ndarray, vb: np.ndarray) -> Tuple[float, float, float, float]:
    """Paired t-test and Wilcoxon signed-rank test.

    Returns (t_stat, t_pvalue, w_stat, w_pvalue). NaN if not computable.
    """
    try:
        from scipy import stats
    except Exception as e:
        print(f"[WARN] scipy not available, cannot run significance tests: {e}")
        return float("nan"), float("nan"), float("nan"), float("nan")

    delta = va - vb
    n = len(delta)
    t_stat = t_p = float("nan")
    w_stat = w_p = float("nan")

    if n >= 2 and not np.allclose(delta, 0):
        try:
            res = stats.ttest_rel(va, vb, nan_policy="omit")
            t_stat, t_p = float(res.statistic), float(res.pvalue)
        except Exception as e:
            print(f"[WARN] ttest_rel failed: {e}")
        try:
            res = stats.wilcoxon(va, vb, zero_method="wilcox")
            w_stat, w_p = float(res.statistic), float(res.pvalue)
        except Exception as e:
            print(f"[WARN] wilcoxon failed: {e}")

    return t_stat, t_p, w_stat, w_p


def run_all(df: pd.DataFrame, n_resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    available_methods = set(df["config_id"].unique().tolist())
    if UCDPA_ID not in available_methods:
        raise ValueError(f"UCDPA rows not found in summary (config_id='{UCDPA_ID}').")

    baselines = [b for b in BASELINES if b in available_methods]
    if not baselines:
        raise ValueError("No baseline methods found in summary CSV.")

    for metric in METRICS:
        if metric not in df.columns:
            print(f"[WARN] metric '{metric}' not in CSV; skipping.")
            continue
        for baseline in baselines:
            try:
                va, vb = paired_values(df, metric, UCDPA_ID, baseline)
            except KeyError as e:
                print(f"[WARN] {e}")
                continue
            n = len(va)
            if n == 0:
                print(f"[WARN] no paired samples for {metric} UCDPA vs {baseline}; skipping.")
                continue

            mean_delta, ci_lo, ci_hi, mean_a, mean_b = bootstrap_delta_ci(va, vb, n_resamples, rng)
            t_stat, t_p, w_stat, w_p = paired_tests(va, vb)

            rows.append({
                "metric": metric,
                "ucdpa_id": UCDPA_ID,
                "baseline_id": baseline,
                "n_paired": n,
                "ucdpa_mean": mean_a,
                "baseline_mean": mean_b,
                "mean_delta": mean_delta,
                "ci_low_2.5": ci_lo,
                "ci_high_97.5": ci_hi,
                "t_statistic": t_stat,
                "t_pvalue": t_p,
                "wilcoxon_statistic": w_stat,
                "wilcoxon_pvalue": w_p,
                "resamples": n_resamples,
                "seed": seed,
            })
            print(f"[INFO] {metric:<26} UCDPA vs {baseline:<9} n={n:<4d} "
                  f"delta={mean_delta:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
                  f"t_p={t_p:.4g} w_p={w_p:.4g}")

    return pd.DataFrame(rows)


def print_summary_table(results: pd.DataFrame) -> None:
    if results.empty:
        print("[WARN] No bootstrap results to print.")
        return
    print("\n" + "=" * 110)
    print("Paired Bootstrap CI Summary (UCDPA - Baseline)")
    print("=" * 110)
    fmt = "{:<26} {:<10} {:>5} {:>12} {:>12} {:>12} {:>14} {:>12} {:>12}"
    print(fmt.format("Metric", "Baseline", "N", "UCDPA_mean", "Base_mean",
                     "Delta", "CI95", "t_pvalue", "w_pvalue"))
    print("-" * 110)
    for _, r in results.iterrows():
        print(fmt.format(
            str(r["metric"]),
            str(r["baseline_id"]),
            int(r["n_paired"]),
            f"{r['ucdpa_mean']:.4f}" if not np.isnan(r["ucdpa_mean"]) else "nan",
            f"{r['baseline_mean']:.4f}" if not np.isnan(r["baseline_mean"]) else "nan",
            f"{r['mean_delta']:+.4f}" if not np.isnan(r["mean_delta"]) else "nan",
            f"[{r['ci_low_2.5']:+.3f},{r['ci_high_97.5']:+.3f}]"
                if not np.isnan(r["ci_low_2.5"]) else "nan",
            f"{r['t_pvalue']:.4f}" if not np.isnan(r["t_pvalue"]) else "nan",
            f"{r['wilcoxon_pvalue']:.4f}" if not np.isnan(r["wilcoxon_pvalue"]) else "nan",
        ))
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap CIs for UCDPA vs baselines.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed (default {DEFAULT_SEED}).")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES,
                        help=f"Number of bootstrap resamples (default {DEFAULT_RESAMPLES}).")
    args = parser.parse_args()

    print("=" * 70)
    print("UCDPA bootstrap_ci.py")
    print("=" * 70)
    print(f"[INFO] resamples={args.resamples}, seed={args.seed}")

    df = load_summary()
    results = run_all(df, n_resamples=args.resamples, seed=args.seed)

    if results.empty:
        print("[WARN] No results produced; exiting without writing CSV.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Wrote {len(results)} rows -> {OUTPUT_CSV}")

    print_summary_table(results)
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
