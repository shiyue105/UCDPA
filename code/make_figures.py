"""Generate figures for the UCDPA TTA experiments.

Produces (PDF + PNG) under results/figures/:
  fig1_accuracy_vs_ece.png         Accuracy vs ECE Pareto plot (7 methods).
  fig2_accuracy_by_severity.png    Accuracy by severity (line plot, 7 methods).
  fig3_accuracy_by_family.png      Accuracy by corruption family (grouped bars).
  fig4_ucb_distribution.png        UCB score distribution: correct vs wrong (UCDPA).
  fig5_auroc_comparison.png        Error detection AUROC comparison (7 methods).
  fig6_reliability_diagram.png     Reliability diagram (source vs ucdpa).
  fig7_mi_vs_uncertainty.png       Mutual information vs uncertainty scatter (UCDPA).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
SUMMARY_CSV = ROOT / "results" / "summaries" / "full_summary.csv"
SAMPLES_DIR = ROOT / "results" / "samples"
FIGURES_DIR = ROOT / "results" / "figures"

METHOD_ORDER = ["source", "tent", "eata", "sar", "come", "tentcome", "ucdpa"]
METHOD_DISPLAY = {
    "source": "Source-only", "tent": "Tent", "eata": "EATA", "sar": "SAR",
    "come": "COME", "tentcome": "Tent-COME", "ucdpa": "UCDPA",
}
METHOD_COLOR = {
    "source": "#888888", "tent": "#4444aa", "eata": "#6a1b9a", "sar": "#6d4c41",
    "come": "#cc7700", "tentcome": "#2266cc", "ucdpa": "#cc1133",
}
METHOD_MARKER = {
    "source": "s", "tent": "v", "eata": "P", "sar": "<",
    "come": "D", "tentcome": "^", "ucdpa": "*",
}

CORRUPTION_FAMILY = {
    "gaussian_noise": "Noise", "shot_noise": "Noise", "impulse_noise": "Noise", "speckle_noise": "Noise",
    "defocus_blur": "Blur", "glass_blur": "Blur", "motion_blur": "Blur", "zoom_blur": "Blur", "gaussian_blur": "Blur",
    "snow": "Weather", "frost": "Weather", "fog": "Weather", "brightness": "Weather",
    "contrast": "Digital", "elastic_transform": "Digital", "pixelate": "Digital", "jpeg_compression": "Digital",
    "saturate": "Other", "spatter": "Other",
}


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {name}.png / .pdf")


def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        print(f"[WARN] Summary not found: {SUMMARY_CSV}. Run aggregate.py first.")
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_CSV)


def _ordered_methods(df: pd.DataFrame) -> list:
    return [m for m in METHOD_ORDER if m in df["config_id"].unique().tolist()]


def fig1_accuracy_vs_ece(df: pd.DataFrame):
    """Accuracy vs ECE Pareto plot."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    agg = df.groupby("config_id").agg(
        acc_mean=("top1_accuracy", "mean"), acc_std=("top1_accuracy", "std"),
        ece_mean=("ece_15bins_percent", "mean"), ece_std=("ece_15bins_percent", "std"),
    ).reset_index()
    for m in _ordered_methods(df):
        row = agg[agg["config_id"] == m]
        if row.empty:
            continue
        ax.errorbar(
            float(row["ece_mean"]), float(row["acc_mean"]),
            xerr=float(row["ece_std"]), yerr=float(row["acc_std"]),
            fmt=METHOD_MARKER.get(m, "o"), color=METHOD_COLOR.get(m, "#333"),
            label=METHOD_DISPLAY.get(m, m), markersize=10, capsize=3, linewidth=1.5,
        )
    ax.set_xlabel("ECE (%)  [lower is better]", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)  [higher is better]", fontsize=12)
    ax.set_title("Accuracy vs Calibration Error", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    _save(fig, "fig1_accuracy_vs_ece")


def fig2_accuracy_by_severity(df: pd.DataFrame):
    """Accuracy by severity (line plot)."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in _ordered_methods(df):
        sub = df[df["config_id"] == m].groupby("severity")["top1_accuracy"].agg(["mean", "std"]).reset_index()
        sub = sub.sort_values("severity")
        ax.errorbar(
            sub["severity"], sub["mean"], yerr=sub["std"],
            fmt=METHOD_MARKER.get(m, "o") + "-", color=METHOD_COLOR.get(m, "#333"),
            label=METHOD_DISPLAY.get(m, m), markersize=8, capsize=3, linewidth=1.5,
        )
    ax.set_xlabel("Severity", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy by Corruption Severity", fontsize=13)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    _save(fig, "fig2_accuracy_by_severity")


def fig3_accuracy_by_family(df: pd.DataFrame):
    """Accuracy by corruption family (grouped bars)."""
    if df.empty:
        return
    df = df.copy()
    df["family"] = df["corruption"].map(CORRUPTION_FAMILY).fillna("Other")
    agg = df.groupby(["family", "config_id"])["top1_accuracy"].mean().reset_index()
    families = sorted(df["family"].unique())
    methods = _ordered_methods(df)
    x = np.arange(len(families))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(methods):
        vals = [agg[(agg["family"] == f) & (agg["config_id"] == m)]["top1_accuracy"].mean()
                if not agg[(agg["family"] == f) & (agg["config_id"] == m)].empty else 0
                for f in families]
        ax.bar(x + i * width, vals, width, label=METHOD_DISPLAY.get(m, m),
               color=METHOD_COLOR.get(m, "#333"), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(families, fontsize=10)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy by Corruption Family", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "fig3_accuracy_by_family")


def fig4_ucb_distribution(df: pd.DataFrame):
    """UCB score distribution: correct vs wrong (UCDPA only)."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ucdpa = df[df["config_id"] == "ucdpa"]
    if ucdpa.empty:
        plt.close(fig)
        return
    ax.bar(["Correct\n(higher UCB)", "Wrong\n(lower UCB)"],
           [ucdpa["correct_ucb_score"].mean(), ucdpa["wrong_ucb_score"].mean()],
           yerr=[ucdpa["correct_ucb_score"].std(), ucdpa["wrong_ucb_score"].std()],
           color=["#33aa55", "#cc1133"], capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("UCB Score", fontsize=12)
    ax.set_title("UCB Score: Correct vs Wrong Predictions (UCDPA)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "fig4_ucb_distribution")


def fig5_auroc_comparison(df: pd.DataFrame):
    """Error detection AUROC comparison (grouped bars)."""
    if df.empty:
        return
    auroc_cols = ["auroc_neg_max_confidence", "auroc_entropy", "auroc_uncertainty",
                  "auroc_neg_mutual_info", "auroc_neg_ucb_score", "auroc_combined"]
    auroc_labels = ["Confidence", "Entropy", "Uncertainty", "Mutual Info", "UCB Score", "Combined"]
    methods = _ordered_methods(df)
    agg = df.groupby("config_id")[auroc_cols].mean()
    x = np.arange(len(auroc_cols))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(methods):
        vals = [agg.loc[m, c] if m in agg.index and c in agg.columns else 0 for c in auroc_cols]
        ax.bar(x + i * width, vals, width, label=METHOD_DISPLAY.get(m, m),
               color=METHOD_COLOR.get(m, "#333"), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(auroc_labels, fontsize=10)
    ax.set_ylabel("AUROC (error detection)", fontsize=12)
    ax.set_title("Error Detection AUROC by Score", fontsize=13)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "fig5_auroc_comparison")


def fig6_reliability_diagram(df: pd.DataFrame):
    """Reliability diagram: source vs UCDPA (accuracy by confidence bin)."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    methods_to_plot = [m for m in ["source", "ucdpa"] if m in df["config_id"].unique().tolist()]
    for m in methods_to_plot:
        sub = df[df["config_id"] == m]
        if sub.empty:
            continue
        ax.scatter(sub["mean_confidence"], sub["top1_accuracy"] / 100.0,
                   label=METHOD_DISPLAY.get(m, m), color=METHOD_COLOR.get(m, "#333"),
                   marker=METHOD_MARKER.get(m, "o"), s=30, alpha=0.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Mean Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Reliability: Confidence vs Accuracy", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _save(fig, "fig6_reliability_diagram")


def fig7_mi_vs_uncertainty(df: pd.DataFrame):
    """Mutual information vs uncertainty scatter (UCDPA, aggregated by run)."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in _ordered_methods(df):
        sub = df[df["config_id"] == m]
        if sub.empty:
            continue
        ax.scatter(sub["mean_uncertainty"], sub["mean_mutual_info"],
                   label=METHOD_DISPLAY.get(m, m), color=METHOD_COLOR.get(m, "#333"),
                   marker=METHOD_MARKER.get(m, "o"), s=20, alpha=0.5)
    ax.set_xlabel("Mean Uncertainty (u)", fontsize=12)
    ax.set_ylabel("Mean Mutual Information", fontsize=12)
    ax.set_title("Mutual Information vs Uncertainty", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, "fig7_mi_vs_uncertainty")


def main():
    print("=" * 60)
    print("UCDPA Figure Generation")
    print("=" * 60)
    df = load_summary()
    if df.empty:
        print("[WARN] No summary data. Run aggregate.py first.")
        return
    print(f"Loaded {len(df)} runs, methods: {sorted(df['config_id'].unique().tolist())}")

    fig1_accuracy_vs_ece(df)
    fig2_accuracy_by_severity(df)
    fig3_accuracy_by_family(df)
    fig4_ucb_distribution(df)
    fig5_auroc_comparison(df)
    fig6_reliability_diagram(df)
    fig7_mi_vs_uncertainty(df)

    print(f"\n[INFO] Figures complete. See {FIGURES_DIR}")


if __name__ == "__main__":
    main()
