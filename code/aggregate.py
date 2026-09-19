"""Aggregate per-run UCDPA summaries into summary CSVs and comparison tables.

Outputs (under results/):
  summaries/full_summary.csv          All per-run summaries (one row per run).
  tables/table1_main_comparison.csv   Main 7-method comparison (mean over seeds/corruptions/severities).
  tables/table2_by_severity.csv       Accuracy/calibration by severity (mean over corruptions/seeds).
  tables/table3_by_corruption.csv     Accuracy by corruption (mean over severities/seeds).
  tables/table4_calibration.csv       Calibration metrics (ECE, MCE, MI, HCE@0.9) by method.
  tables/table5_error_detection.csv   Error detection AUROC scores by method.

All tables are written in CSV, Markdown, and LaTeX formats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
SUMMARY_DIR = ROOT / "results" / "summaries"
TABLES_DIR = ROOT / "results" / "tables"

# Display order for the 7 methods.
METHOD_ORDER = ["source", "tent", "eata", "sar", "come", "tentcome", "ucdpa"]
METHOD_DISPLAY = {
    "source": "Source-only",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "come": "COME",
    "tentcome": "Tent-COME",
    "ucdpa": "UCDPA",
}

CORRUPTION_FAMILY = {
    "gaussian_noise": "Noise", "shot_noise": "Noise", "impulse_noise": "Noise", "speckle_noise": "Noise",
    "defocus_blur": "Blur", "glass_blur": "Blur", "motion_blur": "Blur", "zoom_blur": "Blur", "gaussian_blur": "Blur",
    "snow": "Weather", "frost": "Weather", "fog": "Weather", "brightness": "Weather",
    "contrast": "Digital", "elastic_transform": "Digital", "pixelate": "Digital", "jpeg_compression": "Digital",
    "saturate": "Other", "spatter": "Other",
}

METRIC_COLS = [
    "top1_accuracy", "ece_15bins_percent", "mce_15bins_percent",
    "mean_confidence", "mean_predictive_entropy", "mean_uncertainty",
    "mean_mutual_info", "mean_ucb_score", "mean_come_loss",
    "wrong_confidence", "correct_confidence", "confidence_separation",
    "wrong_uncertainty", "correct_uncertainty", "uncertainty_separation",
    "correct_mutual_info", "wrong_mutual_info", "mutual_info_separation",
    "correct_ucb_score", "wrong_ucb_score", "ucb_separation",
    "hce_rate@0.8", "hce_rate@0.9", "hce_rate@0.95",
    "hce_risk@0.8", "hce_risk@0.9", "hce_risk@0.95",
    "aurc", "e_aurc", "aurc_uncertainty", "aurc_ucb_score",
    "selective_risk@0.80", "selective_risk@0.90", "selective_risk@0.95",
    "auroc_neg_max_confidence", "auroc_entropy", "auroc_neg_margin",
    "auroc_uncertainty", "auroc_neg_mutual_info", "auroc_neg_ucb_score",
    "auroc_combined", "auroc_error_combined",
    "runtime_seconds", "peak_gpu_memory_mb", "optimizer_steps",
]

COLUMN_DISPLAY = {
    "config_id": "Method",
    "method": "Method",
    "top1_accuracy": "Acc(%)",
    "ece_15bins_percent": "ECE(%)",
    "mce_15bins_percent": "MCE(%)",
    "mean_confidence": "Mean Conf",
    "mean_predictive_entropy": "Mean Entropy",
    "mean_uncertainty": "Mean Unc",
    "mean_mutual_info": "Mean MI",
    "mean_ucb_score": "Mean UCB",
    "hce_rate@0.9": "HCE@0.9",
    "aurc": "AURC",
    "e_aurc": "E-AURC",
    "selective_risk@0.90": "Risk@90",
    "auroc_combined": "AUROC_comb",
    "auroc_uncertainty": "AUROC_unc",
    "auroc_neg_ucb_score": "AUROC_ucb",
    "severity": "Severity",
    "family": "Family",
    "corruption": "Corruption",
}


def load_summaries() -> pd.DataFrame:
    """Load all per-run summary CSVs from results/summaries/."""
    rows = []
    if not SUMMARY_DIR.exists():
        print(f"[WARN] Summary directory not found: {SUMMARY_DIR}")
        return pd.DataFrame()
    for p in sorted(SUMMARY_DIR.glob("*.csv")):
        if p.name == "full_summary.csv":
            continue
        if p.name.endswith("_config.json") or "_config.json" in p.name:
            continue
        try:
            d = pd.read_csv(p)
            if len(d):
                rows.append(d.iloc[0].to_dict())
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Save the full summary
    SUMMARY_DIR.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results" / "summaries" / "full_summary.csv", index=False)
    print(f"[INFO] Loaded {len(df)} per-run summaries -> full_summary.csv")
    return df


def _agg(df: pd.DataFrame, group_cols: list, metrics: list) -> pd.DataFrame:
    """Aggregate metrics by group_cols, computing mean and std."""
    available = [c for c in metrics if c in df.columns]
    agg_df = df.groupby(group_cols)[available].agg(["mean", "std"]).reset_index()
    # Flatten multi-index columns
    new_cols = []
    for col in agg_df.columns:
        if isinstance(col, tuple):
            base, stat = col
            new_cols.append(f"{base}_{stat}" if stat else base)
        else:
            new_cols.append(col)
    agg_df.columns = new_cols
    return agg_df


def _write_table(df: pd.DataFrame, name: str, display_map: dict = None):
    """Write a table to CSV, Markdown, and LaTeX."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    # Markdown
    try:
        disp = df.copy()
        if display_map:
            disp = disp.rename(columns=display_map)
        disp.to_markdown(TABLES_DIR / f"{name}.md", index=False)
    except Exception:
        pass
    # LaTeX
    try:
        disp = df.copy()
        if display_map:
            disp = disp.rename(columns=display_map)
        with open(TABLES_DIR / f"{name}.tex", "w") as f:
            f.write(disp.to_latex(index=False, float_format="%.2f"))
    except Exception:
        pass
    print(f"[INFO] Wrote {name} ({len(df)} rows)")


def table1_main_comparison(df: pd.DataFrame):
    """Table 1: Main 7-method comparison (mean +/- std over all runs)."""
    if df.empty:
        return
    metrics = ["top1_accuracy", "ece_15bins_percent", "mce_15bins_percent",
               "mean_mutual_info", "mean_ucb_score", "hce_rate@0.9",
               "aurc", "e_aurc", "auroc_combined", "mean_predictive_entropy"]
    agg = _agg(df, ["config_id"], metrics)
    # Order by method
    agg["__order"] = agg["config_id"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)
    agg = agg.sort_values("__order").drop(columns=["__order"])
    _write_table(agg, "table1_main_comparison", COLUMN_DISPLAY)


def table2_by_severity(df: pd.DataFrame):
    """Table 2: Accuracy and calibration by severity."""
    if df.empty:
        return
    metrics = ["top1_accuracy", "ece_15bins_percent", "mce_15bins_percent",
               "mean_mutual_info", "hce_rate@0.9", "auroc_combined"]
    agg = _agg(df, ["config_id", "severity"], metrics)
    agg["__order"] = agg["config_id"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)
    agg = agg.sort_values(["severity", "__order"]).drop(columns=["__order"])
    _write_table(agg, "table2_by_severity", COLUMN_DISPLAY)


def table3_by_corruption(df: pd.DataFrame):
    """Table 3: Accuracy by corruption (mean over severities and seeds)."""
    if df.empty:
        return
    agg = _agg(df, ["config_id", "corruption"], ["top1_accuracy"])
    # Pivot: rows=corruption, cols=method
    pivot = agg.pivot_table(index="corruption", columns="config_id", values="top1_accuracy_mean")
    pivot = pivot.reindex(columns=[m for m in METHOD_ORDER if m in pivot.columns])
    pivot.reset_index().to_csv(TABLES_DIR / "table3_by_corruption.csv", index=False)
    print(f"[INFO] Wrote table3_by_corruption ({len(pivot)} corruptions)")


def table4_by_family(df: pd.DataFrame):
    """Table 4: Accuracy by corruption family."""
    if df.empty:
        return
    df = df.copy()
    df["family"] = df["corruption"].map(CORRUPTION_FAMILY).fillna("Other")
    agg = _agg(df, ["config_id", "family"], ["top1_accuracy", "ece_15bins_percent", "mean_mutual_info"])
    agg["__order"] = agg["config_id"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)
    agg = agg.sort_values(["family", "__order"]).drop(columns=["__order"])
    _write_table(agg, "table4_by_family", COLUMN_DISPLAY)


def table5_error_detection(df: pd.DataFrame):
    """Table 5: Error detection AUROC by method (mean over all runs)."""
    if df.empty:
        return
    metrics = ["auroc_neg_max_confidence", "auroc_entropy", "auroc_neg_margin",
               "auroc_uncertainty", "auroc_neg_mutual_info", "auroc_neg_ucb_score",
               "auroc_combined"]
    agg = _agg(df, ["config_id"], metrics)
    agg["__order"] = agg["config_id"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)
    agg = agg.sort_values("__order").drop(columns=["__order"])
    _write_table(agg, "table5_error_detection", COLUMN_DISPLAY)


def table6_ucb_analysis(df: pd.DataFrame):
    """Table 6: UCB score analysis (correct vs wrong separation)."""
    if df.empty:
        return
    metrics = ["correct_ucb_score", "wrong_ucb_score", "ucb_separation",
               "correct_mutual_info", "wrong_mutual_info", "mutual_info_separation",
               "correct_uncertainty", "wrong_uncertainty", "uncertainty_separation"]
    agg = _agg(df, ["config_id"], metrics)
    agg["__order"] = agg["config_id"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)
    agg = agg.sort_values("__order").drop(columns=["__order"])
    _write_table(agg, "table6_ucb_analysis", COLUMN_DISPLAY)


def table7_cross_backbone(df: pd.DataFrame):
    """Table 7: Cross-backbone comparison (ViT-B/16 vs ResNet-50 vs CIFAR-ResNet-18).

    Filters to severity 3, seed 0 for fair comparison across backbones.
    """
    if df.empty:
        return
    # Detect backbone from config_id suffix
    df = df.copy()
    df["backbone"] = "vit_base_patch16_224"  # default
    df.loc[df["config_id"].str.contains("_resnet50", na=False), "backbone"] = "resnet50"
    df.loc[df["config_id"].str.contains("_cifar100", na=False), "backbone"] = "cifar_resnet18"
    # Extract base method name
    df["base_method"] = df["config_id"].str.replace("_resnet50", "", regex=False)
    df["base_method"] = df["base_method"].str.replace("_cifar100", "", regex=False)
    # Filter to severity 3, seed 0 for cross-backbone comparison
    cross = df[(df["severity"] == 3) & (df["seed"] == 0)]
    if cross.empty:
        print("[INFO] table7_cross_backbone: no severity-3 seed-0 runs found")
        return
    metrics = ["top1_accuracy", "ece_15bins_percent", "mce_15bins_percent",
               "mean_mutual_info", "mean_ucb_score", "auroc_combined"]
    agg = _agg(cross, ["backbone", "base_method"], metrics)
    agg["__order"] = agg["base_method"].apply(
        lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)
    agg = agg.sort_values(["backbone", "__order"]).drop(columns=["__order"])
    _write_table(agg, "table7_cross_backbone", COLUMN_DISPLAY)


def table8_sensitivity(df: pd.DataFrame):
    """Table 8: Sensitivity analysis for UCB routing hyperparameters.

    Aggregates ucdpa_sens_* runs by hyperparameter and value.
    """
    if df.empty:
        return
    sens = df[df["config_id"].str.startswith("ucdpa_sens_", na=False)].copy()
    if sens.empty:
        print("[INFO] table8_sensitivity: no sensitivity runs found")
        return
    # Parse hyperparameter name and value from config_id
    # Format: ucdpa_sens_{param}_{value}
    def parse_sens(cid):
        parts = cid.split("_")
        if len(parts) >= 4:
            param = parts[2]
            val = parts[3]
            return pd.Series({"param": param, "value": val})
        return pd.Series({"param": "unknown", "value": "0"})
    sens[["param", "value"]] = sens["config_id"].apply(parse_sens)
    metrics = ["top1_accuracy", "ece_15bins_percent", "mean_ucb_score",
               "mean_mutual_info", "auroc_combined"]
    agg = _agg(sens, ["param", "value"], metrics)
    agg = agg.sort_values(["param", "value"])
    _write_table(agg, "table8_sensitivity", {"param": "Hyperparam", "value": "Value",
                                              **COLUMN_DISPLAY})


def table9_batch_size(df: pd.DataFrame):
    """Table 9: Batch size sensitivity.

    Aggregates ucdpa_bs* runs by batch size.
    """
    if df.empty:
        return
    bs = df[df["config_id"].str.startswith("ucdpa_bs", na=False)].copy()
    if bs.empty:
        print("[INFO] table9_batch_size: no batch-size runs found")
        return
    # Extract batch size from config_id (ucdpa_bs1, ucdpa_bs8, etc.)
    bs["batch_size_val"] = bs["config_id"].str.extract(r"ucdpa_bs(\d+)").astype(int)
    metrics = ["top1_accuracy", "ece_15bins_percent", "mce_15bins_percent",
               "mean_ucb_score", "auroc_combined", "runtime_seconds"]
    agg = _agg(bs, ["batch_size_val"], metrics)
    agg = agg.sort_values("batch_size_val")
    _write_table(agg, "table9_batch_size", {"batch_size_val": "BatchSize", **COLUMN_DISPLAY})


def table10_scale_norm_ablation(df: pd.DataFrame):
    """Table 10: Scale normalization ablation (ucdpa vs ucdpa_noscalenorm).

    Compares full UCDPA with and without scale normalization, seed 0.
    """
    if df.empty:
        return
    # Get ucdpa (seed 0) and ucdpa_noscalenorm
    variants = df[df["config_id"].isin(["ucdpa", "ucdpa_noscalenorm"])].copy()
    if variants.empty or "ucdpa_noscalenorm" not in variants["config_id"].values:
        print("[INFO] table10_scale_norm: no noscalenorm runs found")
        return
    variants = variants[variants["seed"] == 0]
    metrics = ["top1_accuracy", "ece_15bins_percent", "mce_15bins_percent",
               "mean_ucb_score", "mean_mutual_info", "auroc_combined"]
    agg = _agg(variants, ["config_id"], metrics)
    _write_table(agg, "table10_scale_norm_ablation", COLUMN_DISPLAY)


def main():
    print("=" * 60)
    print("UCDPA Results Aggregation")
    print("=" * 60)
    df = load_summaries()
    if df.empty:
        print("[WARN] No summaries found. Run experiments first.")
        return
    print(f"\nMethods found: {sorted(df['config_id'].unique().tolist())}")
    print(f"Corruptions: {df['corruption'].nunique()}, Severities: {sorted(df['severity'].unique().tolist())}")
    print(f"Seeds: {sorted(df['seed'].unique().tolist())}, Total runs: {len(df)}")

    # Main tables (7-method comparison on ImageNet-C ViT)
    main_df = df[~df["config_id"].str.contains("_resnet50|_cifar100|ucdpa_sens_|ucdpa_bs|ucdpa_noscalenorm", na=False, regex=True)]
    table1_main_comparison(main_df)
    table2_by_severity(main_df)
    table3_by_corruption(main_df)
    table4_by_family(main_df)
    table5_error_detection(main_df)
    table6_ucb_analysis(main_df)

    # New supplementary tables
    table7_cross_backbone(df)
    table8_sensitivity(df)
    table9_batch_size(df)
    table10_scale_norm_ablation(df)

    print(f"\n[INFO] Aggregation complete. Tables in {TABLES_DIR}")


if __name__ == "__main__":
    main()
