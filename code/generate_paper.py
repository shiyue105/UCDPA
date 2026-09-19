"""Generate the UCDPA paper by filling TBD placeholder macros and recompiling.

Workflow
--------
1. Read aggregated results from results/summaries/full_summary.csv
2. Compute mean metrics per method (averaged over seeds, corruptions, severities)
3. Compute per-severity and per-corruption-family breakdowns for UCDPA and COME
4. Compute ablation metrics (if ablation runs exist)
5. Compute bootstrap confidence intervals for UCDPA / COME accuracy
6. Compute paired t-test p-value (UCDPA vs COME)
7. Fill TBD placeholder macros in paper/main.tex with actual values
8. Recompile the paper with pdflatex + bibtex

Usage:
    python3 generate_paper.py
    python3 generate_paper.py --no-compile      # fill macros only, skip LaTeX
    python3 generate_paper.py --seed 42         # bootstrap RNG seed
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
ROOT = Path("/data/yuehan/outputs/ucdpa_tta")
SUMMARY_CSV = ROOT / "results" / "summaries" / "full_summary.csv"
MAIN_TEX = ROOT / "paper" / "main.tex"
PAPER_DIR = ROOT / "paper"

METHOD_ORDER = ["source", "tent", "eata", "sar", "come", "tentcome", "ucdpa"]

# Macro name suffix per method (config_id -> macro suffix used in main.tex).
METHOD_MACRO_SUFFIX = {
    "source": "source",
    "tent": "tent",
    "eata": "eata",
    "sar": "sar",
    "come": "come",
    "tentcome": "tentcome",
    "ucdpa": "ucdpa",
}

CORRUPTION_FAMILY = {
    "gaussian_noise": "Noise", "shot_noise": "Noise",
    "impulse_noise": "Noise", "speckle_noise": "Noise",
    "defocus_blur": "Blur", "glass_blur": "Blur",
    "motion_blur": "Blur", "zoom_blur": "Blur", "gaussian_blur": "Blur",
    "snow": "Weather", "frost": "Weather", "fog": "Weather", "brightness": "Weather",
    "contrast": "Digital", "elastic_transform": "Digital",
    "pixelate": "Digital", "jpeg_compression": "Digital",
    "saturate": "Other", "spatter": "Other",
}

# Family -> macro suffix used in main.tex.
FAMILY_MACRO_SUFFIX = {
    "Noise": "noise",
    "Blur": "blur",
    "Weather": "weather",
    "Digital": "digital",
    "Other": "extra",
}

SEVERITY_MACRO = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
}

# Ablation variant config_id substrings to detect (config_id -> macro).
ABLATION_VARIANTS = {
    "noucb": "ablationnoucb",
    "nocal": "ablationnocal",
    "nocb": "ablationnocb",
    "nofpa": "ablationnofpa",
}

BOOTSTRAP_RESAMPLES = 10000


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_acc(x: float) -> str:
    """Accuracy to 2 decimal places."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "TBD"
    return f"{float(x):.2f}"


def fmt_delta(x: float) -> str:
    """Delta to 2 decimal places with explicit sign."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "TBD"
    return f"{float(x):+.2f}"


def fmt_ece(x: float) -> str:
    """ECE/MCE to 2 decimal places."""
    return fmt_acc(x)


def fmt_auroc(x: float) -> str:
    """AUROC to 2 decimal places, multiplying by 100 if in [0, 1]."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "TBD"
    v = float(x)
    if 0.0 <= v <= 1.0:
        v = v * 100.0
    return f"{v:.2f}"


def fmt_mi(x: float) -> str:
    """Mutual information to 4 decimal places."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "TBD"
    return f"{float(x):.4f}"


def fmt_pe(x: float) -> str:
    """Predictive entropy to 4 decimal places."""
    return fmt_mi(x)


def fmt_pvalue(x: float) -> str:
    """p-value to 4 decimal places."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "TBD"
    return f"{float(x):.4f}"


def fmt_ci(x: float) -> str:
    """Confidence interval bound to 2 decimal places."""
    return fmt_acc(x)


def fmt_ucb(x: float) -> str:
    """UCB score to 4 decimal places."""
    return fmt_mi(x)


def fmt_time(x: float) -> str:
    """Per-batch runtime to 4 decimal places."""
    return fmt_mi(x)


def fmt_memory(x: float) -> str:
    """Peak GPU memory (MB) to 2 decimal places."""
    return fmt_acc(x)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_summary() -> pd.DataFrame:
    """Load the aggregated full_summary.csv."""
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {SUMMARY_CSV}\n"
            "Run aggregate.py first (or wait for experiments to finish)."
        )
    df = pd.read_csv(SUMMARY_CSV)
    print(f"[INFO] Loaded {len(df)} rows from {SUMMARY_CSV}")
    print(f"[INFO] Methods: {sorted(df['config_id'].unique().tolist())}")
    if "severity" in df.columns:
        print(f"[INFO] Severities: {sorted(df['severity'].unique().tolist())}")
    if "corruption" in df.columns:
        print(f"[INFO] Corruptions: {df['corruption'].nunique()}")
    if "seed" in df.columns:
        print(f"[INFO] Seeds: {sorted(df['seed'].unique().tolist())}")
    return df


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the column if present, else a NaN-filled series."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index, name=col)


# --------------------------------------------------------------------------- #
# Per-method aggregate metrics
# --------------------------------------------------------------------------- #
def method_mean(df: pd.DataFrame, config_id: str, col: str) -> float:
    """Mean of ``col`` for the given method, ignoring NaNs."""
    sub = df.loc[df["config_id"] == config_id, col]
    sub = pd.to_numeric(sub, errors="coerce").dropna()
    if sub.empty:
        return float("nan")
    return float(sub.mean())


def compute_main_metrics(df: pd.DataFrame) -> Dict[str, str]:
    """Compute the main per-method metrics -> macro name -> formatted value."""
    macros: Dict[str, str] = {}

    acc_by_method: Dict[str, float] = {}
    for cid, suffix in METHOD_MACRO_SUFFIX.items():
        acc = method_mean(df, cid, "top1_accuracy")
        acc_by_method[cid] = acc
        macros[f"{suffix}acc"] = fmt_acc(acc)

        macros[f"{suffix}ece"] = fmt_ece(method_mean(df, cid, "ece_15bins_percent"))
        macros[f"{suffix}mce"] = fmt_ece(method_mean(df, cid, "mce_15bins_percent"))
        macros[f"{suffix}auroc"] = fmt_auroc(method_mean(df, cid, "auroc_combined"))
        macros[f"{suffix}mi"] = fmt_mi(method_mean(df, cid, "mean_mutual_info"))
        macros[f"{suffix}pe"] = fmt_pe(method_mean(df, cid, "mean_predictive_entropy"))

    # Delta: UCDPA accuracy - COME accuracy.
    delta = acc_by_method.get("ucdpa", float("nan")) - acc_by_method.get("come", float("nan"))
    macros["ucdpadelta"] = fmt_delta(delta)

    return macros


# --------------------------------------------------------------------------- #
# Per-severity breakdowns (UCDPA and COME)
# --------------------------------------------------------------------------- #
def compute_severity_breakdown(df: pd.DataFrame) -> Dict[str, str]:
    """Per-severity accuracy for UCDPA and COME (averaged over corruptions/seeds)."""
    macros: Dict[str, str] = {}
    for cid in ("ucdpa", "come"):
        suffix = METHOD_MACRO_SUFFIX[cid]
        sub = df[df["config_id"] == cid]
        for sev in (1, 2, 3, 4, 5):
            sev_rows = sub.loc[sub["severity"] == sev, "top1_accuracy"]
            sev_rows = pd.to_numeric(sev_rows, errors="coerce").dropna()
            macro_name = f"{suffix}sev{SEVERITY_MACRO[sev]}"
            macros[macro_name] = fmt_acc(float(sev_rows.mean()) if not sev_rows.empty else float("nan"))
    return macros


# --------------------------------------------------------------------------- #
# Per-corruption-family breakdowns (UCDPA and COME)
# --------------------------------------------------------------------------- #
def compute_family_breakdown(df: pd.DataFrame) -> Dict[str, str]:
    """Per-corruption-family accuracy for UCDPA and COME (averaged over corruptions/seeds/severities)."""
    macros: Dict[str, str] = {}
    df = df.copy()
    df["family"] = df["corruption"].map(CORRUPTION_FAMILY).fillna("Other")
    for cid in ("ucdpa", "come"):
        suffix = METHOD_MACRO_SUFFIX[cid]
        sub = df[df["config_id"] == cid]
        for family, fam_suffix in FAMILY_MACRO_SUFFIX.items():
            fam_rows = sub.loc[sub["family"] == family, "top1_accuracy"]
            fam_rows = pd.to_numeric(fam_rows, errors="coerce").dropna()
            macro_name = f"{suffix}{fam_suffix}"
            macros[macro_name] = fmt_acc(float(fam_rows.mean()) if not fam_rows.empty else float("nan"))
    return macros


# --------------------------------------------------------------------------- #
# UCB score analysis (UCDPA only)
# --------------------------------------------------------------------------- #
def compute_ucb_analysis(df: pd.DataFrame) -> Dict[str, str]:
    """UCB score analysis macros for UCDPA."""
    macros: Dict[str, str] = {}
    sub = df[df["config_id"] == "ucdpa"]

    correct_ucb = pd.to_numeric(sub.get("correct_ucb_score"), errors="coerce").dropna()
    wrong_ucb = pd.to_numeric(sub.get("wrong_ucb_score"), errors="coerce").dropna()
    auroc_ucb = pd.to_numeric(sub.get("auroc_neg_ucb_score"), errors="coerce").dropna()

    mean_certain = float(correct_ucb.mean()) if not correct_ucb.empty else float("nan")
    mean_uncertain = float(wrong_ucb.mean()) if not wrong_ucb.empty else float("nan")
    separation = mean_certain - mean_uncertain

    macros["ucbmeancertain"] = fmt_ucb(mean_certain)
    macros["ucbmeanuncertain"] = fmt_ucb(mean_uncertain)
    macros["ucbauroc"] = fmt_auroc(float(auroc_ucb.mean()) if not auroc_ucb.empty else float("nan"))
    macros["ucbseparation"] = fmt_ucb(separation)
    return macros


# --------------------------------------------------------------------------- #
# Efficiency metrics
# --------------------------------------------------------------------------- #
def compute_efficiency(df: pd.DataFrame) -> Dict[str, str]:
    """Per-batch runtime and peak GPU memory for UCDPA and COME."""
    macros: Dict[str, str] = {}
    for cid in ("ucdpa", "come"):
        suffix = METHOD_MACRO_SUFFIX[cid]
        sub = df[df["config_id"] == cid]
        runtime = pd.to_numeric(sub.get("runtime_seconds"), errors="coerce")
        steps = pd.to_numeric(sub.get("optimizer_steps"), errors="coerce")
        # Per-batch time = runtime / optimizer_steps, then averaged over runs.
        valid = (runtime > 0) & (steps > 0)
        if valid.any():
            per_batch = (runtime[valid] / steps[valid]).mean()
            macros[f"{suffix}timeperbatch"] = fmt_time(float(per_batch))
        else:
            macros[f"{suffix}timeperbatch"] = "TBD"

    # Peak GPU memory (UCDPA only).
    sub = df[df["config_id"] == "ucdpa"]
    mem = pd.to_numeric(sub.get("peak_gpu_memory_mb"), errors="coerce").dropna()
    macros["ucdpamemory"] = fmt_memory(float(mem.mean()) if not mem.empty else float("nan"))
    macros.setdefault("cometimeperbatch", "TBD")
    return macros


# --------------------------------------------------------------------------- #
# Ablation study
# --------------------------------------------------------------------------- #
def compute_ablation(df: pd.DataFrame) -> Dict[str, str]:
    """Ablation metrics. Full UCDPA is the ``ucdpa`` config_id.

    Ablation variants are detected by config_id substrings (e.g. ``ucdpa_noucb``).
    If no ablation runs are present the macros are left as TBD.

    Note: ablations are run on seed 0 only (95 runs each). For a fair
    apples-to-apples comparison, the full-UCDPA baseline is also restricted
    to seed 0. Additionally, since the default config has ``lambda_fpa=0.0``
    (FPA disabled), the ``nofpa`` ablation is identical to the full model,
    so ``ablationnofpa`` is set equal to ``ablationfull``.
    """
    macros: Dict[str, str] = {}

    # Restrict to seed 0 for a fair comparison with the seed-0-only ablations.
    if "seed" in df.columns:
        df_seed0 = df[df["seed"] == 0]
    else:
        df_seed0 = df

    # Full model == plain UCDPA mean accuracy on seed 0.
    full_sub = pd.to_numeric(
        df_seed0.loc[df_seed0["config_id"] == "ucdpa", "top1_accuracy"],
        errors="coerce",
    ).dropna()
    full_acc = float(full_sub.mean()) if not full_sub.empty else float("nan")
    macros["ablationfull"] = fmt_acc(full_acc)

    all_cids = set(df_seed0["config_id"].unique().tolist())
    for substr, macro_name in ABLATION_VARIANTS.items():
        if substr == "nofpa":
            # FPA is disabled (lambda_fpa=0.0) in the default config, so the
            # no-FPA ablation is identical to the full model.
            macros[macro_name] = fmt_acc(full_acc)
            continue
        # Find config_ids containing the variant substring (and not the full model).
        matches = [c for c in all_cids if substr in str(c).lower() and c != "ucdpa"]
        if not matches:
            macros[macro_name] = "TBD"
            continue
        # Average accuracy over all matching ablation config_ids (seed 0 only).
        vals = []
        for c in matches:
            v = pd.to_numeric(
                df_seed0.loc[df_seed0["config_id"] == c, "top1_accuracy"],
                errors="coerce",
            ).dropna()
            if not v.empty:
                vals.extend(v.tolist())
        macros[macro_name] = fmt_acc(float(np.mean(vals)) if vals else float("nan"))
    return macros


# --------------------------------------------------------------------------- #
# Paired tests (UCDPA vs COME)
# --------------------------------------------------------------------------- #
def _paired_accuracy(df: pd.DataFrame, a: str, b: str) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """Return paired per-run accuracies for methods a and b, matched by
    (corruption, severity, seed). Also returns the matched key list."""
    keys = ["corruption", "severity", "seed"]
    da = df[df["config_id"] == a].set_index(keys)["top1_accuracy"]
    db = df[df["config_id"] == b].set_index(keys)["top1_accuracy"]
    common = da.index.intersection(db.index)
    va = pd.to_numeric(da.loc[common], errors="coerce").to_numpy(dtype=float)
    vb = pd.to_numeric(db.loc[common], errors="coerce").to_numpy(dtype=float)
    mask = ~(np.isnan(va) | np.isnan(vb))
    return va[mask], vb[mask], list(common[mask])


def compute_pvalue(df: pd.DataFrame) -> Dict[str, str]:
    """Paired t-test p-value for UCDPA vs COME accuracies."""
    macros: Dict[str, str] = {}
    try:
        from scipy import stats
    except Exception as e:  # pragma: no cover
        print(f"[WARN] scipy not available, cannot compute p-value: {e}")
        macros["pvalue"] = "TBD"
        return macros

    va, vb, _ = _paired_accuracy(df, "ucdpa", "come")
    if len(va) < 2 or np.allclose(va, vb):
        # Fallback to Wilcoxon if there is no variation, else t-test.
        try:
            if len(va) >= 2 and not np.allclose(va - vb, 0):
                w_stat = stats.wilcoxon(va, vb)
                p = float(w_stat.pvalue)
            else:
                p = float("nan")
        except Exception:
            p = float("nan")
        macros["pvalue"] = fmt_pvalue(p)
        return macros

    t_stat, t_p = stats.ttest_rel(va, vb)
    p = float(t_p)
    # If t-test degenerate, fall back to Wilcoxon signed-rank.
    if np.isnan(p):
        try:
            w_stat = stats.wilcoxon(va, vb)
            p = float(w_stat.pvalue)
        except Exception:
            p = float("nan")
    macros["pvalue"] = fmt_pvalue(p)
    return macros


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def compute_bootstrap_ci(df: pd.DataFrame, seed: int = 0) -> Dict[str, str]:
    """Bootstrap CIs for UCDPA and COME mean accuracy.

    Resample (corruption, severity, seed) units with replacement
    ``BOOTSTRAP_RESAMPLES`` times. For each resample compute the mean UCDPA and
    COME accuracy. Report 2.5% / 97.5% percentiles per method.
    """
    macros: Dict[str, str] = {}
    rng = np.random.default_rng(seed)

    va, vb, keys = _paired_accuracy(df, "ucdpa", "come")
    n = len(va)
    if n == 0:
        print("[WARN] No paired UCDPA/COME runs for bootstrap CI.")
        for m in ("ucdpacilow", "ucdpacihigh", "comecilow", "comecihigh"):
            macros[m] = "TBD"
        return macros

    va = np.asarray(va, dtype=float)
    vb = np.asarray(vb, dtype=float)
    n_boot = BOOTSTRAP_RESAMPLES

    boot_u = np.empty(n_boot, dtype=float)
    boot_c = np.empty(n_boot, dtype=float)
    boot_delta = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_u[i] = va[idx].mean()
        boot_c[i] = vb[idx].mean()
        boot_delta[i] = boot_u[i] - boot_c[i]

    u_lo, u_hi = np.percentile(boot_u, [2.5, 97.5])
    c_lo, c_hi = np.percentile(boot_c, [2.5, 97.5])
    d_lo, d_hi = np.percentile(boot_delta, [2.5, 97.5])

    macros["ucdpacilow"] = fmt_ci(float(u_lo))
    macros["ucdpacihigh"] = fmt_ci(float(u_hi))
    macros["comecilow"] = fmt_ci(float(c_lo))
    macros["comecihigh"] = fmt_ci(float(c_hi))

    print(f"[INFO] Bootstrap (n={n} pairs, {n_boot} resamples):")
    print(f"        UCDPA acc CI = [{u_lo:.3f}, {u_hi:.3f}]")
    print(f"        COME  acc CI = [{c_lo:.3f}, {c_hi:.3f}]")
    print(f"        Delta  CI    = [{d_lo:.3f}, {d_hi:.3f}]")
    return macros


# --------------------------------------------------------------------------- #
# LaTeX macro substitution
# --------------------------------------------------------------------------- #
_MACRO_RE_CACHE: Dict[str, re.Pattern] = {}


def _macro_regex(macro: str) -> re.Pattern:
    if macro not in _MACRO_RE_CACHE:
        # Matches \newcommand{\macro}{TBD} allowing arbitrary whitespace.
        _MACRO_RE_CACHE[macro] = re.compile(
            r"(\\newcommand\{\\"
            + re.escape(macro)
            + r"\}\{)[^}]*?(\})"
        )
    return _MACRO_RE_CACHE[macro]


def replace_macro(tex: str, macro: str, value: str) -> Tuple[str, bool]:
    """Replace the body of ``\\newcommand{\\macro}{TBD}`` with ``value``.

    Returns the updated tex and whether a replacement occurred.
    """
    pattern = _macro_regex(macro)
    new_tex, n = pattern.subn(r"\g<1>" + value + r"\g<2>", tex, count=1)
    return new_tex, (n > 0)


def fill_macros(tex: str, macros: Dict[str, str]) -> Tuple[str, List[str], List[str]]:
    """Fill all macros into the tex string.

    Returns (updated_tex, filled_macros, missing_macros).
    """
    filled: List[str] = []
    missing: List[str] = []
    for macro, value in macros.items():
        new_tex, ok = replace_macro(tex, macro, str(value))
        if ok:
            tex = new_tex
            filled.append(macro)
        else:
            missing.append(macro)
    return tex, filled, missing


# --------------------------------------------------------------------------- #
# LaTeX compilation
# --------------------------------------------------------------------------- #
def compile_paper() -> bool:
    """Run pdflatex + bibtex + pdflatex x2 in the paper directory."""
    if not PAPER_DIR.exists():
        print(f"[WARN] Paper directory not found: {PAPER_DIR}")
        return False
    print("\n[INFO] Compiling paper (pdflatex + bibtex + pdflatex x2) ...")
    steps = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    ok = True
    for cmd in steps:
        try:
            proc = subprocess.run(
                cmd, cwd=str(PAPER_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
            print(f"  [{' '.join(cmd)}] -> {status}")
            if proc.returncode != 0:
                # Print the tail of the log to help diagnose.
                tail = "\n".join(proc.stdout.splitlines()[-25:])
                if tail.strip():
                    print("  --- log tail ---")
                    print("  " + tail.replace("\n", "\n  "))
                ok = False
        except FileNotFoundError as e:
            print(f"  [{' '.join(cmd)}] -> NOT FOUND: {e}")
            ok = False
    pdf = PAPER_DIR / "main.pdf"
    if pdf.exists():
        print(f"[INFO] PDF written: {pdf}")
    else:
        print(f"[WARN] PDF not found after compile: {pdf}")
        ok = False
    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Fill UCDPA paper macros and recompile.")
    parser.add_argument("--no-compile", action="store_true",
                        help="Fill macros but skip LaTeX compilation.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the bootstrap resampling.")
    args = parser.parse_args()

    print("=" * 70)
    print("UCDPA generate_paper.py")
    print("=" * 70)

    df = load_summary()

    # Compute all macro values.
    macros: Dict[str, str] = {}
    macros.update(compute_main_metrics(df))
    macros.update(compute_severity_breakdown(df))
    macros.update(compute_family_breakdown(df))
    macros.update(compute_ucb_analysis(df))
    macros.update(compute_efficiency(df))
    macros.update(compute_ablation(df))
    macros.update(compute_pvalue(df))
    macros.update(compute_bootstrap_ci(df, seed=args.seed))

    # Read main.tex.
    if not MAIN_TEX.exists():
        raise FileNotFoundError(f"main.tex not found: {MAIN_TEX}")
    tex = MAIN_TEX.read_text()

    # Fill macros.
    new_tex, filled, missing = fill_macros(tex, macros)
    MAIN_TEX.write_text(new_tex)
    print(f"\n[INFO] Filled {len(filled)} macros in {MAIN_TEX}")
    if missing:
        print(f"[WARN] {len(missing)} macros not found in main.tex: {missing}")

    # Print a summary table of all filled values.
    print("\n" + "=" * 70)
    print("SUMMARY OF FILLED MACROS")
    print("=" * 70)
    print(f"{'Macro':<22} {'Value':<18}")
    print("-" * 40)
    for macro in macros:
        # Only print macros that we attempted to fill.
        value = macros[macro]
        print(f"\\{macro:<21} {value:<18}")

    # Compile.
    if args.no_compile:
        print("\n[INFO] --no-compile set; skipping LaTeX compilation.")
    else:
        compile_paper()

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
