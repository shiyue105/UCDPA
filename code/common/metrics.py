"""Full metric suite for UCDPA Test-Time Adaptation evaluation.

Covers accuracy/calibration (ECE, MCE), predictive entropy, mutual information
(Dirichlet-based), wrong-prediction reliability, high-confidence error (HCE),
selective prediction (AURC / E-AURC / selective risk), and multi-score error
detection AUROC.

The predictive entropy H(p) = -sum p_k log p_k is the softmax entropy. The
mutual information for a Dirichlet opinion is MI = H(u/K) - E[H(b_k)], where
H(u/K) = -K * (u/K) log(u/K) is the entropy of the (uniform) expected mean and
E[H(b_k)] is the mean belief entropy; higher MI => less epistemic uncertainty.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _safe_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def _safe_std(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if x.size else float("nan")


def _roc_auc(y_true, score) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    good = np.isfinite(score) & np.isfinite(y_true.astype(float))
    y_true, score = y_true[good], score[good]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, score))
    except Exception:
        pos = score[y_true == 1]
        neg = score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        total = float(np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :]))
        return total / (len(pos) * len(neg))


def ece_percent(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (percent)."""
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    good = np.isfinite(confidence) & np.isfinite(correct)
    correct, confidence = correct[good], confidence[good]
    if confidence.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence <= hi) if hi == 1.0 else (confidence >= lo) & (confidence < hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece * 100.0)


def mce_percent(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error (percent): the largest per-bin |acc - conf|."""
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    good = np.isfinite(confidence) & np.isfinite(correct)
    correct, confidence = correct[good], confidence[good]
    if confidence.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence <= hi) if hi == 1.0 else (confidence >= lo) & (confidence < hi)
        if mask.any():
            mce = max(mce, abs(correct[mask].mean() - confidence[mask].mean()))
    return float(mce * 100.0)


def hce_rate(correct: np.ndarray, confidence: np.ndarray, c: float) -> float:
    """Fraction of all samples that are wrong AND high-confidence (>=c)."""
    correct = np.asarray(correct)
    confidence = np.asarray(confidence)
    good = np.isfinite(confidence)
    correct, confidence = correct[good], confidence[good]
    if confidence.size == 0:
        return float("nan")
    wrong = correct == 0
    return float((wrong & (confidence >= c)).sum() / confidence.size)


def hce_risk(correct: np.ndarray, confidence: np.ndarray, c: float) -> float:
    """Error rate among high-confidence (>=c) predictions."""
    correct = np.asarray(correct)
    confidence = np.asarray(confidence)
    good = np.isfinite(confidence)
    correct, confidence = correct[good], confidence[good]
    hc = confidence >= c
    if hc.sum() == 0:
        return float("nan")
    return float((correct[hc] == 0).sum() / hc.sum())


def risk_coverage_curve(score: np.ndarray, correct: np.ndarray, higher_is_more_reliable: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coverage, risk) for selective classification."""
    correct = np.asarray(correct).astype(int)
    score = np.asarray(score, dtype=float)
    good = np.isfinite(score)
    correct, score = correct[good], score[good]
    n = len(correct)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-score) if higher_is_more_reliable else np.argsort(score)
    correct_sorted = correct[order]
    cum_correct = np.cumsum(correct_sorted)
    cum_n = np.arange(1, n + 1)
    cum_errors = cum_n - cum_correct
    risk = cum_errors / cum_n
    coverage = cum_n / n
    return coverage, risk


def aurc_from_curve(coverage: np.ndarray, risk: np.ndarray) -> float:
    if coverage.size == 0:
        return float("nan")
    cov = np.concatenate([[0.0], coverage])
    rsk = np.concatenate([[0.0], risk])
    return float(np.trapz(rsk, cov))


def excess_aurc(aurc: float, error_rate: float) -> float:
    err = float(error_rate)
    if err <= 0.0:
        oracle = 0.0
    elif err >= 1.0:
        oracle = 1.0
    else:
        oracle = err + (1.0 - err) * np.log(1.0 - err)
    return float(aurc - oracle)


def selective_risk_at_coverage(score: np.ndarray, correct: np.ndarray, kappa: float, higher_is_more_reliable: bool) -> float:
    coverage, risk = risk_coverage_curve(score, correct, higher_is_more_reliable)
    if coverage.size == 0:
        return float("nan")
    idx = np.searchsorted(coverage, kappa, side="left")
    idx = min(idx, len(risk) - 1)
    return float(risk[idx])


def compute_error_auroc_scores(
    correct: np.ndarray,
    pmax: np.ndarray,
    entropy: np.ndarray,
    margin: np.ndarray,
    uncertainty: np.ndarray,
    mutual_info: np.ndarray,
    ucb_score: np.ndarray,
) -> Dict[str, float]:
    """AUROC for distinguishing wrong (positive=1) from correct predictions.

    Each score is oriented so that HIGHER score => more likely WRONG.
    """
    wrong = (np.asarray(correct) == 0).astype(int)
    out: Dict[str, float] = {}
    out["auroc_neg_max_confidence"] = _roc_auc(wrong, -np.asarray(pmax, dtype=float))
    out["auroc_entropy"] = _roc_auc(wrong, np.asarray(entropy, dtype=float))
    out["auroc_neg_margin"] = _roc_auc(wrong, -np.asarray(margin, dtype=float))
    out["auroc_uncertainty"] = _roc_auc(wrong, np.asarray(uncertainty, dtype=float))
    out["auroc_neg_mutual_info"] = _roc_auc(wrong, -np.asarray(mutual_info, dtype=float))
    out["auroc_neg_ucb_score"] = _roc_auc(wrong, -np.asarray(ucb_score, dtype=float))

    def _zscore(x):
        x = np.asarray(x, dtype=float)
        good = np.isfinite(x)
        s = np.zeros_like(x)
        if good.any():
            mu = float(np.mean(x[good]))
            sd = float(np.std(x[good])) or 1.0
            s[good] = (x[good] - mu) / sd
        return s

    zunc = _zscore(uncertainty)
    zconf = _zscore(pmax)
    zmi = _zscore(mutual_info)
    combined = zunc - zconf - zmi
    out["auroc_combined"] = _roc_auc(wrong, combined)
    return out


def compute_all_metrics(
    correct: np.ndarray,
    pred: np.ndarray,
    pmax: np.ndarray,
    top2_prob: np.ndarray,
    margin: np.ndarray,
    entropy: np.ndarray,
    uncertainty: np.ndarray,
    mutual_info: np.ndarray,
    ucb_score: np.ndarray,
    come_loss: np.ndarray,
) -> Dict[str, float]:
    """Compute the full per-run metric dict for UCDPA evaluation."""
    correct = np.asarray(correct).astype(int)
    pred = np.asarray(pred).astype(int)
    pmax = np.asarray(pmax, dtype=float)
    top2_prob = np.asarray(top2_prob, dtype=float)
    margin = np.asarray(margin, dtype=float)
    entropy = np.asarray(entropy, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    mutual_info = np.asarray(mutual_info, dtype=float)
    ucb_score = np.asarray(ucb_score, dtype=float)
    come_loss = np.asarray(come_loss, dtype=float)

    n = len(correct)
    num_wrong = int((correct == 0).sum())
    num_correct = int(correct.sum())
    err_rate = num_wrong / max(n, 1)

    out: Dict[str, float] = {}
    out["num_samples"] = int(n)
    out["num_correct"] = num_correct
    out["num_wrong"] = num_wrong
    out["error_rate"] = float(err_rate)
    out["top1_accuracy"] = float(num_correct / max(n, 1) * 100.0)
    out["accuracy"] = out["top1_accuracy"]
    out["ece_15bins_percent"] = ece_percent(correct, pmax, n_bins=15)
    out["mce_15bins_percent"] = mce_percent(correct, pmax, n_bins=15)
    out["mean_confidence"] = _safe_mean(pmax)
    out["mean_predictive_entropy"] = _safe_mean(entropy)
    out["mean_entropy"] = _safe_mean(entropy)
    out["mean_uncertainty"] = _safe_mean(uncertainty)
    out["mean_mutual_info"] = _safe_mean(mutual_info)
    out["mean_ucb_score"] = _safe_mean(ucb_score)
    out["mean_come_loss"] = _safe_mean(come_loss)

    # Wrong-prediction reliability
    corr_mask = correct == 1
    wrong_mask = correct == 0
    out["correct_confidence"] = _safe_mean(pmax[corr_mask])
    out["wrong_confidence"] = _safe_mean(pmax[wrong_mask])
    out["confidence_separation"] = (out["correct_confidence"] - out["wrong_confidence"]) if (np.isfinite(out["correct_confidence"]) and np.isfinite(out["wrong_confidence"])) else float("nan")
    out["correct_uncertainty"] = _safe_mean(uncertainty[corr_mask])
    out["wrong_uncertainty"] = _safe_mean(uncertainty[wrong_mask])
    out["uncertainty_separation"] = (out["wrong_uncertainty"] - out["correct_uncertainty"]) if (np.isfinite(out["wrong_uncertainty"]) and np.isfinite(out["correct_uncertainty"])) else float("nan")
    out["correct_mutual_info"] = _safe_mean(mutual_info[corr_mask])
    out["wrong_mutual_info"] = _safe_mean(mutual_info[wrong_mask])
    out["mutual_info_separation"] = (out["correct_mutual_info"] - out["wrong_mutual_info"]) if (np.isfinite(out["correct_mutual_info"]) and np.isfinite(out["wrong_mutual_info"])) else float("nan")
    out["correct_ucb_score"] = _safe_mean(ucb_score[corr_mask])
    out["wrong_ucb_score"] = _safe_mean(ucb_score[wrong_mask])
    out["ucb_separation"] = (out["correct_ucb_score"] - out["wrong_ucb_score"]) if (np.isfinite(out["correct_ucb_score"]) and np.isfinite(out["wrong_ucb_score"])) else float("nan")

    # HCE
    for c in [0.8, 0.9, 0.95]:
        out[f"hce_rate@{c}"] = hce_rate(correct, pmax, c)
        out[f"hce_risk@{c}"] = hce_risk(correct, pmax, c)

    # Selective prediction (confidence-based primary)
    cov_c, risk_c = risk_coverage_curve(pmax, correct, higher_is_more_reliable=True)
    out["aurc"] = aurc_from_curve(cov_c, risk_c)
    out["e_aurc"] = excess_aurc(out["aurc"], err_rate)
    out["selective_risk@0.80"] = selective_risk_at_coverage(pmax, correct, 0.80, True)
    out["selective_risk@0.90"] = selective_risk_at_coverage(pmax, correct, 0.90, True)
    out["selective_risk@0.95"] = selective_risk_at_coverage(pmax, correct, 0.95, True)

    # AURC using uncertainty (lower = more reliable) and UCB score (higher = more reliable)
    cov_u, risk_u = risk_coverage_curve(uncertainty, correct, higher_is_more_reliable=False)
    out["aurc_uncertainty"] = aurc_from_curve(cov_u, risk_u)
    cov_b, risk_b = risk_coverage_curve(ucb_score, correct, higher_is_more_reliable=True)
    out["aurc_ucb_score"] = aurc_from_curve(cov_b, risk_b)

    # Error detection AUROC
    out.update(compute_error_auroc_scores(correct, pmax, entropy, margin, uncertainty, mutual_info, ucb_score))

    # Convenience aliases for the final table
    out["auroc_error_combined"] = out["auroc_combined"]
    out["hce_rate@0.9"] = out.get("hce_rate@0.9", float("nan"))
    return out
