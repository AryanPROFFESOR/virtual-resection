"""
vrpipeline.stats
==================
Every statistical procedure the two papers actually run, real
implementations (not p-value stand-ins):

  - Wilcoxon rank-sum / Mann-Whitney U (scipy passthrough, documented)
  - DeLong's test for comparing two correlated ROC-AUCs from the same
    patients (Kini et al.: "We used the non-parametric DeLong test to
    compare ROC curves and determine whether any single predictive model
    derived from a specific frequency band performed significantly
    better than the others"). Implemented via the fast structural
    components (X/Y placement matrices), not a bootstrap approximation.
  - Permutation-based functional-data-analysis (FDA) curve-difference
    test (Kini et al.: "reassigning surgical outcome to adjacency
    matrices uniformly at random up to 10 000 times and computing the
    median area under the resulting curves").
  - Odds ratio + 95% CI + two-tailed Fisher exact p-value (Kini et al.
    Supplementary Table S3).
  - Bonferroni correction (Sun/Kini: correcting for 6 frequency-band
    comparisons, alpha_corrected = 0.05/6 = 0.0083).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats as sstats


def wilcoxon_rank_sum(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U) test, used throughout
    both papers to compare median network measures between groups
    (pre-ictal vs ictal, good vs poor outcome, lesional vs non-lesional)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    stat, p = sstats.ranksums(a, b)
    return {"statistic": float(stat), "p_value": float(p),
            "n_a": len(a), "n_b": len(b)}


# ---------------------------------------------------------------------
# DeLong test for correlated ROC curves
# ---------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def _fast_delong(pred_pos: np.ndarray, pred_neg: np.ndarray):
    """Structural components for DeLong's AUC variance/covariance
    (Sun & Xiao 2014 fast implementation of DeLong 1988)."""
    m = pred_pos.shape[1]
    n = pred_neg.shape[1]
    k = pred_pos.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        pos = pred_pos[r]
        neg = pred_neg[r]
        z = np.concatenate([pos, neg])
        tz[r] = _midrank(z)
        tx[r] = _midrank(pos)
        ty[r] = _midrank(neg)

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, np.atleast_2d(delong_cov)


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray,
                     score_b: np.ndarray) -> dict:
    """DeLong's test comparing two correlated ROC-AUCs computed on the
    SAME set of patients with two different scores (e.g. broadband vs.
    beta-band synchronizability change), returning z, two-sided p, and
    both AUCs. This is the exact test Kini et al. use to claim broadband
    (AUC 0.89) is significantly better than every other band."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    y_true = y_true[order]
    score_a = np.asarray(score_a, dtype=float)[order]
    score_b = np.asarray(score_b, dtype=float)[order]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative outcome cases")

    preds = np.vstack([score_a, score_b])
    pos = preds[:, :n_pos]
    neg = preds[:, n_pos:]
    aucs, cov = _fast_delong(pos, neg)

    l = np.array([1.0, -1.0])
    diff = l @ aucs
    var = l @ cov @ l
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = diff / np.sqrt(var)
        p = 2 * (1 - sstats.norm.cdf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "z": float(z), "p_value": float(p)}


def roc_auc_ci_delong(y_true: np.ndarray, score: np.ndarray,
                       alpha: float = 0.05) -> dict:
    """AUC + DeLong-derived asymptotic confidence interval for a single
    score, matching the "AUC = 0.89, 95% CI 0.76-1.00" style reporting."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    y_true = y_true[order]
    score = np.asarray(score, dtype=float)[order]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    pos = score[:n_pos].reshape(1, -1)
    neg = score[n_pos:].reshape(1, -1)
    aucs, cov = _fast_delong(pos, neg)
    se = np.sqrt(cov[0, 0])
    z = sstats.norm.ppf(1 - alpha / 2)
    lo, hi = aucs[0] - z * se, aucs[0] + z * se
    return {"auc": float(aucs[0]), "ci_low": float(max(0, lo)),
            "ci_high": float(min(1, hi))}


# ---------------------------------------------------------------------
# Permutation-based functional data analysis (curve-area) test
# ---------------------------------------------------------------------
def permutation_curve_test(curves: np.ndarray, group_labels: np.ndarray,
                            n_perm: int = 10000,
                            rng: Optional[np.random.Generator] = None
                            ) -> dict:
    """Non-parametric test for whether the mean time-normalized curve
    differs between two outcome groups more than chance, matching Kini
    et al.'s functional-data-analysis procedure: "The null model was
    created by reassigning surgical outcome to adjacency matrices
    uniformly at random up to 10 000 times and computing the median area
    under the resulting curves of functional network metrics."

    Parameters
    ----------
    curves : ndarray, shape (n_subjects, n_timebins)
        Time-normalized curves (e.g. from preprocessing.time_normalize),
        one row per subject/seizure.
    group_labels : ndarray, shape (n_subjects,), boolean or 0/1
        True/1 = e.g. "good outcome" group.
    n_perm : int
        Number of label permutations (paper uses 10000).

    Returns
    -------
    dict with 'observed_area' (|area between group-mean curves|),
    'p_value', and 'null_distribution' (ndarray, len n_perm).
    """
    rng = rng or np.random.default_rng(0)
    curves = np.asarray(curves, dtype=float)
    labels = np.asarray(group_labels).astype(bool)

    def _area(lab):
        m1 = np.nanmean(curves[lab], axis=0)
        m0 = np.nanmean(curves[~lab], axis=0)
        return np.nansum(np.abs(m1 - m0))

    observed = _area(labels)
    null = np.empty(n_perm)
    n = len(labels)
    for p in range(n_perm):
        perm = rng.permutation(labels)
        null[p] = _area(perm)
    p_value = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return {"observed_area": float(observed), "p_value": float(p_value),
            "null_distribution": null}


# ---------------------------------------------------------------------
# Odds ratios (Fisher exact) and Bonferroni correction
# ---------------------------------------------------------------------
def odds_ratio_fisher(exposed_pos: int, exposed_neg: int,
                       unexposed_pos: int, unexposed_neg: int,
                       alpha: float = 0.05) -> dict:
    """2x2 odds ratio with 95% CI (Woolf/log method) and two-tailed
    Fisher's exact test p-value, matching Supplementary Table S3's
    "OR (95% CI), p value" format for each predictor of good outcome."""
    table = [[exposed_pos, exposed_neg], [unexposed_pos, unexposed_neg]]
    odds_ratio, p_value = sstats.fisher_exact(table)

    # Continuity-corrected log-OR CI (Haldane-Anscombe if any cell is 0).
    a, b, c, d = exposed_pos, exposed_neg, unexposed_pos, unexposed_neg
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        odds_ratio = (a * d) / (b * c)
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = sstats.norm.ppf(1 - alpha / 2)
    ci_low = np.exp(log_or - z * se_log_or)
    ci_high = np.exp(log_or + z * se_log_or)
    return {"odds_ratio": float(odds_ratio), "ci_low": float(ci_low),
            "ci_high": float(ci_high), "p_value": float(p_value)}


def bonferroni_alpha(n_comparisons: int, family_alpha: float = 0.05
                      ) -> float:
    """Bonferroni-corrected per-comparison alpha (both papers: 6
    frequency-band comparisons -> alpha = 0.05/6 = 0.0083)."""
    return family_alpha / n_comparisons
