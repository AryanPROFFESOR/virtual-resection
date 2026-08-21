"""
vrpipeline.outcome
=====================
Surgical-outcome prediction from a scalar per-patient feature (Kini et
al.'s Delta s(t) pre-ictal->ictal broadband synchronizability change, or
any other summary feature you derive, e.g. Sun et al.'s resection-zone
control-centrality value or asymmetric-stability minimum). Real
scikit-learn ROC machinery plus the exact Youden-optimal thresholding and
confusion-matrix reporting used in both papers' Tables S2/S3.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def roc_analysis(y_true: np.ndarray, feature: np.ndarray,
                  higher_is_positive: bool = True) -> dict:
    """Full ROC/AUC sweep over `feature` (Kini et al.: "We varied the
    threshold of Delta s(t) to predict patients as having either good or
    poor surgical outcome, which generated a receiver operating
    characteristic (ROC) curve").

    Returns dict with fpr, tpr, thresholds, auc, and the Youden-optimal
    threshold/operating point.
    """
    y_true = np.asarray(y_true).astype(int)
    feature = np.asarray(feature, dtype=float)
    score = feature if higher_is_positive else -feature
    fpr, tpr, thr = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    youden = tpr - fpr
    best = int(np.argmax(youden))
    best_thr = thr[best]
    if not higher_is_positive:
        best_thr = -best_thr
    return {"fpr": fpr, "tpr": tpr, "thresholds": thr, "auc": float(roc_auc),
            "best_threshold": float(best_thr),
            "best_sensitivity": float(tpr[best]),
            "best_specificity": float(1 - fpr[best])}


def predict_at_threshold(feature: np.ndarray, threshold: float,
                          higher_is_positive: bool = True) -> np.ndarray:
    feature = np.asarray(feature, dtype=float)
    if higher_is_positive:
        return (feature >= threshold).astype(int)
    return (feature <= threshold).astype(int)


def confusion_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """TP/FP/FN/TN + accuracy/TPR/TNR, matching Supplementary Table S2's
    layout exactly (rows = predicted, columns = actual)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fp = cm[0, 0], cm[0, 1]
    fn, tn = cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "accuracy": float(acc), "TPR": float(tpr), "TNR": float(tnr)}


def per_band_roc(y_true: np.ndarray, feature_by_band: dict,
                  higher_is_positive: bool = True) -> dict:
    """Run `roc_analysis` independently per frequency band, matching Kini
    et al. Fig. 3C (per-band ROC overlay with broadband as best AUC)."""
    return {band: roc_analysis(y_true, feat, higher_is_positive)
            for band, feat in feature_by_band.items()}
