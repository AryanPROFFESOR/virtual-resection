import numpy as np
from vrpipeline.stats import (wilcoxon_rank_sum, delong_roc_test,
                               roc_auc_ci_delong, permutation_curve_test,
                               odds_ratio_fisher, bonferroni_alpha)


def test_wilcoxon_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50)
    b = rng.normal(2, 1, 50)
    r = wilcoxon_rank_sum(a, b)
    assert r["p_value"] < 0.01


def test_delong_roc_test_better_score_wins():
    rng = np.random.default_rng(1)
    n = 60
    y = np.array([1] * 30 + [0] * 30)
    score_good = y + rng.normal(0, 0.3, n)
    score_bad = rng.normal(0, 1, n)
    out = delong_roc_test(y, score_good, score_bad)
    assert out["auc_a"] > out["auc_b"]


def test_roc_auc_ci_delong_bounds():
    rng = np.random.default_rng(2)
    n = 60
    y = np.array([1] * 30 + [0] * 30)
    score = y + rng.normal(0, 0.3, n)
    out = roc_auc_ci_delong(y, score)
    assert 0 <= out["ci_low"] <= out["auc"] <= out["ci_high"] <= 1


def test_permutation_curve_test_detects_difference():
    rng = np.random.default_rng(3)
    n_bins = 10
    good = np.array([np.linspace(0, 1, n_bins) + rng.normal(0, 0.05, n_bins)
                      for _ in range(15)])
    poor = np.array([np.linspace(0, -1, n_bins) + rng.normal(0, 0.05, n_bins)
                      for _ in range(15)])
    curves = np.vstack([good, poor])
    labels = np.array([True] * 15 + [False] * 15)
    out = permutation_curve_test(curves, labels, n_perm=500)
    assert out["p_value"] < 0.05


def test_odds_ratio_fisher():
    out = odds_ratio_fisher(15, 2, 2, 9)
    assert out["odds_ratio"] > 1
    assert out["p_value"] < 1


def test_bonferroni_alpha():
    assert np.isclose(bonferroni_alpha(6), 0.05 / 6)
