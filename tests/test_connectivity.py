import numpy as np
from vrpipeline.connectivity import (multitaper_coherence_band,
                                      coherence_adjacency,
                                      broadband_crosscorr_adjacency,
                                      transfer_entropy,
                                      transfer_entropy_matrix,
                                      high_order_te)


def test_coherence_identical_signals_is_nan_diag_handling():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    c = multitaper_coherence_band(x, x, fs=500, band=(15, 25))
    assert np.isnan(c)


def test_coherence_correlated_signals_high():
    rng = np.random.default_rng(1)
    fs = 500.0
    t = np.arange(4000) / fs
    base = np.sin(2 * np.pi * 20 * t) + 0.1 * rng.standard_normal(len(t))
    x = base + 0.05 * rng.standard_normal(len(t))
    y = base + 0.05 * rng.standard_normal(len(t))
    c = multitaper_coherence_band(x, y, fs, band=(15, 25))
    assert c > 0.5


def test_coherence_independent_signals_low():
    rng = np.random.default_rng(2)
    fs = 500.0
    x = rng.standard_normal(4000)
    y = rng.standard_normal(4000)
    c = multitaper_coherence_band(x, y, fs, band=(15, 25))
    assert c < 0.5


def test_coherence_adjacency_symmetric():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((2000, 4))
    adj = coherence_adjacency(data, fs=500, band=(15, 25), n_taper=3)
    assert adj.shape == (4, 4)
    assert np.allclose(adj, adj.T)
    assert np.allclose(np.diag(adj), 0)


def test_broadband_crosscorr():
    rng = np.random.default_rng(4)
    data = rng.standard_normal((500, 3))
    adj = broadband_crosscorr_adjacency(data)
    assert adj.shape == (3, 3)
    assert np.allclose(np.diag(adj), 0)


def test_transfer_entropy_directional_detects_coupling():
    rng = np.random.default_rng(5)
    n = 3000
    y = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * y[i - 1] + 0.2 * rng.standard_normal()
    te_yx = transfer_entropy(y, x, k=1, l=1, n_bins=4)
    te_xy = transfer_entropy(x, y, k=1, l=1, n_bins=4)
    assert te_yx > te_xy


def test_transfer_entropy_matrix_shape():
    rng = np.random.default_rng(6)
    data = rng.standard_normal((800, 4))
    te = transfer_entropy_matrix(data, n_bins=4)
    assert te.shape == (4, 4)
    assert np.allclose(np.diag(te), 0)


def test_high_order_te_composition():
    hte1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    mats = high_order_te(hte1, order=3)
    assert len(mats) == 3
    assert np.allclose(mats[0], hte1)
    assert np.allclose(mats[1], hte1 @ hte1)
    assert np.allclose(mats[2], hte1 @ mats[1])
