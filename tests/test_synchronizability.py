import numpy as np
from vrpipeline.synchronizability import (synchronizability,
                                           outdegree_laplacian,
                                           asymmetric_stability)


def test_synchronizability_complete_graph():
    n = 6
    adj = np.ones((n, n)) - np.eye(n)
    s = synchronizability(adj)
    # Complete graph Laplacian eigenvalues: 0 once, n (multiplicity n-1)
    assert np.isclose(s, 1.0, atol=1e-6)


def test_synchronizability_nan_on_missing_edges():
    adj = np.array([[0, np.nan], [np.nan, 0]])
    assert np.isnan(synchronizability(adj))


def test_synchronizability_rejects_asymmetric():
    adj = np.array([[0, 1], [0, 0]], dtype=float)
    try:
        synchronizability(adj)
        assert False, "should have raised"
    except ValueError:
        pass


def test_outdegree_laplacian_shape():
    hte = np.random.rand(5, 5)
    np.fill_diagonal(hte, 0)
    lap = outdegree_laplacian(hte)
    assert lap.shape == (5, 5)
    assert np.allclose(lap.sum(axis=0), 0, atol=1e-8)


def test_asymmetric_stability_runs():
    rng = np.random.default_rng(0)
    hte = rng.random((8, 8))
    np.fill_diagonal(hte, 0)
    r = asymmetric_stability(hte)
    assert np.isfinite(r) or np.isnan(r)
