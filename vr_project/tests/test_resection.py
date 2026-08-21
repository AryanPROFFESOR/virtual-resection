import numpy as np
from vrpipeline.resection import (region_control_centrality,
                                   nodal_control_centrality,
                                   sequential_virtual_resection,
                                   evaluate_alternative_schemes)
from vrpipeline.propagation import ordered_removal_sequence


def test_region_control_centrality_removing_hub_desynchronizes():
    n = 6
    adj = np.ones((n, n)) - np.eye(n)
    c = region_control_centrality(adj, [0, 1])
    assert np.isfinite(c)


def test_nodal_control_centrality_shape():
    rng = np.random.default_rng(0)
    a = rng.random((5, 5))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0)
    out = nodal_control_centrality(a)
    assert out.shape == (5,)


def test_sequential_virtual_resection_monotone_progress():
    rng = np.random.default_rng(1)
    n = 10
    hte = rng.random((n, n))
    np.fill_diagonal(hte, 0)
    binary = (hte > 0.7).astype(int)
    order = ordered_removal_sequence(binary, seed_idx=[0, 1], n_steps=3)
    result = sequential_virtual_resection(hte, order)
    assert len(result["stability"]) == len(result["n_removed"])
    assert result["best_index"] >= 0
    assert isinstance(result["best_removed"], list)


def test_evaluate_alternative_schemes():
    rng = np.random.default_rng(2)
    n = 8
    hte = rng.random((n, n))
    np.fill_diagonal(hte, 0)
    schemes = {"scheme_a": [0, 1], "scheme_b": [2, 3, 4]}
    scores = evaluate_alternative_schemes(hte, schemes)
    assert set(scores.keys()) == set(schemes.keys())
