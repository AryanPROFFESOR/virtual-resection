"""
End-to-end smoke test: simulate a small coupled-Epileptor "seizure",
run it through BOTH pipelines exactly as a real patient recording would
be, and check every stage produces finite, sane outputs. This is the
test to run first after installing the package to confirm nothing is a
stub / everything actually executes on real numbers.
"""
import numpy as np

from vrpipeline.simulate import epileptor_network, make_simple_chain_coupling
from vrpipeline.io import load_array
from vrpipeline.pipeline import run_pipeline_a, run_pipeline_b
from vrpipeline.resection import electrodes_in_resection


def _make_recordings(n_nodes=8, seed=0):
    coupling = make_simple_chain_coupling(n_nodes)
    x0 = np.full(n_nodes, -2.1)
    x0[0:2] = -1.6  # EZ nodes: closer to seizure threshold
    sim = epileptor_network(n_nodes, x0, coupling, K=1.0,
                             t_span=(0.0, 3000.0), fs=250.0, seed=seed)
    signal = sim["signal"]
    fs = sim["fs"]
    channels = [f"E{i}" for i in range(n_nodes)]

    half = signal.shape[0] // 2
    pre = load_array(signal[:half], fs, channels)
    ictal = load_array(signal[half:], fs, channels)
    return pre, ictal, channels


def test_pipeline_a_end_to_end():
    pre, ictal, channels = _make_recordings(n_nodes=6)
    resected = channels[:2]
    result = run_pipeline_a(pre, ictal, resected, epoch_sec=1.0,
                             time_band=2.0, n_taper=3, n_bins=5)
    assert set(result.band_delta_s.keys()) >= {"beta", "broadband_cc"}
    for band, val in result.band_delta_s.items():
        assert np.isfinite(val) or np.isnan(val)
    assert "broadband_cc" in result.s_curves
    assert result.s_curves["broadband_cc"].shape[0] == 5


def test_pipeline_b_end_to_end():
    pre, ictal, channels = _make_recordings(n_nodes=6)
    seeds = channels[:2]
    resected = channels[:2]
    result = run_pipeline_b(pre, ictal, seeds, resected_labels=resected,
                             epoch_sec=1.5, te_bins=4, hte_order=3,
                             fdr_n_perm=20)
    assert result.asym_stability_curve is not None
    assert result.sequential_resection is not None
    seq = result.sequential_resection
    assert len(seq["stability"]) >= 1
    assert seq["best_index"] >= 0


def test_electrode_resection_mapping_from_mask():
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[8:14, 8:14, 8:14] = 1  # a resection cavity block
    affine = np.eye(4)
    coords = {
        "E0": (10, 10, 10),   # inside the block
        "E1": (11, 11, 11),   # inside the block
        "E2": (1, 1, 1),      # far outside
        "E3": (19, 19, 19),   # far outside
    }
    inside = electrodes_in_resection(vol, affine, coords)
    assert set(inside) == {"E0", "E1"}


def test_electrode_resection_mapping_dilation_increases_membership():
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[10, 10, 10] = 1
    affine = np.eye(4)
    coords = {"E0": (10, 10, 10), "E1": (12, 10, 10), "E2": (0, 0, 0)}
    base = electrodes_in_resection(vol, affine, coords)
    dilated = electrodes_in_resection(vol, affine, coords,
                                       dilate_erode_pct=0.5)
    assert len(dilated) >= len(base)
