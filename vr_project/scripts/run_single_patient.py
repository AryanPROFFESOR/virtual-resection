#!/usr/bin/env python3
"""
run_single_patient.py
======================
Run BOTH virtual-resection pipelines (Kini et al. symmetric/coherence and
Sun et al. directed/transfer-entropy) on one patient's matched pre-ictal +
ictal clips, and print/plot the combined report.

You supply your own downloaded data. Nothing here is simulated unless you
pass --simulate.

Usage
-----
    python run_single_patient.py \\
        --pre-ictal patient01_preictal.h5 \\
        --ictal patient01_ictal.h5 \\
        --format h5 \\
        --resected E1,E2,E3 \\
        --seed E1,E2 \\
        --electrode-coords patient01_electrodes.csv \\
        --resection-mask patient01_resection_mask.nii.gz \\
        --out-dir results/patient01

    # or, to validate the pipeline end-to-end without any real data:
    python run_single_patient.py --simulate --out-dir results/sim_demo

Input clip format (--format h5) matches the HDF5 layout used throughout
the original EpiVR codebase (evData: T x N array, Fs: scalar, channels:
list of channel-name strings) -- exactly what you get by exporting a clip
from IEEG.org via their MATLAB/Python client and re-saving to HDF5, or by
adapting `vrpipeline.io.load_h5_ieeg` to your own exporter's layout.
--format edf uses MNE to read a standard EDF/EDF+ file directly.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrpipeline.io import (load_h5_ieeg, load_edf, load_electrode_coordinates,
                            load_resection_mask)
from vrpipeline.pipeline import run_pipeline_a, run_pipeline_b
from vrpipeline.resection import electrodes_in_resection
from vrpipeline.simulate import epileptor_network, make_simple_chain_coupling
from vrpipeline.io import load_array


def _load_clip(path, fmt):
    if fmt == "h5":
        return load_h5_ieeg(path)
    elif fmt == "edf":
        return load_edf(path)
    else:
        raise ValueError(f"unsupported format: {fmt}")


def _simulate_demo():
    n_nodes = 10
    # Denser-than-chain coupling for the demo so the short synthetic clip
    # yields a well-connected (non-degenerate) propagation network; a bare
    # nearest-neighbor chain over only 10 nodes and a few seconds of data
    # can legitimately leave the high-order TE graph disconnected, which
    # `asymmetric_stability` correctly reports as NaN (undefined
    # stability) rather than papering over it.
    coupling = make_simple_chain_coupling(n_nodes)
    rng = np.random.default_rng(7)
    extra = (rng.random((n_nodes, n_nodes)) < 0.3).astype(float)
    extra = np.triu(extra, 1)
    coupling = np.clip(coupling + extra + extra.T, 0, 1)
    x0 = np.full(n_nodes, -2.1)
    x0[0:2] = -1.6
    sim = epileptor_network(n_nodes, x0, coupling, K=2.0,
                             t_span=(0.0, 8000.0), fs=250.0, seed=42)
    signal, fs = sim["signal"], sim["fs"]
    channels = [f"E{i}" for i in range(n_nodes)]
    half = signal.shape[0] // 2
    pre = load_array(signal[:half], fs, channels)
    ictal = load_array(signal[half:], fs, channels)
    return pre, ictal, channels, channels[:2], channels[:2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre-ictal", type=str)
    ap.add_argument("--ictal", type=str)
    ap.add_argument("--format", choices=["h5", "edf"], default="h5")
    ap.add_argument("--resected", type=str,
                     help="comma-separated resected electrode labels")
    ap.add_argument("--seed", type=str,
                     help="comma-separated seed/SOZ electrode labels for "
                          "the propagation-path pipeline")
    ap.add_argument("--electrode-coords", type=str,
                     help="CSV of id,x,y,z,label (world coords, "
                          "co-registered to --resection-mask)")
    ap.add_argument("--resection-mask", type=str,
                     help="NIfTI binary post-resection segmentation mask")
    ap.add_argument("--out-dir", type=str, default="results/patient")
    ap.add_argument("--simulate", action="store_true",
                     help="ignore all data flags, run on synthetic "
                          "Epileptor-simulated data for smoke-testing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.simulate:
        pre, ictal, channels, resected, seeds = _simulate_demo()
    else:
        if not (args.pre_ictal and args.ictal):
            ap.error("--pre-ictal and --ictal are required unless --simulate")
        pre = _load_clip(args.pre_ictal, args.format)
        ictal = _load_clip(args.ictal, args.format)
        channels = pre.channels

        if args.resected:
            resected = [s.strip() for s in args.resected.split(",")]
        elif args.electrode_coords and args.resection_mask:
            coords = load_electrode_coordinates(args.electrode_coords)
            vol, affine = load_resection_mask(args.resection_mask)
            resected = electrodes_in_resection(vol, affine, coords)
            print(f"[imaging pipeline] resected electrodes inferred from "
                  f"mask: {resected}")
        else:
            ap.error("must supply --resected OR both --electrode-coords "
                      "and --resection-mask")

        seeds = ([s.strip() for s in args.seed.split(",")]
                 if args.seed else resected)

    print(f"Channels ({len(channels)}): {channels}")
    print(f"Resected: {resected}")
    print(f"Seed/SOZ: {seeds}")

    print("\n== Pipeline A (Kini et al.: symmetric coherence, en-bloc "
          "resection) ==")
    result_a = run_pipeline_a(pre, ictal, resected)
    for band, d in result_a.band_delta_s.items():
        cres = result_a.band_cres[band]
        print(f"  {band:14s}  Delta s(t) = {d:+.4f}   "
              f"c_res(ictal, median) = {cres:+.4f}")

    print("\n== Pipeline B (Sun et al.: directed TE, sequential resection) ==")
    # NB: --simulate uses a relaxed FDR alpha (vs. the paper's 0.001
    # default) purely so this short synthetic demo clip reliably yields a
    # non-empty propagation graph; pass your own fdr_alpha via the
    # vrpipeline.pipeline.run_pipeline_b API for real analyses.
    pb_kwargs = {"fdr_alpha": 0.2, "epoch_sec": 3.0} if args.simulate else {}
    result_b = run_pipeline_b(pre, ictal, seeds, resected_labels=resected,
                               **pb_kwargs)
    r_curve = result_b.asym_stability_curve
    print(f"  Asymmetric stability R(t) across {len(r_curve)} epochs: "
          f"min={np.nanmin(r_curve):.4f} max={np.nanmax(r_curve):.4f}")
    seq = result_b.sequential_resection
    print(f"  Sequential resection: inflection point at "
          f"{seq['best_index']} node(s) removed, "
          f"R = {seq['stability'][seq['best_index']]:.4f} "
          f"(baseline R = {seq['stability'][0]:.4f})")
    best_labels = [channels[i] for i in seq["best_removed"]]
    print(f"  Minimum-intervention target (electrode labels): {best_labels}")

    report = {
        "channels": channels,
        "resected": resected,
        "seed": seeds,
        "pipeline_a": {
            "band_delta_s": result_a.band_delta_s,
            "band_cres": result_a.band_cres,
        },
        "pipeline_b": {
            "stability_curve": list(map(float, r_curve)),
            "sequential_resection": {
                "n_removed": seq["n_removed"].tolist(),
                "stability": seq["stability"].tolist(),
                "best_index": seq["best_index"],
                "best_removed_labels": best_labels,
            },
        },
    }
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
