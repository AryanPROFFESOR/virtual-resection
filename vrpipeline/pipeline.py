"""
vrpipeline.pipeline
======================
End-to-end orchestration tying every module together into the two
complete pipelines, run side-by-side on the same raw recordings so their
outputs are directly comparable per patient/seizure.

    Pipeline A (Kini et al., symmetric/coherence/en-bloc):
        raw iEEG
          -> preprocessing (CAR, notch, band split)
          -> multiband coherence adjacency per 1-s epoch
          -> s(t) synchronizability time series, per band + broadband
          -> Delta s(t) = median(ictal) - median(pre-ictal)
          -> en-bloc resection-zone control centrality c_res(t)
          -> per-patient scalar feature -> cohort ROC/AUC/outcome

    Pipeline B (Sun et al., directed/transfer-entropy/sequential):
        raw iEEG
          -> preprocessing (CAR, notch, optional single broadband pass)
          -> first-order transfer-entropy matrix -> FDR-binarize
          -> high-order (K-step) TE composition
          -> asymmetric stability R(t)
          -> propagation-path extraction from seed (SOZ) electrodes
          -> sequential virtual resection -> inflection-point minimum
             intervention target
          -> three-stage (pre-seizure/seizure/VR-seizure) comparison

Everything below operates on `io.IEEGRecording` objects and real node
labels; nothing here is templated with placeholder numbers.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

import numpy as np

from . import connectivity as conn
from . import synchronizability as sync
from . import resection as resec
from . import propagation as prop
from .io import IEEGRecording
from .preprocessing import (common_average_reference, notch_filter,
                             bandpass_filter, FREQ_BANDS, sliding_windows,
                             time_normalize)


@dataclasses.dataclass
class PatientResult:
    patient_id: str
    band_delta_s: dict            # band -> Delta s(t) (Pipeline A feature)
    band_cres: dict                # band -> median c_res(t) during ictal
    s_curves: dict                 # band -> (pre+ictal) time-normalized s(t)
    asym_stability_curve: Optional[np.ndarray] = None   # Pipeline B, R(t)
    sequential_resection: Optional[dict] = None          # Pipeline B result
    resected_labels: Optional[list] = None


# ---------------------------------------------------------------------
# Pipeline A: Kini et al.
# ---------------------------------------------------------------------
def run_pipeline_a(rec_pre: IEEGRecording, rec_ictal: IEEGRecording,
                    resected_labels: Iterable[str], epoch_sec: float = 1.0,
                    time_band: float = 4.0, n_taper: int = 7,
                    n_bins: int = 10, notch_hz: float = 60.0) -> PatientResult:
    """Run the full symmetric-coherence / en-bloc virtual-resection
    pipeline on a matched pair of pre-ictal and ictal clips from one
    seizure, for one patient.

    `rec_pre` / `rec_ictal` must share the same channel list and order.
    """
    if rec_pre.channels != rec_ictal.channels:
        raise ValueError("pre-ictal and ictal channel lists must match")
    channels = rec_pre.channels
    resected_idx = [i for i, c in enumerate(channels) if c in set(resected_labels)]

    def _epoch_adjacencies(rec: IEEGRecording):
        d = common_average_reference(rec.data)
        d = notch_filter(d, rec.fs, notch_hz)
        per_epoch = {b: [] for b in list(FREQ_BANDS) + ["broadband_cc"]}
        for start, stop in sliding_windows(d.shape[0], rec.fs, epoch_sec):
            seg = d[start:stop]
            adj = conn.multiband_adjacency(seg, rec.fs, time_band, n_taper)
            for band, a in adj.items():
                per_epoch[band].append(a)
        return per_epoch

    pre_adj = _epoch_adjacencies(rec_pre)
    ictal_adj = _epoch_adjacencies(rec_ictal)

    band_delta_s, band_cres, s_curves = {}, {}, {}
    for band in pre_adj:
        s_pre = sync.synchronizability_timeseries(pre_adj[band])
        s_ictal = sync.synchronizability_timeseries(ictal_adj[band])
        band_delta_s[band] = float(np.nanmedian(s_ictal) - np.nanmedian(s_pre))

        cres_ictal = resec.resection_zone_control_timeseries(
            ictal_adj[band], resected_idx)
        band_cres[band] = float(np.nanmedian(cres_ictal))

        full_curve = np.concatenate([s_pre, s_ictal])
        s_curves[band] = time_normalize(full_curve, n_bins)

    return PatientResult(patient_id=getattr(rec_pre, "patient_id", "unknown"),
                          band_delta_s=band_delta_s, band_cres=band_cres,
                          s_curves=s_curves, resected_labels=list(resected_labels))


# ---------------------------------------------------------------------
# Pipeline B: Sun et al.
# ---------------------------------------------------------------------
def run_pipeline_b(rec_pre: IEEGRecording, rec_ictal: IEEGRecording,
                    seed_labels: Iterable[str],
                    resected_labels: Optional[Iterable[str]] = None,
                    epoch_sec: float = 2.0, te_k: int = 1, te_l: int = 1,
                    te_bins: int = 6, hte_order: int = 3,
                    fdr_alpha: float = 0.001, fdr_n_perm: int = 100,
                    notch_hz: float = 60.0, smallest_index: int = 1
                    ) -> PatientResult:
    """Run the full directed transfer-entropy / sequential virtual-
    resection pipeline on a matched pre-ictal + ictal clip pair.

    `seed_labels` = clinically marked EZ/SOZ contacts to seed propagation-
    path extraction (Sun et al.: "using TB1, TB'1, and B'1 as seed
    nodes" -- their analogue of the clinician-marked seizure-onset zone).
    """
    if rec_pre.channels != rec_ictal.channels:
        raise ValueError("pre-ictal and ictal channel lists must match")
    channels = rec_pre.channels
    seed_idx = [i for i, c in enumerate(channels) if c in set(seed_labels)]
    if not seed_idx:
        raise ValueError("none of seed_labels found in channel list")

    d = common_average_reference(rec_ictal.data)
    d = notch_filter(d, rec_ictal.fs, notch_hz)

    epoch_hte = []
    epoch_r = []
    last_te1 = None
    for start, stop in sliding_windows(d.shape[0], rec_ictal.fs, epoch_sec):
        seg = d[start:stop]
        te1 = conn.transfer_entropy_matrix(seg, te_k, te_l, te_bins)
        mask = conn.fdr_binarize(te1, fdr_alpha, fdr_n_perm, data=seg,
                                  k=te_k, l=te_l, n_bins=te_bins)
        weighted_binary = te1 * mask
        hte_all = conn.high_order_te(weighted_binary, hte_order)
        hte_final = hte_all[-1]
        epoch_hte.append(hte_final)
        epoch_r.append(sync.asymmetric_stability(hte_final, smallest_index))
        last_te1 = mask

    r_curve = np.array(epoch_r)

    seq_result = None
    if last_te1 is not None and epoch_hte:
        order = prop.ordered_removal_sequence(last_te1, seed_idx, n_steps=3)
        # Run sequential resection on the LAST epoch's HTE as the
        # representative propagation-path snapshot (Sun et al. Fig. 2).
        seq_result = resec.sequential_virtual_resection(
            epoch_hte[-1], order, smallest_index)

    resected_idx = None
    if resected_labels is not None:
        resected_idx = [i for i, c in enumerate(channels)
                         if c in set(resected_labels)]

    return PatientResult(
        patient_id=getattr(rec_pre, "patient_id", "unknown"),
        band_delta_s={}, band_cres={}, s_curves={},
        asym_stability_curve=r_curve,
        sequential_resection=seq_result,
        resected_labels=list(resected_labels) if resected_labels else None,
    )


# ---------------------------------------------------------------------
# Cohort-level aggregation (both pipelines feed the same ROC machinery)
# ---------------------------------------------------------------------
def build_cohort_feature_table(results: list, band: str = "broadband_cc"
                                ) -> dict:
    """Assemble the per-patient Delta-s(t) feature vector (Pipeline A) and
    outcome-ready arrays for `outcome.roc_analysis` / `stats.delong_roc_test`.
    """
    patient_ids = [r.patient_id for r in results]
    delta_s = np.array([r.band_delta_s.get(band, np.nan) for r in results])
    cres = np.array([r.band_cres.get(band, np.nan) for r in results])
    return {"patient_ids": patient_ids, "delta_s": delta_s, "c_res": cres}
