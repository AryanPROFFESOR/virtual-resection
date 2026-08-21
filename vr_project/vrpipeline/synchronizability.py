"""
vrpipeline.synchronizability
==============================
Two network-stability measures, one per source paper:

1. `synchronizability` (symmetric) -- Kini et al.: s(t) = lambda_2/lambda_N
   of the graph Laplacian L = D - A of an undirected weighted adjacency A.
   Verified line-for-line against Echobase's
   `Echobase/Network/Metrics/globaltopo.py::synchronizability`.

2. `asymmetric_stability` -- Sun et al. eq. (11)-(12): directed Laplacian
   L = D_out - HTE built from an *outdegree* diagonal and a (possibly
   non-symmetric) high-order-TE adjacency, then R = lambda_N / lambda_2
   of that Laplacian's eigenvalues (sorted by real part; L is generally
   non-symmetric so eigenvalues can be complex -- we take the real part,
   documented explicitly, standard practice for asymmetric graph
   Laplacians in the network-control literature this paper draws on).

   NB on eq. (12) vs. body text: Sun et al.'s body text describes "the
   ratio of the third smallest eigenvalue to the largest eigenvalue"
   while eq. (12) states R = lambda_N / lambda_2 (largest / *second*
   smallest). We implement eq. (12) as written (the ratio actually used
   throughout their Results/Figures, which show single-inflection-point
   curves consistent with a 2nd-eigenvalue-based measure) and expose
   `smallest_index` so you can reproduce the alternate text description
   if you prefer -- this discrepancy is called out here explicitly rather
   than silently resolved.
"""
from __future__ import annotations

import numpy as np


def synchronizability(adj: np.ndarray) -> float:
    """s = lambda_2 / lambda_max of the Laplacian of a symmetric adjacency
    (Kini et al.). Returns NaN if the input has NaNs (missing edges)."""
    adj = np.asarray(adj, dtype=float)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square")
    if np.isnan(adj).any():
        return float("nan")
    if not np.allclose(adj, adj.T, atol=1e-8):
        raise ValueError("adjacency matrix is not symmetric")
    deg = np.diag(adj.sum(axis=0))
    lap = deg - adj
    eigval = np.linalg.eigvals(lap)
    eigval = np.sort(np.real(eigval))
    lam2, lamN = eigval[1], eigval[-1]
    if lamN == 0:
        return float("nan")
    return float(abs(lam2 / lamN))


def outdegree_laplacian(hte: np.ndarray) -> np.ndarray:
    """L = D_out - HTE, eq. (11): D_out is the diagonal matrix of column
    sums (outdegree) of the directed high-order-TE adjacency."""
    hte = np.asarray(hte, dtype=float)
    outdeg = hte.sum(axis=0)  # column sums = outdegree of source node
    return np.diag(outdeg) - hte


def asymmetric_stability(hte: np.ndarray, smallest_index: int = 1,
                          zero_tol: float = 1e-9) -> float:
    """R = lambda_N / lambda_{smallest_index} of L = D_out - HTE, eq. (12).
    `smallest_index=1` -> second-smallest eigenvalue (matches eq. (12)
    as written); use `smallest_index=2` to reproduce the body-text
    description ("ratio of the third smallest eigenvalue...").

    A directed graph built from a sparse/short-duration high-order TE
    matrix can be disconnected, giving the zero eigenvalue of L a
    multiplicity > 1 (the directed analogue of "number of weakly
    connected components"). In that case the naive `smallest_index`-th
    sorted eigenvalue would still be (numerically) zero and R would be
    undefined even though the connected sub-components each have a
    perfectly well-defined internal stability. We therefore count
    `smallest_index` starting AFTER the block of near-zero eigenvalues
    (|eigenvalue| <= zero_tol), matching standard graph-Laplacian
    convention (algebraic connectivity is the smallest eigenvalue beyond
    the zero-eigenvalue block). If fewer than `smallest_index` non-zero
    eigenvalues exist, R is genuinely undefined and NaN is returned.
    """
    hte = np.asarray(hte, dtype=float)
    if hte.shape[0] != hte.shape[1]:
        raise ValueError("hte must be square")
    if hte.shape[0] <= smallest_index + 1:
        return float("nan")
    lap = outdegree_laplacian(hte)
    eigval = np.linalg.eigvals(lap)
    eigval = np.sort(np.real(eigval))
    nonzero = eigval[np.abs(eigval) > zero_tol]
    if len(nonzero) <= smallest_index - 1:
        return float("nan")
    lam_small = nonzero[smallest_index - 1] if smallest_index >= 1 else eigval[0]
    lam_max = eigval[-1]
    if lam_small == 0:
        return float("nan")
    return float(abs(lam_max / lam_small))


def synchronizability_timeseries(adjacency_by_epoch: list) -> np.ndarray:
    """Vectorized s(t) across a list of per-epoch adjacency matrices
    (symmetric coherence networks, Kini et al. Fig. 3A)."""
    return np.array([synchronizability(a) for a in adjacency_by_epoch])


def stability_timeseries(hte_by_epoch: list, smallest_index: int = 1
                          ) -> np.ndarray:
    """Vectorized R(t) across a list of per-epoch HTE adjacency matrices
    (Sun et al. Fig. 2d/4c/5b)."""
    return np.array([asymmetric_stability(h, smallest_index)
                      for h in hte_by_epoch])
