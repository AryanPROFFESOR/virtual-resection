"""
vrpipeline.simulate
======================
The Jirsa "Epileptor" coupled neural-mass model, eq. (1)-(8) of Sun et al.
(originally Jirsa et al. 2014, Brain), used by Sun et al. to build a
"simulation dataset" with ground-truth propagation paths for validating
the virtual-resection pipeline BEFORE applying it to real patient data.

This gives you an immediate, fully-synthetic, mathematically real
multi-node "seizure" dataset (with a known excitability/coupling
ground truth) to smoke-test every stage of the pipeline while you
separately obtain and prepare real iEEG data.

    x1_i' = y1_i - f1(x1_i, x2_i) - z_i + I1
    y1_i' = 1 - 5 x1_i^2 - y1_i
    z_i'  = (1/tau0) [ 4(x1_i - x0) - z_i - K * sum_j C_ij (x1_j - x1_i) ]
    x2_i' = -y2_i + x2_i - x2_i^3 + I2 + 0.002 g(x1_i) - 0.3(z_i - 3.5)
    y2_i' = (1/tau2) [ -y2_i + f2(x2_i) ]

    f1(x1,x2) = x1^3 - 3x1^2                    if x1 < 0
              = (x2 - 0.6(z-4)^2) x1             if x1 >= 0
    f2(x2)    = 0                                if x2 < -0.25
              = 6(x2 + 0.25)                     if x2 >= -0.25
    g(x1)     = integral_{t0}^{t} exp(-gamma(t-tau)) x1(tau) dtau
                (approximated here as an exponential leaky-integrator
                ODE state, dg/dt = -gamma*g + x1, exactly equivalent for
                gamma > 0 and identical initial condition g(t0)=0)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp


def epileptor_network(n_nodes: int, x0: np.ndarray, coupling: np.ndarray,
                       K: float = 1.0, tau0: float = 2857.0, tau2: float = 10.0,
                       I1: float = 3.1, I2: float = 0.45, gamma: float = 0.01,
                       t_span: tuple = (0.0, 6000.0), fs: float = 500.0,
                       seed: Optional[int] = None) -> dict:
    """Simulate `n_nodes` coupled Epileptors.

    Parameters
    ----------
    x0 : ndarray, shape (n_nodes,)
        Per-node excitability parameter (more negative/lower ~ further
        from seizure threshold; x0 close to -1.6...-2.0 typical of EZ in
        the original Epileptor literature -- Sun et al. set this
        per-node to define EZ / early-propagation / late-propagation
        roles in their five simulation configurations, see their
        Table S4).
    coupling : ndarray, shape (n_nodes, n_nodes)
        Structural coupling matrix C_ij (symmetric, non-negative).
    K : float
        Global coupling scaling.
    t_span : tuple
        (t_start, t_end) in model time units (same units as tau0/tau2).
    fs : float
        Output sampling rate for the returned time series (resampled
        from the adaptive-step ODE solution via linear interpolation
        onto a uniform grid).

    Returns
    -------
    dict with 'signal' (T, n_nodes) using x1(t) as the surface-recording
    proxy (standard Epileptor convention: -x1(t) approximates the
    observed field potential), 'fs', and 't'.
    """
    rng = np.random.default_rng(seed)
    n = n_nodes
    x0 = np.asarray(x0, dtype=float)
    coupling = np.asarray(coupling, dtype=float)

    def rhs(t, state):
        x1 = state[0:n]
        y1 = state[n:2 * n]
        z = state[2 * n:3 * n]
        x2 = state[3 * n:4 * n]
        y2 = state[4 * n:5 * n]
        g = state[5 * n:6 * n]

        f1 = np.where(x1 < 0, x1 ** 3 - 3 * x1 ** 2,
                       (x2 - 0.6 * (z - 4) ** 2) * x1)
        f2 = np.where(x2 < -0.25, 0.0, 6 * (x2 + 0.25))

        coupling_term = K * (coupling * (x1[None, :] - x1[:, None])).sum(axis=1)

        dx1 = y1 - f1 - z + I1
        dy1 = 1 - 5 * x1 ** 2 - y1
        dz = (1.0 / tau0) * (4 * (x1 - x0) - z - coupling_term)
        dx2 = -y2 + x2 - x2 ** 3 + I2 + 0.002 * g - 0.3 * (z - 3.5)
        dy2 = (1.0 / tau2) * (-y2 + f2)
        dg = -gamma * g + x1

        return np.concatenate([dx1, dy1, dz, dx2, dy2, dg])

    y0 = np.concatenate([
        rng.normal(0, 0.1, n) - 1.0,   # x1
        rng.normal(0, 0.1, n) - 5.0,   # y1
        rng.normal(0, 0.1, n) + 3.0,   # z
        rng.normal(0, 0.1, n),         # x2
        rng.normal(0, 0.1, n),         # y2
        np.zeros(n),                   # g
    ])

    # Uniform output grid: fs samples per 1000 model-time-units, matching
    # the tau0/tau2 time constants' natural scale.
    n_samples = int(round(fs * (t_span[1] - t_span[0]) / 1000.0))
    n_samples = max(n_samples, 100)
    t_eval = np.linspace(t_span[0], t_span[1], n_samples)

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="RK45",
                     max_step=(t_span[1] - t_span[0]) / n_samples)

    x1 = sol.y[0:n, :].T  # (T, n)
    return {"signal": -x1, "fs": fs, "t": sol.t}


def make_simple_chain_coupling(n_nodes: int) -> np.ndarray:
    """A simple nearest-neighbor chain coupling matrix C_ij (symmetric),
    a minimal stand-in structural connectome for smoke-testing; replace
    with a real coupling matrix (e.g. derived from your own DTI/atlas
    data) for anything beyond pipeline validation."""
    c = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes - 1):
        c[i, i + 1] = c[i + 1, i] = 1.0
    return c
