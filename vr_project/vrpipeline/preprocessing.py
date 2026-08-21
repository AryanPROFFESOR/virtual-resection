"""
vrpipeline.preprocessing
=========================
Signal-conditioning steps described in Kini et al. (Materials & Methods,
"Network methods") and Sun et al. (Materials & Methods, "Preprocessing
with the Brainstorm tool"):

  - common-average re-reference
  - 60 Hz (or 50 Hz) notch filter for line noise
  - band-pass filtering into the five canonical bands used by both papers
  - bad/artefact channel exclusion
  - resampling
  - time-normalization of a variable-length seizure into K sequential
    bins (used for the functional-data-analysis comparison of s(t)/R(t)
    curves across patients with different seizure durations)
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy import signal

# Canonical frequency bands, identical across both papers.
FREQ_BANDS = {
    "alpha_theta": (5.0, 15.0),
    "beta": (15.0, 25.0),
    "low_gamma": (30.0, 40.0),
    "high_gamma": (95.0, 105.0),
    "very_high": (105.0, 500.0),  # upper bound clipped to Nyquist at runtime
}


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the across-channel mean at each time sample (Kini et al.:
    'a common average reference was applied ... by first computing a
    time-varying signal averaged across all electrodes and then by
    subtracting this signal from each electrode')."""
    car = data.mean(axis=1, keepdims=True)
    return data - car


def notch_filter(data: np.ndarray, fs: float, freq: float = 60.0,
                  quality: float = 30.0) -> np.ndarray:
    """IIR notch filter to remove power-line noise (Kini et al.: 'All ECoG
    signals were notch filtered at 60 Hz to remove power line noise')."""
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.filtfilt(b, a, data, axis=0)


def bandpass_filter(data: np.ndarray, fs: float, low: float, high: float,
                     order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    nyq = fs / 2.0
    high = min(high, nyq * 0.999)
    low = max(low, 0.1)
    sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass",
                         output="sos")
    return signal.sosfiltfilt(sos, data, axis=0)


def exclude_channels(data: np.ndarray, channels: list,
                      exclude: Iterable[str]) -> tuple:
    """Drop artefact/bad channels by name. Returns (data, channels)."""
    exclude = set(exclude)
    keep_idx = [i for i, c in enumerate(channels) if c not in exclude]
    return data[:, keep_idx], [channels[i] for i in keep_idx]


def resample(data: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling to a target sampling rate (e.g. 1000 -> 500 Hz,
    matching Sun et al.'s down-sampling step)."""
    if fs_in == fs_out:
        return data
    from math import gcd
    g = gcd(int(round(fs_in)), int(round(fs_out)))
    up = int(round(fs_out)) // g
    down = int(round(fs_in)) // g
    return signal.resample_poly(data, up, down, axis=0)


def time_normalize(curve: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Interpolate a variable-length 1-D (or 2-D, time-major) curve onto
    `n_bins` sequential normalized-time bins, matching the
    "interpolated to fit into 10 sequential time bins spanning the
    pre-seizure and seizure epochs" step used for cross-patient/cross-
    seizure functional data analysis in both papers."""
    curve = np.asarray(curve, dtype=float)
    t_in = np.linspace(0.0, 1.0, curve.shape[0])
    t_out = np.linspace(0.0, 1.0, n_bins)
    if curve.ndim == 1:
        return np.interp(t_out, t_in, curve)
    out = np.zeros((n_bins,) + curve.shape[1:])
    for idx in np.ndindex(*curve.shape[1:]):
        sl = (slice(None),) + idx
        out[(slice(None),) + idx] = np.interp(t_out, t_in, curve[sl])
    return out


def sliding_windows(n_samples: int, fs: float, win_sec: float,
                     step_sec: Optional[float] = None):
    """Yield (start, stop) sample index pairs for a sliding analysis window
    (Kini et al. used 1-s time windows for the Laplacian/s(t) computation)."""
    step_sec = win_sec if step_sec is None else step_sec
    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += step
