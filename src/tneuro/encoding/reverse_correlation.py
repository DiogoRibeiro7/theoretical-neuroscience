from __future__ import annotations

import numpy as np

from tneuro.utils.validate import require_1d_float_array, require_positive_scalar


def _collect_windows(
    stim: np.ndarray,
    spike_times: np.ndarray,
    fs_hz: float,
    window_s: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x = require_1d_float_array(stim, name="stim")
    t_spike = require_1d_float_array(spike_times, name="spike_times")
    fs = require_positive_scalar(fs_hz, name="fs_hz")

    if len(window_s) != 2:
        raise ValueError("window_s must be a 2-tuple (t_pre_s, t_post_s).")
    t_pre_s, t_post_s = float(window_s[0]), float(window_s[1])
    if t_pre_s < 0.0 or t_post_s < 0.0 or not np.isfinite(t_pre_s) or not np.isfinite(t_post_s):
        raise ValueError("window_s values must be finite and non-negative.")

    n_pre = int(np.rint(t_pre_s * fs))
    n_post = int(np.rint(t_post_s * fs))
    if n_pre < 0 or n_post < 0:
        raise ValueError("window_s values must be non-negative.")

    lags_s = np.arange(-n_pre, n_post + 1, dtype=float) / fs
    if t_spike.size == 0:
        raise ValueError("spike_times is empty.")

    spike_idx = np.rint(t_spike * fs).astype(int)
    valid = (spike_idx >= n_pre) & (spike_idx + n_post < x.size)
    spike_idx = spike_idx[valid]
    if spike_idx.size == 0:
        raise ValueError("No spikes have a full window within the stimulus bounds.")

    windows = np.empty((spike_idx.size, lags_s.size), dtype=float)
    for i, idx in enumerate(spike_idx):
        windows[i] = x[idx - n_pre : idx + n_post + 1]

    return windows, lags_s


def spike_triggered_average(
    stim: np.ndarray,
    spike_times: np.ndarray,
    fs_hz: float,
    window_s: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spike-triggered average (STA) of a 1D stimulus.

    Spikes too close to the stimulus boundaries are discarded so that each
    included spike has a full window available.
    """
    windows, lags_s = _collect_windows(stim, spike_times, fs_hz, window_s)
    sta = np.mean(windows, axis=0)
    return sta, lags_s


def spike_triggered_covariance(
    stim: np.ndarray,
    spike_times: np.ndarray,
    fs_hz: float,
    window_s: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spike-triggered covariance (STC) of a 1D stimulus."""
    windows, lags_s = _collect_windows(stim, spike_times, fs_hz, window_s)
    if windows.shape[0] < 2:
        raise ValueError("Need at least 2 spikes to compute covariance.")
    cov = np.cov(windows, rowvar=False, bias=False)
    return cov, lags_s


__all__ = [
    "spike_triggered_average",
    "spike_triggered_covariance",
]
