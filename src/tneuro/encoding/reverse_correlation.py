from __future__ import annotations

from typing import Tuple

import numpy as np

from tneuro.utils.validate import require_1d_float_array, require_positive_scalar


def spike_triggered_average(
    stim: np.ndarray,
    spike_times: np.ndarray,
    fs_hz: float,
    window_s: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the spike-triggered average (STA) of a 1D stimulus.

    Spikes too close to the stimulus boundaries are discarded so that each
    included spike has a full window available.

    Parameters
    ----------
    stim:
        1D stimulus array sampled at ``fs_hz``.
    spike_times:
        1D array of spike times in seconds.
    fs_hz:
        Sampling rate in Hz.
    window_s:
        Tuple ``(t_pre_s, t_post_s)`` specifying the window before and after
        each spike.

    Returns
    -------
    sta:
        Spike-triggered average over the valid spikes.
    lags_s:
        Time lags (seconds) corresponding to ``sta``.
    """
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

    sta = np.zeros(lags_s.shape, dtype=float)
    for idx in spike_idx:
        sta += x[idx - n_pre : idx + n_post + 1]
    sta /= float(spike_idx.size)

    return sta, lags_s
