from __future__ import annotations

from math import lgamma

import numpy as np

from tneuro.typing import ArrayF, ArrayI
from tneuro.utils.validate import require_1d_float_array, require_non_negative_scalar


def gaussian_tuning_curves(
    pos_grid: ArrayF | np.ndarray,
    place_centers: ArrayF | np.ndarray,
    place_width: float | ArrayF | np.ndarray,
    peak_rate_hz: float | ArrayF | np.ndarray = 20.0,
) -> ArrayF:
    """Build Gaussian tuning curves for 1D place cells."""
    pos: ArrayF = require_1d_float_array(pos_grid, name="pos_grid")
    centers: ArrayF = require_1d_float_array(place_centers, name="place_centers")
    n_neurons = centers.size

    widths: ArrayF = np.asarray(place_width, dtype=float)
    if widths.ndim == 0:
        widths = np.full(n_neurons, float(widths), dtype=float)
    if widths.ndim != 1 or widths.shape[0] != n_neurons:
        raise ValueError("place_width must be a scalar or shape (n_neurons,).")
    if np.any(widths <= 0.0) or not np.all(np.isfinite(widths)):
        raise ValueError("place_width must be finite and positive.")

    peaks: ArrayF = np.asarray(peak_rate_hz, dtype=float)
    if peaks.ndim == 0:
        peaks = np.full(n_neurons, float(peaks), dtype=float)
    if peaks.ndim != 1 or peaks.shape[0] != n_neurons:
        raise ValueError("peak_rate_hz must be a scalar or shape (n_neurons,).")
    if np.any(peaks < 0.0) or not np.all(np.isfinite(peaks)):
        raise ValueError("peak_rate_hz must be finite and non-negative.")

    pos_row = pos[None, :]
    centers_col = centers[:, None]
    widths_col = widths[:, None]
    rate = peaks[:, None] * np.exp(-0.5 * ((pos_row - centers_col) / widths_col) ** 2)
    return rate.astype(float)


def decode_bayes_grid_1d(
    spike_counts: ArrayI | ArrayF | np.ndarray,
    *,
    rate_hz: ArrayF | np.ndarray,
    dt_s: float,
    prior: ArrayF | np.ndarray | None = None,
) -> tuple[ArrayF, ArrayI, ArrayF]:
    """Decode 1D position on a grid using a Poisson model.

    Parameters
    ----------
    spike_counts:
        Array of spike counts with shape (n_time, n_neurons) or (n_neurons, n_time).
    rate_hz:
        Tuning curves with shape (n_neurons, n_pos), in Hz.
    dt_s:
        Bin width in seconds.
    prior:
        Optional prior over positions (length = n_pos). If None, uniform.

    Returns
    -------
    posterior:
        Posterior over positions with shape (n_time, n_pos).
    map_index:
        MAP index for each time bin, shape (n_time,).
    posterior_mean:
        Posterior mean position (index-weighted), shape (n_time,).
    """
    rates = np.asarray(rate_hz, dtype=float)
    if rates.ndim != 2:
        raise ValueError("rate_hz must be 2D with shape (n_neurons, n_pos).")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("rate_hz must be finite and non-negative.")

    counts = np.asarray(spike_counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("spike_counts must be 2D.")
    if counts.shape[0] == rates.shape[0]:
        counts = counts.T
    elif counts.shape[1] != rates.shape[0]:
        raise ValueError("spike_counts must have n_neurons matching rate_hz.")
    if np.any(counts < 0.0) or not np.all(np.isfinite(counts)):
        raise ValueError("spike_counts must be finite and non-negative.")

    n_neurons, n_pos = rates.shape
    dt = require_non_negative_scalar(dt_s, name="dt_s")
    if dt == 0.0:
        raise ValueError("dt_s must be positive.")

    if prior is None:
        prior_arr = np.full(n_pos, 1.0 / n_pos, dtype=float)
    else:
        prior_arr = require_1d_float_array(prior, name="prior")
        if prior_arr.shape[0] != n_pos:
            raise ValueError("prior must match the position grid length.")
        if np.any(prior_arr < 0.0) or not np.all(np.isfinite(prior_arr)):
            raise ValueError("prior must be finite and non-negative.")
        s = float(np.sum(prior_arr))
        if s <= 0.0:
            raise ValueError("prior must sum to a positive value.")
        prior_arr = prior_arr / s

    rate_dt = rates * dt
    log_rate_dt = np.where(rate_dt > 0.0, np.log(rate_dt), -np.inf)
    log_prior = np.where(prior_arr > 0.0, np.log(prior_arr), -np.inf)

    n_time = counts.shape[0]
    posterior = np.empty((n_time, n_pos), dtype=float)
    map_index = np.empty(n_time, dtype=np.int64)
    posterior_mean = np.empty(n_time, dtype=float)

    for t in range(n_time):
        k = counts[t][:, None]
        log_fact = np.vectorize(lgamma)(k + 1.0)
        term = np.where(rate_dt > 0.0, k * log_rate_dt, np.where(k == 0.0, 0.0, -np.inf))
        log_like = np.sum(term - rate_dt - log_fact, axis=0)
        log_post = log_like + log_prior
        log_post -= float(np.max(log_post))
        post = np.exp(log_post)
        s = float(np.sum(post))
        if s <= 0.0 or not np.isfinite(s):
            raise ValueError("Failed to normalize posterior.")
        post = post / s
        posterior[t] = post
        map_index[t] = int(np.argmax(post))
        posterior_mean[t] = float(np.sum(post * np.arange(n_pos)))

    return posterior, map_index, posterior_mean


__all__ = [
    "gaussian_tuning_curves",
    "decode_bayes_grid_1d",
]
