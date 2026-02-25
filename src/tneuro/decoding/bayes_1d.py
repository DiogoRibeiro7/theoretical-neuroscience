from __future__ import annotations

from math import lgamma

import numpy as np

from tneuro.typing import ArrayF, ArrayI
from tneuro.utils.validate import require_1d_float_array, require_non_negative_scalar


def decode_position(
    spike_counts: ArrayI | ArrayF | np.ndarray,
    *,
    pos_grid: ArrayF | np.ndarray,
    place_centers: ArrayF | np.ndarray,
    place_width: float | ArrayF | np.ndarray,
    peak_rate_hz: float | ArrayF | np.ndarray = 20.0,
    dt_s: float = 0.1,
    prior: ArrayF | np.ndarray | None = None,
) -> tuple[ArrayF, ArrayF]:
    """Decode position from place-cell spike counts with a Poisson model.

    Parameters
    ----------
    spike_counts:
        Array of spike counts with shape ``(n_neurons,)`` or ``(n_neurons, n_time)``.
        If provided as ``(n_time, n_neurons)``, it will be transposed.
    pos_grid:
        1D array of position grid points.
    place_centers:
        1D array of place field centers for each neuron.
    place_width:
        Scalar or 1D array of Gaussian widths (same units as ``pos_grid``).
    peak_rate_hz:
        Scalar or 1D array of peak firing rates for each neuron (Hz).
    dt_s:
        Bin width in seconds for ``spike_counts``.
    prior:
        Optional prior over positions (length = ``pos_grid``). If None, uniform.

    Returns
    -------
    posterior:
        Posterior over positions with shape ``(n_time, n_pos)``.
    pos_grid:
        The position grid (returned for convenience).
    """
    pos: ArrayF = require_1d_float_array(pos_grid, name="pos_grid")
    centers: ArrayF = require_1d_float_array(place_centers, name="place_centers")
    n_neurons = centers.size
    dt = require_non_negative_scalar(dt_s, name="dt_s")
    if dt == 0.0:
        raise ValueError("dt_s must be positive.")

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

    spikes: ArrayF = np.asarray(spike_counts, dtype=float)
    if spikes.ndim == 1:
        spikes = spikes[:, None]
    elif spikes.ndim == 2:
        if spikes.shape[0] == n_neurons:
            pass
        elif spikes.shape[1] == n_neurons:
            spikes = spikes.T
        else:
            raise ValueError("spike_counts must have shape (n_neurons, n_time).")
    else:
        raise ValueError("spike_counts must be 1D or 2D.")
    if np.any(spikes < 0.0) or not np.all(np.isfinite(spikes)):
        raise ValueError("spike_counts must be finite and non-negative.")

    if prior is None:
        prior_arr: ArrayF = np.full(pos.size, 1.0 / pos.size, dtype=float)
    else:
        prior_arr = require_1d_float_array(prior, name="prior")
        if prior_arr.shape != pos.shape:
            raise ValueError("prior must have the same shape as pos_grid.")
        if np.any(prior_arr < 0.0) or not np.all(np.isfinite(prior_arr)):
            raise ValueError("prior must be finite and non-negative.")
        s = float(np.sum(prior_arr))
        if s == 0.0:
            raise ValueError("prior must sum to a positive value.")
        prior_arr = np.asarray(prior_arr / s, dtype=float)

    pos_row = pos[None, :]
    centers_col = centers[:, None]
    widths_col = widths[:, None]
    rate = peaks[:, None] * np.exp(-0.5 * ((pos_row - centers_col) / widths_col) ** 2)
    rate_dt = rate * dt

    log_rate_dt = np.where(rate_dt > 0.0, np.log(rate_dt), -np.inf)
    log_prior = np.log(prior_arr)

    n_time = spikes.shape[1]
    posterior: ArrayF = np.empty((n_time, pos.size), dtype=float)
    for t in range(n_time):
        k = spikes[:, t]
        log_fact = np.vectorize(lgamma)(k + 1.0)[:, None]
        term1 = np.where(rate_dt > 0.0, k[:, None] * log_rate_dt, np.where(k[:, None] == 0.0, 0.0, -np.inf))
        ll = np.sum(term1 - rate_dt - log_fact, axis=0)
        log_post = ll + log_prior
        log_post -= float(np.max(log_post))
        post = np.exp(log_post)
        post_sum = float(np.sum(post))
        posterior[t] = post / post_sum if post_sum > 0.0 else np.full(pos.size, 1.0 / pos.size)

    return posterior, pos


__all__ = ["decode_position"]
