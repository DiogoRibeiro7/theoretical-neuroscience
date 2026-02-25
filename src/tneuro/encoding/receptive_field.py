from __future__ import annotations

import numpy as np

from tneuro.utils.validate import require_1d_float_array, require_non_negative_scalar


def fit_linear_rf(
    stim: np.ndarray,
    spikes: np.ndarray,
    lags: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a linear receptive field using ridge regression.

    Parameters
    ----------
    stim:
        1D stimulus array.
    spikes:
        1D spike response array aligned to ``stim`` (e.g., binned counts).
    lags:
        1D integer lags in samples (negative values look back in time).
    alpha:
        Ridge regularization strength (non-negative).

    Returns
    -------
    rf:
        Estimated receptive field weights with shape ``(n_lags,)``.
    lags:
        The lags used (returned for convenience).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> stim = rng.normal(0.0, 1.0, size=1000)
    >>> lags = np.arange(-5, 1)
    >>> true_rf = np.linspace(0.1, 1.0, lags.size)
    >>> X = np.stack([np.roll(stim, -lag) for lag in lags], axis=1)
    >>> spikes = X @ true_rf + rng.normal(0.0, 0.1, size=stim.size)
    >>> rf, _ = fit_linear_rf(stim, spikes, lags, alpha=1e-2)
    >>> rf.shape
    (6,)
    """
    # TODO: Add optional standardization for stimulus. LABELS:encoding,enhancement ASSIGNEE:diogoribeiro7
    # TODO: Add cross-validation helper for alpha. LABELS:encoding,analysis ASSIGNEE:diogoribeiro7
    x = require_1d_float_array(stim, name="stim")
    y = require_1d_float_array(spikes, name="spikes")
    lags_arr = np.asarray(lags, dtype=int)
    alpha_val = require_non_negative_scalar(alpha, name="alpha")

    if x.shape != y.shape:
        raise ValueError("stim and spikes must have the same shape.")
    if lags_arr.ndim != 1 or lags_arr.size == 0:
        raise ValueError("lags must be a non-empty 1D array of integers.")

    min_lag = int(np.min(lags_arr))
    max_lag = int(np.max(lags_arr))
    start = max(0, -min_lag)
    stop = x.size - max_lag
    if stop <= start:
        raise ValueError("lags are incompatible with stim length.")

    n_rows = stop - start
    n_lags = lags_arr.size
    x_mat = np.empty((n_rows, n_lags), dtype=float)
    for j, lag in enumerate(lags_arr):
        idx = np.arange(start, stop) + lag
        x_mat[:, j] = x[idx]

    y_valid = y[start:stop]
    xtx = x_mat.T @ x_mat
    xtx.flat[:: n_lags + 1] += alpha_val
    rf = np.linalg.solve(xtx, x_mat.T @ y_valid)

    return rf, lags_arr
