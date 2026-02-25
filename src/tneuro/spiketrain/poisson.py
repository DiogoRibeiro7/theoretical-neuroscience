from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tneuro.spiketrain.core import SpikeTrain
from tneuro.utils.validate import require_1d_float_array, require_non_negative_scalar

RateFunc = Callable[[np.ndarray], np.ndarray]


def _require_increasing_grid(t_grid_s: np.ndarray) -> None:
    if t_grid_s.size < 2:
        raise ValueError("t_grid_s must have at least 2 points.")
    if not np.all(np.isfinite(t_grid_s)):
        raise ValueError("t_grid_s must be finite.")
    if np.any(np.diff(t_grid_s) <= 0.0):
        raise ValueError("t_grid_s must be strictly increasing.")


def _validate_rate_array(rate_hz: np.ndarray, *, name: str) -> np.ndarray:
    rate = np.asarray(rate_hz, dtype=float)
    if rate.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={rate.shape}")
    if not np.all(np.isfinite(rate)):
        raise ValueError(f"{name} must be finite.")
    if np.any(rate < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return rate


def _rate_from_grid(t: np.ndarray, *, t_grid_s: np.ndarray, rate_hz: np.ndarray) -> np.ndarray:
    return np.interp(t, t_grid_s, rate_hz, left=rate_hz[0], right=rate_hz[-1])


def _rate_from_callable(t: np.ndarray, *, rate_fn: RateFunc) -> np.ndarray:
    rate = np.asarray(rate_fn(t), dtype=float)
    if rate.shape != t.shape:
        raise ValueError("rate_hz callable must return an array with the same shape as input.")
    if not np.all(np.isfinite(rate)):
        raise ValueError("rate_hz callable must return finite values.")
    if np.any(rate < 0.0):
        raise ValueError("rate_hz callable must return non-negative values.")
    return rate


def generate_inhom_poisson(
    rate_hz: RateFunc | np.ndarray,
    *,
    t_start_s: float,
    t_stop_s: float,
    t_grid_s: np.ndarray | None = None,
    rate_hz_max: float | None = None,
    seed: int | None = None,
) -> SpikeTrain:
    """Generate an inhomogeneous Poisson spike train using thinning.

    Parameters
    ----------
    rate_hz:
        Either a callable ``rate_hz(t)`` (Hz) or a 1D array of rates defined
        on ``t_grid_s``.
    t_start_s:
        Start time (seconds).
    t_stop_s:
        Stop time (seconds), must be > ``t_start_s``.
    t_grid_s:
        1D time grid (seconds) for array-based rates or for estimating
        ``rate_hz_max`` when ``rate_hz`` is callable.
    rate_hz_max:
        Optional upper bound on the rate (Hz). If not provided for callable
        rates, it will be estimated from ``t_grid_s``.
    seed:
        Seed for reproducible random number generation.
    """
    # TODO: Add an option to return the thinning acceptance ratio. LABELS:spiketrain,enhancement ASSIGNEE:diogoribeiro7
    # TODO: Add a fast path for constant rate without thinning. LABELS:spiketrain,performance ASSIGNEE:diogoribeiro7
    t0 = float(t_start_s)
    t1 = float(t_stop_s)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("Require finite times with t_stop_s > t_start_s.")

    rate_fn: Callable[[np.ndarray], np.ndarray]
    if callable(rate_hz):
        if rate_hz_max is None:
            if t_grid_s is None:
                raise ValueError("Provide t_grid_s or rate_hz_max for callable rate_hz.")
            t_grid = require_1d_float_array(t_grid_s, name="t_grid_s")
            _require_increasing_grid(t_grid)
            rate_grid = _rate_from_callable(t_grid, rate_fn=rate_hz)
            rate_hz_max = float(np.max(rate_grid)) if rate_grid.size > 0 else 0.0
        def rate_fn(t: np.ndarray) -> np.ndarray:
            return _rate_from_callable(t, rate_fn=rate_hz)
    else:
        if t_grid_s is None:
            raise ValueError("t_grid_s is required when rate_hz is an array.")
        t_grid = require_1d_float_array(t_grid_s, name="t_grid_s")
        _require_increasing_grid(t_grid)
        rate_grid = _validate_rate_array(rate_hz, name="rate_hz")
        if rate_grid.shape != t_grid.shape:
            raise ValueError("rate_hz and t_grid_s must have the same shape.")
        if t_grid[0] > t0 or t_grid[-1] < t1:
            raise ValueError("t_grid_s must cover [t_start_s, t_stop_s].")
        rate_hz_max = float(np.max(rate_grid)) if rate_grid.size > 0 else 0.0
        def rate_fn(t: np.ndarray) -> np.ndarray:
            return _rate_from_grid(t, t_grid_s=t_grid, rate_hz=rate_grid)

    rate_hz_max = require_non_negative_scalar(rate_hz_max, name="rate_hz_max")
    if rate_hz_max == 0.0:
        return SpikeTrain(times_s=np.asarray([], dtype=float), t_start_s=t0, t_stop_s=t1)

    rng = np.random.default_rng(seed)
    times: list[float] = []
    t = t0
    while True:
        t += float(rng.exponential(1.0 / rate_hz_max))
        if t >= t1:
            break
        accept_prob = float(rate_fn(np.asarray([t], dtype=float))[0] / rate_hz_max)
        if accept_prob > 1.0:
            raise ValueError("rate_hz exceeds rate_hz_max; increase rate_hz_max.")
        if rng.random() < accept_prob:
            times.append(t)

    return SpikeTrain(times_s=np.asarray(times, dtype=float), t_start_s=t0, t_stop_s=t1, sorted=True)
