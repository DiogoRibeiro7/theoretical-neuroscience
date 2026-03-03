from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tneuro.spiketrain.core import SpikeTrain
from tneuro.spiketrain.poisson import generate_inhom_poisson as _generate_inhom_poisson
from tneuro.typing import ArrayF
from tneuro.utils.validate import require_non_negative_scalar, require_positive_scalar

RateFunc = Callable[[ArrayF | np.ndarray], ArrayF]


def _validate_window(t_start_s: float, t_stop_s: float) -> tuple[float, float]:
    t0 = float(t_start_s)
    t1 = float(t_stop_s)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("Require finite times with t_stop_s > t_start_s.")
    return t0, t1


def _apply_refractory(times: np.ndarray, *, refractory_s: float) -> np.ndarray:
    if times.size == 0 or refractory_s <= 0.0:
        return times
    keep = [times[0]]
    last = times[0]
    for t in times[1:]:
        if t - last >= refractory_s:
            keep.append(t)
            last = t
    return np.asarray(keep, dtype=float)


def generate_hom_poisson(
    rate_hz: float,
    *,
    t_start_s: float,
    t_stop_s: float,
    seed: int | None = None,
    refractory_s: float = 0.0,
) -> SpikeTrain:
    """Generate a homogeneous Poisson spike train.

    Parameters
    ----------
    rate_hz:
        Constant rate in Hz.
    t_start_s:
        Start time (seconds).
    t_stop_s:
        Stop time (seconds), must be > ``t_start_s``.
    seed:
        Random seed for reproducibility.
    refractory_s:
        Absolute refractory period (seconds). If > 0, each ISI is at least this long.
    """
    t0, t1 = _validate_window(t_start_s, t_stop_s)
    rate = require_non_negative_scalar(rate_hz, name="rate_hz")
    ref = require_non_negative_scalar(refractory_s, name="refractory_s")
    if rate == 0.0:
        return SpikeTrain(times_s=np.asarray([], dtype=float), t_start_s=t0, t_stop_s=t1)

    rng = np.random.default_rng(seed)
    times: list[float] = []
    t = t0
    while True:
        isi = float(rng.exponential(1.0 / rate))
        if ref > 0.0:
            isi += ref
        t += isi
        if t >= t1:
            break
        times.append(t)

    return SpikeTrain(times_s=np.asarray(times, dtype=float), t_start_s=t0, t_stop_s=t1, sorted=True)


def generate_inhom_poisson(
    rate_hz: RateFunc | ArrayF | np.ndarray,
    *,
    t_start_s: float,
    t_stop_s: float,
    t_grid_s: ArrayF | np.ndarray | None = None,
    rate_hz_max: float | None = None,
    seed: int | None = None,
    refractory_s: float = 0.0,
) -> SpikeTrain:
    """Generate an inhomogeneous Poisson spike train using thinning.

    This wraps :func:`tneuro.spiketrain.poisson.generate_inhom_poisson` and adds an
    optional absolute refractory filter.
    """
    st = _generate_inhom_poisson(
        rate_hz,
        t_start_s=t_start_s,
        t_stop_s=t_stop_s,
        t_grid_s=t_grid_s,
        rate_hz_max=rate_hz_max,
        seed=seed,
    )
    ref = require_non_negative_scalar(refractory_s, name="refractory_s")
    if ref == 0.0 or st.times_s.size == 0:
        return st
    times = _apply_refractory(st.times_s, refractory_s=ref)
    return SpikeTrain(times_s=times, t_start_s=st.t_start_s, t_stop_s=st.t_stop_s, sorted=True)


def generate_renewal_gamma(
    rate_hz: float,
    *,
    shape_k: float,
    t_start_s: float,
    t_stop_s: float,
    seed: int | None = None,
    refractory_s: float = 0.0,
) -> SpikeTrain:
    """Generate a renewal process with Gamma-distributed ISIs.

    Parameters
    ----------
    rate_hz:
        Mean firing rate (Hz).
    shape_k:
        Gamma shape parameter (k). CV = 1 / sqrt(k).
    t_start_s:
        Start time (seconds).
    t_stop_s:
        Stop time (seconds), must be > ``t_start_s``.
    seed:
        Random seed for reproducibility.
    refractory_s:
        Absolute refractory period (seconds). If > 0, each ISI is at least this long.
    """
    t0, t1 = _validate_window(t_start_s, t_stop_s)
    rate = require_positive_scalar(rate_hz, name="rate_hz")
    k = require_positive_scalar(shape_k, name="shape_k")
    ref = require_non_negative_scalar(refractory_s, name="refractory_s")
    scale = 1.0 / (rate * k)

    rng = np.random.default_rng(seed)
    times: list[float] = []
    t = t0
    while True:
        isi = float(rng.gamma(shape=k, scale=scale))
        if ref > 0.0:
            isi += ref
        t += isi
        if t >= t1:
            break
        times.append(t)

    return SpikeTrain(times_s=np.asarray(times, dtype=float), t_start_s=t0, t_stop_s=t1, sorted=True)


__all__ = [
    "generate_hom_poisson",
    "generate_inhom_poisson",
    "generate_renewal_gamma",
]
