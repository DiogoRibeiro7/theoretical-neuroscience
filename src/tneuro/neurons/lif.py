from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from tneuro.utils.random import make_rng
from tneuro.utils.validate import require_non_negative_scalar, require_positive_scalar


@dataclass(frozen=True, slots=True)
class LIFParams:
    """Leaky integrate-and-fire (LIF) parameters.

    Model
    -----
    dv/dt = -(v - v_rest)/tau_m + (r_m * I(t))/tau_m

    Threshold/reset:
    - if v >= v_th: spike, set v <- v_reset and hold for refractory_s.

    Units:
    - tau_m_s: seconds
    - voltages: arbitrary units (consistent)
    - r_m_ohm: arbitrary scale (consistent with I)
    - I: arbitrary units (e.g., A)
    """
    tau_m_s: float
    v_rest: float
    v_reset: float
    v_th: float
    r_m_ohm: float
    refractory_s: float = 0.0

    def validate(self) -> None:
        tau = require_positive_scalar(self.tau_m_s, name="tau_m_s")
        _ = float(self.v_rest)
        _ = float(self.v_reset)
        _ = float(self.v_th)
        _ = float(self.r_m_ohm)
        ref = require_non_negative_scalar(self.refractory_s, name="refractory_s")

        if not (np.isfinite(tau) and np.isfinite(ref)):
            raise ValueError("tau_m_s and refractory_s must be finite.")
        if not np.isfinite(self.v_th) or not np.isfinite(self.v_rest) or not np.isfinite(self.v_reset):
            raise ValueError("Voltages must be finite.")
        if not np.isfinite(self.r_m_ohm):
            raise ValueError("r_m_ohm must be finite.")
        if self.v_th <= self.v_reset:
            raise ValueError("Require v_th > v_reset for a meaningful threshold/reset.")


Current: TypeAlias = float | np.ndarray | Callable[[np.ndarray], np.ndarray]


def _evaluate_current(
    i_inj: Current,
    t_s: np.ndarray,
) -> np.ndarray:
    if callable(i_inj):
        out = np.asarray(i_inj(t_s), dtype=float)
    else:
        out = np.asarray(i_inj, dtype=float)
        if out.ndim == 0:
            out = np.full_like(t_s, float(out), dtype=float)
        elif out.shape != t_s.shape:
            raise ValueError("If i_inj_a is an array, it must match t_s shape.")
    if out.shape != t_s.shape:
        raise ValueError("Current must evaluate to an array with same shape as t_s.")
    return out


def simulate_lif(
    *,
    params: LIFParams,
    t_stop_s: float,
    dt_s: float,
    i_inj_a: Current,
    v0: float | None = None,
    noise_std_a: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a LIF neuron with Euler integration.

    Parameters
    ----------
    params:
        LIF parameters.
    t_stop_s:
        Simulation duration in seconds.
    dt_s:
        Time step in seconds (>0).
    i_inj_a:
        Injected current. Can be:
        - scalar (constant current),
        - array of same length as the time grid,
        - callable f(t) returning array of currents.
    v0:
        Initial membrane potential. Defaults to v_rest.
    noise_std_a:
        Standard deviation of additive Gaussian noise on current (same units as current).
    seed:
        Optional random seed.

    Returns
    -------
    t_s:
        Time grid (seconds).
    v:
        Membrane potential trace.
    spike_times_s:
        Spike times (seconds).
    """
    # TODO: Add option to return refractory mask. LABELS:neurons,enhancement ASSIGNEE:diogoribeiro7
    # TODO: Add option to return input current trace. LABELS:neurons,enhancement ASSIGNEE:diogoribeiro7
    params.validate()
    t_stop = require_positive_scalar(t_stop_s, name="t_stop_s")
    dt = require_positive_scalar(dt_s, name="dt_s")
    noise_std = float(noise_std_a)
    if noise_std < 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std_a must be finite and >= 0.")

    n = int(np.floor(t_stop / dt)) + 1
    t = np.arange(n, dtype=float) * dt
    i = _evaluate_current(i_inj_a, t)

    rng = make_rng(seed)
    if noise_std > 0.0:
        i = i + rng.normal(loc=0.0, scale=noise_std, size=i.shape)

    v = np.empty_like(t)
    v[0] = float(params.v_rest if v0 is None else v0)

    tau = float(params.tau_m_s)
    v_rest = float(params.v_rest)
    v_reset = float(params.v_reset)
    v_th = float(params.v_th)
    r_m = float(params.r_m_ohm)
    ref = float(params.refractory_s)

    spike_times: list[float] = []
    refractory_steps = int(np.round(ref / dt)) if ref > 0.0 else 0
    ref_left = 0

    for k in range(1, n):
        if ref_left > 0:
            v[k] = v_reset
            ref_left -= 1
            continue

        dv = (-(v[k - 1] - v_rest) + r_m * i[k - 1]) * (dt / tau)
        v_k = v[k - 1] + dv

        if v_k >= v_th:
            spike_times.append(t[k])
            v[k] = v_reset
            ref_left = refractory_steps
        else:
            v[k] = v_k

    return t, v, np.asarray(spike_times, dtype=float)
