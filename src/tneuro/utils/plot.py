from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _require_matplotlib() -> object:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - exercised in import error tests
        raise ImportError(
            "matplotlib is required for plotting. Install with "
            "`pip install \"theoretical-neuroscience[plot]\"`."
        ) from exc
    return plt


def plot_spike_raster(
    spike_times_s: Sequence[np.ndarray] | np.ndarray,
    *,
    ax: object | None = None,
    t_start_s: float | None = None,
    t_stop_s: float | None = None,
    color: str = "k",
    linewidth: float = 1.0,
) -> object:
    """Plot a simple spike raster.

    Parameters
    ----------
    spike_times_s:
        Either a 1D array of spike times for a single unit, or a sequence of
        1D arrays (one per unit / trial).
    ax:
        Optional matplotlib Axes to plot into.
    t_start_s, t_stop_s:
        Optional x-limits (seconds).
    color, linewidth:
        Line styling.
    """
    # TODO: Add support for per-trial colors. LABELS:utils,enhancement ASSIGNEE:diogoribeiro7
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots()

    if isinstance(spike_times_s, np.ndarray) and spike_times_s.ndim == 1:
        trains = [spike_times_s]
    elif isinstance(spike_times_s, np.ndarray) and spike_times_s.ndim != 1:
        raise ValueError("spike_times_s must be 1D or a sequence of 1D arrays.")
    else:
        trains = list(spike_times_s)

    for idx, times in enumerate(trains):
        t = np.asarray(times, dtype=float)
        if t.ndim != 1:
            raise ValueError("Each spike train must be 1D.")
        if t.size == 0:
            continue
        ax.vlines(t, idx + 0.5, idx + 1.5, color=color, linewidth=linewidth)

    ax.set_ylabel("trial")
    ax.set_xlabel("time (s)")
    if t_start_s is not None or t_stop_s is not None:
        ax.set_xlim(left=t_start_s, right=t_stop_s)
    return ax


def plot_voltage_trace(
    t_s: np.ndarray,
    v: np.ndarray,
    *,
    ax: object | None = None,
    color: str = "C0",
    linewidth: float = 1.5,
) -> object:
    """Plot a voltage trace."""
    # TODO: Add optional spike overlay markers. LABELS:utils,enhancement ASSIGNEE:diogoribeiro7
    plt = _require_matplotlib()

    t = np.asarray(t_s, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if t.shape != v_arr.shape:
        raise ValueError("t_s and v must have the same shape.")
    if t.ndim != 1:
        raise ValueError("t_s and v must be 1D arrays.")

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, v_arr, color=color, linewidth=linewidth)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("voltage")
    return ax
