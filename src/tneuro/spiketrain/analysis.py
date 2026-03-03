from __future__ import annotations

import numpy as np

from tneuro.spiketrain.core import SpikeTrain
from tneuro.utils.validate import require_non_negative_scalar, require_positive_scalar


def _to_times(spike_times: np.ndarray | SpikeTrain) -> np.ndarray:
    if isinstance(spike_times, SpikeTrain):
        return np.asarray(spike_times.times_s, dtype=float)
    return np.asarray(spike_times, dtype=float)


def isi_values(spike_times: np.ndarray | SpikeTrain) -> np.ndarray:
    """Return inter-spike intervals (seconds)."""
    times = _to_times(spike_times)
    if times.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(np.sort(times))


def cv_isi(isi_s: np.ndarray) -> float:
    """Coefficient of variation (CV) of ISIs."""
    isi = np.asarray(isi_s, dtype=float)
    if isi.size < 2:
        return float("nan")
    mu = float(np.mean(isi))
    if mu == 0.0:
        return float("nan")
    return float(np.std(isi, ddof=1) / mu)


def lv_isi(isi_s: np.ndarray) -> float:
    """Local variation (LV) of ISIs."""
    isi = np.asarray(isi_s, dtype=float)
    if isi.size < 2:
        return float("nan")
    num = isi[:-1] - isi[1:]
    denom = isi[:-1] + isi[1:]
    valid = denom > 0.0
    if not np.any(valid):
        return float("nan")
    lv = np.mean((num[valid] / denom[valid]) ** 2)
    return float(3.0 * lv)


def isi_histogram(
    spike_times: np.ndarray | SpikeTrain,
    *,
    bin_width_s: float,
    max_isi_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of ISIs.

    Returns
    -------
    edges_s:
        Bin edges (seconds).
    counts:
        ISI counts per bin.
    """
    w = require_positive_scalar(bin_width_s, name="bin_width_s")
    isi = isi_values(spike_times)
    if isi.size == 0:
        edges = np.arange(0.0, w + 1e-12, w, dtype=float)
        return edges, np.zeros(edges.size - 1, dtype=int)

    if max_isi_s is None:
        max_isi = float(np.max(isi))
    else:
        max_isi = require_non_negative_scalar(max_isi_s, name="max_isi_s")
    if max_isi == 0.0:
        edges = np.arange(0.0, w + 1e-12, w, dtype=float)
        return edges, np.zeros(edges.size - 1, dtype=int)

    edges = np.arange(0.0, max_isi + w, w, dtype=float)
    counts, _ = np.histogram(isi, bins=edges)
    return edges, counts.astype(int)


def autocorrelogram(
    spike_times: np.ndarray | SpikeTrain,
    *,
    bin_width_s: float,
    max_lag_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Binned autocorrelogram of spike times.

    Returns
    -------
    lags_s:
        Bin centers (seconds), symmetric around zero.
    counts:
        Counts per lag bin (same length as ``lags_s``).
    """
    w = require_positive_scalar(bin_width_s, name="bin_width_s")
    max_lag = require_positive_scalar(max_lag_s, name="max_lag_s")
    times = np.sort(_to_times(spike_times))
    lags = np.arange(-max_lag, max_lag + 1e-12, w, dtype=float)
    if times.size < 2:
        return lags, np.zeros(lags.size, dtype=int)

    diffs: list[float] = []
    for i in range(times.size):
        dt = times - times[i]
        valid = (dt != 0.0) & (np.abs(dt) <= max_lag)
        diffs.extend(dt[valid].tolist())

    if len(diffs) == 0:
        return lags, np.zeros(lags.size, dtype=int)

    diffs_arr = np.asarray(diffs, dtype=float)
    # Map diffs to bin indices centered at multiples of w for symmetry.
    idx = np.rint((diffs_arr + max_lag) / w).astype(int)
    idx = idx[(idx >= 0) & (idx < lags.size)]
    counts = np.bincount(idx, minlength=lags.size)
    return lags, counts.astype(int)


__all__ = [
    "autocorrelogram",
    "cv_isi",
    "isi_histogram",
    "isi_values",
    "lv_isi",
]
