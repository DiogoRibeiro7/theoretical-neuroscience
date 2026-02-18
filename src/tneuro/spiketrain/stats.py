from __future__ import annotations

import numpy as np

from .core import SpikeTrain


def fano_factor_counts(counts: np.ndarray) -> float:
    """Fano factor of binned spike counts: Var / Mean.

    Returns NaN if mean is 0 or if input is empty.
    """
    x = np.asarray(counts, dtype=float)
    if x.size == 0:
        return float("nan")
    mu = float(np.mean(x))
    if mu == 0.0:
        return float("nan")
    return float(np.var(x, ddof=1) / mu) if x.size > 1 else float("nan")


def fano_factor_spiketrain(st: SpikeTrain, *, bin_width_s: float) -> float:
    """Fano factor computed from a SpikeTrain using fixed-width bins."""
    _, counts = st.bin_counts(bin_width_s=bin_width_s)
    return fano_factor_counts(counts)
