from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tneuro.utils.validate import require_1d_float_array, require_positive_scalar


@dataclass(frozen=True, slots=True)
class SpikeTrain:
    """Spike train defined by spike times in seconds.

    Notes
    -----
    - Times are assumed to be in seconds.
    - This class does not assume any sampling grid.
    - Times are expected to be sorted; use `sorted=True` if you guarantee this.

    Parameters
    ----------
    times_s:
        1D array of spike times in seconds.
    t_start_s:
        Start time of the recording window (seconds).
    t_stop_s:
        Stop time of the recording window (seconds). Must be > t_start_s.
    sorted:
        If False, times will be sorted on construction.
    """

    times_s: np.ndarray
    t_start_s: float
    t_stop_s: float

    def __init__(
        self,
        times_s: np.ndarray,
        *,
        t_start_s: float,
        t_stop_s: float,
        sorted: bool = False,
    ) -> None:
        t0 = float(t_start_s)
        t1 = float(t_stop_s)
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            raise ValueError("Require finite times with t_stop_s > t_start_s.")

        ts = require_1d_float_array(times_s, name="times_s")
        if not sorted:
            ts = np.sort(ts)

        # Allow empty trains; otherwise enforce bounds
        if ts.size > 0 and (ts[0] < t0 or ts[-1] > t1):
            raise ValueError("Spike times must lie within [t_start_s, t_stop_s].")

        object.__setattr__(self, "times_s", ts)
        object.__setattr__(self, "t_start_s", t0)
        object.__setattr__(self, "t_stop_s", t1)

    def duration_s(self) -> float:
        """Recording duration in seconds."""
        return self.t_stop_s - self.t_start_s

    def n_spikes(self) -> int:
        """Number of spikes."""
        return int(self.times_s.size)

    def rate_hz(self) -> float:
        """Mean firing rate (Hz) over the recording window.

        Examples
        --------
        >>> import numpy as np
        >>> st = SpikeTrain(times_s=np.array([0.1, 0.6, 0.9]), t_start_s=0.0, t_stop_s=1.0)
        >>> st.rate_hz()
        3.0
        """
        dur = self.duration_s()
        if dur <= 0.0:
            return float("nan")
        return self.n_spikes() / dur

    def isi_s(self) -> np.ndarray:
        """Inter-spike intervals (seconds). Empty if fewer than 2 spikes.

        Examples
        --------
        >>> import numpy as np
        >>> st = SpikeTrain(times_s=np.array([0.1, 0.4, 1.0]), t_start_s=0.0, t_stop_s=2.0)
        >>> st.isi_s()
        array([0.3, 0.6])
        """
        if self.times_s.size < 2:
            return np.asarray([], dtype=float)
        return np.diff(self.times_s)

    def cv_isi(self) -> float:
        """Coefficient of variation (CV) of ISI.

        Returns NaN if fewer than 2 spikes or if mean ISI is 0.
        """
        isi = self.isi_s()
        if isi.size == 0:
            return float("nan")
        mu = float(np.mean(isi))
        if mu == 0.0:
            return float("nan")
        return float(np.std(isi, ddof=1) / mu) if isi.size > 1 else float("nan")

    # TODO: Add burstiness metric helper. LABELS:spiketrain,enhancement ASSIGNEE:diogoribeiro7

    def bin_counts(self, *, bin_width_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Bin spikes into counts.

        Parameters
        ----------
        bin_width_s:
            Bin width in seconds (>0).

        Returns
        -------
        edges_s:
            Bin edges (seconds), length = n_bins + 1
        counts:
            Spike counts per bin, length = n_bins

        Examples
        --------
        >>> import numpy as np
        >>> st = SpikeTrain(times_s=np.array([0.1, 0.4, 1.2]), t_start_s=0.0, t_stop_s=2.0)
        >>> edges_s, counts = st.bin_counts(bin_width_s=0.5)
        >>> edges_s
        array([0. , 0.5, 1. , 1.5, 2. ])
        >>> counts
        array([2, 0, 1, 0])
        """
        # TODO: Add optional return of bin centers. LABELS:spiketrain,enhancement ASSIGNEE:diogoribeiro7
        w = require_positive_scalar(bin_width_s, name="bin_width_s")
        edges = np.arange(self.t_start_s, self.t_stop_s + w, w, dtype=float)
        # Ensure last edge is exactly t_stop for reproducibility
        edges[-1] = self.t_stop_s
        counts, _ = np.histogram(self.times_s, bins=edges)
        return edges, counts.astype(int)
