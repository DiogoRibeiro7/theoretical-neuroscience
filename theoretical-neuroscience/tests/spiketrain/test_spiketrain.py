from __future__ import annotations

import numpy as np

from tneuro.spiketrain.core import SpikeTrain


def test_rate_and_isi() -> None:
    st = SpikeTrain(np.array([0.1, 0.2, 0.4]), t_start_s=0.0, t_stop_s=1.0)
    assert st.n_spikes() == 3
    assert np.isclose(st.rate_hz(), 3.0)

    isi = st.isi_s()
    assert np.allclose(isi, np.array([0.1, 0.2]))

def test_bin_counts() -> None:
    st = SpikeTrain(np.array([0.05, 0.15, 0.95]), t_start_s=0.0, t_stop_s=1.0)
    edges, counts = st.bin_counts(bin_width_s=0.5)
    assert len(edges) == 3
    assert counts.sum() == 3
