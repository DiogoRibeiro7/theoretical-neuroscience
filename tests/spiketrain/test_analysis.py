import numpy as np

from tneuro.spiketrain.analysis import autocorrelogram, cv_isi, isi_histogram, isi_values, lv_isi


def test_isi_histogram_counts() -> None:
    spike_times = np.array([0.1, 0.3, 0.9])
    edges, counts = isi_histogram(spike_times, bin_width_s=0.2)
    # ISIs are [0.2, 0.6]
    idx_02 = np.where(np.isclose(edges, 0.2))[0]
    assert counts.sum() == 2
    assert idx_02.size > 0


def test_autocorrelogram_symmetry() -> None:
    spike_times = np.array([0.0, 0.1, 0.2])
    lags, counts = autocorrelogram(spike_times, bin_width_s=0.05, max_lag_s=0.2)
    # Symmetry around zero
    assert np.allclose(counts, counts[::-1])
    assert lags.shape == counts.shape


def test_cv_lv_basic() -> None:
    isi = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isfinite(cv_isi(isi))
    assert np.isfinite(lv_isi(isi))
    isi_vals = isi_values(np.cumsum(np.concatenate([[0.0], isi])))
    assert np.allclose(isi_vals, isi)
