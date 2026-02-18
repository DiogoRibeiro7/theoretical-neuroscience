import numpy as np

from tneuro.encoding.reverse_correlation import spike_triggered_average


def test_spike_triggered_average_correlates_with_filter() -> None:
    rng = np.random.default_rng(123)
    fs = 1000.0
    n_samples = 20000
    stim = rng.normal(0.0, 1.0, size=n_samples)

    t = np.arange(0, 0.05, 1.0 / fs)
    true_filter = np.exp(-t / 0.01)
    true_filter /= np.linalg.norm(true_filter)

    drive = np.convolve(stim, true_filter, mode="full")[:n_samples]
    rate = 30.0 * np.exp(0.5 * drive)
    rate = np.clip(rate, 0.0, 120.0)
    p_spike = np.clip(rate / fs, 0.0, 0.3)
    spikes = rng.random(n_samples) < p_spike
    spike_times = np.flatnonzero(spikes) / fs

    sta, lags_s = spike_triggered_average(
        stim,
        spike_times,
        fs_hz=fs,
        window_s=(t[-1], 0.0),
    )

    assert lags_s.size == true_filter.size
    sta_rev = sta[::-1]
    sta_rev /= np.linalg.norm(sta_rev)
    corr = float(np.dot(sta_rev, true_filter))
    assert corr > 0.7


def test_spike_triggered_average_discards_edge_spikes() -> None:
    fs = 1000.0
    stim = np.zeros(1000, dtype=float)
    spike_times = np.array([0.0, 0.5, 0.999])

    sta, lags_s = spike_triggered_average(
        stim,
        spike_times,
        fs_hz=fs,
        window_s=(0.1, 0.05),
    )

    assert sta.shape == lags_s.shape
    assert np.allclose(sta, 0.0)
