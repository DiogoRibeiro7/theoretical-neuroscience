from __future__ import annotations

import numpy as np

from tneuro.decoding.bayes import decode_bayes_poisson


def test_decode_bayes_poisson_prefers_correct_state() -> None:
    rate_hz = np.array([[5.0, 1.0], [1.0, 5.0]])
    spike_counts = np.array([3.0, 0.0])
    posterior, map_idx = decode_bayes_poisson(rate_hz, spike_counts, dt_s=1.0)

    assert posterior.shape == (2,)
    assert np.isclose(posterior.sum(), 1.0)
    assert map_idx == 0
    assert posterior[0] > posterior[1]


def test_decode_bayes_poisson_respects_prior() -> None:
    rate_hz = np.array([[5.0, 5.0], [5.0, 5.0]])
    spike_counts = np.array([1.0, 1.0])
    prior = np.array([0.9, 0.1])
    posterior, map_idx = decode_bayes_poisson(rate_hz, spike_counts, dt_s=1.0, prior=prior)

    assert np.isclose(posterior.sum(), 1.0)
    assert map_idx == 0
    assert np.allclose(posterior, prior / prior.sum())


def test_decode_bayes_poisson_zero_rate_with_spikes_impossible() -> None:
    rate_hz = np.array([[0.0, 5.0], [5.0, 5.0]])
    spike_counts = np.array([1.0, 0.0])
    posterior, map_idx = decode_bayes_poisson(rate_hz, spike_counts, dt_s=1.0)

    assert np.isclose(posterior.sum(), 1.0)
    assert posterior[0] == 0.0
    assert map_idx == 1
