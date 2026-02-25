from __future__ import annotations

import numpy as np

from tneuro.decoding.bayes import decode_bayes_poisson, decode_bayes_poisson_batch


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


def test_decode_bayes_poisson_batch_matches_single() -> None:
    rate_hz = np.array([[5.0, 1.0], [1.0, 5.0]])
    counts = np.array([[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    post_batch, map_batch = decode_bayes_poisson_batch(rate_hz, counts, dt_s=1.0)

    assert post_batch.shape == (3, 2)
    assert map_batch.shape == (3,)
    assert np.allclose(post_batch.sum(axis=1), 1.0)

    for i in range(counts.shape[0]):
        post_single, map_single = decode_bayes_poisson(rate_hz, counts[i], dt_s=1.0)
        assert np.allclose(post_batch[i], post_single)
        assert map_batch[i] == map_single


# TODO: Add a seeded Monte Carlo decoding accuracy test. LABELS:decoding,tests ASSIGNEE:diogoribeiro7
