import numpy as np

from tneuro.decoding.bayes_grid_1d import decode_bayes_grid_1d, gaussian_tuning_curves


def test_posterior_normalization() -> None:
    rng = np.random.default_rng(0)
    pos = np.linspace(0.0, 1.0, 25)
    centers = np.linspace(0.1, 0.9, 5)
    rate = gaussian_tuning_curves(pos, centers, place_width=0.15, peak_rate_hz=20.0)

    n_time = 10
    counts = rng.poisson(0.2, size=(n_time, centers.size))
    post, map_idx, mean_idx = decode_bayes_grid_1d(counts, rate_hz=rate, dt_s=0.1)

    assert post.shape == (n_time, pos.size)
    assert map_idx.shape == (n_time,)
    assert mean_idx.shape == (n_time,)
    assert np.allclose(np.sum(post, axis=1), 1.0)


def test_more_spikes_improves_map() -> None:
    rng = np.random.default_rng(1)
    pos = np.linspace(0.0, 1.0, 41)
    centers = np.linspace(0.1, 0.9, 8)
    rate = gaussian_tuning_curves(pos, centers, place_width=0.1, peak_rate_hz=25.0)

    true_idx = 20
    true_rate = rate[:, true_idx]
    dt = 0.1
    n_trials = 200
    post_true_low = []
    post_true_high = []
    for _ in range(n_trials):
        counts_low = rng.poisson(true_rate * dt)
        counts_high = rng.poisson(true_rate * dt * 3.0)
        post_low, _, _ = decode_bayes_grid_1d(
            counts_low[None, :], rate_hz=rate, dt_s=dt
        )
        post_high, _, _ = decode_bayes_grid_1d(
            counts_high[None, :], rate_hz=rate, dt_s=dt
        )
        post_true_low.append(post_low[0, true_idx])
        post_true_high.append(post_high[0, true_idx])

    # On average, higher counts should concentrate mass at the true position.
    assert float(np.mean(post_true_high)) >= float(np.mean(post_true_low))
