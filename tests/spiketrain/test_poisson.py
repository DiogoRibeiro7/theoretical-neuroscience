import numpy as np

from tneuro.spiketrain.poisson import generate_inhom_poisson


def test_generate_inhom_poisson_callable_mean_count() -> None:
    def rate_fn(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, 20.0, dtype=float)

    t_start = 0.0
    t_stop = 5.0
    t_grid = np.linspace(t_start, t_stop, 501)
    seeds = np.arange(50)
    counts = []
    for seed in seeds:
        st = generate_inhom_poisson(
            rate_fn,
            t_start_s=t_start,
            t_stop_s=t_stop,
            t_grid_s=t_grid,
            seed=int(seed),
        )
        counts.append(st.n_spikes())

    mean_count = float(np.mean(counts))
    expected = 20.0 * (t_stop - t_start)
    assert abs(mean_count - expected) / expected < 0.15


def test_generate_inhom_poisson_array_grid_bounds() -> None:
    t_start = 0.0
    t_stop = 2.0
    t_grid = np.linspace(t_start, t_stop, 201)
    rate = np.full_like(t_grid, 15.0, dtype=float)
    st = generate_inhom_poisson(
        rate,
        t_start_s=t_start,
        t_stop_s=t_stop,
        t_grid_s=t_grid,
        seed=123,
    )

    assert np.all(st.times_s >= t_start)
    assert np.all(st.times_s <= t_stop)
