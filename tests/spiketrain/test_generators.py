import numpy as np

from tneuro.spiketrain.generators import (
    generate_hom_poisson,
    generate_inhom_poisson,
    generate_renewal_gamma,
)


def test_generate_hom_poisson_mean_count() -> None:
    t_start = 0.0
    t_stop = 5.0
    rate = 15.0
    seeds = np.arange(50)
    counts = []
    for seed in seeds:
        st = generate_hom_poisson(rate, t_start_s=t_start, t_stop_s=t_stop, seed=int(seed))
        counts.append(st.n_spikes())

    mean_count = float(np.mean(counts))
    expected = rate * (t_stop - t_start)
    assert abs(mean_count - expected) / expected < 0.2


def test_generate_hom_poisson_refractory_enforced() -> None:
    st = generate_hom_poisson(50.0, t_start_s=0.0, t_stop_s=2.0, seed=1, refractory_s=0.01)
    if st.times_s.size > 1:
        isi = np.diff(st.times_s)
        assert np.all(isi >= 0.01 - 1e-9)


def test_generate_inhom_poisson_refractory_enforced() -> None:
    def rate_fn(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, 40.0, dtype=float)

    st = generate_inhom_poisson(
        rate_fn,
        t_start_s=0.0,
        t_stop_s=2.0,
        t_grid_s=np.linspace(0.0, 2.0, 201),
        seed=2,
        refractory_s=0.005,
    )
    if st.times_s.size > 1:
        isi = np.diff(st.times_s)
        assert np.all(isi >= 0.005 - 1e-9)


def test_generate_renewal_gamma_cv() -> None:
    st = generate_renewal_gamma(
        20.0,
        shape_k=4.0,
        t_start_s=0.0,
        t_stop_s=20.0,
        seed=3,
    )
    isi = st.isi_s()
    if isi.size > 20:
        cv = float(np.std(isi, ddof=1) / np.mean(isi))
        expected = 1.0 / np.sqrt(4.0)
        assert abs(cv - expected) < 0.2
