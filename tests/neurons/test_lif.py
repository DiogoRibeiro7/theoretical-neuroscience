from __future__ import annotations

import numpy as np

from tneuro.neurons.lif import LIFParams, simulate_lif


def _validate_spike_times(spikes: np.ndarray, t: np.ndarray, t_stop_s: float) -> None:
    assert spikes.ndim == 1
    if spikes.size == 0:
        return
    assert np.all(np.diff(spikes) > 0.0)
    assert spikes.min() >= t[0] - 1e-12
    assert spikes.max() <= t_stop_s + 1e-12


def test_lif_spikes_with_strong_current() -> None:
    params = LIFParams(
        tau_m_s=0.02,
        v_rest=0.0,
        v_reset=0.0,
        v_th=1.0,
        r_m_ohm=1.0,
        refractory_s=0.0,
    )
    t, v, spikes = simulate_lif(
        params=params,
        t_stop_s=0.5,
        dt_s=1e-4,
        i_inj_a=2.0,
        noise_std_a=0.0,
        seed=0,
    )
    assert t.shape == v.shape
    _validate_spike_times(spikes, t, 0.5)
    assert spikes.size > 0  # should spike

def test_lif_no_spikes_with_zero_current() -> None:
    params = LIFParams(
        tau_m_s=0.02,
        v_rest=0.0,
        v_reset=0.0,
        v_th=1.0,
        r_m_ohm=1.0,
        refractory_s=0.0,
    )
    t, _, spikes = simulate_lif(
        params=params,
        t_stop_s=0.2,
        dt_s=1e-4,
        i_inj_a=0.0,
        noise_std_a=0.0,
        seed=0,
    )
    _validate_spike_times(spikes, t, 0.2)
    assert spikes.size == 0


def test_lif_callable_and_array_current_inputs() -> None:
    params = LIFParams(
        tau_m_s=0.02,
        v_rest=0.0,
        v_reset=0.0,
        v_th=1.0,
        r_m_ohm=1.0,
        refractory_s=0.0,
    )
    t_stop_s = 0.3
    dt_s = 1e-3

    def i_inj(t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.1, 0.0, 2.0)

    t, _, spikes_callable = simulate_lif(
        params=params,
        t_stop_s=t_stop_s,
        dt_s=dt_s,
        i_inj_a=i_inj,
        noise_std_a=0.0,
        seed=1,
    )
    _validate_spike_times(spikes_callable, t, t_stop_s)
    assert spikes_callable.size > 0

    i_arr = np.full_like(t, 2.0)
    t2, _, spikes_array = simulate_lif(
        params=params,
        t_stop_s=t_stop_s,
        dt_s=dt_s,
        i_inj_a=i_arr,
        noise_std_a=0.0,
        seed=1,
    )
    _validate_spike_times(spikes_array, t2, t_stop_s)
    assert spikes_array.size > 0


def test_lif_refractory_enforces_minimum_isi() -> None:
    params = LIFParams(
        tau_m_s=0.02,
        v_rest=0.0,
        v_reset=0.0,
        v_th=1.0,
        r_m_ohm=1.0,
        refractory_s=0.005,
    )
    t_stop_s = 0.2
    dt_s = 0.001
    t, _, spikes = simulate_lif(
        params=params,
        t_stop_s=t_stop_s,
        dt_s=dt_s,
        i_inj_a=3.0,
        noise_std_a=0.0,
        seed=2,
    )
    _validate_spike_times(spikes, t, t_stop_s)
    assert spikes.size > 1

    refractory_steps = int(np.round(params.refractory_s / dt_s))
    min_isi = (refractory_steps + 1) * dt_s
    assert np.all(np.diff(spikes) >= min_isi - 1e-12)


def test_lif_time_grid_edges() -> None:
    params = LIFParams(
        tau_m_s=0.02,
        v_rest=0.0,
        v_reset=0.0,
        v_th=1.0,
        r_m_ohm=1.0,
        refractory_s=0.0,
    )
    t_stop_s = 0.035
    dt_s = 0.01
    t, v, spikes = simulate_lif(
        params=params,
        t_stop_s=t_stop_s,
        dt_s=dt_s,
        i_inj_a=0.0,
        noise_std_a=0.0,
        seed=3,
    )
    assert t.shape == v.shape
    assert t[-1] <= t_stop_s + 1e-12
    _validate_spike_times(spikes, t, t_stop_s)
    assert spikes.size == 0

    t2, v2, spikes2 = simulate_lif(
        params=params,
        t_stop_s=0.001,
        dt_s=0.01,
        i_inj_a=0.0,
        noise_std_a=0.0,
        seed=4,
    )
    assert t2.shape == v2.shape
    assert t2.size == 1
    _validate_spike_times(spikes2, t2, 0.001)
    assert spikes2.size == 0
