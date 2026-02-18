from __future__ import annotations

import numpy as np

from tneuro.neurons.lif import LIFParams, simulate_lif


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
    assert spikes.ndim == 1
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
    _, _, spikes = simulate_lif(
        params=params,
        t_stop_s=0.2,
        dt_s=1e-4,
        i_inj_a=0.0,
        noise_std_a=0.0,
        seed=0,
    )
    assert spikes.size == 0
