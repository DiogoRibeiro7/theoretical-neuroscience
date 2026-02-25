from __future__ import annotations

import numpy as np

from tneuro.neurons.lif import LIFParams, simulate_lif
from tneuro.utils.plot import plot_spike_raster, plot_voltage_trace


def main() -> None:
    params = LIFParams(
        tau_m_s=0.02,
        v_rest=0.0,
        v_reset=0.0,
        v_th=1.0,
        r_m_ohm=1.0,
        refractory_s=0.002,
    )
    t, v, spikes = simulate_lif(
        params=params,
        t_stop_s=0.5,
        dt_s=1e-4,
        i_inj_a=1.2,
        noise_std_a=0.0,
        seed=123,
    )

    ax_v = plot_voltage_trace(t, v)
    ax_v.set_title("LIF voltage trace")

    ax_r = plot_spike_raster(spikes)
    ax_r.set_title("Spike raster")

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
