# theoretical-neuroscience

A PyPI-ready Python package implementing core ideas from *Theoretical Neuroscience*
(Dayan & Abbott): spike trains, simple neuron models, and information-theory tools.

- **PyPI name**: `theoretical-neuroscience`
- **Import name**: `tneuro`

## Install

```bash
pip install theoretical-neuroscience
```

Optional plotting extras:

```bash
pip install "theoretical-neuroscience[plot]"
```

## Quick start

```python
import numpy as np
from tneuro import SpikeTrain, LIFParams, simulate_lif
from tneuro.information import entropy_discrete

# 1) Spike train from spike times (seconds)
st = SpikeTrain(times_s=np.array([0.1, 0.13, 0.9, 1.2]), t_start_s=0.0, t_stop_s=2.0)
print(st.rate_hz())

# 2) LIF simulation
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
    t_stop_s=1.0,
    dt_s=1e-4,
    i_inj_a=0.75,   # constant input current
    noise_std_a=0.05,
    seed=123,
)
print(len(spikes))

# 3) Discrete entropy
x = np.array([0, 0, 1, 1, 1, 2])
print(entropy_discrete(x, base=2.0))
```

## API overview (stable)

Current stable API surface (import paths shown):

- `tneuro`: `SpikeTrain`, `LIFParams`, `simulate_lif`
- `tneuro.spiketrain`: `SpikeTrain`, `fano_factor_counts`, `fano_factor_spiketrain`
- `tneuro.neurons`: `LIFParams`, `simulate_lif`
- `tneuro.information`: `entropy_discrete`

Short examples:

```python
import numpy as np
from tneuro import SpikeTrain

st = SpikeTrain(times_s=np.array([0.1, 0.13, 0.9, 1.2]), t_start_s=0.0, t_stop_s=2.0)
edges_s, counts = st.bin_counts(bin_width_s=0.5)
print(edges_s, counts)
```

```python
import numpy as np
from tneuro import SpikeTrain
from tneuro.spiketrain.stats import fano_factor_spiketrain

st = SpikeTrain(times_s=np.array([0.1, 0.13, 0.9, 1.2]), t_start_s=0.0, t_stop_s=2.0)
ff = fano_factor_spiketrain(st, bin_width_s=0.5)
print(ff)
```

## Development

This repo uses **Poetry** and a **src/** layout.

```bash
poetry install
poetry run ruff check .
poetry run mypy src/tneuro
poetry run pytest
poetry run pre-commit install
```

## Roadmap

- Encoding: reverse correlation / receptive fields
- GLM spiking models
- Decoding: Bayesian decoding utilities
- Information-theory estimators with bias corrections
- Learning: delta rule / simple RL primitives

## License

MIT
