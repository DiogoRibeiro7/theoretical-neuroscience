# Book map (living plan)

Mapping *Theoretical Neuroscience* (Dayan & Abbott) topics to repo modules and example notebooks.
This is a living plan; items are tentative and will evolve.

Format:
- Topic
  - Target module path
  - Example notebook(s)
  - Notes (datasets/assumptions)

---

- Spike trains, statistics, variability
  - Target: `src/tneuro/spiketrain/`
  - Examples: `examples/spiketrain_basic.ipynb` (planned)
  - Notes: assumes time units in seconds; no datasets required.

- LIF neuron and basic dynamics
  - Target: `src/tneuro/neurons/lif.py`
  - Examples: `examples/plot_lif.py` (script), `examples/lif_basics.ipynb` (planned)
  - Notes: assumes Euler integration with fixed `dt_s`.

- Encoding: receptive fields (linear models)
  - Target: `src/tneuro/encoding/receptive_field.py`
  - Examples: `examples/receptive_field.ipynb` (planned)
  - Notes: requires 1D stimulus arrays; synthetic stimuli by default.

- Encoding: GLM (Poisson)
  - Target: `src/tneuro/encoding/glm_poisson.py`
  - Examples: `examples/glm_poisson.ipynb` (planned)
  - Notes: assumes Poisson spiking and log link; synthetic data sufficient.

- Information theory: entropy and mutual information
  - Target: `src/tneuro/information/`
  - Examples: `examples/information_basic.ipynb` (planned)
  - Notes: discrete estimators; bias corrections approximate for small samples.

- Decoding: Bayesian decoding (planned)
  - Target: `src/tneuro/decoding/`
  - Examples: `examples/bayesian_decoding.ipynb`
  - Notes: uses synthetic tuning curves and Poisson spike counts.
  - TODO: Add multi-bin decoding notebook. LABELS:decoding,docs ASSIGNEE:diogoribeiro7

- Reverse correlation / STA/STC (planned)
  - Target: `src/tneuro/encoding/reverse_correlation.py` (to be created)
  - Examples: `examples/sta_stc.ipynb` (planned)
  - Notes: requires stimulus-response datasets or simulated stimuli.

- Learning: delta rule / simple RL (planned)
  - Target: `src/tneuro/learning/`
  - Examples: `examples/learning_delta_rule.ipynb`
  - Notes: synthetic tasks by default; no external datasets required.
