# Milestones (v0.2.0 -> v1.0.0)

This plan is a living document. Each milestone has 3–6 concrete deliverables and is ordered
from foundational to advanced.

## v0.2.0 (stabilize current core)

- Spike train core + statistics + Poisson generators are implemented and tested.
- LIF neuron simulation API is stable with deterministic tests.
- Encoding core (reverse correlation, receptive field, Poisson GLM) is implemented with tests.
- Decoding core (Bayesian decoding utilities + 1D example) is implemented with tests.
- Information theory core (entropy + mutual information with Miller-Madow) is implemented with tests.

## v0.3.0 (notebook standard + early chapters)

- Notebook template and standard are enforced in `examples/`.
- Chapter notebook: reverse correlation + LNP (STA/GLM comparison).
- Chapter notebook: Bayesian decoding (1D grid + plots).
- Chapter notebook: learning (delta rule + Oja + TD(0) or Q-learning).

## v0.4.0 (spike train depth + LNP)

- Spike train generators expanded (renewal + refractory) with tests.
- Spike train analysis tools (autocorrelogram, ISI hist, CV/LV) with tests.
- LNP model module (design matrix + IRLS fit + prediction) with tests.

## v0.5.0 (decoding + networks)

- Bayesian decoding on a grid with numerically stable log-space implementation.
- Kalman filter (plus smoother if feasible) with tests on a simulated LDS.
- Rate and spiking network modules with tests and a chapter notebook.

## v0.6.0 (information theory + noise)

- Information theory expansion: KL, conditional MI, bias corrections.
- Noise process utilities (OU and filtered noise) with tests.
- Chapter notebook: information theory with finite-sample bias demo.

## v1.0.0 (book-complete baseline)

- All planned chapter notebooks exist and run within time budgets.
- API docs and "reproduce results" guide are complete and accurate.
- Definition of Done criteria are fully satisfied; CI green; release checklist executed.
