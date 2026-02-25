# Changelog

## Unreleased
- Docs: Add MkDocs site with minimal API pages (spike trains, neurons, encoding, decoding placeholder, information theory).
- Docs: Add release checklist and Dayan & Abbott book map (living plan).
- Utilities: Add optional plotting helpers for spike rasters and voltage traces.
- Tests: Expand LIF tests (input modes, refractory behavior, time grid edge cases).

## 0.1.0
- Initial release:
  - `SpikeTrain` container with basic statistics.
  - LIF simulator (`simulate_lif`) with optional Gaussian current noise.
  - Discrete entropy helper (`entropy_discrete`) for quick experiments.
