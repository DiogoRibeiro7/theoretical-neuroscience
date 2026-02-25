# Definition of Done (book-complete)

This defines what “book-complete” means for this repo and how to review work
against that bar. Keep this concrete and measurable.

## Required module coverage

Must have implemented and tested modules for these areas:
- Spike trains: `src/tneuro/spiketrain/`
- Neurons (LIF + extensions as needed): `src/tneuro/neurons/`
- Information theory: `src/tneuro/information/`
- Encoding (at least linear RF + Poisson GLM): `src/tneuro/encoding/`
- Decoding: `src/tneuro/decoding/` (at least one working decoder)
- Learning: `src/tneuro/learning/` (at least one reusable learning rule)

## Notebooks (examples)

Minimum notebooks:
- At least 6 notebooks (one per major area above).
- Every notebook must be referenced in `docs/book_map.md`.

Notebook rules:
- Naming: `snake_case.ipynb`.
- Required sections (markdown cells): **Goal**, **Method**, **Results**.
- Reproducibility cell near the top:
  - Set all random seeds used in the notebook.
  - Print versions for `numpy` and `tneuro`.
- Expected outputs:
  - Include at least one plot or table in **Results**.
  - Results should be deterministic within tolerances (seeded).
- Runtime limits:
  - Notebook should run in <= 2 minutes on a typical laptop.

Rule: **No chapter notebook without adding at least one reusable function + tests.**

## Testing rules

- Every new public function must have at least one test.
- New estimators or simulators need:
  - one deterministic unit test, and
  - one statistical or Monte Carlo test (seeded).
- All tests must pass with `poetry run pytest`.

## Release criteria

For a release tag (e.g., `vX.Y.Z`):
- CI is green.
- Docs build succeeds (`mkdocs build`).
- `CHANGELOG.md` updated with the release notes.
- Version bumped in the single source of truth (`pyproject.toml`).
- Release checklist in `docs/releasing.md` followed.

## PR reviewer checklist

- [ ] The change maps to a topic in `docs/book_map.md`.
- [ ] New modules have tests, and test coverage is reasonable for behavior.
- [ ] Notebooks (if added) follow the rules above and are reproducible.
- [ ] Docs updated if public APIs or notebooks changed.
- [ ] CI passes (or failures are understood and non-blocking).
- [ ] TODO issues updated if new TODOs were added. LABELS:meta,process ASSIGNEE:diogoribeiro7
