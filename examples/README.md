# Examples notebook guidelines

These rules keep notebooks consistent and easy to review.

Naming convention:
- Use `snake_case.ipynb` (e.g., `lif_basics.ipynb`, `glm_poisson.ipynb`).
- Prefer a short, descriptive name over chapter numbers.

Required sections (as markdown cells):
- **Goal**: what question the notebook answers.
- **Method**: modeling/analysis approach and assumptions.
- **Results**: key outputs, plots, or tables.

Reproducibility cell (first or second code cell):
- Set all random seeds used in the notebook.
- Record library versions (at least `numpy` and `tneuro`).

Optional CI:
- A lightweight notebook CI check can be added later (e.g., to verify metadata
  or run a smoke execution). This is intentionally not enforced yet.
