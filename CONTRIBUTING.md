# Contributing

Thanks for contributing. This project uses Poetry and a `src/` layout.

## Setup

1. Install dependencies:
   - `poetry install`
2. Activate the virtual environment:
   - `poetry shell`

## Checks

Run these before opening a PR:
- `poetry run ruff check .`
- `poetry run mypy src/tneuro`
- `poetry run pytest`

Optional:
- `poetry run pre-commit install`

## Proposing new modules

Guidelines for adding a new module:
- Add the module under `src/tneuro/<area>/`.
- Add minimal tests under `tests/<area>/`.
- Keep APIs small and documented with docstrings.
- Update `README.md` and `docs/` if the module is public.
- Add a brief example under `examples/` if it helps adoption.
