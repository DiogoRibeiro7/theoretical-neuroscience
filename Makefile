.PHONY: install lint typecheck test coverage build clean

install:
	poetry install

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src/tneuro

test:
	poetry run pytest

coverage:
	poetry run pytest --cov=src/tneuro --cov-report=term-missing

build:
	poetry build

clean:
	@powershell -NoProfile -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist,build,.pytest_cache,.mypy_cache,.ruff_cache,.coverage,coverage.xml"
	@powershell -NoProfile -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
