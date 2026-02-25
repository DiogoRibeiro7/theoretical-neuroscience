# Releasing

Step-by-step checklist for a release.

1. Bump the version in one place:
   - Single source of truth: `pyproject.toml` (`[tool.poetry].version`).
   - If `src/tneuro/_version.py` is used, update it to match or switch it to read from
     `importlib.metadata` to avoid duplication.
2. Update `CHANGELOG.md` with the new version and date.
3. Commit the changes and tag the release: `git tag vX.Y.Z`.
4. Push the tag to GitHub: `git push origin vX.Y.Z`.
5. Create a GitHub Release for the tag with release notes.

PyPI publishing:
- `publish.yml` uses PyPI Trusted Publishing (OIDC). Ensure the PyPI project is
  configured with a Trusted Publisher for this repo and workflow.
- Once the GitHub Release is published, the workflow should build and publish
  the package automatically.
