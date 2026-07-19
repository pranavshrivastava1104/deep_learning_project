# Exported requirements

Environment-specific requirement files are generated from the canonical
`pyproject.toml` and `uv.lock`; they must not be maintained as independent
dependency sources.

- `colab.txt` describes the explicit model and observability environment used by
  Colab.
- `serving.txt` describes the CPU API/worker and observability environment.

Both exports contain exact transitive versions and artifact hashes. Regenerate
them after an approved dependency change; never edit them manually.

```powershell
uv lock --python 3.11.15
uv export --frozen --format requirements.txt --no-emit-project --extra model --extra observability --output-file requirements/colab.txt
uv export --frozen --format requirements.txt --no-emit-project --extra serving --extra observability --output-file requirements/serving.txt
```
