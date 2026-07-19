# Colab launchers

This directory will contain the ordered Colab notebooks defined by the implementation guide. Each notebook must:

- identify its notebook name, Git SHA, selected configuration, and UTC start time;
- install dependencies from the committed Colab requirements;
- fail clearly when a required capability such as CUDA is unavailable;
- invoke reusable functions from Python packages instead of duplicating them;
- write machine-readable outputs and durable checkpoints where applicable;
- contain no embedded credentials or machine-specific paths.

## Phase 1 runtime check

`00_runtime_check.ipynb` is the Colab readiness gate. It checks out an exact Git
revision, installs the hash-pinned Colab requirements, proves a small CUDA tensor
operation works, invokes `pipelines.environment_report`, and validates the resulting
machine-readable report.

The committed notebook intentionally starts with `GIT_SHA` set to
`REPLACE_WITH_TESTED_COMMIT_SHA`. Pinning requires two commits because a commit cannot
contain its own SHA:

1. Commit and locally validate the notebook and environment-report implementation.
2. Copy that commit's full SHA from `git rev-parse HEAD` into the notebook.
3. Commit only the SHA update with
   `chore(colab): pin runtime check to tested revision`.

In Colab, select **Runtime > Change runtime type > GPU** before running all cells.
The notebook contains no credentials; a private repository must be authenticated by a
separate secure Colab mechanism.
