# Colab launchers

This directory will contain the ordered Colab notebooks defined by the implementation guide. Each notebook must:

- identify its notebook name, Git SHA, selected configuration, and UTC start time;
- install dependencies from the committed Colab requirements;
- fail clearly when a required capability such as CUDA is unavailable;
- invoke reusable functions from Python packages instead of duplicating them;
- write machine-readable outputs and durable checkpoints where applicable;
- contain no embedded credentials or machine-specific paths.

The Phase 1 runtime notebook will be added as `00_runtime_check.ipynb` in its dedicated step.
