# Computer Vision Platform

This repository contains the production-oriented computer vision learning project defined in the Phase 00 documents. Phase 1 establishes a package-first foundation so notebooks, command-line pipelines, services, and tests can reuse the same Python modules.

## Package-first rule

Reusable project behavior belongs in importable Python modules. Notebooks and future process entry points should orchestrate those modules rather than duplicate their implementation.

For example, a future classification preprocessor will live under `models/classification/` and can be imported by training, evaluation, and serving code:

```python
from models.classification.preprocessing import preprocess_image
```

The preprocessor itself is intentionally not implemented during this scaffold step.

## Current layout

```text
configs/                 Frozen model, dataset, runtime, and acceptance contracts
docs/                    Architecture, API, governance, and phase documents
models/                  Shared and task-specific model behavior
  common/                Cross-task model utilities
  classification/        EfficientNet classification modules
  detection/             YOLOX detection modules
  similarity/            OpenCLIP similarity modules
pipelines/               Reusable workflow and command-line orchestration
src/common/              Cross-process logging and runtime infrastructure
services/
  api/                   Future synchronous FastAPI service
  worker/                Future asynchronous worker service
notebooks/
  colab/                 Thin Colab launchers that call Python modules
tests/
  unit/                  Fast deterministic unit tests
requirements/            Exported environment-specific requirement files
```

## Dependency boundaries

- Importing a package must not download data, load a model, connect to infrastructure, or start a process.
- `models.common` contains behavior genuinely shared by multiple model tasks.
- Task packages own their preprocessing, model wrappers, and postprocessing contracts.
- `pipelines` orchestrates reusable modules; notebooks remain thin launchers.
- `services` consumes model/runtime interfaces and must not contain the only copy of ML logic.
- Tests mirror package behavior and use small deterministic fixtures.

## Project status

Phase 00 contracts are complete. Phase 1 now includes locked dependencies,
deterministic seed configuration, common structured logging, and the environment-report
pipeline. Podman service definitions and executable model behavior remain later tasks.
