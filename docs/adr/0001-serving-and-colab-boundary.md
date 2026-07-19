# ADR 0001: Serving and Colab boundary

- Status: Accepted
- Date: 2026-07-18
- Decision owners: ML, backend, platform, and QA leads
- Scope: Phase 00 project definition

## Context

The project needs GPU-backed model development without making a temporary notebook runtime part of the production system. It also needs model tracking without coupling online predictions to the availability of a tracking server.

The team must support contributors with CPU-only development machines while retaining a path to CUDA and TensorRT deployment.

## Decision

1. Google Colab is an interactive GPU executor for training, GPU export checks, calibration experiments, and development benchmarks. It is not an API host, scheduler, artifact source of truth, or production runtime.
2. VS Code is the primary local development environment. The default runnable inference profile is FastAPI backed by ONNX Runtime CPU.
3. FastAPI owns synchronous prediction contracts. Celery workers use the same model/runtime modules for asynchronous work.
4. MLflow records experiments and model versions and exposes `candidate`, `challenger`, and `champion` aliases. MLflow is a control-plane dependency, not part of the prediction request path.
5. Redis owns ephemeral cache, rate-limit, and queue coordination state. PostgreSQL owns durable application, job, deployment, and audit state. PostgreSQL with pgvector owns similarity embeddings.
6. Large model artifacts are stored in Google Drive initially and are transferred as immutable, checksum-verified bundles. Git stores code, configuration, small manifests, and documentation only.
7. ONNX is the portable serving artifact. A final TensorRT engine is compiled and validated on the declared target deployment GPU; a Colab-built engine is development evidence only.
8. Changing an MLflow alias never changes a live API process by itself. Promotion must trigger a deployment workflow that verifies, warms, tests, and atomically activates an immutable bundle.

## Supported profiles

| Profile | Request path | Intended environment |
|---|---|---|
| Local CPU | FastAPI -> ONNX Runtime CPU | Developer PC and default reviewer demo |
| Portable GPU | FastAPI -> ONNX Runtime CUDA | GPU host before TensorRT hardening |
| Optimized GPU | FastAPI -> TensorRT | Controlled target deployment GPU |

Runtime fallback is permitted only for the same task, model name, semantic version, preprocessing version, and postprocessing version. The service must never silently fall back to a semantically different model.

## Promotion and rollback

```text
MLflow champion alias approved
  -> deployment workflow resolves immutable bundle
  -> bundle schema and SHA-256 checksums verified
  -> compatible runtime selected
  -> model loaded in inactive slot and warmed
  -> golden and readiness tests executed
  -> traffic atomically switched
  -> SLOs monitored
```

Rollback reactivates the previous verified bundle and image digest. It does not rebuild the previous model.

## Consequences

### Positive

- CPU-only contributors can run and test the complete service contract.
- Training logic can move from Colab to another GPU executor without changing the model modules.
- Registry or Colab outages do not directly interrupt already-loaded synchronous inference.
- ONNX artifacts remain portable and can be used to rebuild target-specific TensorRT engines.

### Costs

- Artifact manifests and environment reports must be maintained carefully.
- Promotion requires explicit deployment automation instead of an alias-only hot swap.
- CPU and GPU dependency sets and container images must remain separate.
- TensorRT performance cannot be claimed until a target GPU is identified and measured.

## Rejected alternatives

- **Serve directly from Colab:** rejected because runtimes are temporary and unsuitable for reliable public serving.
- **Serve directly through MLflow:** rejected because the product needs custom task schemas, preprocessing, auth, cache, batch, pgvector, and operational behavior.
- **Use TensorRT as the only artifact:** rejected because serialized engines are target-dependent and are not the reconstructable portable source.
- **Store artifacts in Git:** rejected because model binaries and datasets are large and need immutable artifact storage rather than source-control history.

## Review trigger

Review this ADR if the team adopts a managed training platform, an authenticated object store, a different model registry, or a deployment platform that changes these responsibility boundaries.
