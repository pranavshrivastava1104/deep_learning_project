# Phase 00 architecture

## Product boundary

The project is a versioned computer-vision platform with three initial baselines:

| Task | Baseline | Data | Initial decision |
|---|---|---|---|
| Classification | EfficientNet-B0 | Tiny ImageNet | Fine-tune in Colab |
| Detection | YOLOX-Tiny | Deterministic COCO subset | Use pinned pretrained weights and validate |
| Similarity | OpenCLIP ViT-B/32 | Versioned query/gallery set | Use pinned pretrained encoder and index normalized embeddings |

The first implementation freezes these choices. Replacement models require a new model version, evidence bundle, and promotion decision; they do not overwrite an existing version.

## Responsibilities and execution locations

| Component | Location | Responsibility |
|---|---|---|
| GPU training | Google Colab | Fine-tuning, AMP, checkpointing |
| GPU export and development benchmarks | Google Colab | CUDA validation, export parity, calibration experiments, qualified benchmark evidence |
| Final TensorRT compilation | Target deployment GPU | Hardware-specific engine build, parity, load, and performance tests |
| Local development | VS Code | Packages, contracts, tests, API, workers, persistence, and operations |
| Default inference | ONNX Runtime CPU | Portable local and reviewer runtime |
| Online serving | FastAPI behind Nginx | Authentication, validation, routing, synchronous inference, response contracts |
| Batch processing | Celery | Idempotent long-running and batch work |
| Coordination | Redis | Broker, bounded-TTL cache, distributed rate limiting |
| Durable application state | PostgreSQL | Jobs, API identities, deployments, experiments, audit, inference metadata |
| Similarity vectors | PostgreSQL with pgvector | Versioned normalized embeddings and Top-K retrieval |
| Experiment tracking and registry | MLflow | Runs, metrics, artifacts, versions, lifecycle aliases |
| Large artifacts | Google Drive initially | Immutable handoff bundles; replaceable by object storage |
| CI/CD | GitHub Actions | CPU checks, image release, promotion approvals, deployment orchestration |
| Monitoring | Prometheus and Grafana | Bounded-cardinality metrics, dashboards, and alerts |

## Logical architecture

```text
Clients
  -> Nginx
  -> FastAPI
       -> model manager -> PyTorch / ONNX Runtime / TensorRT adapter
       -> Redis (cache and rate limit)
       -> PostgreSQL (durable metadata and jobs)
       -> pgvector (versioned similarity gallery)
       -> Celery submission -> Redis broker -> CPU/GPU worker

Colab GPU -> immutable artifact bundle -> Drive -> verification/import -> MLflow registry
MLflow promotion -> GitHub deployment workflow -> verified runtime activation
Prometheus <- API/workers/exporters -> Grafana
```

## Data ownership

| Data | Source of truth | Notes |
|---|---|---|
| Source code and configuration | Git | No credentials or large binaries |
| Dataset membership | Versioned manifests | Stable sample IDs, hashes, seed, and provenance |
| Model bundle | Drive/artifact store | Immutable version path and SHA-256 checksums |
| Model lifecycle | MLflow | Concrete versions plus candidate/challenger/champion aliases |
| Live deployment | Deployment record in PostgreSQL and release metadata | Records actual bundle and image digest, not only mutable alias |
| Batch job state | PostgreSQL | Redis may cache status but is not durable truth |
| Similarity vectors | pgvector | Every row and query is constrained by embedding model version |
| Metrics | Prometheus | No user IDs, request IDs, or image hashes as labels |
| Correlated diagnostic detail | Structured logs | Redacted and subject to retention policy |

## Synchronous request flow

1. Nginx applies coarse body limits, timeouts, and request-ID propagation.
2. FastAPI authenticates `X-API-Key`, rate-limits the identity, and validates the decoded image.
3. The model manager resolves the concrete champion bundle and runtime for the requested task.
4. The cache key includes the image hash, normalized parameters, model version, preprocessing version, and postprocessing version.
5. A bounded runtime adapter executes inference. Detection preprocessing and postprocessing remain shared and versioned across backends.
6. FastAPI returns the schema in `docs/api-contract.md`, records bounded operational metadata, and caches only safe response data.

## Batch flow

1. FastAPI validates the request, persists a job and item identities, stores large inputs outside Redis, and returns `202 Accepted`.
2. Celery receives references and identifiers, not large image payloads.
3. Workers execute idempotently with retry classification and bounded exponential backoff.
4. Progress and the terminal state are committed to PostgreSQL. Duplicate delivery cannot duplicate a committed item result.

## Model lifecycle

```text
experiment -> candidate -> challenger -> champion -> archived
                                      \-> rejected
champion -> rolled back to retained prior champion
```

- **Experiment:** incomplete or exploratory run; never served.
- **Candidate:** offline metrics, manifest, checksum, and parity gates passed.
- **Challenger:** staging/integration/security/performance gates passed; eligible for shadow or A/B traffic.
- **Champion:** approved production version with an immutable deployable bundle.
- **Archived:** rejected, superseded, or retained only for evidence.

Aliases are lifecycle references. Runtime activation is a separate audited action.

## Artifact contract

```text
artifacts/<task>/<model>/<version>/
  checkpoint.pt
  model.onnx
  model.int8.onnx          # only if its quality gate passes
  model.plan               # only for the declared target GPU
  preprocessing.json
  labels.json
  metrics.json
  parity.json
  benchmark.json
  manifest.json
  MODEL_CARD.md
  checksums.sha256
```

The manifest must identify the Git SHA, dataset and split hashes, configuration hash, model and weight provenance, preprocessing and output signatures, quality gates, environment, supported runtimes, and the SHA-256 and size of every file.

## Failure and degradation policy

| Failure | Required behavior |
|---|---|
| TensorRT unavailable | Use explicitly configured ONNX for the exact same semantic model version; report the fallback runtime |
| Redis cache unavailable | Continue safe synchronous inference without cache; emit dependency metrics and use conservative local protection |
| Redis broker unavailable | Reject new batch submissions with `503`; do not break independent synchronous inference |
| PostgreSQL audit write unavailable | Preserve a valid synchronous prediction where policy allows; emit an error and use a bounded retry/buffer |
| Champion cannot load or warm | Readiness is false; return the stable `503` error envelope |
| Promotion health/SLO gate fails | Keep or restore the previous warmed immutable bundle |

## Phase 00 acceptance record

- Model and dataset baselines: accepted in model/data configuration.
- Colab/local/target-GPU boundary: accepted in ADR 0001.
- Serving profiles: accepted in runtime configuration.
- API requests, responses, and errors: accepted in `api-contract.md`.
- Quality and performance metrics: accepted in `configs/acceptance.yaml`; baseline-dependent thresholds remain explicitly `null`.
- Split and calibration rules: accepted in data configuration.
- Lifecycle, deployment, rollback, and artifact structure: accepted in this document and ADR 0001.
- Security, retention, and licensing controls: accepted in `SECURITY.md` and `model-data-licenses.md`.

Phase 01 must not silently change these contracts. A material change requires an ADR or an explicit Phase 00 amendment reviewed by the affected owners.
