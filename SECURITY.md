# Security and data-handling policy

## Scope

This policy covers source code, notebooks, APIs, model bundles, datasets, user uploads, batch work, registries, containers, logs, and operational services.

## Security boundary

- FastAPI is reachable only through the configured gateway in deployed profiles.
- `X-API-Key` is required for inference. Only salted cryptographic hashes are stored; plaintext keys are shown once at issuance and never logged.
- MLflow, PostgreSQL, Redis, Prometheus, Grafana, and worker control ports are private services and are not exposed publicly by default.
- Colab is an interactive development executor. It must not receive production database credentials or act as a production scheduler/server.
- Google Drive is the initial artifact handoff location, not a secrets store. Access is limited to the project artifact folder.

## Secrets

- Provide secrets through environment variables locally and an approved secret manager in deployment.
- Commit `.env.example` with names and safe placeholders only; never commit `.env`, credentials, tokens, private keys, database dumps, or signed URLs.
- Never place secrets in notebook cells, outputs, model manifests, MLflow parameters, command history examples, logs, or container images.
- Rotate a secret immediately if it is exposed and remove it from all retained outputs and logs; deleting only the latest Git version is insufficient.

## Upload and decoder controls

- Enforce the limits and accepted formats in `docs/api-contract.md` at the gateway and application layers.
- Validate decoded content rather than filename extension or supplied MIME type.
- Normalize EXIF orientation and RGB mode; reject malformed, truncated, decompression-bomb, oversized-dimension, and excessive-pixel inputs.
- Do not accept serialized Python objects, unrestricted archives, remote URLs, or filesystem paths as inference inputs.
- Use bounded request, decode, queue, and inference timeouts and bounded model concurrency.

## Retention decisions

| Data class | Policy |
|---|---|
| Synchronous request image | Process in memory; do not persist by default |
| Accepted batch input | Store in private encrypted blob/shared storage; delete within 24 hours after terminal job state, and no later than 7 days after acceptance if a job is abandoned |
| Batch result payload | Retain for 7 days by default, then delete; durable job metadata may remain without image bytes |
| Inference event metadata | Retain for 30 days; exclude raw image, API key, and directly identifying content |
| Application/security logs | Retain for 30 days with secret and PII redaction |
| Drift data | Retain sampled derived features only for 30 days by default; raw user images require explicit opt-in and a separately documented purpose |
| Similarity gallery item | Retain only under an explicit gallery/source policy until deletion or dataset-version retirement; never add an inference upload automatically |
| Training/evaluation dataset | Retain according to its verified license and team storage policy; never commit large data to Git |

Deletion jobs must be observable and auditable. Production policy may shorten these periods; extending them requires a documented purpose and review.

## Artifact and model supply chain

- Load only bundles from an approved source and immutable version path.
- Verify manifest schema, file allowlist, byte sizes, and SHA-256 before deserializing any model.
- Do not load untrusted pickle/checkpoint files in an online process. Served ONNX/TensorRT artifacts must be produced by the controlled pipeline.
- Pin direct dependencies and container base digests. Run dependency, source, secret, license, and image scans; generate an SBOM for releases.
- Record code, pretrained-weight, and dataset provenance as required by `docs/model-data-licenses.md`.

## Service and container controls

- Run containers as non-root, drop unnecessary Linux capabilities, use a read-only root filesystem where practical, and declare only required writable volumes.
- Separate CPU and GPU images and dependency groups.
- Apply least-privilege database roles, parameterized queries, migration controls, private networks, connection limits, and TLS where traffic crosses a trust boundary.
- Namespace Redis uses and apply TTL, maximum-memory, and eviction policies. PostgreSQL remains the durable source of truth.
- Rate-limit by authenticated identity with coarse pre-authentication gateway protection. Return `429` with `Retry-After`.

## Logging, metrics, and errors

- Structured logs may contain request ID, job ID, task, concrete model version, runtime, duration, and stable error code.
- Logs must not contain plaintext API keys, authorization headers, image bytes, secrets, signed URLs, full database strings, or raw model inputs.
- Prometheus labels must have bounded cardinality. Never use request ID, user ID, email, API key, or image hash as a metric label.
- Client errors follow the stable envelope in `docs/api-contract.md` and never expose stack traces, model paths, SQL, or internal service addresses.

## Security verification gates

- Unit tests cover upload validation, cache-key isolation, authentication, authorization state, and error redaction.
- Integration tests cover Redis rate limiting, database least privilege, bundle verification, duplicate batch delivery, and dependency failure behavior.
- Release checks include source/secret/dependency/image scans, SBOM generation, non-root execution, and clean-environment container startup.
- No critical unaccepted finding, missing license decision, or embedded secret may pass champion promotion or release.

## Incident ownership

Report suspected credential exposure, malicious uploads, artifact tampering, unauthorized retention, or dependency compromise to the project security owner. Preserve bounded evidence, revoke affected access, quarantine the artifact/version, roll back to a verified bundle when necessary, and record the incident and remediation.
