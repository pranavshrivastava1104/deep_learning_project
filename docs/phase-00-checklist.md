# Phase 00 approval checklist

Status: **Ready for team review**  
Decision date: 2026-07-18

## Frozen decisions

- [x] Classification baseline is EfficientNet-B0 on Tiny ImageNet, fine-tuned in Google Colab.
- [x] Detection baseline is YOLOX-Tiny with pinned pretrained COCO weights and a deterministic COCO evaluation subset.
- [x] Similarity baseline is OpenCLIP ViT-B/32 with a versioned query/gallery set and pgvector retrieval.
- [x] Colab is restricted to interactive GPU development work.
- [x] VS Code and the CPU Docker profile are the default development and reviewer environments.
- [x] FastAPI is the online serving layer; MLflow is tracking and model-registry control plane.
- [x] Local serving uses ONNX Runtime CPU; portable GPU uses ONNX Runtime CUDA.
- [x] Final TensorRT engines are compiled and tested on the actual target GPU.
- [x] API request, response, upload-limit, compatibility, and error contracts are defined.
- [x] Quality metrics and maximum export/quantization regressions are defined.
- [x] Training, validation, calibration, and test responsibilities are separate and deterministic.
- [x] Experiment, candidate, challenger, champion, archive, promotion, and rollback behavior is defined.
- [x] Immutable artifact structure and verification behavior is defined.
- [x] Security, upload retention, secrets, logging, and supply-chain controls are defined.
- [x] Model, weight, and dataset provenance requirements and redistribution gates are recorded.

## Intentionally deferred measurements/evidence

These items do not reopen the Phase 00 architecture. Their current `null` or `pending` values block the relevant later gate until evidence exists.

- [ ] Record each downloaded dataset/archive URL, byte size, SHA-256, and final license evidence before Phase 02 acquisition.
- [ ] Record each exact pretrained-weight URL, byte size, SHA-256, and redistribution decision before candidate registration.
- [ ] Approve the similarity query/gallery source inventory before building manifests or an index.
- [ ] Measure baseline task quality before replacing `minimum_*` acceptance values.
- [ ] Measure local and target performance before setting `maximum_api_p95_ms` and runtime concurrency.
- [ ] Record the deployment GPU, driver, CUDA, TensorRT, platform, and build flags before producing a deployable `model.plan`.

## Approval rule

Phase 01 may begin after the ML, backend, platform, and QA owners review this checklist and the linked Phase 00 artifacts. Any material contract change must be made through a new ADR or an explicit amendment to ADR 0001.
