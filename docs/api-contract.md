# API contract v1

## Conventions

- Base path: `/v1`
- Authentication: `X-API-Key` header
- Normal image transport: `multipart/form-data`
- Request correlation: the client may provide `X-Request-ID`; the service always returns a validated/generated UUID.
- Coordinates: pixel coordinates in the EXIF-normalized original image, origin at the top-left, `x2 > x1`, `y2 > y1`.
- Scores: finite JSON numbers in the inclusive range `[0.0, 1.0]`.
- Runtime values: `onnx-cpu`, `onnx-cuda`, or `tensorrt`.
- Model version: semantic version such as `0.1.0`. Similarity result vectors additionally carry the full embedding namespace identifier.
- All successful inference responses expose the concrete model and runtime used; aliases such as `champion` are never returned as versions.

## Shared upload limits

| Control | Frozen v1 limit |
|---|---:|
| Encoded image bytes | 10 MiB |
| Decoded pixels | 25,000,000 |
| Width or height | 10,000 pixels |
| Accepted formats | JPEG, PNG, WebP |
| Synchronous images per request | 1 |
| Inference deadline | 30 seconds |

The service validates decoded content rather than trusting the filename or declared MIME type. EXIF orientation is normalized and inputs are converted to RGB before versioned model preprocessing.

## `POST /v1/classify`

### Request

`multipart/form-data`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `image` | file | yes | Shared upload limits |
| `top_k` | integer | no | Default `5`, range `1..20` |

### `200 OK`

```json
{
  "request_id": "7eb9d5a6-acde-4e4f-a3f8-53c902a84e83",
  "model_name": "efficientnet-b0",
  "model_version": "0.1.0",
  "runtime": "onnx-cpu",
  "predictions": [
    {
      "class_id": 42,
      "label": "example",
      "confidence": 0.91
    }
  ],
  "processing_ms": 35.2
}
```

Predictions are ordered by descending confidence and contain at most `top_k` entries.

## `POST /v1/detect`

### Request

`multipart/form-data`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `image` | file | yes | Shared upload limits |
| `confidence` | number | no | Default `0.25`, range `0.0..1.0` |
| `iou` | number | no | Default `0.45`, range `0.0..1.0` |
| `max_detections` | integer | no | Default `100`, range `1..300` |

### `200 OK`

```json
{
  "request_id": "8bcaf3ea-e77c-46a8-b46e-166a4135541a",
  "model_name": "yolox-tiny",
  "model_version": "0.1.0",
  "runtime": "onnx-cpu",
  "detections": [
    {
      "class_id": 1,
      "label": "person",
      "confidence": 0.94,
      "bounding_box": {
        "x1": 20.0,
        "y1": 35.0,
        "x2": 280.0,
        "y2": 410.0
      }
    }
  ],
  "processing_ms": 74.8
}
```

Detections are ordered by descending confidence and bounded by `max_detections`. Letterboxing, raw-output decoding, thresholding, and NMS are part of the versioned preprocessing/postprocessing contract.

## `POST /v1/similar`

### Request

`multipart/form-data`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `image` | file | yes | Shared upload limits |
| `top_k` | integer | no | Default `10`, range `1..50` |

### `200 OK`

```json
{
  "request_id": "bd42e78b-e010-4784-9bba-3dd349b6e760",
  "model_name": "openclip-vit-b32",
  "model_version": "0.1.0",
  "runtime": "onnx-cpu",
  "results": [
    {
      "image_id": "image-123",
      "similarity_score": 0.89,
      "model_version": "openclip-vit-b32-0.1.0"
    }
  ],
  "processing_ms": 41.6
}
```

The per-result `model_version` is the embedding namespace stored with the gallery vector. It must match the active query encoder namespace; cross-version vector comparison is forbidden.

## Error contract

Every error uses the same envelope:

```json
{
  "request_id": "dbf5659c-1bec-4e11-9942-70ccdc787eeb",
  "error": {
    "code": "UNSUPPORTED_IMAGE_TYPE",
    "message": "The decoded image format is not supported.",
    "retryable": false,
    "details": {}
  }
}
```

`details` contains bounded, non-sensitive field errors only. Responses never expose stack traces, filesystem paths, SQL, secrets, or internal dependency addresses.

| HTTP | Stable code | Meaning | Retryable |
|---:|---|---|---|
| 400 | `INVALID_REQUEST` | Malformed request or parameters | no |
| 401 | `AUTHENTICATION_FAILED` | Missing, invalid, or inactive API key | no |
| 413 | `IMAGE_TOO_LARGE` | Encoded, decoded-pixel, or dimension limit exceeded | no |
| 415 | `UNSUPPORTED_IMAGE_TYPE` | Decoded image is not JPEG, PNG, or WebP | no |
| 422 | `VALIDATION_FAILED` | Structurally valid request violates field rules | no |
| 429 | `RATE_LIMIT_EXCEEDED` | Identity tier quota exhausted | yes; include `Retry-After` |
| 503 | `MODEL_UNAVAILABLE` | Required model/runtime is not ready | yes |
| 504 | `INFERENCE_TIMEOUT` | Bounded inference deadline exceeded | yes |

## Compatibility policy

- Additive optional response fields may be introduced within v1.
- Removing or renaming a field, changing its meaning/type, changing coordinate semantics, or changing a default that affects predictions requires a new API version or an explicitly approved compatibility plan.
- Backend changes must not alter the response schema. Runtime parity is judged using the task-aware gates in `configs/acceptance.yaml`.
