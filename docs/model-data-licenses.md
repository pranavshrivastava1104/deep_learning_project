# Model, weight, and data provenance register

This file is an engineering inventory, not legal advice. A `pending` row blocks redistribution or production promotion until the team verifies the exact downloaded version and records evidence. SHA-256 values are captured immediately after an approved download and then frozen in the model/data manifest.

## Baseline models and weights

| Component | Frozen source | Code license | Weight/training-data consideration | Redistribution decision | Status |
|---|---|---|---|---|---|
| EfficientNet-B0 implementation | `torchvision.models.efficientnet_b0` from https://github.com/pytorch/vision | BSD-3-Clause | `EfficientNet_B0_Weights.IMAGENET1K_V1` was trained on ImageNet; torchvision explicitly requires users to verify pretrained-model terms for their use case | Do not redistribute upstream weights until terms are reviewed; store a checksum and provenance even when downloading at build time | Pending weight review |
| YOLOX-Tiny implementation and checkpoint | Official YOLOX repository/release: https://github.com/Megvii-BaseDetection/YOLOX | Apache-2.0 | Official pretrained checkpoint is evaluated on COCO; exact release URL and SHA-256 must be pinned | Code redistribution allowed subject to Apache-2.0 notices; checkpoint redistribution remains blocked until exact release terms are recorded | Pending checkpoint review |
| OpenCLIP ViT-B/32 implementation | https://github.com/mlfoundations/open_clip, architecture `ViT-B-32`, pretrained tag `openai` | MIT for OpenCLIP code | The tag resolves original OpenAI CLIP-compatible weights; exact resolved artifact, upstream license/terms, and SHA-256 must be captured | Do not mirror weights until the resolved artifact terms are reviewed | Pending resolved-weight review |

Implementation licenses do not automatically grant rights to pretrained weights or their training datasets.

## Datasets

| Dataset | Source | Intended use | License/terms finding | Redistribution decision | Status |
|---|---|---|---|---|---|
| Tiny ImageNet | Stanford CS231n Tiny ImageNet challenge material, https://cs231n.stanford.edu/ | Fine-tuning and evaluation of EfficientNet-B0 | Tiny ImageNet is derived from ImageNet/WordNet imagery; the project sources reviewed do not provide a clear standalone redistribution grant | Local/team learning use only; do not redistribute images or publish a dataset archive until written terms are verified | Pending; production release blocker |
| COCO 2017 | https://cocodataset.org/ | Deterministic detector validation/test subset and representative calibration | COCO annotations and individual Flickr-hosted images may have different attribution/license obligations; preserve original image/license metadata | Commit only sample IDs, annotations allowed by verified terms, hashes, and scripts; do not mirror the image corpus in Git or model bundles | Per-image/annotation review required |
| Similarity query/gallery set | Team-approved source inventory, to be assigned before Phase 02 | Retrieval validation, test, calibration, and gallery indexing | Rights depend on each source item; user/customer uploads are not automatically reusable evaluation data | Only explicitly approved items with provenance and retention class may enter a committed manifest or durable gallery | Source inventory pending |

## Required acquisition record

Before any dataset or weight is used, its manifest must include:

- upstream project and exact artifact URL;
- release/tag/version and retrieval timestamp;
- SHA-256 and byte size;
- code license and exact weight/data terms;
- required attribution and notice files;
- allowed purposes and redistribution decision;
- reviewer name/date and evidence link or stored notice;
- dataset sample IDs and original license metadata where terms vary per item.

## Release gates

1. A missing checksum blocks candidate registration.
2. A missing or ambiguous license blocks redistribution and champion promotion; it does not get converted into assumed permission.
3. Model cards must list code, weights, training/evaluation data, intended use, limitations, and license status.
4. Bundles include required license/notice files but never include source data merely because the model was trained or evaluated on it.
5. Changing the weight tag, dataset version, or source requires a new provenance review and manifest hash.

## Initial decision record

The selected baselines are approved for implementation and internal learning evaluation. Commercial use and external redistribution are not approved by Phase 00; those actions remain gated on the pending evidence above.
