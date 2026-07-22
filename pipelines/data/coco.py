"""Validate raw COCO images and instance annotations before splitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pipelines.data.common import (
    ImageValidationError,
    require_disjoint_splits,
    validate_image,
    validate_manifest,
    write_json,
    write_jsonl,
)


class CocoValidationError(ValueError):
    """Raised when COCO images or annotations violate the raw-data contract."""


@dataclass(frozen=True, slots=True)
class CocoValidationSummary:
    """Counts collected from valid COCO images and annotations."""

    images_root: Path
    annotation_path: Path
    image_count: int
    annotation_count: int
    category_count: int


@dataclass(frozen=True, slots=True)
class CocoManifestSummary:
    """Paths and counts for generated COCO manifests."""

    output_directory: Path
    calibration_count: int
    validation_count: int
    test_count: int
    category_count: int


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CocoValidationError(f"{context} must be a JSON object")
    return cast(dict[str, object], value)


def _list_field(document: dict[str, object], key: str) -> list[object]:
    value = document.get(key)
    if not isinstance(value, list):
        raise CocoValidationError(f"COCO field {key!r} must be a JSON array")
    return value


def _integer(record: dict[str, object], key: str, context: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CocoValidationError(f"{context}.{key} must be an integer")
    return value


def _load_annotation_document(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise CocoValidationError(f"COCO annotation file is missing: {path}")
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CocoValidationError(f"COCO annotation JSON cannot be loaded: {path}") from error
    return _mapping(loaded, "COCO document")


def _category_ids(document: dict[str, object], expected_categories: int) -> set[int]:
    category_ids = {
        _integer(_mapping(item, "COCO category"), "id", "COCO category")
        for item in _list_field(document, "categories")
    }
    if len(category_ids) != expected_categories:
        raise CocoValidationError(
            f"expected {expected_categories} unique COCO categories, "
            f"found {len(category_ids)}"
        )
    return category_ids


def _image_records(
    document: dict[str, object],
    images_root: Path,
) -> dict[int, Path]:
    records: dict[int, Path] = {}
    filenames: set[str] = set()
    resolved_root = images_root.resolve()

    for item in _list_field(document, "images"):
        record = _mapping(item, "COCO image")
        image_id = _integer(record, "id", "COCO image")
        filename = record.get("file_name")
        if not isinstance(filename, str) or not filename.strip():
            raise CocoValidationError(f"COCO image {image_id} has an invalid file_name")
        if image_id in records:
            raise CocoValidationError(f"duplicate COCO image ID: {image_id}")
        if filename in filenames:
            raise CocoValidationError(f"duplicate COCO image filename: {filename}")

        image_path = (images_root / filename).resolve()
        if not image_path.is_relative_to(resolved_root):
            raise CocoValidationError(
                f"COCO image {image_id} escapes the image root: {filename!r}"
            )
        try:
            validate_image(image_path)
        except ImageValidationError as error:
            raise CocoValidationError(f"invalid COCO image {image_id}: {error}") from error

        records[image_id] = image_path
        filenames.add(filename)

    if not records:
        raise CocoValidationError("COCO annotation document contains no images")
    return records


def _validate_bbox(value: object, annotation_id: int) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise CocoValidationError(
            f"COCO annotation {annotation_id} bbox must contain four numbers"
        )
    numbers: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise CocoValidationError(
                f"COCO annotation {annotation_id} bbox contains a non-numeric value"
            )
        numeric_coordinate = float(coordinate)
        if not math.isfinite(numeric_coordinate):
            raise CocoValidationError(
                f"COCO annotation {annotation_id} bbox contains a non-finite value"
            )
        numbers.append(numeric_coordinate)
    if numbers[2] <= 0 or numbers[3] <= 0:
        raise CocoValidationError(
            f"COCO annotation {annotation_id} bbox width and height must be positive"
        )


def _annotation_count(
    document: dict[str, object],
    image_ids: set[int],
    category_ids: set[int],
) -> int:
    annotation_ids: set[int] = set()
    for item in _list_field(document, "annotations"):
        record = _mapping(item, "COCO annotation")
        annotation_id = _integer(record, "id", "COCO annotation")
        image_id = _integer(record, "image_id", f"COCO annotation {annotation_id}")
        category_id = _integer(record, "category_id", f"COCO annotation {annotation_id}")
        if annotation_id in annotation_ids:
            raise CocoValidationError(f"duplicate COCO annotation ID: {annotation_id}")
        if image_id not in image_ids:
            raise CocoValidationError(
                f"COCO annotation {annotation_id} references unknown image ID {image_id}"
            )
        if category_id not in category_ids:
            raise CocoValidationError(
                f"COCO annotation {annotation_id} references unknown "
                f"category ID {category_id}"
            )
        _validate_bbox(record.get("bbox"), annotation_id)
        annotation_ids.add(annotation_id)
    return len(annotation_ids)


def validate_coco(
    images_root: Path,
    annotation_path: Path,
    *,
    expected_categories: int = 80,
) -> CocoValidationSummary:
    """Validate COCO IDs, image files, category references, and bounding boxes."""

    if expected_categories <= 0:
        raise ValueError("expected_categories must be positive")
    if not images_root.is_dir():
        raise CocoValidationError(f"COCO image directory is missing: {images_root}")

    document = _load_annotation_document(annotation_path)
    category_ids = _category_ids(document, expected_categories)
    images = _image_records(document, images_root)
    annotation_count = _annotation_count(document, set(images), category_ids)
    return CocoValidationSummary(
        images_root=images_root,
        annotation_path=annotation_path,
        image_count=len(images),
        annotation_count=annotation_count,
        category_count=len(category_ids),
    )


def _stable_image_ids(image_ids: list[int], seed: int) -> list[int]:
    return sorted(
        image_ids,
        key=lambda image_id: hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest(),
    )


def prepare_coco_manifests(
    images_root: Path,
    annotation_path: Path,
    data_root: Path,
    manifest_root: Path,
    *,
    seed: int = 42,
    calibration_count: int = 500,
    validation_count: int = 1000,
    test_count: int = 3500,
    validation_summary: CocoValidationSummary | None = None,
) -> CocoManifestSummary:
    """Partition val2017 image IDs and create COCO evaluation manifests."""

    validation_summary = validation_summary or validate_coco(
        images_root,
        annotation_path,
    )
    if (
        validation_summary.images_root.resolve() != images_root.resolve()
        or validation_summary.annotation_path.resolve() != annotation_path.resolve()
    ):
        raise ValueError("validation_summary does not belong to the COCO inputs")
    expected_total = calibration_count + validation_count + test_count
    if validation_summary.image_count != expected_total:
        raise CocoValidationError(
            f"COCO split counts require {expected_total} images, "
            f"but the annotation document contains {validation_summary.image_count}"
        )

    document = _load_annotation_document(annotation_path)
    image_records: dict[int, dict[str, object]] = {}
    for item in _list_field(document, "images"):
        record = _mapping(item, "COCO image")
        image_records[_integer(record, "id", "COCO image")] = record

    annotation_ids: defaultdict[int, list[int]] = defaultdict(list)
    for item in _list_field(document, "annotations"):
        record = _mapping(item, "COCO annotation")
        image_id = _integer(record, "image_id", "COCO annotation")
        annotation_id = _integer(record, "id", "COCO annotation")
        annotation_ids[image_id].append(annotation_id)

    ordered_ids = _stable_image_ids(list(image_records), seed)
    calibration_end = calibration_count
    validation_end = calibration_end + validation_count
    split_image_ids = {
        "calibration": ordered_ids[:calibration_end],
        "validation": ordered_ids[calibration_end:validation_end],
        "test": ordered_ids[validation_end:],
    }

    output_directory = manifest_root / "coco-2017" / "v1"
    split_ids: dict[str, set[str]] = {}
    required_fields = {
        "sample_id",
        "coco_image_id",
        "relative_path",
        "annotation_ids",
    }
    filenames = {
        "calibration": "calibration.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
    for split_name, image_ids in split_image_ids.items():
        records: list[dict[str, object]] = []
        for image_id in image_ids:
            filename = image_records[image_id]["file_name"]
            if not isinstance(filename, str):
                raise CocoValidationError(
                    f"COCO image {image_id} has an invalid file_name"
                )
            image_path = images_root / filename
            try:
                relative_path = (
                    image_path.resolve().relative_to(data_root.resolve()).as_posix()
                )
            except ValueError as error:
                raise CocoValidationError(
                    f"COCO image is outside the data root: {image_path}"
                ) from error
            records.append(
                {
                    "sample_id": f"coco-{image_id:012d}",
                    "coco_image_id": image_id,
                    "relative_path": relative_path,
                    "annotation_ids": sorted(annotation_ids[image_id]),
                }
            )

        records.sort(key=lambda item: str(item["sample_id"]))
        write_jsonl(output_directory / filenames[split_name], records)
        split_ids[split_name] = validate_manifest(
            records,
            required_fields=required_fields,
            data_root=data_root,
        )
    require_disjoint_splits(split_ids)

    categories = [
        _mapping(item, "COCO category")
        for item in _list_field(document, "categories")
    ]
    categories.sort(key=lambda item: _integer(item, "id", "COCO category"))
    write_json(
        output_directory / "categories.json",
        {"schema_version": 1, "categories": categories},
    )
    return CocoManifestSummary(
        output_directory=output_directory,
        calibration_count=len(split_image_ids["calibration"]),
        validation_count=len(split_image_ids["validation"]),
        test_count=len(split_image_ids["test"]),
        category_count=len(categories),
    )
