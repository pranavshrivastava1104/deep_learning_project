"""Validate the raw Tiny ImageNet directory before splitting it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pipelines.data.common import (
    ImageValidationError,
    find_image_files,
    require_disjoint_splits,
    validate_image,
    validate_manifest,
    write_json,
    write_jsonl,
)


class TinyImageNetValidationError(ValueError):
    """Raised when Tiny ImageNet files or labels violate the raw-data contract."""


@dataclass(frozen=True, slots=True)
class TinyImageNetValidationSummary:
    """Counts collected from a valid raw Tiny ImageNet dataset."""

    root: Path
    class_ids: tuple[str, ...]
    training_image_count: int
    validation_image_count: int

    @property
    def class_count(self) -> int:
        return len(self.class_ids)


@dataclass(frozen=True, slots=True)
class TinyImageNetManifestSummary:
    """Paths and counts for generated Tiny ImageNet manifests."""

    output_directory: Path
    train_count: int
    validation_count: int
    calibration_count: int
    test_count: int


def _read_class_ids(path: Path, expected_classes: int) -> tuple[str, ...]:
    if not path.is_file():
        raise TinyImageNetValidationError(f"Tiny ImageNet class file is missing: {path}")

    class_ids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(class_ids) != expected_classes:
        raise TinyImageNetValidationError(
            f"expected {expected_classes} class IDs in {path}, found {len(class_ids)}"
        )
    if len(set(class_ids)) != len(class_ids):
        raise TinyImageNetValidationError(f"duplicate class IDs found in {path}")
    return class_ids


def _validate_training_images(root: Path, class_ids: tuple[str, ...]) -> int:
    image_count = 0
    for class_id in class_ids:
        image_directory = root / "train" / class_id / "images"
        try:
            image_paths = find_image_files(image_directory)
        except ImageValidationError as error:
            raise TinyImageNetValidationError(
                f"training directory is invalid for class {class_id}: {error}"
            ) from error
        if not image_paths:
            raise TinyImageNetValidationError(
                f"training class contains no supported images: {class_id}"
            )

        for image_path in image_paths:
            try:
                validate_image(image_path)
            except ImageValidationError as error:
                raise TinyImageNetValidationError(
                    f"invalid Tiny ImageNet training image for class {class_id}: {error}"
                ) from error
        image_count += len(image_paths)
    return image_count


def _read_validation_annotations(
    annotation_path: Path,
    known_classes: set[str],
) -> dict[str, str]:
    if not annotation_path.is_file():
        raise TinyImageNetValidationError(
            f"Tiny ImageNet validation annotations are missing: {annotation_path}"
        )

    annotations: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        if len(fields) < 2 or not fields[0] or not fields[1]:
            raise TinyImageNetValidationError(
                f"invalid validation annotation at {annotation_path}:{line_number}"
            )
        filename, class_id = fields[0], fields[1]
        if class_id not in known_classes:
            raise TinyImageNetValidationError(
                f"unknown validation class {class_id!r} at "
                f"{annotation_path}:{line_number}"
            )
        if filename in annotations:
            raise TinyImageNetValidationError(
                f"duplicate validation filename {filename!r} in {annotation_path}"
            )
        annotations[filename] = class_id
    return annotations


def _validate_validation_images(root: Path, known_classes: set[str]) -> int:
    image_directory = root / "val" / "images"
    try:
        image_paths = find_image_files(image_directory)
    except ImageValidationError as error:
        raise TinyImageNetValidationError(
            f"Tiny ImageNet validation image directory is invalid: {error}"
        ) from error

    annotations = _read_validation_annotations(
        root / "val" / "val_annotations.txt",
        known_classes,
    )
    filenames = {path.name for path in image_paths}
    annotated_filenames = set(annotations)
    if filenames != annotated_filenames:
        missing_annotations = sorted(filenames - annotated_filenames)
        missing_images = sorted(annotated_filenames - filenames)
        raise TinyImageNetValidationError(
            "Tiny ImageNet validation images and annotations do not match; "
            f"images_without_annotations={missing_annotations[:5]}, "
            f"annotations_without_images={missing_images[:5]}"
        )

    for image_path in image_paths:
        try:
            validate_image(image_path)
        except ImageValidationError as error:
            raise TinyImageNetValidationError(
                f"invalid Tiny ImageNet validation image: {error}"
            ) from error
    return len(image_paths)


def validate_tiny_imagenet(
    root: Path,
    *,
    expected_classes: int = 200,
) -> TinyImageNetValidationSummary:
    """Validate Tiny ImageNet classes, image files, and validation labels."""

    if expected_classes <= 0:
        raise ValueError("expected_classes must be positive")
    if not root.is_dir():
        raise TinyImageNetValidationError(f"Tiny ImageNet root is missing: {root}")

    class_ids = _read_class_ids(root / "wnids.txt", expected_classes)
    known_classes = set(class_ids)
    training_image_count = _validate_training_images(root, class_ids)
    validation_image_count = _validate_validation_images(root, known_classes)
    return TinyImageNetValidationSummary(
        root=root,
        class_ids=class_ids,
        training_image_count=training_image_count,
        validation_image_count=validation_image_count,
    )


def _stable_order(paths: tuple[Path, ...], root: Path, seed: int) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(
            f"{seed}:{path.relative_to(root).as_posix()}".encode()
        ).hexdigest(),
    )


def _manifest_record(
    image_path: Path,
    *,
    data_root: Path,
    class_id: str,
    source_split: str,
) -> dict[str, object]:
    try:
        relative_path = image_path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as error:
        raise TinyImageNetValidationError(
            f"Tiny ImageNet image is outside the data root: {image_path}"
        ) from error
    return {
        "sample_id": image_path.stem,
        "relative_path": relative_path,
        "class_id": class_id,
        "source_split": source_split,
    }


def prepare_tiny_imagenet_manifests(
    dataset_root: Path,
    data_root: Path,
    manifest_root: Path,
    *,
    seed: int = 42,
    validation_summary: TinyImageNetValidationSummary | None = None,
) -> TinyImageNetManifestSummary:
    """Create stratified train, validation, calibration, and test manifests."""

    summary = validation_summary or validate_tiny_imagenet(dataset_root)
    if summary.root.resolve() != dataset_root.resolve():
        raise ValueError("validation_summary does not belong to dataset_root")
    records: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "calibration": [],
        "test": [],
    }

    for class_id in summary.class_ids:
        image_directory = dataset_root / "train" / class_id / "images"
        ordered = _stable_order(find_image_files(image_directory), dataset_root, seed)
        train_end = int(len(ordered) * 0.90)
        validation_end = train_end + int(len(ordered) * 0.05)
        class_splits = {
            "train": ordered[:train_end],
            "validation": ordered[train_end:validation_end],
            "calibration": ordered[validation_end:],
        }
        if any(not paths for paths in class_splits.values()):
            raise TinyImageNetValidationError(
                f"class {class_id} is too small for the 90/5/5 split"
            )
        for split_name, paths in class_splits.items():
            records[split_name].extend(
                _manifest_record(
                    path,
                    data_root=data_root,
                    class_id=class_id,
                    source_split="official_train",
                )
                for path in paths
            )

    annotations = _read_validation_annotations(
        dataset_root / "val" / "val_annotations.txt",
        set(summary.class_ids),
    )
    validation_images = {
        path.name: path for path in find_image_files(dataset_root / "val" / "images")
    }
    records["test"] = [
        _manifest_record(
            validation_images[filename],
            data_root=data_root,
            class_id=class_id,
            source_split="official_validation",
        )
        for filename, class_id in sorted(annotations.items())
    ]

    output_directory = manifest_root / "tiny-imagenet" / "v1"
    filenames = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "calibration": "calibration.jsonl",
        "test": "test.jsonl",
    }
    split_ids: dict[str, set[str]] = {}
    required_fields = {"sample_id", "relative_path", "class_id", "source_split"}
    for split_name, filename in filenames.items():
        split_records = sorted(records[split_name], key=lambda item: str(item["sample_id"]))
        write_jsonl(output_directory / filename, split_records)
        split_ids[split_name] = validate_manifest(
            split_records,
            required_fields=required_fields,
            data_root=data_root,
        )
    require_disjoint_splits(split_ids)

    write_json(
        output_directory / "labels.json",
        {
            "schema_version": 1,
            "classes": [
                {"class_id": class_id, "index": index}
                for index, class_id in enumerate(summary.class_ids)
            ],
        },
    )
    return TinyImageNetManifestSummary(
        output_directory=output_directory,
        train_count=len(records["train"]),
        validation_count=len(records["validation"]),
        calibration_count=len(records["calibration"]),
        test_count=len(records["test"]),
    )
