"""Validate sample images used to construct the similarity dataset."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance

from pipelines.data.common import (
    ImageValidationError,
    find_image_files,
    require_disjoint_splits,
    validate_image,
    validate_manifest,
    write_jsonl,
)


class SimilarityValidationError(ValueError):
    """Raised when the similarity source inventory is unusable."""


@dataclass(frozen=True, slots=True)
class SimilarityValidationSummary:
    """Validated source images available for later query/gallery construction."""

    root: Path
    image_paths: tuple[Path, ...]

    @property
    def image_count(self) -> int:
        return len(self.image_paths)


@dataclass(frozen=True, slots=True)
class SimilarityManifestSummary:
    """Paths and counts for generated similarity manifests."""

    output_directory: Path
    processed_directory: Path
    calibration_count: int
    query_count: int
    gallery_count: int
    positive_pair_count: int


def validate_similarity_source(
    root: Path,
    *,
    minimum_images: int = 2,
) -> SimilarityValidationSummary:
    """Validate the source image inventory before query/gallery generation."""

    if minimum_images <= 0:
        raise ValueError("minimum_images must be positive")
    try:
        image_paths = find_image_files(root)
    except ImageValidationError as error:
        raise SimilarityValidationError(
            f"similarity image directory is invalid: {error}"
        ) from error

    if len(image_paths) < minimum_images:
        raise SimilarityValidationError(
            f"similarity source requires at least {minimum_images} images, "
            f"found {len(image_paths)}"
        )

    for image_path in image_paths:
        try:
            validate_image(image_path)
        except ImageValidationError as error:
            raise SimilarityValidationError(f"invalid similarity source image: {error}") from error

    return SimilarityValidationSummary(root=root, image_paths=image_paths)


def _stable_source_order(paths: tuple[Path, ...], root: Path, seed: int) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(
            f"{seed}:{path.relative_to(root).as_posix()}".encode()
        ).hexdigest(),
    )


def _relative_to_data_root(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as error:
        raise SimilarityValidationError(
            f"generated similarity image is outside the data root: {path}"
        ) from error


def _create_query_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        with image.convert("RGB") as rgb_image:
            width, height = rgb_image.size
            crop_width = max(1, int(width * 0.90))
            crop_height = max(1, int(height * 0.90))
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            cropped = rgb_image.crop(
                (left, top, left + crop_width, top + crop_height)
            )
            resized = cropped.resize((width, height), Image.Resampling.LANCZOS)
            adjusted = ImageEnhance.Brightness(resized).enhance(1.05)
            adjusted.save(destination, format="JPEG", quality=95)
            cropped.close()
            resized.close()
            adjusted.close()


def prepare_similarity_manifests(
    source_root: Path,
    data_root: Path,
    manifest_root: Path,
    *,
    seed: int = 42,
    calibration_fraction: float = 0.10,
    validation_summary: SimilarityValidationSummary | None = None,
) -> SimilarityManifestSummary:
    """Create calibration images and deterministic query/gallery pairs."""

    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    source_summary = validation_summary or validate_similarity_source(source_root)
    if source_summary.root.resolve() != source_root.resolve():
        raise ValueError("validation_summary does not belong to source_root")
    ordered_sources = _stable_source_order(
        source_summary.image_paths,
        source_root,
        seed,
    )
    calibration_count = max(1, int(len(ordered_sources) * calibration_fraction))
    if calibration_count >= len(ordered_sources):
        raise SimilarityValidationError(
            "similarity source does not contain enough retrieval images "
            "after calibration selection"
        )

    calibration_sources = ordered_sources[:calibration_count]
    retrieval_sources = ordered_sources[calibration_count:]
    processed_directory = data_root / "similarity-gallery" / "v1"
    calibration_directory = processed_directory / "calibration"
    query_directory = processed_directory / "queries"
    gallery_directory = processed_directory / "gallery"
    for directory in (
        calibration_directory,
        query_directory,
        gallery_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    calibration_records: list[dict[str, object]] = []
    for index, source in enumerate(calibration_sources):
        sample_id = f"similarity-calibration-{index:04d}"
        destination = calibration_directory / f"{sample_id}{source.suffix.casefold()}"
        shutil.copy2(source, destination)
        validate_image(destination)
        calibration_records.append(
            {
                "sample_id": sample_id,
                "relative_path": _relative_to_data_root(destination, data_root),
            }
        )

    query_records: list[dict[str, object]] = []
    gallery_records: list[dict[str, object]] = []
    positive_pairs: list[dict[str, object]] = []
    for index, source in enumerate(retrieval_sources):
        gallery_id = f"similarity-gallery-{index:04d}"
        query_id = f"similarity-query-{index:04d}"
        gallery_path = gallery_directory / f"{gallery_id}{source.suffix.casefold()}"
        query_path = query_directory / f"{query_id}.jpg"
        shutil.copy2(source, gallery_path)
        _create_query_image(source, query_path)
        validate_image(gallery_path)
        validate_image(query_path)

        gallery_records.append(
            {
                "sample_id": gallery_id,
                "relative_path": _relative_to_data_root(gallery_path, data_root),
            }
        )
        query_records.append(
            {
                "sample_id": query_id,
                "relative_path": _relative_to_data_root(query_path, data_root),
            }
        )
        positive_pairs.append(
            {
                "query_sample_id": query_id,
                "gallery_sample_id": gallery_id,
            }
        )

    output_directory = manifest_root / "similarity-gallery" / "v1"
    write_jsonl(output_directory / "calibration.jsonl", calibration_records)
    write_jsonl(output_directory / "test_queries.jsonl", query_records)
    write_jsonl(output_directory / "gallery.jsonl", gallery_records)
    write_jsonl(output_directory / "positive_pairs.jsonl", positive_pairs)

    required_fields = {"sample_id", "relative_path"}
    split_ids = {
        "calibration": validate_manifest(
            calibration_records,
            required_fields=required_fields,
            data_root=data_root,
        ),
        "test_queries": validate_manifest(
            query_records,
            required_fields=required_fields,
            data_root=data_root,
        ),
        "gallery": validate_manifest(
            gallery_records,
            required_fields=required_fields,
            data_root=data_root,
        ),
    }
    require_disjoint_splits(split_ids)
    query_ids = split_ids["test_queries"]
    gallery_ids = split_ids["gallery"]
    paired_query_ids: set[str] = set()
    for pair in positive_pairs:
        query_id = pair["query_sample_id"]
        gallery_id = pair["gallery_sample_id"]
        if not isinstance(query_id, str) or query_id not in query_ids:
            raise SimilarityValidationError(f"unknown positive-pair query: {query_id}")
        if not isinstance(gallery_id, str) or gallery_id not in gallery_ids:
            raise SimilarityValidationError(
                f"unknown positive-pair gallery item: {gallery_id}"
            )
        paired_query_ids.add(query_id)
    if paired_query_ids != query_ids:
        raise SimilarityValidationError(
            "every similarity query must have at least one positive gallery item"
        )

    return SimilarityManifestSummary(
        output_directory=output_directory,
        processed_directory=processed_directory,
        calibration_count=len(calibration_records),
        query_count=len(query_records),
        gallery_count=len(gallery_records),
        positive_pair_count=len(positive_pairs),
    )
