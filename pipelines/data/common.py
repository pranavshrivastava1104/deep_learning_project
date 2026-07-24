"""Common filesystem utilities for Phase 2 data preparation."""

from __future__ import annotations

import json
import shutil
import stat
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast
from zipfile import ZipFile, ZipInfo

from PIL import Image

SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})


class UnsafeArchiveError(ValueError):
    """Raised when an archive member could escape or alter the extraction root."""


class ImageValidationError(ValueError):
    """Raised when an image is missing, corrupt or unsupported."""


class ManifestValidationError(ValueError):
    """Raised when generated manifest records are missing required data."""


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Validated image properties needed by later data preparation."""

    path: Path
    width: int
    height: int
    format: str


def validate_image(
    path: Path,
    *,
    supported_formats: Collection[str] = SUPPORTED_IMAGE_FORMATS,
) -> ImageMetadata:
    """Verify an image, reopen it, and fully decode an RGB conversion."""

    if not path.is_file():
        raise ImageValidationError(f"image does not exist or is not a file: {path}")

    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ImageValidationError(
                    f"image dimensions must be positive, got {width}x{height}: {path}"
                )
            with image.convert("RGB") as rgb_image:
                rgb_image.load()
    except ImageValidationError:
        raise
    except (OSError, ValueError) as error:
        raise ImageValidationError(f"image cannot be decoded: {path}: {error}") from error

    if image_format is None:
        raise ImageValidationError(f"image format could not be detected: {path}")
    normalized_format = image_format.upper()
    normalized_supported = {value.upper() for value in supported_formats}
    if normalized_format not in normalized_supported:
        supported = ", ".join(sorted(normalized_supported))
        raise ImageValidationError(
            f"unsupported image format {normalized_format!r} for {path}; expected: {supported}"
        )

    return ImageMetadata(
        path=path,
        width=width,
        height=height,
        format=normalized_format,
    )


def find_image_files(root: Path) -> tuple[Path, ...]:
    """Return supported image files below a directory in stable path order."""

    if not root.is_dir():
        raise ImageValidationError(f"image directory does not exist: {root}")
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        )
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Write JSONL atomically with stable keys."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True))
            output.write("\n")
    temporary_path.replace(path)


def write_json(path: Path, payload: object) -> None:
    """Write formatted JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL manifest and reject non-object records."""

    records: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            loaded: object = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ManifestValidationError(
                f"invalid JSON at {path}:{line_number}"
            ) from error
        if not isinstance(loaded, dict):
            raise ManifestValidationError(
                f"manifest record must be an object at {path}:{line_number}"
            )
        records.append(cast(dict[str, object], loaded))
    return records


def validate_manifest(
    records: Iterable[Mapping[str, object]],
    *,
    required_fields: Collection[str],
    data_root: Path,
) -> set[str]:
    """Validate required fields, unique sample IDs, and referenced paths."""

    sample_ids: set[str] = set()
    resolved_root = data_root.resolve()
    for index, record in enumerate(records, start=1):
        missing = sorted(set(required_fields) - record.keys())
        if missing:
            raise ManifestValidationError(
                f"manifest record {index} is missing fields: {', '.join(missing)}"
            )

        sample_id = record.get("sample_id")
        relative_path = record.get("relative_path")
        if not isinstance(sample_id, str) or not sample_id:
            raise ManifestValidationError(
                f"manifest record {index} has an invalid sample_id"
            )
        if sample_id in sample_ids:
            raise ManifestValidationError(f"duplicate sample_id: {sample_id}")
        if not isinstance(relative_path, str) or not relative_path:
            raise ManifestValidationError(
                f"manifest record {index} has an invalid relative_path"
            )

        image_path = (data_root / relative_path).resolve()
        if not image_path.is_relative_to(resolved_root) or not image_path.is_file():
            raise ManifestValidationError(
                f"manifest path does not resolve to an image: {relative_path}"
            )
        sample_ids.add(sample_id)
    return sample_ids


def require_disjoint_splits(split_ids: Mapping[str, set[str]]) -> None:
    """Reject sample IDs that occur in more than one named split."""

    names = list(split_ids)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            overlap = split_ids[first_name] & split_ids[second_name]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ManifestValidationError(
                    f"splits {first_name!r} and {second_name!r} overlap: {examples}"
                )


def _safe_member_target(destination: Path, member: ZipInfo) -> Path:
    normalized = PurePosixPath(member.filename.replace("\\", "/"))
    parts = tuple(part for part in normalized.parts if part not in {"", "."})

    if (
        not parts
        or normalized.is_absolute()
        or ".." in parts
        or parts[0].endswith(":")
    ):
        raise UnsafeArchiveError(f"unsafe ZIP member path: {member.filename!r}")

    unix_mode = (member.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode):
        raise UnsafeArchiveError(
            f"symbolic links are not allowed in ZIP files: {member.filename!r}"
        )

    root = destination.resolve()
    target = (root / Path(*parts)).resolve()
    if not target.is_relative_to(root):
        raise UnsafeArchiveError(f"ZIP member escapes extraction directory: {member.filename!r}")
    return target


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP after validating every member path.

    Extraction is performed manually so absolute paths, parent traversal, symbolic
    links and duplicate targets cannot overwrite files outside the staging directory.
    """

    destination.mkdir(parents=True, exist_ok=True)
    members: list[tuple[ZipInfo, Path]] = []
    targets: set[Path] = set()

    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = _safe_member_target(destination, member)
            if target in targets:
                raise UnsafeArchiveError(f"duplicate ZIP member target: {member.filename!r}")
            targets.add(target)
            members.append((member, target))

        for member, target in members:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
