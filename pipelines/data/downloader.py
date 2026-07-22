"""Download and extract the public datasets used by Phase 2."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from zipfile import BadZipFile

import requests
from tqdm import tqdm

from pipelines.data.common import UnsafeArchiveError, safe_extract_zip

DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT = (10.0, 120.0)
USER_AGENT = "ml-engineer-challenge-dataset-downloader/0.1.0"


class DatasetError(RuntimeError):
    """Base error for dataset acquisition failures."""


class DatasetDownloadError(DatasetError):
    """Raised when an archive cannot be downloaded."""


class DatasetExtractionError(DatasetError):
    """Raised when an archive cannot be safely extracted."""


class DatasetValidationError(DatasetError):
    """Raised when extracted data does not match its expected layout."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Download and extraction contract for one public dataset archive."""

    name: str
    url: str
    filename: str
    archive_directory: str
    extracted_directory: str
    required_paths: tuple[str, ...] = ()
    required_glob: str | None = None


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Result returned for a created or already-present dataset."""

    dataset: str
    path: Path
    created: bool


class HttpResponse(Protocol):
    """Subset of requests.Response needed by the streaming downloader."""

    headers: Mapping[str, str]

    def raise_for_status(self) -> None: ...

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...


class HttpGet(Protocol):
    """Injectable HTTP GET operation used by network-free unit tests."""

    def __call__(
        self,
        url: str,
        *,
        stream: bool,
        timeout: tuple[float, float],
        headers: Mapping[str, str],
    ) -> AbstractContextManager[HttpResponse]: ...


DATASETS: dict[str, DatasetSpec] = {
    "tiny_imagenet": DatasetSpec(
        name="tiny_imagenet",
        url="https://cs231n.stanford.edu/tiny-imagenet-200.zip",
        filename="tiny-imagenet-200.zip",
        archive_directory="tiny-imagenet-200",
        extracted_directory="tiny-imagenet-200",
        required_paths=("wnids.txt", "train", "val"),
        required_glob="**/*.JPEG",
    ),
    "coco_val2017": DatasetSpec(
        name="coco_val2017",
        url="https://images.cocodataset.org/zips/val2017.zip",
        filename="val2017.zip",
        archive_directory="val2017",
        extracted_directory="val2017",
        required_glob="*.jpg",
    ),
    "coco_annotations": DatasetSpec(
        name="coco_annotations",
        url=(
            "https://images.cocodataset.org/"
            "annotations/annotations_trainval2017.zip"
        ),
        filename="annotations_trainval2017.zip",
        archive_directory="annotations",
        extracted_directory="annotations",
        required_paths=("instances_val2017.json",),
    ),
    "test_images": DatasetSpec(
        name="test_images",
        url=(
            "https://github.com/EliSchwartz/imagenet-sample-images/"
            "archive/refs/heads/master.zip"
        ),
        filename="imagenet-sample-images.zip",
        archive_directory="imagenet-sample-images-master",
        extracted_directory="imagenet-sample-images",
        required_glob="*.JPEG",
    ),
}


def _content_length(headers: Mapping[str, str]) -> int | None:
    raw_value = headers.get("content-length")
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None


def _download_archive(
    spec: DatasetSpec,
    archive_path: Path,
    *,
    http_get: HttpGet | None,
    show_progress: bool,
) -> None:
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    request = http_get or cast(HttpGet, requests.get)

    try:
        with request(
            spec.url,
            stream=True,
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        ) as response:
            response.raise_for_status()
            total = _content_length(response.headers)
            with (
                partial_path.open("wb") as output,
                tqdm(
                    desc=f"Downloading {spec.filename}",
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    disable=not show_progress,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                    if not chunk:
                        continue
                    output.write(chunk)
                    progress.update(len(chunk))
        partial_path.replace(archive_path)
    except (OSError, requests.RequestException) as error:
        partial_path.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"failed to download {spec.name} from {spec.url}: {error}"
        ) from error


def validate_dataset_root(spec: DatasetSpec, root: Path) -> None:
    """Validate the lightweight directory contract for an extracted dataset."""

    if not root.is_dir():
        raise DatasetValidationError(
            f"{spec.name} did not create expected directory: {root}"
        )

    missing = [relative for relative in spec.required_paths if not (root / relative).exists()]
    if missing:
        joined = ", ".join(missing)
        raise DatasetValidationError(
            f"{spec.name} is missing required extracted paths under {root}: {joined}"
        )

    if spec.required_glob is not None and next(root.glob(spec.required_glob), None) is None:
        raise DatasetValidationError(
            f"{spec.name} contains no files matching {spec.required_glob!r} under {root}"
        )


def _discard_archive(archive_path: Path) -> None:
    try:
        archive_path.unlink(missing_ok=True)
    except OSError:
        pass


def download_dataset(
    dataset: str,
    data_dir: Path,
    *,
    http_get: HttpGet | None = None,
    show_progress: bool = True,
) -> DownloadResult:
    """Download, validate, extract and normalize one configured dataset."""

    try:
        spec = DATASETS[dataset]
    except KeyError as error:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown dataset {dataset!r}; expected one of: {supported}") from error

    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / spec.extracted_directory
    archive_path = data_dir / spec.filename

    if destination.exists():
        validate_dataset_root(spec, destination)
        if archive_path.exists() and not archive_path.is_file():
            raise DatasetValidationError(
                f"archive cleanup path exists but is not a file: {archive_path}"
            )
        archive_path.unlink(missing_ok=True)
        return DownloadResult(dataset=dataset, path=destination, created=False)

    if archive_path.exists() and not archive_path.is_file():
        raise DatasetDownloadError(f"archive path exists but is not a file: {archive_path}")

    if not archive_path.exists():
        _download_archive(
            spec,
            archive_path,
            http_get=http_get,
            show_progress=show_progress,
        )

    try:
        with tempfile.TemporaryDirectory(
            dir=data_dir,
            prefix=f".{dataset}-extract-",
        ) as temporary_directory:
            staging_root = Path(temporary_directory)
            safe_extract_zip(archive_path, staging_root)
            extracted_root = staging_root / spec.archive_directory
            validate_dataset_root(spec, extracted_root)

            archive_path.unlink()
            extracted_root.replace(destination)
    except (BadZipFile, OSError, RuntimeError, UnsafeArchiveError) as error:
        _discard_archive(archive_path)
        raise DatasetExtractionError(
            f"failed to extract {dataset} from {archive_path}: {error}"
        ) from error

    validate_dataset_root(spec, destination)
    return DownloadResult(dataset=dataset, path=destination, created=True)


def download_datasets(
    datasets: Iterable[str],
    data_dir: Path,
    *,
    show_progress: bool = True,
) -> list[DownloadResult]:
    """Download multiple datasets in order and stop on the first failure."""

    return [
        download_dataset(dataset, data_dir, show_progress=show_progress)
        for dataset in datasets
    ]
