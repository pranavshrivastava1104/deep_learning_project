"""Generate a machine-readable report for the active Python environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import structlog

from src.common.logging import LogEvent, configure_logging, get_logger

DEPENDENCIES: dict[str, str] = {
    "project": "ml-engineer-challenge",
    "structlog": "structlog",
    "torch": "torch",
    "torchvision": "torchvision",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "open_clip_torch": "open-clip-torch",
}


class _CudaApi(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def get_device_name(self, device: int) -> str: ...


class _TorchVersionApi(Protocol):
    cuda: str | None


class _TorchModule(Protocol):
    __version__: str
    cuda: _CudaApi
    version: _TorchVersionApi


class _OnnxRuntimeModule(Protocol):
    def get_available_providers(self) -> list[str]: ...


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return os.getenv("GIT_SHA", "unknown")

    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else os.getenv("GIT_SHA", "unknown")


def _optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _collect_torch_and_gpu_information() -> tuple[dict[str, object], dict[str, object]]:
    module = _optional_module("torch")
    if module is None:
        return (
            {
                "installed": False,
                "version": None,
                "cuda_available": False,
                "cuda_version": None,
            },
            {"name": None, "device_count": 0},
        )

    torch = cast(_TorchModule, module)
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available and device_count else None
    return (
        {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
        },
        {"name": gpu_name, "device_count": device_count},
    )


def _collect_onnx_provider_information() -> dict[str, object]:
    module = _optional_module("onnxruntime")
    if module is None:
        return {"installed": False, "version": None, "available_providers": []}

    onnxruntime = cast(_OnnxRuntimeModule, module)
    return {
        "installed": True,
        "version": _distribution_version("onnxruntime"),
        "available_providers": onnxruntime.get_available_providers(),
    }


def _path_status(path: Path, *, create: bool) -> dict[str, object]:
    if create:
        path.mkdir(parents=True, exist_ok=True)

    exists = path.is_dir()
    writable = False
    if exists:
        try:
            with tempfile.NamedTemporaryFile(prefix=".write-check-", dir=path):
                writable = True
        except OSError:
            writable = False

    return {"path": str(path), "exists": exists, "writable": writable}


def _collect_path_information(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve()
    disk = shutil.disk_usage(root)
    return {
        "repository_root": _path_status(root, create=False),
        "configs": _path_status(root / "configs", create=False),
        "artifacts": _path_status(root / "artifacts", create=True),
        "checkpoints": _path_status(root / "checkpoints", create=True),
        "data": _path_status(root / "data", create=True),
        "google_drive": {
            "path": "/content/drive",
            "mounted": Path("/content/drive").is_dir(),
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def collect_environment_information(
    logger: structlog.stdlib.BoundLogger,
    *,
    environment: str,
    git_sha: str,
    repository_root: Path,
) -> dict[str, object]:
    """Collect safe runtime metadata without dumping environment variables."""

    python_information = {
        "version": platform.python_version(),
        "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }
    logger.info(LogEvent.PYTHON_ENVIRONMENT_DETECTED, **python_information)

    packages: dict[str, str | None] = {}
    for package_name, distribution in DEPENDENCIES.items():
        version = _distribution_version(distribution)
        packages[package_name] = version
        logger.info(
            LogEvent.DEPENDENCY_CHECK_COMPLETED,
            dependency=package_name,
            installed=version is not None,
            version=version,
        )

    torch, gpu = _collect_torch_and_gpu_information()
    cuda_log = logger.info if torch["cuda_available"] else logger.warning
    cuda_log(
        LogEvent.CUDA_CHECK_COMPLETED,
        cuda_available=torch["cuda_available"],
        cuda_version=torch["cuda_version"],
        device_count=gpu["device_count"],
        gpu_name=gpu["name"],
        fallback_device=None if torch["cuda_available"] else "cpu",
    )

    onnxruntime = _collect_onnx_provider_information()
    logger.info(
        LogEvent.ONNX_PROVIDER_CHECK_COMPLETED,
        installed=onnxruntime["installed"],
        providers=onnxruntime["available_providers"],
    )

    return {
        "schema_version": 1,
        "generated_at": _utc_timestamp(),
        "environment": environment,
        "git": {"sha": git_sha},
        "python": python_information,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
        },
        "packages": packages,
        "torch": torch,
        "gpu": gpu,
        "onnxruntime": onnxruntime,
        "paths": _collect_path_information(repository_root),
    }


def _write_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/environment-local.json"),
        help="Path for the JSON environment report.",
    )
    parser.add_argument(
        "--environment",
        help="Execution environment label; defaults to APP_ENV or local.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root whose required directories are checked.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Write the report, then fail when PyTorch cannot execute on CUDA.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the environment check and write its report."""

    args = _parse_args(argv)
    output_path = cast(Path, args.output)
    repository_root = cast(Path, args.repository_root)
    environment = cast(str | None, args.environment) or os.getenv("APP_ENV") or "local"
    require_cuda = cast(bool, args.require_cuda)
    git_sha = _git_sha()

    configure_logging(
        service=os.getenv("SERVICE_NAME", "environment-report"),
        environment=environment,
        git_sha=git_sha,
    )
    logger = get_logger("pipelines.environment_report")
    logger.info(LogEvent.ENVIRONMENT_CHECK_STARTED, output_path=str(output_path))
    logger.info(LogEvent.GIT_IDENTITY_DETECTED, git_sha=git_sha)

    try:
        report = collect_environment_information(
            logger,
            environment=environment,
            git_sha=git_sha,
            repository_root=repository_root,
        )
        try:
            _write_report(report, output_path)
        except OSError:
            logger.exception(
                LogEvent.ENVIRONMENT_REPORT_WRITE_FAILED,
                operation="write_environment_report",
                output_path=str(output_path),
            )
            raise

        torch_information = cast(dict[str, object], report["torch"])
        cuda_available = cast(bool, torch_information["cuda_available"])
        logger.info(
            LogEvent.ENVIRONMENT_REPORT_WRITTEN,
            output_path=str(output_path),
            cuda_available=cuda_available,
        )
        if require_cuda and not cuda_available:
            raise RuntimeError(
                "CUDA is required but unavailable. In Colab select Runtime > "
                "Change runtime type > GPU, then restart and rerun the notebook."
            )

        logger.info(LogEvent.ENVIRONMENT_CHECK_COMPLETED, output_path=str(output_path))
    except Exception:
        logger.exception(
            LogEvent.ENVIRONMENT_CHECK_FAILED,
            operation="collect_and_write_environment_report",
            output_path=str(output_path),
        )
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
