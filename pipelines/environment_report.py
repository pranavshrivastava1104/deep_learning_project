"""Generate a machine-readable report for the active Python environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import structlog

from src.common.logging import LogEvent, configure_logging, get_logger

DEPENDENCIES = (
    "structlog",
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "open-clip-torch",
)


class _CudaApi(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...


class _TorchModule(Protocol):
    cuda: _CudaApi


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


def _collect_cuda_information() -> dict[str, object]:
    module = _optional_module("torch")
    if module is None:
        return {"available": False, "device_count": 0, "torch_installed": False}

    torch = cast(_TorchModule, module)
    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "torch_installed": True,
    }


def _collect_onnx_provider_information() -> dict[str, object]:
    module = _optional_module("onnxruntime")
    if module is None:
        return {"installed": False, "providers": []}

    onnxruntime = cast(_OnnxRuntimeModule, module)
    return {"installed": True, "providers": onnxruntime.get_available_providers()}


def collect_environment_information(
    logger: structlog.stdlib.BoundLogger,
    *,
    environment: str,
    git_sha: str,
) -> dict[str, object]:
    """Collect safe runtime metadata without dumping environment variables."""

    python_information = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    logger.info(LogEvent.PYTHON_ENVIRONMENT_DETECTED, **python_information)

    dependencies: dict[str, str | None] = {}
    for dependency in DEPENDENCIES:
        version = _distribution_version(dependency)
        dependencies[dependency] = version
        logger.info(
            LogEvent.DEPENDENCY_CHECK_COMPLETED,
            dependency=dependency,
            installed=version is not None,
            version=version,
        )

    cuda = _collect_cuda_information()
    cuda_log = logger.info if cuda["available"] else logger.warning
    cuda_log(
        LogEvent.CUDA_CHECK_COMPLETED,
        cuda_available=cuda["available"],
        device_count=cuda["device_count"],
        fallback_device=None if cuda["available"] else "cpu",
    )

    onnxruntime = _collect_onnx_provider_information()
    logger.info(
        LogEvent.ONNX_PROVIDER_CHECK_COMPLETED,
        installed=onnxruntime["installed"],
        providers=onnxruntime["providers"],
    )

    return {
        "schema_version": 1,
        "generated_at": _utc_timestamp(),
        "environment": environment,
        "git_sha": git_sha,
        "python": python_information,
        "dependencies": dependencies,
        "cuda": cuda,
        "onnxruntime": onnxruntime,
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the environment check and write its report."""

    args = _parse_args(argv)
    output_path = cast(Path, args.output)
    git_sha = _git_sha()
    environment = os.getenv("APP_ENV", "local")

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

        logger.info(
            LogEvent.ENVIRONMENT_REPORT_WRITTEN,
            output_path=str(output_path),
            cuda_available=cast(dict[str, object], report["cuda"])["available"],
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
