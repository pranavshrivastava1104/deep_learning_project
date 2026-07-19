"""Unit tests for the shared local and Colab environment report."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from pipelines import environment_report


def _events(captured: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in captured.splitlines():
        loaded: object = json.loads(line)
        assert isinstance(loaded, dict)
        records.append(cast(dict[str, object], loaded))
    return records


def _mock_runtime_collectors(monkeypatch: pytest.MonkeyPatch, *, cuda: bool) -> None:
    monkeypatch.setattr(
        environment_report,
        "_collect_torch_and_gpu_information",
        lambda: (
            {
                "installed": True,
                "version": "2.13.0",
                "cuda_available": cuda,
                "cuda_version": "13.0" if cuda else None,
            },
            {"name": "Test GPU" if cuda else None, "device_count": 1 if cuda else 0},
        ),
    )
    monkeypatch.setattr(
        environment_report,
        "_collect_onnx_provider_information",
        lambda: {
            "installed": True,
            "version": "1.27.0",
            "available_providers": ["CPUExecutionProvider"],
        },
    )


def test_environment_report_writes_stable_schema_and_lifecycle_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "environment.json"
    (tmp_path / "configs").mkdir()
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setattr(environment_report, "_git_sha", lambda: "abc123")
    _mock_runtime_collectors(monkeypatch, cuda=True)

    assert (
        environment_report.main(
            [
                "--environment",
                "colab",
                "--require-cuda",
                "--repository-root",
                str(tmp_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    report = cast(
        dict[str, object],
        json.loads(output_path.read_text(encoding="utf-8")),
    )
    records = _events(capsys.readouterr().out)
    event_names = [record["event"] for record in records]
    git = cast(dict[str, object], report["git"])
    torch = cast(dict[str, object], report["torch"])
    paths = cast(dict[str, object], report["paths"])
    artifacts = cast(dict[str, object], paths["artifacts"])

    assert report["schema_version"] == 1
    assert report["environment"] == "colab"
    assert git["sha"] == "abc123"
    assert torch["cuda_available"] is True
    assert artifacts["writable"] is True
    assert (tmp_path / "checkpoints").is_dir()
    assert (tmp_path / "data").is_dir()
    assert "environment_check_started" in event_names
    assert "cuda_check_completed" in event_names
    assert "environment_report_written" in event_names
    assert event_names[-1] == "environment_check_completed"
    assert records[0]["logger"] == "pipelines.environment_report"


def test_require_cuda_writes_diagnostic_report_then_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "environment.json"
    (tmp_path / "configs").mkdir()
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setattr(environment_report, "_git_sha", lambda: "abc123")
    _mock_runtime_collectors(monkeypatch, cuda=False)

    with pytest.raises(RuntimeError, match="CUDA is required but unavailable"):
        environment_report.main(
            [
                "--environment",
                "colab",
                "--require-cuda",
                "--repository-root",
                str(tmp_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.is_file()
    records = _events(capsys.readouterr().out)
    assert records[-1]["event"] == "environment_check_failed"
    assert "CUDA is required" in cast(str, records[-1]["exception"])


def test_environment_report_logs_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    output_path = parent_file / "environment.json"
    (tmp_path / "configs").mkdir()
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setattr(environment_report, "_git_sha", lambda: "abc123")
    _mock_runtime_collectors(monkeypatch, cuda=False)

    with pytest.raises(OSError):
        environment_report.main(
            [
                "--repository-root",
                str(tmp_path),
                "--output",
                str(output_path),
            ]
        )

    records = _events(capsys.readouterr().out)
    failed = [record for record in records if record["event"] == "environment_check_failed"]
    write_failed = [
        record for record in records if record["event"] == "environment_report_write_failed"
    ]
    assert len(write_failed) == 1
    assert len(failed) == 1
    assert "exception" in write_failed[0]
    assert write_failed[0]["operation"] == "write_environment_report"


def test_git_sha_uses_repository_identity_and_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "full-git-sha\n", ""),
    )
    assert environment_report._git_sha() == "full-git-sha"

    def raise_os_error(*args: object, **kwargs: object) -> None:
        raise OSError("git unavailable")

    monkeypatch.setenv("GIT_SHA", "sha-from-environment")
    monkeypatch.setattr(subprocess, "run", raise_os_error)
    assert environment_report._git_sha() == "sha-from-environment"


def test_missing_distribution_and_module_are_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    def missing_module(_name: str) -> ModuleType:
        raise ImportError

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)
    monkeypatch.setattr(importlib, "import_module", missing_module)

    assert environment_report._distribution_version("not-installed") is None
    assert environment_report._optional_module("not_installed") is None


def test_runtime_collectors_cover_available_and_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment_report, "_optional_module", lambda _name: None)
    assert environment_report._collect_torch_and_gpu_information() == (
        {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
        },
        {"name": None, "device_count": 0},
    )
    assert environment_report._collect_onnx_provider_information() == {
        "installed": False,
        "version": None,
        "available_providers": [],
    }

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda _device: "Test GPU",
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        version=SimpleNamespace(cuda="13.0"),
        __version__="2.13.0",
    )
    fake_onnxruntime = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    modules = {"torch": fake_torch, "onnxruntime": fake_onnxruntime}
    monkeypatch.setattr(environment_report, "_optional_module", modules.__getitem__)
    monkeypatch.setattr(
        environment_report,
        "_distribution_version",
        lambda distribution: "1.27.0" if distribution == "onnxruntime" else None,
    )

    assert environment_report._collect_torch_and_gpu_information() == (
        {
            "installed": True,
            "version": "2.13.0",
            "cuda_available": True,
            "cuda_version": "13.0",
        },
        {"name": "Test GPU", "device_count": 2},
    )
    assert environment_report._collect_onnx_provider_information() == {
        "installed": True,
        "version": "1.27.0",
        "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
