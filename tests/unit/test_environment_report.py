"""Unit tests for the first structured-logging consumer."""

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


def test_environment_report_writes_json_and_lifecycle_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "environment.json"
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setattr(environment_report, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(
        environment_report,
        "_collect_cuda_information",
        lambda: {"available": False, "device_count": 0, "torch_installed": True},
    )
    monkeypatch.setattr(
        environment_report,
        "_collect_onnx_provider_information",
        lambda: {"installed": True, "providers": ["CPUExecutionProvider"]},
    )

    assert environment_report.main(["--output", str(output_path)]) == 0

    report = cast(
        dict[str, object],
        json.loads(output_path.read_text(encoding="utf-8")),
    )
    records = _events(capsys.readouterr().out)
    event_names = [record["event"] for record in records]

    assert report["schema_version"] == 1
    assert report["environment"] == "test"
    assert report["git_sha"] == "abc123"
    assert "environment_check_started" in event_names
    assert "cuda_check_completed" in event_names
    assert "environment_report_written" in event_names
    assert event_names[-1] == "environment_check_completed"
    assert records[0]["logger"] == "pipelines.environment_report"


def test_environment_report_logs_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    output_path = parent_file / "environment.json"
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setattr(environment_report, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(
        environment_report,
        "_collect_cuda_information",
        lambda: {"available": False, "device_count": 0, "torch_installed": False},
    )
    monkeypatch.setattr(
        environment_report,
        "_collect_onnx_provider_information",
        lambda: {"installed": False, "providers": []},
    )

    with pytest.raises(OSError):
        environment_report.main(["--output", str(output_path)])

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


def test_optional_runtime_collectors_cover_available_and_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment_report, "_optional_module", lambda _name: None)
    assert environment_report._collect_cuda_information() == {
        "available": False,
        "device_count": 0,
        "torch_installed": False,
    }
    assert environment_report._collect_onnx_provider_information() == {
        "installed": False,
        "providers": [],
    }

    fake_cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    fake_onnxruntime = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    modules = {"torch": fake_torch, "onnxruntime": fake_onnxruntime}
    monkeypatch.setattr(environment_report, "_optional_module", modules.__getitem__)

    assert environment_report._collect_cuda_information() == {
        "available": True,
        "device_count": 2,
        "torch_installed": True,
    }
    assert environment_report._collect_onnx_provider_information() == {
        "installed": True,
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
