"""Unit tests for the shared structured logging contract."""

from __future__ import annotations

import json
import logging
from typing import cast

import pytest

from src.common.logging import REDACTED, configure_logging, get_logger


def _json_record(captured: str) -> dict[str, object]:
    lines = [line for line in captured.splitlines() if line]
    assert len(lines) == 1
    loaded: object = json.loads(lines[0])
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def test_json_log_contains_required_fields_and_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(
        service="environment-report",
        environment="test",
        git_sha="abc123",
        log_format="json",
    )

    get_logger("pipelines.environment_report").info(
        "cuda_check_completed",
        cuda_available=False,
        request_id="req-123",
    )
    record = _json_record(capsys.readouterr().out)

    assert record["event"] == "cuda_check_completed"
    assert record["level"] == "info"
    assert record["service"] == "environment-report"
    assert record["environment"] == "test"
    assert record["git_sha"] == "abc123"
    assert record["logger"] == "pipelines.environment_report"
    assert cast(str, record["timestamp"]).endswith("Z")
    assert record["cuda_available"] is False
    assert record["request_id"] == "req-123"


def test_sensitive_fields_are_redacted_recursively(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(log_format="json")

    get_logger(__name__).info(
        "configuration_loaded",
        password="unsafe-password",
        nested={"api_token": "unsafe-token", "safe_host": "localhost"},
        values=({"client_secret": "unsafe-secret"},),
        authorization_header="unsafe-authorization",
    )
    captured = capsys.readouterr().out
    record = _json_record(captured)
    nested = cast(dict[str, object], record["nested"])

    assert record["password"] == REDACTED
    assert record["authorization_header"] == REDACTED
    assert nested == {"api_token": REDACTED, "safe_host": "localhost"}
    assert record["values"] == [{"client_secret": REDACTED}]
    assert "unsafe-password" not in captured
    assert "unsafe-token" not in captured
    assert "unsafe-authorization" not in captured
    assert "unsafe-secret" not in captured


def test_exception_log_contains_type_message_and_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(log_format="json")
    logger = get_logger(__name__)

    try:
        raise ValueError("invalid test value")
    except ValueError:
        logger.exception("test_operation_failed", operation="test")

    record = _json_record(capsys.readouterr().out)
    exception = cast(str, record["exception"])
    assert record["event"] == "test_operation_failed"
    assert record["operation"] == "test"
    assert "ValueError" in exception
    assert "invalid test value" in exception
    assert "Traceback" in exception


def test_standard_library_logging_uses_the_same_json_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(
        service="api",
        environment="test",
        git_sha="def456",
        log_level=logging.INFO,
        log_format="json",
    )

    logging.getLogger("third_party.library").warning(
        "dependency_fallback",
        extra={"request_id": "req-stdlib", "api_token": "unsafe-stdlib-token"},
    )
    captured = capsys.readouterr().out
    record = _json_record(captured)

    assert record["event"] == "dependency_fallback"
    assert record["level"] == "warning"
    assert record["logger"] == "third_party.library"
    assert record["service"] == "api"
    assert record["request_id"] == "req-stdlib"
    assert record["api_token"] == REDACTED
    assert "unsafe-stdlib-token" not in captured


def test_environment_variables_supply_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SERVICE_NAME", "worker")
    monkeypatch.setenv("APP_ENV", "ci")
    monkeypatch.setenv("GIT_SHA", "sha-from-env")
    configure_logging()
    logger = get_logger(__name__)

    logger.info("filtered_event")
    logger.warning("worker_started", job_id="job-456")
    record = _json_record(capsys.readouterr().out)

    assert record["event"] == "worker_started"
    assert record["service"] == "worker"
    assert record["environment"] == "ci"
    assert record["git_sha"] == "sha-from-env"
    assert record["job_id"] == "job-456"


def test_console_format_is_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(log_format="console", service="local-service")
    get_logger(__name__).info("application_started")

    captured = capsys.readouterr().out
    assert "application_started" in captured
    assert "local-service" in captured
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("log_level", "VERBOSE"), ("log_format", "xml"), ("service", "  ")],
)
def test_invalid_configuration_is_rejected(keyword: str, value: str) -> None:
    with pytest.raises(ValueError):
        configure_logging(**{keyword: value})
