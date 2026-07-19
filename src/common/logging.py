"""Secret-safe structured logging shared by all Python entry points."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, Literal, cast

import structlog
from structlog.typing import EventDict, WrappedLogger

REDACTED = "[REDACTED]"
LogFormat = Literal["json", "console"]
Processor = Callable[[WrappedLogger, str, EventDict], EventDict]

_SENSITIVE_KEY_PARTS = (
    "password",
    "token",
    "secret",
    "authorization",
    "api_key",
    "access_key",
    "connection_string",
)


class LogEvent(StrEnum):
    """Phase 1 event catalogue for consistent machine-readable names."""

    APPLICATION_STARTED = "application_started"
    APPLICATION_STOPPED = "application_stopped"
    APPLICATION_STARTUP_FAILED = "application_startup_failed"

    ENVIRONMENT_CHECK_STARTED = "environment_check_started"
    PYTHON_ENVIRONMENT_DETECTED = "python_environment_detected"
    DEPENDENCY_CHECK_COMPLETED = "dependency_check_completed"
    CUDA_CHECK_COMPLETED = "cuda_check_completed"
    ONNX_PROVIDER_CHECK_COMPLETED = "onnx_provider_check_completed"
    GIT_IDENTITY_DETECTED = "git_identity_detected"
    ENVIRONMENT_REPORT_WRITTEN = "environment_report_written"
    ENVIRONMENT_REPORT_WRITE_FAILED = "environment_report_write_failed"
    ENVIRONMENT_CHECK_COMPLETED = "environment_check_completed"
    ENVIRONMENT_CHECK_FAILED = "environment_check_failed"

    CONFIGURATION_LOADED = "configuration_loaded"
    CONFIGURATION_MISSING = "configuration_missing"
    CONFIGURATION_INVALID = "configuration_invalid"

    API_STARTED = "api_started"
    API_HEALTH_CHECK = "api_health_check"
    WORKER_STARTED = "worker_started"
    DEPENDENCY_HEALTH_CHECK_PASSED = "dependency_health_check_passed"
    DEPENDENCY_HEALTH_CHECK_FAILED = "dependency_health_check_failed"


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).casefold().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_sensitive_key(key) else _redact_nested(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_nested(item) for item in value)
    return value


def _redact_sensitive_fields(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    for key in list(event_dict):
        event_dict[key] = REDACTED if _is_sensitive_key(key) else _redact_nested(event_dict[key])
    return event_dict


def _standard_fields(service: str, environment: str, git_sha: str) -> Processor:
    def add_fields(
        _logger: WrappedLogger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        event_dict["service"] = service
        event_dict["environment"] = environment
        event_dict["git_sha"] = git_sha
        return event_dict

    return add_fields


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level

    configured = level or os.getenv("LOG_LEVEL") or "INFO"
    numeric_level = logging.getLevelNamesMapping().get(configured.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"invalid log level: {configured!r}")
    return numeric_level


def _resolve_format(log_format: str | None) -> LogFormat:
    configured = (log_format or os.getenv("LOG_FORMAT") or "console").casefold()
    if configured not in {"json", "console"}:
        raise ValueError(f"invalid log format: {configured!r}; expected 'json' or 'console'")
    return cast(LogFormat, configured)


def _required_text(explicit: str | None, environment_key: str, default: str) -> str:
    value = explicit if explicit is not None else os.getenv(environment_key, default)
    value = value.strip()
    if not value:
        raise ValueError(f"{environment_key} must not be empty")
    return value


def configure_logging(
    *,
    service: str | None = None,
    environment: str | None = None,
    git_sha: str | None = None,
    log_level: str | int | None = None,
    log_format: str | None = None,
) -> None:
    """Configure structlog and standard logging to emit one format to stdout.

    Explicit arguments override environment variables. The supported variables are
    ``SERVICE_NAME``, ``APP_ENV``, ``GIT_SHA``, ``LOG_LEVEL`` and ``LOG_FORMAT``.
    Repeated calls replace the root handler, which keeps tests and process bootstrap
    deterministic.
    """

    resolved_service = _required_text(service, "SERVICE_NAME", "ml-challenge")
    resolved_environment = _required_text(environment, "APP_ENV", "local")
    resolved_git_sha = _required_text(git_sha, "GIT_SHA", "unknown")
    resolved_level = _resolve_level(log_level)
    resolved_format = _resolve_format(log_format)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp")
    standard_fields = _standard_fields(
        resolved_service,
        resolved_environment,
        resolved_git_sha,
    )
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        standard_fields,
        _redact_sensitive_fields,
    ]

    renderer: structlog.types.Processor
    if resolved_format == "json":
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[structlog.stdlib.ExtraAdder(), *shared_processors],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_sensitive_fields,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(resolved_level)
    logging.captureWarnings(True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a named structured logger after process-level configuration."""

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
