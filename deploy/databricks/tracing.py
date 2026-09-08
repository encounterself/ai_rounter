"""Optional, failure-isolated MLflow GenAI tracing for the serving runtime."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_SECRET_PARTS = (
    "authorization",
    "access_token",
    "api_key",
    "client_secret",
    "password",
    "secret",
    "token",
)
_MAX_VALUE_LENGTH = 512


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_ratio(value: Any, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TracingConfig:
    enabled: bool = False
    experiment_name: str | None = None
    sampling_ratio: float = 0.0
    capture_content: bool = False
    beta: bool = False
    uc_catalog: str | None = None
    uc_schema: str | None = None
    uc_table: str | None = None
    uc_volume: str | None = None
    tracking_uri: str | None = None
    destination: str | None = None
    sql_warehouse_id: str | None = None

    @classmethod
    def from_env(cls, profile: Mapping[str, Any] | None = None) -> "TracingConfig":
        profile = profile or {}
        configured = profile.get("tracing", {})
        if not isinstance(configured, Mapping):
            configured = {}

        def value(key: str, env_name: str, default: Any = None) -> Any:
            environment_value = os.getenv(env_name)
            if environment_value not in (None, ""):
                return environment_value
            return configured.get(key, default)

        enabled = _as_bool(value("enabled", "MLFLOW_TRACING_ENABLED", False))
        sampling_default = 1.0 if enabled else 0.0
        return cls(
            enabled=enabled,
            experiment_name=_optional_text(value("experiment_name", "MLFLOW_EXPERIMENT_NAME")),
            sampling_ratio=_as_ratio(value("sampling_ratio", "MLFLOW_TRACE_SAMPLING_RATIO", sampling_default)),
            capture_content=_as_bool(value("capture_content", "MLFLOW_TRACE_CAPTURE_CONTENT", False)),
            beta=_as_bool(value("beta", "MLFLOW_TRACE_BETA", False)),
            uc_catalog=_optional_text(value("uc_catalog", "MLFLOW_TRACE_UC_CATALOG")),
            uc_schema=_optional_text(value("uc_schema", "MLFLOW_TRACE_UC_SCHEMA")),
            uc_table=_optional_text(value("uc_table", "MLFLOW_TRACE_UC_TABLE")),
            uc_volume=_optional_text(value("uc_volume", "MLFLOW_TRACE_UC_VOLUME")),
            tracking_uri=_optional_text(value("tracking_uri", "MLFLOW_TRACKING_URI")),
            destination=_optional_text(value("destination", "MLFLOW_TRACING_DESTINATION")),
            sql_warehouse_id=_optional_text(value("sql_warehouse_id", "MLFLOW_TRACING_SQL_WAREHOUSE_ID")),
        )


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)[:_MAX_VALUE_LENGTH]


def _safe_identifiers(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [_safe_identifier(item) for item in list(value)[:32]]
    if value is None:
        return []
    return [_safe_identifier(value)]


def _safe_identifier(value: Any) -> str:
    value = str(value)[:_MAX_VALUE_LENGTH]
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _attribute_value(value: Any) -> Any:
    """Convert a redacted value to an MLflow attribute-safe scalar."""

    value = redact(value)
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))[:_MAX_VALUE_LENGTH]
    return value


def _is_secret_key(key: Any) -> bool:
    lowered = str(key).lower().replace("-", "_")
    return any(part in lowered for part in _SECRET_PARTS)


def redact(value: Any, *, max_length: int = _MAX_VALUE_LENGTH) -> Any:
    """Recursively redact secret-looking fields and bound trace values."""

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _is_secret_key(key) else redact(item, max_length=max_length)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, max_length=max_length) for item in value]
    if isinstance(value, str):
        return value[:max_length]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value)[:max_length]


class TraceSpan:
    """A best-effort span which never changes serving behavior on tracing failure."""

    def __init__(self, config: TracingConfig, attributes: Mapping[str, Any]):
        self.config = config
        self.attributes = dict(attributes)
        self.span: Any = None
        self.active = False
        self._started = False

    def __enter__(self) -> "TraceSpan":
        if not self.config.enabled or self.config.sampling_ratio <= 0:
            return self
        if self.config.sampling_ratio < 1 and random.random() > self.config.sampling_ratio:
            return self
        try:
            import mlflow

            for env_name, value in {
                "MLFLOW_TRACING_DESTINATION": self.config.destination,
                "MLFLOW_TRACING_SQL_WAREHOUSE_ID": self.config.sql_warehouse_id,
            }.items():
                if value:
                    try:
                        os.environ[env_name] = value
                    except Exception:
                        pass

            if self.config.experiment_name:
                set_experiment = getattr(mlflow, "set_experiment", None)
                if callable(set_experiment):
                    try:
                        set_experiment(experiment_name=self.config.experiment_name)
                    except Exception:
                        pass

            start_span = getattr(mlflow, "start_span", None)
            if not callable(start_span):
                return self
            span_type: Any = "AGENT"
            try:
                from mlflow.entities import SpanType

                span_type = getattr(SpanType, "AGENT", span_type)
            except Exception:
                pass
            self.span = start_span(name="haystack-agent-router", span_type=span_type)
            self.active = True
            self._started = True
            safe_attributes = {
                key: _attribute_value(value) for key, value in self.attributes.items()
            }
            safe_attributes = {
                key: value for key, value in safe_attributes.items() if value is not None
            }
            if self.config.experiment_name:
                safe_attributes["mlflow.experiment_name"] = self.config.experiment_name
            if self.config.beta:
                safe_attributes["mlflow.trace.beta"] = True
            for key, value in {
                "mlflow.trace.uc_catalog": self.config.uc_catalog,
                "mlflow.trace.uc_schema": self.config.uc_schema,
                "mlflow.trace.uc_table": self.config.uc_table,
                "mlflow.trace.uc_volume": self.config.uc_volume,
                "mlflow.tracking_uri": self.config.tracking_uri,
                "mlflow.trace.destination": self.config.destination,
                "mlflow.trace.sql_warehouse_id": self.config.sql_warehouse_id,
            }.items():
                if value:
                    safe_attributes[key] = value
            self._set_attributes(safe_attributes)
        except Exception:
            end = getattr(self.span, "end", None)
            if callable(end):
                try:
                    end()
                except Exception:
                    pass
            self.span = None
            self.active = False
        return self

    def _set_attributes(self, attributes: Mapping[str, Any]) -> None:
        if self.span is None:
            return
        setter = getattr(self.span, "set_attributes", None)
        if callable(setter):
            setter(dict(attributes))
            return
        setter = getattr(self.span, "set_attribute", None)
        if callable(setter):
            for key, value in attributes.items():
                setter(key, value)

    def finish(self, *, error: Mapping[str, Any] | None = None) -> None:
        if not self._started or self.span is None:
            return
        try:
            if error:
                error_attributes = {
                    f"error.{key}": _attribute_value(value)
                    for key, value in error.items()
                }
                self._set_attributes({"status": "error", **error_attributes})
            else:
                self._set_attributes({"status": "ok"})
        except Exception:
            pass
        finally:
            end = getattr(self.span, "end", None)
            if callable(end):
                try:
                    end()
                except Exception:
                    pass
            self.active = False
            self._started = False

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        if exc is None:
            self.finish()
        else:
            self.finish(error={"exception_type": exc_type.__name__ if exc_type else "Exception"})
        return False


def start_span(
    config: TracingConfig,
    *,
    correlation_id: str,
    request_id: str,
    model: str | None,
    route: str | None,
    provider: str | None,
    metadata: Mapping[str, Any] | None = None,
    payload: Any = None,
) -> TraceSpan:
    metadata = metadata or {}
    attributes: dict[str, Any] = {
        "correlation_id": correlation_id,
        "request_id": request_id,
        "model": model,
        "route": route,
        "provider": provider,
        "tool_count": metadata.get("tool_count", 0),
        "tool_names": _safe_identifiers(metadata.get("tool_names")),
        "mcp_binding": _safe_identifier(metadata["mcp_binding"]) if metadata.get("mcp_binding") else None,
        "skill_binding": _safe_identifier(metadata["skill_binding"]) if metadata.get("skill_binding") else None,
    }
    if config.capture_content and payload is not None:
        attributes["input"] = redact(payload)
    return TraceSpan(config, attributes)


def safe_error_attributes(
    *,
    correlation_id: str,
    route: str | None,
    host: str | None,
    path: str | None,
    exception_type: str,
    http_status: int | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "correlation_id": correlation_id,
            "route": route,
            "provider_host": host,
            "provider_path": path,
            "exception_type": exception_type,
            "http_status": http_status,
            "error_code": error_code,
        }.items()
        if value is not None
    }
