"""Optional, failure-isolated MLflow GenAI tracing for the serving runtime."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

_SECRET_PARTS = ("authorization", "access_token", "api_key", "client_secret", "password", "secret", "token")
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
        """Build tracing configuration from a profile and environment overrides."""

        profile = profile or {}
        configured: dict[str, Any] = {}
        observability = profile.get("observability", {})
        if isinstance(observability, Mapping):
            configured.update(
                {
                    "enabled": observability.get("trace_enabled", False),
                    "sampling_ratio": observability.get("sampling_ratio", 0.0),
                    "capture_content": observability.get("capture_content", False),
                }
            )
        tracing = profile.get("tracing", {})
        if isinstance(tracing, Mapping):
            configured.update(tracing)

        def value(key: str, env_name: str, default: Any = None) -> Any:
            environment_value = os.getenv(env_name)
            if environment_value not in (None, ""):
                return environment_value
            return configured.get(key, default)

        enabled_environment = os.getenv("MLFLOW_TRACING_ENABLED") or os.getenv("ENABLE_MLFLOW_TRACING")
        enabled = _as_bool(
            enabled_environment if enabled_environment not in (None, "") else configured.get("enabled", False)
        )
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

    def __init__(self, config: TracingConfig, attributes: Mapping[str, Any]) -> None:
        self.config = config
        self.attributes = dict(attributes)
        self.span: Any = None
        self.active = False
        self._started = False
        self._started_at = 0.0
        self._first_token_at: float | None = None

    @property
    def trace_id(self) -> str | None:
        """Return the logical trace identifier attached to this span."""

        value = self.attributes.get("trace_id")
        return str(value) if value is not None else None

    @property
    def span_id(self) -> str | None:
        """Return the logical span identifier attached to this span."""

        value = self.attributes.get("span_id")
        return str(value) if value is not None else None

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
                    with suppress(Exception):
                        os.environ[env_name] = value

            if self.config.experiment_name:
                set_experiment = getattr(mlflow, "set_experiment", None)
                if callable(set_experiment):
                    with suppress(Exception):
                        set_experiment(experiment_name=self.config.experiment_name)

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
            self._started_at = time.monotonic()
            safe_attributes = {key: _attribute_value(value) for key, value in self.attributes.items()}
            safe_attributes = {key: value for key, value in safe_attributes.items() if value is not None}
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
                with suppress(Exception):
                    end()
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

    def _trace_metadata(self, attributes: Mapping[str, Any] | None = None) -> dict[str, str]:
        values = {**self.attributes, **dict(attributes or {})}
        metadata: dict[str, str] = {}
        metadata_values = {
            "mlflow.trace.user": values.get("user_id"),
            "mlflow.trace.session": values.get("session_id"),
            "haystack.trace_id": values.get("trace_id"),
            "haystack.span_id": values.get("span_id"),
            "haystack.correlation_id": values.get("correlation_id"),
            "haystack.request_id": values.get("request_id"),
            "haystack.turn_id": values.get("turn_id"),
            "haystack.endpoint_name": values.get("endpoint_name"),
            "haystack.served_entity_name": values.get("served_entity_name"),
            "haystack.model": values.get("model"),
            "haystack.models": values.get("models"),
            "haystack.model_target": values.get("model_target"),
            "haystack.model_targets": values.get("model_targets"),
            "haystack.model_version": values.get("model_version"),
            "haystack.protocol": values.get("protocol"),
            "haystack.protocols": values.get("protocols"),
            "haystack.provider": values.get("provider"),
            "haystack.providers": values.get("providers"),
            "haystack.provider_kind": values.get("provider_kind"),
            "haystack.provider_kinds": values.get("provider_kinds"),
            "haystack.route": values.get("route"),
            "haystack.routes": values.get("routes"),
            "haystack.route_attempts": values.get("route_attempts"),
            "haystack.scenario": values.get("scenario"),
            "haystack.tenant_id": values.get("tenant_id"),
            "haystack.policy_route": values.get("policy_route"),
            "haystack.mcp_binding": values.get("mcp_binding"),
            "haystack.mcp_bindings": values.get("mcp_bindings"),
            "haystack.skill_binding": values.get("skill_binding"),
            "haystack.skill_bindings": values.get("skill_bindings"),
            "haystack.prompt_binding_ids": values.get("prompt_binding_ids"),
            "haystack.tool_count": values.get("tool_count"),
            "haystack.tool_names": values.get("tool_names"),
            "haystack.node_count": values.get("node_count"),
            "haystack.nodes": values.get("nodes"),
            "haystack.status": values.get("status"),
            "haystack.streamed": values.get("streamed"),
        }
        for singular_key, plural_key in (
            ("haystack.model", "models"),
            ("haystack.model_target", "model_targets"),
            ("haystack.protocol", "protocols"),
            ("haystack.provider", "providers"),
            ("haystack.provider_kind", "provider_kinds"),
            ("haystack.route", "routes"),
            ("haystack.mcp_binding", "mcp_bindings"),
            ("haystack.skill_binding", "skill_bindings"),
        ):
            if values.get(plural_key):
                metadata_values.pop(singular_key, None)
        for key, value in metadata_values.items():
            if value is not None:
                metadata[key] = str(_attribute_value(value))
        return metadata

    def finish(self, *, error: Mapping[str, Any] | None = None, attributes: Mapping[str, Any] | None = None) -> None:
        """Finish the span and attach safe outcome and latency attributes."""

        if not self._started or self.span is None:
            return
        try:
            if error:
                error_attributes = {f"error.{key}": _attribute_value(value) for key, value in error.items()}
                self._set_attributes(
                    {
                        "status": "error",
                        "error_code": _attribute_value(error.get("error_code")),
                        "http_status": _attribute_value(error.get("http_status")),
                        **error_attributes,
                    }
                )
            else:
                self._set_attributes({"status": "ok"})
            elapsed_ms = (time.monotonic() - self._started_at) * 1000 if self._started_at else None
            first_token_ms = (
                (self._first_token_at - self._started_at) * 1000
                if self._first_token_at is not None and self._started_at
                else None
            )
            final_attributes = {
                "status": "error" if error else "ok",
                "latency_ms": elapsed_ms,
                "time_to_first_token_ms": first_token_ms,
                **dict(attributes or {}),
            }
            self._set_attributes(
                {key: _attribute_value(value) for key, value in final_attributes.items() if value is not None}
            )
            update_current_trace = None
            with suppress(Exception):
                import mlflow

                update_current_trace = getattr(mlflow, "update_current_trace", None)
            if callable(update_current_trace) and not self.attributes.get("parent_span_id"):
                trace_metadata = self._trace_metadata(final_attributes)
                if trace_metadata:
                    with suppress(Exception):
                        update_current_trace(metadata=trace_metadata)
        except Exception:
            pass
        finally:
            end = getattr(self.span, "end", None)
            if callable(end):
                with suppress(Exception):
                    end()
            self.active = False
            self._started = False

    def mark_first_token(self) -> None:
        """Record the first emitted stream item once."""

        if self._started and self._first_token_at is None:
            self._first_token_at = time.monotonic()

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
    **context: Any,
) -> TraceSpan:
    """Create a sampled, redacting trace span for one serving request."""

    metadata = context.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    attributes: dict[str, Any] = {
        "trace_id": str(uuid4()),
        "span_id": str(uuid4()),
        "parent_span_id": context.get("parent_span_id") or metadata.get("parent_span_id"),
        "correlation_id": correlation_id,
        "request_id": request_id,
        "session_id": context.get("session_id"),
        "turn_id": context.get("turn_id"),
        "user_id": context.get("user_id"),
        "tenant_id": context.get("tenant_id"),
        "scenario": context.get("scenario"),
        "agent_profile": context.get("agent_profile"),
        "component": "haystack-agent-router",
        "protocol": context.get("protocol"),
        "endpoint_name": context.get("endpoint_name") or os.getenv("HAYSTACK_SERVING_ENDPOINT_NAME"),
        "served_entity_name": context.get("served_entity_name") or os.getenv("HAYSTACK_SERVED_ENTITY_NAME"),
        "model": model,
        "model_alias": metadata.get("model_alias"),
        "model_version": metadata.get("model_version") or os.getenv("HAYSTACK_SERVED_MODEL_VERSION"),
        "route": route,
        "route_attempts": _safe_identifiers(metadata.get("route_attempts")),
        "fallback_reason": metadata.get("fallback_reason"),
        "provider": provider,
        "streamed": bool(context.get("streamed", False)),
        "tool_count": metadata.get("tool_count", 0),
        "tool_names": _safe_identifiers(metadata.get("tool_names")),
        "mcp_binding": _safe_identifier(metadata["mcp_binding"]) if metadata.get("mcp_binding") else None,
        "skill_binding": _safe_identifier(metadata["skill_binding"]) if metadata.get("skill_binding") else None,
        "prompt_binding_ids": _safe_identifiers(metadata.get("prompt_binding_ids")),
        "prompt_content_sha256": _safe_identifiers(metadata.get("prompt_content_sha256")),
        "mcp_server": _safe_identifier(metadata["mcp_server"]) if metadata.get("mcp_server") else None,
        "skill_id": _safe_identifier(metadata["skill_id"]) if metadata.get("skill_id") else None,
        "tool_call_id": _safe_identifier(metadata["tool_call_id"]) if metadata.get("tool_call_id") else None,
        "tool_name": _safe_identifier(metadata["tool_name"]) if metadata.get("tool_name") else None,
    }
    if config.capture_content and context.get("payload") is not None:
        attributes["input"] = redact(context["payload"])
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
    """Build bounded error attributes safe to expose to serving callers."""

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
