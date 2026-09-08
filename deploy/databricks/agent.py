"""MLflow 3.x python_function/ResponsesAgent entry point for the router."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

if __package__:
    from .gateway_client import GatewayClient, GatewayError, GatewaySettings, new_correlation_id, safe_error
    from .profile_loader import ProfileError, load_profile, validate_profile
    from .serving_runtime import LocalServingRuntime
    from .tracing import TracingConfig
else:
    def _code_path_package() -> str:
        """Create a package namespace for MLflow's top-level script loading mode."""

        package_name = "_haystack_databricks_code"
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            setattr(package, "__path__", [str(Path(__file__).resolve().parent)])
            package.__package__ = package_name
            sys.modules[package_name] = package
        return package_name


    _package_name = _code_path_package()
    _gateway_client = importlib.import_module(f"{_package_name}.gateway_client")
    _profile_loader = importlib.import_module(f"{_package_name}.profile_loader")
    _serving_runtime = importlib.import_module(f"{_package_name}.serving_runtime")
    _tracing = importlib.import_module(f"{_package_name}.tracing")

    GatewayClient = _gateway_client.GatewayClient
    GatewayError = _gateway_client.GatewayError
    GatewaySettings = _gateway_client.GatewaySettings
    new_correlation_id = _gateway_client.new_correlation_id
    safe_error = _gateway_client.safe_error
    ProfileError = _profile_loader.ProfileError
    load_profile = _profile_loader.load_profile
    validate_profile = _profile_loader.validate_profile
    LocalServingRuntime = _serving_runtime.LocalServingRuntime
    TracingConfig = _tracing.TracingConfig

try:
    from mlflow.models import set_model as _set_model
    from mlflow.pyfunc import ResponsesAgent
    from mlflow.types.responses import (
        ResponsesAgentRequest,
        ResponsesAgentResponse,
        ResponsesAgentStreamEvent,
        output_to_responses_items_stream,
    )
except ImportError:  # Keep local diagnostics and stub tests usable without MLflow.
    _set_model = None
    ResponsesAgentRequest = Any
    ResponsesAgentResponse = None
    ResponsesAgentStreamEvent = None
    output_to_responses_items_stream = None

    class ResponsesAgent:  # type: ignore[no-redef]
        """Offline-only base class used when MLflow is not installed."""


def _request_value(request: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping, Pydantic model, or lightweight stub object."""

    if isinstance(request, Mapping):
        return request.get(name, default)
    model_dump = getattr(request, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, Mapping) and name in dumped:
            return dumped[name]
    return getattr(request, name, default)


def _request_mapping(request: Any) -> dict[str, Any]:
    """Serialize supported Responses request objects without copying secrets to errors."""

    values = {
        name: _request_value(request, name)
        for name in ("input", "stream", "model", "correlation_id")
    }
    return {key: value for key, value in values.items() if value is not None}


def normalize_request(request: Any) -> dict[str, Any]:
    """Normalize an agent/v1/responses request without retaining prompt text in errors."""

    items = _request_value(request, "input", [])
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list):
        raise ValueError("input must be a list or string")
    return {
        "input": items,
        "stream": bool(_request_value(request, "stream", False)),
        "model": _request_value(request, "model"),
        "correlation_id": _request_value(request, "correlation_id"),
        "request_id": _request_value(request, "request_id"),
    }


def _offline_text_item(text: str, item_id: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


class Agent(ResponsesAgent):
    """ResponsesAgent wrapper with explicit runtime diagnostics."""

    def __init__(self, profile: Mapping[str, Any] | None = None):
        self.profile: dict[str, Any] | None = None
        self.runtime: LocalServingRuntime | None = None
        if profile is not None:
            self._configure(profile)

    def _configure(self, profile: Mapping[str, Any]) -> None:
        loaded_profile = dict(profile)
        validate_profile(loaded_profile)
        settings = GatewaySettings.from_env(loaded_profile)
        self.runtime = LocalServingRuntime(
            GatewayClient(settings),
            tracing=TracingConfig.from_env(loaded_profile),
        )
        self.profile = loaded_profile

    def load_context(self, context: Any) -> None:
        """Load and validate the injected profile artifact/model configuration."""

        artifacts = getattr(context, "artifacts", {}) or {}
        config = getattr(context, "model_config", None)
        profile_path = artifacts.get("profile") or artifacts.get("profile.json")
        if profile_path:
            profile = load_profile(profile_path)
        elif isinstance(config, Mapping):
            profile = dict(config)
            validate_profile(profile)
        else:
            raise ProfileError("profile_missing", "profile artifact or model config is required")
        self._configure(profile)

    def predict(self, request: ResponsesAgentRequest) -> Any:
        correlation_id = str(_request_value(request, "correlation_id") or new_correlation_id())
        try:
            normalized = normalize_request(request)
            if self.profile is None or self.runtime is None:
                raise ProfileError("profile_missing", "load_context must load a profile before predict")
            if normalized["stream"]:
                return self._error_response(correlation_id, "use_predict_stream", "StreamingRequestError")
            result = self.runtime._execute({
                "payload": {"input": normalized["input"], "stream": False},
                "stream": False,
                "model": normalized.get("model"),
                "request_id": normalized.get("request_id"),
                "route": self.profile["default_route"],
                "route_metadata": self.profile["route_metadata"],
                "correlation_id": correlation_id,
            })
            if isinstance(result, Mapping) and result.get("status") == "failed":
                return self._error_response(
                    correlation_id,
                    result.get("error", {}).get("error_code", "request_failed"),
                    result.get("error", {}).get("exception_type", "GatewayError"),
                    result.get("error", {}),
                )
            return self._responses_response(result)
        except Exception as exc:
            return self._error_response(correlation_id, getattr(exc, "code", "request_error"), type(exc).__name__)

    def predict_stream(self, request: ResponsesAgentRequest) -> Iterator[Any]:
        correlation_id = str(_request_value(request, "correlation_id") or new_correlation_id())
        try:
            normalized = normalize_request(request)
            if self.profile is None or self.runtime is None:
                raise ProfileError("profile_missing", "load_context must load a profile before predict_stream")
            result = self.runtime._execute({
                "payload": {"input": normalized["input"], "stream": True},
                "stream": True,
                "model": normalized.get("model"),
                "request_id": normalized.get("request_id"),
                "route": self.profile["default_route"],
                "route_metadata": self.profile["route_metadata"],
                "correlation_id": correlation_id,
            })
            if isinstance(result, Mapping):
                yield from self._stream_error_event(result.get("error", {}), correlation_id)
                return
            yield from self._responses_stream(result, correlation_id)
        except Exception as exc:
            yield from self._stream_error_event(
                safe_error(
                    exc,
                    correlation_id=correlation_id,
                    route=self.profile.get("default_route") if self.profile else None,
                    base_url=self.runtime.gateway.settings.base_url if self.runtime else None,
                    path="/mlflow/v1/responses",
                    error_code=getattr(exc, "code", "stream_runtime_error"),
                ),
                correlation_id,
            )

    def _responses_response(self, result: Any) -> Any:
        if isinstance(result, Mapping) and "output" in result:
            if ResponsesAgentResponse is not None:
                return ResponsesAgentResponse(**dict(result))
            return dict(result)
        text = ""
        if isinstance(result, Mapping):
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "") if isinstance(message, Mapping) else ""
        item_id = f"msg_{uuid4().hex}"
        if ResponsesAgentResponse is not None:
            item = self.create_text_output_item(text=text, id=item_id)
            return ResponsesAgentResponse(output=[item])
        return {"output": [_offline_text_item(text, item_id)]}

    def _error_response(
        self,
        correlation_id: str,
        error_code: str,
        exception_type: str,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> Any:
        safe_diagnostic = dict(diagnostic or {})
        safe_diagnostic.setdefault("correlation_id", correlation_id)
        safe_diagnostic.setdefault("route", self.profile.get("default_route") if self.profile else None)
        safe_diagnostic.setdefault("provider_host", self.runtime.gateway.settings.base_url if self.runtime else None)
        safe_diagnostic.setdefault("provider_path", "/mlflow/v1/responses")
        safe_diagnostic.setdefault("exception_type", exception_type)
        safe_diagnostic.setdefault("error_code", error_code)
        safe_diagnostic = {key: value for key, value in safe_diagnostic.items() if value is not None}
        if ResponsesAgentResponse is not None:
            return ResponsesAgentResponse(output=[], custom_outputs={"error": safe_diagnostic})
        return {"status": "failed", "error": safe_diagnostic}

    def _responses_stream(self, result: Iterator[dict[str, Any]], correlation_id: str) -> Iterator[Any]:
        protocol = self.runtime.gateway.settings.protocol if self.runtime else "responses"
        if protocol == "chat" and output_to_responses_items_stream is not None:
            yield from output_to_responses_items_stream(result)
            return
        for chunk in result:
            if not isinstance(chunk, Mapping) or not isinstance(chunk.get("type"), str):
                yield from self._stream_error_event(
                    {"error_code": "invalid_stream_event", "correlation_id": correlation_id},
                    correlation_id,
                )
                return
            if ResponsesAgentStreamEvent is not None:
                yield ResponsesAgentStreamEvent(**dict(chunk))
            else:
                yield dict(chunk)

    def _stream_error_event(self, diagnostic: Mapping[str, Any], correlation_id: str) -> Iterator[Any]:
        safe_diagnostic = dict(diagnostic)
        safe_diagnostic.setdefault("correlation_id", correlation_id)
        if ResponsesAgentStreamEvent is not None:
            try:
                yield ResponsesAgentStreamEvent(
                    type="response.failed",
                    response={"status": "failed", "custom_outputs": {"error": safe_diagnostic}},
                )
                return
            except (TypeError, ValueError):
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text="The request failed; inspect the response diagnostics.",
                        id=f"error_{uuid4().hex}",
                    ),
                )
            return
        yield {
            "type": "response.output_item.done",
            "item": _offline_text_item(
                "The request failed; inspect custom_outputs for diagnostics.",
                f"error_{uuid4().hex}",
            ),
            "custom_outputs": {"error": safe_diagnostic},
        }


def load_agent_context(profile_path: str) -> Agent:
    return Agent(load_profile(profile_path))


def _load_pyfunc(model_config: Mapping[str, Any] | None = None) -> Agent:
    """Compatibility loader for python_function model packaging."""

    if not isinstance(model_config, Mapping):
        raise ProfileError("profile_missing", "profile artifact or model config is required")
    return Agent(model_config)


MODEL = Agent()


def _register_model() -> None:
    if _set_model is not None:
        _set_model(MODEL)


_register_model()
