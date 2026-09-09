"""MLflow 3.x python_function/ResponsesAgent entry point for the router."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import queue
import sys
import threading
from collections.abc import Iterator, Mapping
from contextvars import copy_context
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

if __package__:
    from integrations.databricks.session_store import SessionConflictError, session_store_from_env

    from .gateway_client import GatewayClient, GatewayError, GatewaySettings, new_correlation_id, safe_error
    from .profile_loader import ProfileError, load_profile, validate_profile
    from .serving_runtime import LocalServingRuntime
    from .tracing import TracingConfig, configure_haystack_mlflow_tracing, start_span
else:

    def _code_path_directories() -> list[str]:
        """Find the directory containing sibling modules in an MLflow model image."""

        model_root = Path(__file__).resolve().parent
        directories: list[str] = []
        for filename in (
            "gateway_client.py",
            "profile_loader.py",
            "serving_runtime.py",
            "tracing.py",
            "session_store.py",
        ):
            for candidate in model_root.rglob(filename):
                directory = candidate.parent
                if str(directory) not in directories:
                    directories.append(str(directory))
        return directories

    def _code_path_package() -> str:
        """Create a package namespace for MLflow's top-level script loading mode."""

        package_name = "_haystack_databricks_code"
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            directories = _code_path_directories()
            if not directories:
                raise ModuleNotFoundError("gateway_client.py is not present in the MLflow code paths")
            package.__path__ = directories
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
    configure_haystack_mlflow_tracing = _tracing.configure_haystack_mlflow_tracing
    start_span = _tracing.start_span
    try:
        _session_store = importlib.import_module(f"{_package_name}.session_store")
    except ModuleNotFoundError as error:
        if error.name != f"{_package_name}.session_store":
            raise
        search_roots = (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
        session_store_path = next(
            (
                candidate
                for root in search_roots
                if root.is_dir()
                for candidate in root.rglob("session_store.py")
                if candidate.is_file()
            ),
            None,
        )
        if session_store_path is None:
            raise
        session_store_spec = importlib_util.spec_from_file_location(
            f"{_package_name}.session_store", session_store_path
        )
        if session_store_spec is None or session_store_spec.loader is None:
            raise ModuleNotFoundError("session_store.py is not present in the MLflow code paths")
        _session_store = importlib_util.module_from_spec(session_store_spec)
        sys.modules[session_store_spec.name] = _session_store
        session_store_spec.loader.exec_module(_session_store)
    SessionConflictError = _session_store.SessionConflictError
    session_store_from_env = _session_store.session_store_from_env

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


try:
    from contracts.runtime.haystack_agent_serving.loader import AgentServingSnapshotLoader, SnapshotValidationError
    from haystack.dataclasses import ChatMessage
    from integrations.databricks.agent_orchestrator import ConfiguredAgentOrchestrator
    from integrations.databricks.ai_gateway import GatewayConfig
    from integrations.databricks.mcp_toolset import HttpMcpToolset
except ImportError:  # Optional for the legacy single-route wrapper.
    ChatMessage = None
    ConfiguredAgentOrchestrator = None
    GatewayConfig = None
    HttpMcpToolset = None
    AgentServingSnapshotLoader = None
    SnapshotValidationError = ValueError


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

    values = {name: _request_value(request, name) for name in ("input", "stream", "model", "correlation_id")}
    return {key: value for key, value in values.items() if value is not None}


def normalize_request(request: Any) -> dict[str, Any]:
    """Normalize an agent/v1/responses request without retaining prompt text in errors."""

    items = _request_value(request, "input", [])
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list):
        raise TypeError("input must be a list or string")
    custom_inputs = _request_value(request, "custom_inputs", {})
    if not isinstance(custom_inputs, Mapping):
        custom_inputs = {}
    requested_model = _request_value(request, "model") or custom_inputs.get("model")
    metadata = _request_value(request, "metadata", {})
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    for key in ("session_id", "turn_id", "tenant_id", "user_id", "scenario", "route"):
        if key in custom_inputs and key not in metadata:
            metadata[key] = custom_inputs[key]
    return {
        "input": items,
        "stream": bool(_request_value(request, "stream", False)),
        "model": requested_model,
        "metadata": metadata,
        "custom_inputs": dict(custom_inputs),
        "session_id": custom_inputs.get("session_id") or metadata.get("session_id"),
        "turn_id": custom_inputs.get("turn_id") or metadata.get("turn_id") or str(uuid4()),
        "tenant_id": custom_inputs.get("tenant_id") or metadata.get("tenant_id") or "default",
        "user_id": custom_inputs.get("user_id") or metadata.get("user_id"),
        "scenario": custom_inputs.get("scenario") or metadata.get("scenario"),
        "correlation_id": _request_value(request, "correlation_id"),
        "request_id": _request_value(request, "request_id"),
    }


def _offline_text_item(text: str, item_id: str) -> dict[str, Any]:
    return {"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _response_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            _response_content_text(part.get("text", part.get("content", "")))
            for part in value
            if isinstance(part, Mapping)
        )
    return _json_text(value)


def _canonical_message(item: Any) -> dict[str, Any]:
    """Convert Chat Completions and Responses input items to one session format."""

    if isinstance(item, Mapping):
        item_type = item.get("type")
        if item_type == "function_call_output":
            return {
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id") or "tool-call",
                "content": _response_content_text(item.get("output", "")),
            }
        if item_type == "function_call":
            arguments = item.get("arguments", {})
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": item.get("call_id") or item.get("id") or "tool-call",
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "tool",
                            "arguments": arguments if isinstance(arguments, str) else _json_text(arguments),
                        },
                    }
                ],
            }
        role = item.get("role")
        if role is not None:
            canonical = dict(item)
            if isinstance(canonical.get("content"), list):
                canonical["content"] = _response_content_text(canonical["content"])
            if canonical.get("role") == "developer":
                canonical["role"] = "system"
            return canonical
    raise TypeError("session messages must be mapping values with a role or Responses item type")


def _canonical_messages(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [_canonical_message(item) for item in items]


def _add_custom_outputs(response: Any, custom_outputs: Mapping[str, Any]) -> Any:
    """Attach diagnostics without assuming the MLflow response is mutable."""

    model_copy = getattr(response, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"custom_outputs": dict(custom_outputs)})
    copy_method = getattr(response, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(update={"custom_outputs": dict(custom_outputs)})
        except (TypeError, ValueError):
            pass
    try:
        response.custom_outputs = dict(custom_outputs)
    except (AttributeError, TypeError):
        payload = response.model_dump() if callable(getattr(response, "model_dump", None)) else dict(response)
        payload["custom_outputs"] = dict(custom_outputs)
        return ResponsesAgentResponse(**payload) if ResponsesAgentResponse is not None else payload
    return response


class Agent(ResponsesAgent):
    """ResponsesAgent wrapper with explicit runtime diagnostics."""

    def __init__(self, profile: Mapping[str, Any] | None = None) -> None:
        self.profile: dict[str, Any] | None = None
        self.runtime: LocalServingRuntime | None = None
        self.orchestrator: Any = None
        self.session_store = session_store_from_env()
        self.tracing = TracingConfig.from_env()
        if profile is not None:
            self._configure(profile)

    def _configure(self, profile: Mapping[str, Any]) -> None:
        loaded_profile = dict(profile)
        if "schema_version" in loaded_profile and "agent_graph" in loaded_profile:
            self.tracing = TracingConfig.from_env(loaded_profile)
            configure_haystack_mlflow_tracing(self.tracing)
            self._configure_orchestrator(loaded_profile)
            self.profile = loaded_profile
            return
        validate_profile(loaded_profile)
        self.tracing = TracingConfig.from_env(loaded_profile)
        configure_haystack_mlflow_tracing(self.tracing)
        settings = GatewaySettings.from_env(loaded_profile)
        self.runtime = LocalServingRuntime(GatewayClient(settings), tracing=TracingConfig.from_env(loaded_profile))
        self.profile = loaded_profile

    def _configure_orchestrator(self, profile: Mapping[str, Any]) -> None:
        if AgentServingSnapshotLoader is None or ConfiguredAgentOrchestrator is None or GatewayConfig is None:
            raise ProfileError("haystack_unavailable", "Haystack orchestration dependencies are not installed")
        loader = AgentServingSnapshotLoader()
        result = loader.load_mapping(profile)
        if not result.accepted or result.snapshot is None:
            detail = "; ".join(issue.message for issue in result.issues) or "invalid orchestration profile"
            raise ProfileError("orchestration_profile_invalid", detail)
        snapshot = result.snapshot
        providers: list[dict[str, Any]] = []
        routes: list[dict[str, Any]] = []
        seen_protocols: set[str] = set()
        host = os.getenv("DATABRICKS_HOST") or profile.get("workspace_host")
        for binding in snapshot.model_bindings.values():
            target = binding.ref
            serving_endpoint = target.startswith(("serving-endpoint/", "serving-endpoints/"))
            provider_id = "databricks-serving" if serving_endpoint else "databricks-model-services"
            provider = next((item for item in providers if item["id"] == provider_id), None)
            if provider is None:
                provider = {
                    "id": provider_id,
                    "kind": "databricks_serving" if serving_endpoint else "model_service",
                    "protocols": [],
                }
                providers.append(provider)
            if binding.protocol not in provider["protocols"]:
                provider["protocols"].append(binding.protocol)
            routes.append(
                {
                    "id": binding.id,
                    "provider": provider_id,
                    "target": target,
                    "protocols": [binding.protocol],
                    "priority": len(routes),
                }
            )
            seen_protocols.add(binding.protocol)
        gateway_config = GatewayConfig.from_mapping(
            {"providers": providers, "routes": routes, "default_protocols": sorted(seen_protocols)}
        )
        headers: Mapping[str, str] = {}
        if host and routes:
            auth_profile = {
                "default_route": routes[0]["target"],
                "route_metadata": {"protocol": "chat"},
                "gateway": {"workspace_host": host, "route": routes[0]["target"]},
            }
            try:
                headers = GatewayClient(GatewaySettings.from_env(auth_profile)).auth.headers()
            except Exception:
                headers = {}
        self.orchestrator = ConfiguredAgentOrchestrator(
            snapshot,
            gateway_config,
            host=host,
            headers=headers,
            prompt_sources=profile.get("prompt_sources", {}),
            skill_sources=profile.get("skill_sources", {}),
            toolsets=self._mcp_toolsets(snapshot, profile, headers),
        )

    @staticmethod
    def _mcp_toolsets(snapshot: Any, profile: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        if HttpMcpToolset is None:
            return {}
        endpoints = profile.get("mcp_endpoints", {})
        if not isinstance(endpoints, Mapping):
            return {}
        result: dict[str, Any] = {}
        for binding in snapshot.mcp_bindings.values():
            endpoint = endpoints.get(binding.id) or endpoints.get(binding.ref)
            if isinstance(endpoint, str) and endpoint:
                result[binding.id] = HttpMcpToolset(
                    endpoint, headers=headers, allowed_tools=binding.tools, use_databricks_auth=True
                )
        return result

    def _route_metadata(self, route: str) -> Mapping[str, Any]:
        route_metadata = self.profile.get("route_metadata", {}) if self.profile else {}
        by_route = self.profile.get("route_metadata_by_route", {}) if self.profile else {}
        if isinstance(by_route, Mapping) and isinstance(by_route.get(route), Mapping):
            return by_route[route]
        if self.profile and route != self.profile.get("default_route"):
            return {}
        return route_metadata if isinstance(route_metadata, Mapping) else {}

    def _require_configured(self, operation: str) -> None:
        if (self.profile is None or self.runtime is None) and self.orchestrator is None:
            raise ProfileError("profile_missing", f"load_context must load a profile before {operation}")

    def load_context(self, context: Any) -> None:
        """Load and validate the injected profile artifact/model configuration."""

        artifacts = getattr(context, "artifacts", {}) or {}
        config = getattr(context, "model_config", None)
        profile_path = artifacts.get("profile") or artifacts.get("profile.json")
        if profile_path:
            try:
                raw_profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProfileError("profile_invalid", "profile could not be read") from exc
            if isinstance(raw_profile, Mapping) and "schema_version" in raw_profile and "agent_graph" in raw_profile:
                profile = dict(raw_profile)
            else:
                profile = load_profile(profile_path)
        elif isinstance(config, Mapping):
            profile = dict(config)
            validate_profile(profile)
        else:
            raise ProfileError("profile_missing", "profile artifact or model config is required")
        self._configure(profile)

    def predict(self, request: ResponsesAgentRequest) -> Any:
        """Execute one non-streaming Responses request."""

        correlation_id = str(_request_value(request, "correlation_id") or new_correlation_id())
        try:
            normalized = normalize_request(request)
            self._require_configured("predict")
            if self.orchestrator is not None:
                return self._predict_orchestrated(normalized, correlation_id=correlation_id)
            if normalized["stream"]:
                return self._error_response(correlation_id, "use_predict_stream", "StreamingRequestError")
            route = str(normalized.get("model") or self.profile["default_route"])
            payload_input, previous_revision, _ = self._restore_session(normalized)
            result = self.runtime._execute(
                {
                    "payload": {"input": payload_input, "stream": False},
                    "stream": False,
                    "model": normalized.get("model"),
                    "request_id": normalized.get("request_id"),
                    "route": route,
                    "route_metadata": self._route_metadata(route),
                    "correlation_id": correlation_id,
                    "session_id": normalized["session_id"],
                    "turn_id": normalized["turn_id"],
                    "tenant_id": normalized["tenant_id"],
                    "user_id": normalized["user_id"],
                    "scenario": normalized["scenario"] or "daily",
                    "agent_profile": self.profile.get("name") if self.profile else None,
                    "endpoint_name": os.getenv("HAYSTACK_SERVING_ENDPOINT_NAME"),
                    "served_entity_name": os.getenv("HAYSTACK_SERVED_ENTITY_NAME"),
                    "model_version": os.getenv("HAYSTACK_SERVED_MODEL_VERSION"),
                }
            )
            if isinstance(result, Mapping) and result.get("status") == "failed":
                return self._error_response(
                    correlation_id,
                    result.get("error", {}).get("error_code", "request_failed"),
                    result.get("error", {}).get("exception_type", "GatewayError"),
                    result.get("error", {}),
                )
            response = self._responses_response(result)
            self._save_session(normalized, payload_input, result, previous_revision)
            return response
        except Exception as exc:
            return self._error_response(
                correlation_id,
                getattr(exc, "code", "request_error"),
                type(exc).__name__,
                getattr(exc, "diagnostic", None),
            )

    def predict_stream(self, request: ResponsesAgentRequest) -> Iterator[Any]:
        """Execute one streaming Responses request."""

        correlation_id = str(_request_value(request, "correlation_id") or new_correlation_id())
        output_item_id = f"msg_{correlation_id}"
        streamed_any = False
        try:
            normalized = normalize_request(request)
            self._require_configured("predict_stream")
            if self.orchestrator is not None:
                events: queue.Queue[tuple[str, Any]] = queue.Queue()
                cancelled = threading.Event()
                stream_span = start_span(
                    self.tracing,
                    correlation_id=correlation_id,
                    request_id=str(normalized.get("request_id") or new_correlation_id()),
                    model=str(normalized.get("model")) if normalized.get("model") else None,
                    route=self.profile.get("default_route") if self.profile else None,
                    provider="haystack",
                    session_id=normalized.get("session_id"),
                    turn_id=normalized.get("turn_id"),
                    user_id=normalized.get("user_id"),
                    tenant_id=normalized.get("tenant_id"),
                    scenario=normalized.get("scenario"),
                    agent_profile=self.profile.get("name") if self.profile else None,
                    protocol="haystack_agent",
                    endpoint_name=os.getenv("HAYSTACK_SERVING_ENDPOINT_NAME"),
                    served_entity_name=os.getenv("HAYSTACK_SERVED_ENTITY_NAME"),
                    model_version=os.getenv("HAYSTACK_SERVED_MODEL_VERSION"),
                    streamed=True,
                    payload={"input": normalized.get("input"), "stream": True},
                )
                stream_span.__enter__()

                def capture_chunk(chunk: Any) -> None:
                    if not cancelled.is_set():
                        stream_span.mark_first_token()
                        events.put(("chunk", chunk))

                request_context = copy_context()

                def run_request() -> None:
                    try:
                        response = request_context.run(
                            self._predict_orchestrated,
                            normalized,
                            correlation_id=correlation_id,
                            streaming_callback=capture_chunk,
                            output_item_id=output_item_id,
                        )
                    except BaseException as error:
                        events.put(("error", error))
                    else:
                        events.put(("done", response))

                worker = threading.Thread(target=run_request, name="haystack-agent-stream", daemon=True)
                worker.start()
                try:
                    while True:
                        kind, payload = events.get()
                        if kind == "chunk":
                            text = getattr(payload, "content", None)
                            if not text:
                                continue
                            event = {
                                "type": "response.output_text.delta",
                                "delta": str(text),
                                "item_id": output_item_id,
                                "output_index": 0,
                                "content_index": 0,
                            }
                            if ResponsesAgentStreamEvent is not None:
                                try:
                                    streamed_any = True
                                    yield ResponsesAgentStreamEvent(**event)
                                    continue
                                except (TypeError, ValueError):
                                    pass
                            streamed_any = True
                            yield event
                            continue
                        if kind == "error":
                            stream_span.finish(error={"exception_type": type(payload).__name__})
                            raise payload
                        data = payload if isinstance(payload, Mapping) else payload.model_dump()
                        output = data.get("output", [])
                        stream_span.finish(attributes=self._stream_trace_attributes(data))
                        if output:
                            item = dict(output[0]) if isinstance(output[0], Mapping) else output[0]
                            if isinstance(item, dict):
                                item.setdefault("id", output_item_id)
                            event = {"type": "response.output_item.done", "item": item}
                            if ResponsesAgentStreamEvent is not None:
                                try:
                                    streamed_any = True
                                    yield ResponsesAgentStreamEvent(**event)
                                except (TypeError, ValueError):
                                    streamed_any = True
                                    yield event
                            else:
                                streamed_any = True
                                yield event
                            completed = {"type": "response.completed", "response": data}
                            if ResponsesAgentStreamEvent is not None:
                                try:
                                    streamed_any = True
                                    yield ResponsesAgentStreamEvent(**completed)
                                    return
                                except (TypeError, ValueError):
                                    streamed_any = True
                                    yield completed
                                    return
                            streamed_any = True
                            yield completed
                        return
                finally:
                    cancelled.set()
            route = str(normalized.get("model") or self.profile["default_route"])
            payload_input, previous_revision, _ = self._restore_session(normalized)
            result = self.runtime._execute(
                {
                    "payload": {"input": payload_input, "stream": True},
                    "stream": True,
                    "model": normalized.get("model"),
                    "request_id": normalized.get("request_id"),
                    "route": route,
                    "route_metadata": self._route_metadata(route),
                    "correlation_id": correlation_id,
                    "session_id": normalized["session_id"],
                    "turn_id": normalized["turn_id"],
                    "tenant_id": normalized["tenant_id"],
                    "user_id": normalized["user_id"],
                    "scenario": normalized["scenario"] or "daily",
                    "agent_profile": self.profile.get("name") if self.profile else None,
                }
            )
            if isinstance(result, Mapping):
                yield from self._stream_error_event(result.get("error", {}), correlation_id)
                return
            streamed_events: list[Any] = []
            try:
                for event in self._responses_stream(result, correlation_id):
                    streamed_events.append(event)
                    yield event
            finally:
                self._save_session(
                    normalized,
                    payload_input,
                    {"output": self._stream_session_messages(streamed_events)},
                    previous_revision,
                )
        except Exception as exc:
            if self.orchestrator is not None and not streamed_any:
                try:
                    fallback_response = self._predict_orchestrated(
                        normalized, correlation_id=correlation_id, output_item_id=output_item_id
                    )
                    fallback_data = (
                        fallback_response.model_dump()
                        if callable(getattr(fallback_response, "model_dump", None))
                        else fallback_response
                    )
                    fallback_output = fallback_data.get("output", []) if isinstance(fallback_data, Mapping) else []
                    if fallback_output:
                        stream_span.finish(
                            attributes={
                                **self._stream_trace_attributes(fallback_response),
                                "fallback_reason": type(exc).__name__,
                            }
                        )
                        fallback_item = (
                            dict(fallback_output[0]) if isinstance(fallback_output[0], Mapping) else fallback_output[0]
                        )
                        if isinstance(fallback_item, dict):
                            fallback_item.setdefault("id", output_item_id)
                        yield {"type": "response.output_item.done", "item": fallback_item}
                        yield {"type": "response.completed", "response": fallback_data}
                        return
                except Exception:
                    pass
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

    def _responses_response(self, result: Any, *, item_id: str | None = None) -> Any:
        if isinstance(result, Mapping) and "output" in result:
            if ResponsesAgentResponse is not None:
                return ResponsesAgentResponse(**dict(result))
            return dict(result)
        text = ""
        tool_calls: list[Mapping[str, Any]] = []
        if isinstance(result, Mapping):
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                if isinstance(message, Mapping):
                    text = _response_content_text(message.get("content", ""))
                    raw_tool_calls = message.get("tool_calls", [])
                    if isinstance(raw_tool_calls, list):
                        tool_calls = [call for call in raw_tool_calls if isinstance(call, Mapping)]
        item_id = item_id or f"msg_{uuid4().hex}"
        output_items: list[Any] = []
        if text:
            output_items.append(
                self.create_text_output_item(text=text, id=item_id)
                if ResponsesAgentResponse
                else _offline_text_item(text, item_id)
            )
        for call in tool_calls:
            function = call.get("function", {})
            function = function if isinstance(function, Mapping) else {}
            output_items.append(
                {
                    "type": "function_call",
                    "id": call.get("id") or f"call_{uuid4().hex}",
                    "call_id": call.get("id") or f"call_{uuid4().hex}",
                    "name": function.get("name") or "tool",
                    "arguments": function.get("arguments", "{}"),
                }
            )
        if ResponsesAgentResponse is not None:
            return ResponsesAgentResponse(output=output_items or [self.create_text_output_item(text="", id=item_id)])
        return {"output": output_items or [_offline_text_item(text, item_id)]}

    @staticmethod
    def _stream_session_messages(events: list[Any]) -> list[dict[str, Any]]:
        text_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for event in events:
            payload = event.model_dump() if callable(getattr(event, "model_dump", None)) else event
            if not isinstance(payload, Mapping):
                continue
            if payload.get("type") == "response.output_text.delta":
                text_parts.append(str(payload.get("delta", "")))
            item = payload.get("item")
            if isinstance(item, Mapping) and item.get("type") in {"message", "function_call", "function_call_output"}:
                try:
                    messages.append(_canonical_message(item))
                except TypeError:
                    continue
        if text_parts and not messages:
            messages.append({"role": "assistant", "content": "".join(text_parts)})
        return messages

    @staticmethod
    def _stream_trace_attributes(response: Any) -> dict[str, Any]:
        data = response.model_dump() if callable(getattr(response, "model_dump", None)) else response
        if not isinstance(data, Mapping):
            return {"status": "completed"}
        custom_outputs = data.get("custom_outputs")
        orchestration = custom_outputs.get("orchestration") if isinstance(custom_outputs, Mapping) else None
        if not isinstance(orchestration, Mapping):
            return {"status": "completed"}
        nodes = orchestration.get("nodes", [])
        node_items = [node for node in nodes if isinstance(node, Mapping)] if isinstance(nodes, (list, tuple)) else []
        return {
            "status": "completed",
            "policy_route": orchestration.get("policy_route"),
            "scenario": orchestration.get("scenario", "unknown"),
            "node_count": len(node_items),
            "routing_mode": (orchestration.get("routing") or {}).get("mode", "unknown")
            if isinstance(orchestration.get("routing"), Mapping)
            else "unknown",
            "router_model": (orchestration.get("routing") or {}).get("router_model", "none")
            if isinstance(orchestration.get("routing"), Mapping)
            else "none",
            "selected_scenario": (orchestration.get("routing") or {}).get(
                "scenario", orchestration.get("scenario", "unknown")
            )
            if isinstance(orchestration.get("routing"), Mapping)
            else orchestration.get("scenario", "unknown"),
            "routing_confidence": (orchestration.get("routing") or {}).get("confidence", 0.0)
            if isinstance(orchestration.get("routing"), Mapping)
            else 0.0,
            "routing_reason": (orchestration.get("routing") or {}).get("reason", "not supplied")
            if isinstance(orchestration.get("routing"), Mapping)
            else "not supplied",
            "models": [node.get("model") for node in node_items if node.get("model")],
            "model_targets": [node.get("model_target") for node in node_items if node.get("model_target")],
            "routes": [node.get("route") for node in node_items if node.get("route")],
            "protocols": [node.get("protocol") for node in node_items if node.get("protocol")],
            "providers": [node.get("provider") for node in node_items if node.get("provider")],
            "provider_kinds": [node.get("provider_kind") for node in node_items if node.get("provider_kind")],
            "route_attempts": [attempt for node in node_items for attempt in node.get("route_attempts", []) if attempt],
            "mcp_bindings": [node.get("mcp_binding") for node in node_items if node.get("mcp_binding")],
            "skill_bindings": [binding for node in node_items for binding in node.get("skill_bindings", []) if binding],
            "prompt_binding_ids": [
                binding for node in node_items for binding in node.get("prompt_binding_ids", []) if binding
            ],
            "tool_count": sum(int(node.get("tool_count", 0) or 0) for node in node_items),
            "tool_names": [tool for node in node_items for tool in node.get("tool_names", []) if tool],
            "nodes": nodes,
        }

    def _predict_orchestrated(
        self,
        normalized: Mapping[str, Any],
        *,
        correlation_id: str | None = None,
        streaming_callback: Any = None,
        output_item_id: str | None = None,
    ) -> Any:
        if ChatMessage is None or self.orchestrator is None:
            raise ProfileError("haystack_unavailable", "Haystack orchestration dependencies are not installed")
        orchestration_request: dict[str, Any] = {
            "scenario": normalized.get("scenario"),
            "tenant_id": normalized.get("tenant_id"),
            "user_id": normalized.get("user_id"),
            "session_id": normalized.get("session_id"),
        }
        span = start_span(
            self.tracing,
            correlation_id=str(correlation_id or normalized.get("correlation_id") or new_correlation_id()),
            request_id=str(normalized.get("request_id") or new_correlation_id()),
            model=str(normalized.get("model")) if normalized.get("model") else None,
            route=self.profile.get("default_route") if self.profile else None,
            provider="haystack",
            session_id=str(normalized.get("session_id")) if normalized.get("session_id") else None,
            turn_id=str(normalized.get("turn_id")) if normalized.get("turn_id") else None,
            user_id=str(normalized.get("user_id")) if normalized.get("user_id") else None,
            tenant_id=str(normalized.get("tenant_id")) if normalized.get("tenant_id") else None,
            scenario=str(normalized.get("scenario")) if normalized.get("scenario") else None,
            agent_profile=self.profile.get("name") if self.profile else None,
            protocol="haystack_agent",
            streamed=streaming_callback is not None,
            payload={"input": normalized.get("input"), "stream": streaming_callback is not None},
            metadata={"agent_profile": self.profile.get("name") if self.profile else None},
        )
        span.__enter__()

        def trace_child(metadata: Mapping[str, Any]) -> Any:
            child_metadata = dict(metadata)
            return start_span(
                self.tracing,
                correlation_id=str(correlation_id or normalized.get("correlation_id") or new_correlation_id()),
                request_id=str(normalized.get("request_id") or new_correlation_id()),
                model=str(child_metadata.get("model")) if child_metadata.get("model") else None,
                route=str(child_metadata.get("route")) if child_metadata.get("route") else None,
                provider=str(child_metadata.get("provider") or "haystack"),
                parent_span_id=(
                    span.span_id if child_metadata.get("node_id") and not child_metadata.get("span_name") else None
                ),
                session_id=normalized.get("session_id"),
                turn_id=normalized.get("turn_id"),
                tenant_id=normalized.get("tenant_id"),
                user_id=normalized.get("user_id"),
                scenario=orchestration_request.get("scenario"),
                agent_profile=self.profile.get("name") if self.profile else None,
                protocol=child_metadata.get("protocol"),
                model_target=child_metadata.get("model_target"),
                provider_kind=child_metadata.get("provider_kind"),
                endpoint_name=os.getenv("HAYSTACK_SERVING_ENDPOINT_NAME"),
                served_entity_name=os.getenv("HAYSTACK_SERVED_ENTITY_NAME"),
                model_version=os.getenv("HAYSTACK_SERVED_MODEL_VERSION"),
                streamed=streaming_callback is not None,
                routing_mode=(orchestration_request.get("routing") or {}).get("mode", "unknown"),
                router_model=(orchestration_request.get("routing") or {}).get("router_model", "none"),
                selected_scenario=(orchestration_request.get("routing") or {}).get(
                    "scenario", orchestration_request.get("scenario", "unknown")
                ),
                routing_confidence=(orchestration_request.get("routing") or {}).get("confidence", 0.0),
                routing_reason=(orchestration_request.get("routing") or {}).get("reason", "not supplied"),
                span_name=child_metadata.get("span_name"),
                span_type=child_metadata.get("span_type"),
                payload=child_metadata.get("payload"),
                metadata=child_metadata,
            )

        try:
            input_messages, revision, previous_state = self._restore_session(normalized, trace_span_factory=trace_child)
            messages = [ChatMessage.from_openai_dict_format(_canonical_message(item)) for item in input_messages]
            run_kwargs: dict[str, Any] = {"request": orchestration_request, "streaming_callback": streaming_callback}
            if "state" in inspect.signature(self.orchestrator.run).parameters:
                run_kwargs["state"] = previous_state
            if "trace_span_factory" in inspect.signature(self.orchestrator.run).parameters:
                run_kwargs["trace_span_factory"] = trace_child
            result = self.orchestrator.run(messages, **run_kwargs)
            reply = result.replies[-1]
            response = self._responses_response(
                {"choices": [{"message": reply.to_openai_dict_format()}]}, item_id=output_item_id
            )
            self._save_session(
                normalized,
                input_messages,
                {"output": [reply.to_openai_dict_format()]},
                revision,
                agent_state=getattr(result, "state", {}),
                trace_span_factory=trace_child,
            )
            nodes = result.meta.get("nodes", [])
            node_items = (
                [node for node in nodes if isinstance(node, Mapping)] if isinstance(nodes, (list, tuple)) else []
            )
            models = [node["model"] for node in node_items if node.get("model")]
            model_targets = [node["model_target"] for node in node_items if node.get("model_target")]
            routes = [node["route"] for node in node_items if node.get("route")]
            protocols = [node["protocol"] for node in node_items if node.get("protocol")]
            providers = [node["provider"] for node in node_items if node.get("provider")]
            provider_kinds = [node["provider_kind"] for node in node_items if node.get("provider_kind")]
            route_attempts = [attempt for node in node_items for attempt in node.get("route_attempts", [])]
            mcp_bindings = [node["mcp_binding"] for node in node_items if node.get("mcp_binding")]
            skill_bindings = [binding for node in node_items for binding in node.get("skill_bindings", []) if binding]
            prompt_binding_ids = [
                binding for node in node_items for binding in node.get("prompt_binding_ids", []) if binding
            ]
            tool_names = [tool for node in node_items for tool in node.get("tool_names", []) if tool]
            routing = result.meta.get("routing", {})
            routing = routing if isinstance(routing, Mapping) else {}
            orchestration_meta = dict(result.meta)
            orchestration_meta["trace_id"] = span.trace_id
            orchestration_meta["root_span_id"] = span.span_id
            span.finish(
                attributes={
                    "status": "completed",
                    "response_id": _request_value(response, "id"),
                    "policy_route": result.meta.get("policy_route"),
                    "scenario": result.meta.get("scenario", "unknown"),
                    "node_count": len(node_items),
                    "models": models,
                    "model_targets": model_targets,
                    "routes": routes,
                    "protocols": protocols,
                    "providers": providers,
                    "provider_kinds": provider_kinds,
                    "route_attempts": route_attempts,
                    "mcp_bindings": mcp_bindings,
                    "skill_bindings": skill_bindings,
                    "prompt_binding_ids": prompt_binding_ids,
                    "tool_count": sum(int(node.get("tool_count", 0) or 0) for node in node_items),
                    "tool_names": tool_names,
                    "nodes": nodes,
                    "routing_mode": routing.get("mode", "unknown"),
                    "router_model": routing.get("router_model", "none"),
                    "selected_scenario": routing.get("scenario", result.meta.get("scenario", "unknown")),
                    "routing_confidence": routing.get("confidence", 0.0),
                    "routing_reason": routing.get("reason", "not supplied"),
                }
            )
            if ResponsesAgentResponse is not None and isinstance(response, ResponsesAgentResponse):
                return _add_custom_outputs(
                    response,
                    {
                        "orchestration": orchestration_meta,
                        "trace": {"trace_id": span.trace_id, "root_span_id": span.span_id},
                    },
                )
            response["custom_outputs"] = {
                "orchestration": orchestration_meta,
                "trace": {"trace_id": span.trace_id, "root_span_id": span.span_id},
            }
            return response
        except Exception as exc:
            span.finish(error={"exception_type": type(exc).__name__, "error_code": "orchestration_error"})
            raise

    def _restore_session(
        self, normalized: Mapping[str, Any], *, trace_span_factory: Any = None
    ) -> tuple[list[Any], int | None, dict[str, Any]]:
        session_id = normalized.get("session_id")
        if not session_id:
            return list(normalized["input"]), None, {}
        session_span = (
            trace_span_factory(
                {
                    "span_name": "haystack.session.load",
                    "span_type": "MEMORY",
                    "route": "lakebase-session",
                    "provider": "lakebase",
                    "provider_kind": "postgres",
                    "payload": {"session_id": str(session_id)},
                }
            )
            if callable(trace_span_factory)
            else None
        )
        if session_span is not None:
            session_span.__enter__()
        try:
            snapshot = self.session_store.load(str(session_id), str(normalized.get("tenant_id") or "default"))
        except Exception as error:
            if session_span is not None:
                session_span.finish(error={"exception_type": type(error).__name__, "error_code": "session_load_error"})
            raise
        if session_span is not None:
            session_span.finish(
                attributes={
                    "status": "hit" if snapshot is not None else "miss",
                    "revision": snapshot.revision if snapshot else 0,
                }
            )
        if snapshot is None:
            return _canonical_messages(normalized["input"]), None, {}
        stored_state = snapshot.state if isinstance(snapshot.state, Mapping) else {}
        previous = stored_state.get("messages", [])
        if not isinstance(previous, list):
            previous = []
        agent_state = stored_state.get("agent_state", {})
        return (
            [*_canonical_messages(previous), *_canonical_messages(normalized["input"])],
            snapshot.revision,
            dict(agent_state) if isinstance(agent_state, Mapping) else {},
        )

    def _save_session(
        self,
        normalized: Mapping[str, Any],
        input_messages: list[Any],
        result: Mapping[str, Any] | None,
        previous_revision: int | None,
        *,
        agent_state: Mapping[str, Any] | None = None,
        trace_span_factory: Any = None,
    ) -> None:
        session_id = normalized.get("session_id")
        if not session_id:
            return
        state_messages = [_canonical_message(item) for item in input_messages]
        if isinstance(result, Mapping):
            if isinstance(result.get("output"), list):
                state_messages.extend(_canonical_messages(result["output"]))
            elif isinstance(result.get("choices"), list):
                for choice in result["choices"]:
                    if isinstance(choice, Mapping) and isinstance(choice.get("message"), Mapping):
                        state_messages.append(_canonical_message(choice["message"]))
        session_span = (
            trace_span_factory(
                {
                    "span_name": "haystack.session.save",
                    "span_type": "MEMORY",
                    "route": "lakebase-session",
                    "provider": "lakebase",
                    "provider_kind": "postgres",
                    "payload": {"session_id": str(session_id), "expected_revision": previous_revision},
                }
            )
            if callable(trace_span_factory)
            else None
        )
        if session_span is not None:
            session_span.__enter__()
        try:
            self.session_store.save(
                str(session_id),
                str(normalized.get("tenant_id") or "default"),
                {"schema_version": 1, "messages": state_messages, "agent_state": dict(agent_state or {})},
                expected_revision=previous_revision,
                agent_profile=self.profile.get("name") if self.profile else None,
                route=self.profile.get("default_route") if self.profile else None,
                model_alias=normalized.get("model"),
                last_turn_id=str(normalized.get("turn_id")),
            )
            if session_span is not None:
                session_span.finish(attributes={"status": "completed", "revision": (previous_revision or 0) + 1})
        except SessionConflictError as exc:
            if session_span is not None:
                session_span.finish(error={"exception_type": type(exc).__name__, "error_code": "session_conflict"})
            raise RuntimeError("session was updated concurrently; retry with the same session_id") from exc
        except Exception as error:
            if session_span is not None:
                session_span.finish(error={"exception_type": type(error).__name__, "error_code": "session_save_error"})
            raise

    def _error_response(
        self, correlation_id: str, error_code: str, exception_type: str, diagnostic: Mapping[str, Any] | None = None
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
            item = self.create_text_output_item(
                text=f"Haystack runtime request failed: {error_code}", id=f"error_{uuid4().hex}"
            )
            return ResponsesAgentResponse(output=[item], status="failed", custom_outputs={"error": safe_diagnostic})
        return {"status": "failed", "error": safe_diagnostic}

    def _responses_stream(self, result: Iterator[dict[str, Any]], correlation_id: str) -> Iterator[Any]:
        protocol = self.runtime.gateway.settings.protocol if self.runtime else "responses"
        if protocol == "chat" and output_to_responses_items_stream is not None:
            yield from output_to_responses_items_stream(result)
            return
        for chunk in result:
            if not isinstance(chunk, Mapping) or not isinstance(chunk.get("type"), str):
                yield from self._stream_error_event(
                    {"error_code": "invalid_stream_event", "correlation_id": correlation_id}, correlation_id
                )
                return
            payload = dict(chunk)
            if payload.get("type") == "response.output_text.delta":
                payload.setdefault("item_id", f"msg_{correlation_id}")
            if ResponsesAgentStreamEvent is not None:
                try:
                    yield ResponsesAgentStreamEvent(**payload)
                    continue
                except (TypeError, ValueError):
                    pass
            yield payload

    def _stream_error_event(self, diagnostic: Mapping[str, Any], correlation_id: str) -> Iterator[Any]:
        safe_diagnostic = dict(diagnostic)
        safe_diagnostic.setdefault("correlation_id", correlation_id)
        if ResponsesAgentStreamEvent is not None:
            try:
                yield ResponsesAgentStreamEvent(
                    type="response.failed", response={"status": "failed", "custom_outputs": {"error": safe_diagnostic}}
                )
                return
            except (TypeError, ValueError):
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text="The request failed; inspect the response diagnostics.", id=f"error_{uuid4().hex}"
                    ),
                )
            return
        yield {
            "type": "response.output_item.done",
            "item": _offline_text_item(
                "The request failed; inspect custom_outputs for diagnostics.", f"error_{uuid4().hex}"
            ),
            "custom_outputs": {"error": safe_diagnostic},
        }


def load_agent_context(profile_path: str) -> Agent:
    """Create an agent from a validated profile artifact."""

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
