"""
Haystack ChatGenerator adapter for the Databricks gateway runtime.

The adapter deliberately depends on the provider-neutral runtime in this
repository instead of putting Databricks imports into Haystack core.  A model
binding can therefore select a Serving Endpoint, a Model Service, an
OpenAI-compatible route, or an Anthropic route while Haystack Agent sees one
stable ``ChatGenerator`` contract.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from haystack import component, default_to_dict
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message, _normalize_messages
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    StreamingCallbackT,
    StreamingChunk,
    ToolCall,
    ToolCallDelta,
    select_streaming_callback,
)
from haystack.tools import ToolsType, flatten_tools_or_toolsets

from .ai_gateway import GatewayConfig, resolve_route_reference
from .gateway import AnthropicBackend, DatabricksServingBackend, GatewayRuntime, OpenAICompatibleBackend, RequestContext
from .protocols import (
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_CURSOR_CHAT_COMPLETIONS,
    PROTOCOL_MLFLOW_CHAT,
    PROTOCOL_MLFLOW_RESPONSES,
    PROTOCOL_OPENAI_RESPONSES,
    PROTOCOL_RESPONSES,
    InternalMessage,
    InternalResponse,
    InternalStreamEnvelope,
    parse_stream,
)


class DatabricksChatGeneratorError(RuntimeError):
    """Raised when a configured Databricks route returns an upstream error."""

    def __init__(
        self, message: str, *, code: str = "upstream_error", diagnostic: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostic = dict(diagnostic or {})


def _canonical_protocol(protocol: str) -> str:
    if protocol in {PROTOCOL_CHAT_COMPLETIONS, "chat", PROTOCOL_CURSOR_CHAT_COMPLETIONS}:
        return PROTOCOL_MLFLOW_CHAT if protocol != PROTOCOL_CURSOR_CHAT_COMPLETIONS else "cursor_chat"
    if protocol in {PROTOCOL_RESPONSES, "openai_responses"}:
        return PROTOCOL_OPENAI_RESPONSES
    if protocol == "mlflow_responses":
        return PROTOCOL_MLFLOW_RESPONSES
    return protocol


def _auth_headers(token: str | None, headers: Mapping[str, str] | None) -> dict[str, str]:
    resolved = dict(headers or {})
    if token:
        resolved.setdefault("Authorization", f"Bearer {token}")
    return resolved


def _provider_base_url(provider: Any, host: str | None) -> str:
    if provider.base_url:
        return provider.base_url
    if not host:
        raise ValueError(f"provider {provider.id} requires base_url or host")
    return f"{host.rstrip('/')}/ai-gateway"


def build_gateway_runtime(
    config: GatewayConfig | Mapping[str, Any],
    *,
    host: str | None = None,
    token: str | None = None,
    headers: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> GatewayRuntime:
    """
    Build HTTP backends for a validated route configuration.

    ``transport`` is intentionally injectable so protocol and fallback behavior
    can be tested without contacting a workspace.
    """

    gateway_config = config if isinstance(config, GatewayConfig) else GatewayConfig.from_mapping(config)
    shared_headers = _auth_headers(token, headers)
    backends: dict[str, Any] = {}
    for route in gateway_config.routes:
        provider = next(item for item in gateway_config.providers if item.id == route.provider)
        protocol = _canonical_protocol(
            route.protocols[0] if route.protocols else provider.protocols[0] if provider.protocols else "chat"
        )
        reference = resolve_route_reference(route.target)
        if provider.kind == "databricks_serving" or reference.kind == "serving_endpoint":
            if reference.kind != "serving_endpoint":
                raise ValueError(f"Serving provider route must target a serving endpoint: {route.target}")
            backends[route.id] = DatabricksServingBackend(
                host or provider.base_url or "",
                reference.name,
                protocol=protocol,
                headers=shared_headers,
                transport=transport,
                timeout=route.timeout_seconds,
                model=provider.model,
            )
        elif provider.kind == "anthropic" or protocol == PROTOCOL_ANTHROPIC_MESSAGES:
            backends[route.id] = AnthropicBackend(
                _provider_base_url(provider, host),
                protocol=PROTOCOL_ANTHROPIC_MESSAGES,
                headers=shared_headers,
                transport=transport,
                timeout=route.timeout_seconds,
                model=provider.model or reference.name,
            )
        else:
            backends[route.id] = OpenAICompatibleBackend(
                _provider_base_url(provider, host),
                protocol=protocol,
                headers=shared_headers,
                transport=transport,
                timeout=route.timeout_seconds,
                model=provider.model or reference.name,
            )
    return GatewayRuntime(gateway_config, backends)


def _tool_payload(tools: ToolsType | None, protocol: str) -> list[dict[str, Any]]:
    """Render Haystack tools in the envelope required by the selected protocol."""

    canonical = _canonical_protocol(protocol)
    result: list[dict[str, Any]] = []
    for tool in flatten_tools_or_toolsets(tools):
        spec = dict(tool.tool_spec)
        function = spec.get("function")
        if isinstance(function, Mapping):
            spec = dict(function)
        name = str(spec.get("name") or tool.name)
        description = spec.get("description")
        parameters = spec.get("parameters", spec.get("input_schema", {}))
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}}
        if canonical == PROTOCOL_ANTHROPIC_MESSAGES:
            payload: dict[str, Any] = {"name": name, "input_schema": dict(parameters)}
            if description:
                payload["description"] = str(description)
            result.append(payload)
        elif canonical in {PROTOCOL_OPENAI_RESPONSES, PROTOCOL_MLFLOW_RESPONSES}:
            payload = {"type": "function", "name": name, "parameters": dict(parameters)}
            if description:
                payload["description"] = str(description)
            result.append(payload)
        else:
            result.append(
                {
                    "type": "function",
                    "function": {"name": name, "description": description, "parameters": dict(parameters)},
                }
            )
    return result


def _message_payload(messages: list[ChatMessage], protocol: str) -> dict[str, Any]:
    """Build the request envelope expected by the selected protocol adapter."""

    canonical = _canonical_protocol(protocol)
    if canonical == PROTOCOL_ANTHROPIC_MESSAGES:
        system_parts: list[str] = []
        rendered_messages: list[dict[str, Any]] = []
        for message in messages:
            rendered = message.to_openai_dict_format(require_tool_call_ids=False)
            if rendered.get("role") == "system":
                content = rendered.get("content", "")
                system_parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
                continue
            rendered_messages.append(_anthropic_message(rendered))
        payload: dict[str, Any] = {"messages": rendered_messages}
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        return payload

    if canonical in {PROTOCOL_OPENAI_RESPONSES, PROTOCOL_MLFLOW_RESPONSES}:
        instructions: list[str] = []
        input_messages: list[ChatMessage] = []
        for message in messages:
            rendered = message.to_openai_dict_format(require_tool_call_ids=False)
            if rendered.get("role") in {"system", "developer"}:
                content = rendered.get("content", "")
                instructions.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
            else:
                input_messages.append(message)
        payload: dict[str, Any] = {"input": _responses_input(input_messages)}
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        return payload

    return {"messages": [message.to_openai_dict_format() for message in messages]}


def _anthropic_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI-shaped message into an Anthropic message block."""

    role = str(message.get("role", "user"))
    if role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id") or "tool-call",
                    "content": message.get("content", ""),
                }
            ],
        }
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text if isinstance(text, str) else str(text)})
    for call in message.get("tool_calls", []) or []:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function", {})
        if not isinstance(function, Mapping):
            function = {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "tool-call",
                "name": function.get("name") or "tool",
                "input": arguments if isinstance(arguments, Mapping) else {"value": arguments},
            }
        )
    return {"role": role, "content": content or [{"type": "text", "text": ""}]}


def _responses_input(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Translate Haystack messages into OpenAI Responses input items."""

    items: list[dict[str, Any]] = []
    for message in messages:
        rendered = message.to_openai_dict_format(require_tool_call_ids=False)
        role = rendered.get("role", "user")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": rendered.get("tool_call_id") or "tool-call",
                    "output": rendered.get("content", ""),
                }
            )
            continue
        calls = rendered.get("tool_calls") or []
        if calls:
            if rendered.get("content"):
                items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": str(rendered["content"])}],
                    }
                )
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function", {})
                if not isinstance(function, Mapping):
                    function = {}
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "tool-call",
                        "name": function.get("name") or "tool",
                        "arguments": arguments,
                    }
                )
            continue
        content = rendered.get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
        items.append({"type": "message", "role": role, "content": [{"type": "input_text", "text": str(content)}]})
    return items


def _tool_calls(message: InternalMessage) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for call in message.tool_calls:
        arguments = call.arguments
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                decoded = {"raw": arguments}
        else:
            decoded = dict(arguments)
        calls.append(ToolCall(id=call.id, tool_name=call.name, arguments=decoded))
    return calls


def _response_message(response: InternalResponse, metadata: Mapping[str, Any]) -> ChatMessage:
    message = response.messages[0] if response.messages else InternalMessage(role="assistant", content="")
    meta = {**dict(metadata), "model": response.model, "finish_reason": response.finish_reason}
    if response.usage:
        meta["usage"] = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return ChatMessage.from_assistant(text=message.content or None, tool_calls=_tool_calls(message), meta=meta)


def _streaming_chunk(event: InternalStreamEnvelope, component_info: ComponentInfo) -> StreamingChunk:
    tool_calls = None
    if event.tool_call_id or event.tool_name or event.arguments_delta:
        tool_calls = [
            ToolCallDelta(
                index=event.index or 0,
                tool_name=event.tool_name,
                arguments=event.arguments_delta,
                id=event.tool_call_id,
            )
        ]
    return StreamingChunk(
        content=event.text or "",
        component_info=component_info,
        index=event.index,
        tool_calls=tool_calls,
        finish_reason=event.finish_reason,
        start=event.role == "assistant",
        meta=dict(event.metadata),
    )


@component
class DatabricksChatGenerator:
    """Expose a configured Databricks route as a Haystack ChatGenerator."""

    def __init__(
        self,
        runtime: GatewayRuntime,
        protocol: str = PROTOCOL_CHAT_COMPLETIONS,
        model: str | None = None,
        route_id: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        trace_span_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.protocol = protocol
        self.model = model
        self.route_id = route_id
        self.streaming_callback = streaming_callback
        self.generation_kwargs = generation_kwargs or {}
        self.trace_span_factory = trace_span_factory
        self.last_metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the non-transport component configuration."""

        return default_to_dict(
            self,
            protocol=self.protocol,
            model=self.model,
            route_id=self.route_id,
            generation_kwargs=self.generation_kwargs,
        )

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    def run(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        """Generate Haystack replies through the selected Databricks route."""

        callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )
        normalized = _normalize_messages(messages=messages)
        kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        canonical_protocol = _canonical_protocol(self.protocol)
        tool_enabled = bool(tools)
        payload = {
            **_message_payload(normalized, canonical_protocol),
            "stream": callback is not None and not tool_enabled,
            **kwargs,
        }
        if self.model:
            payload["model"] = self.model
        if tools:
            payload["tools"] = _tool_payload(tools, canonical_protocol)
        metadata_context = {"route_id": self.route_id} if self.route_id else {}
        context = RequestContext(metadata=metadata_context)
        metadata: dict[str, Any] = {
            "protocol": self.protocol,
            "route_id": self.route_id,
            "tool_count": len(flatten_tools_or_toolsets(tools)) if tools else 0,
            "tool_names": [tool.name for tool in flatten_tools_or_toolsets(tools)] if tools else [],
        }
        if callback is None or tool_enabled:
            call_span = self._start_trace_span(
                metadata,
                span_name=f"haystack.llm.{self.route_id or canonical_protocol}",
                span_type="CHAT_MODEL",
                payload=payload,
            )
            try:
                result = self.runtime.handle_internal(canonical_protocol, payload, context=context)
            except Exception as error:
                if call_span is not None:
                    call_span.finish(error={"exception_type": type(error).__name__, "error_code": "llm_call_error"})
                raise
            metadata.update({"route": result.selection.primary.id, "route_attempts": result.attempts})
            self.last_metadata = metadata
            if result.response.error:
                error = result.response.error
                code = str(error.code) if error.code is not None else "upstream_error"
                self.last_metadata["error_code"] = code
                if call_span is not None:
                    call_span.finish(
                        error={"error_code": code, "error_type": error.type, "route_attempts": list(result.attempts)}
                    )
                raise DatabricksChatGeneratorError(
                    error.message,
                    code=code,
                    diagnostic={
                        "error_type": error.type,
                        "error_code": code,
                        "route": result.selection.primary.id,
                        "route_attempts": list(result.attempts),
                    },
                )
            if call_span is not None:
                call_span.finish(
                    attributes={
                        "status": "completed",
                        "route": result.selection.primary.id,
                        "route_attempts": list(result.attempts),
                        "model": result.response.model,
                        "finish_reason": result.response.finish_reason,
                        "usage": (
                            {
                                "input_tokens": result.response.usage.input_tokens,
                                "output_tokens": result.response.usage.output_tokens,
                                "total_tokens": result.response.usage.total_tokens,
                            }
                            if result.response.usage
                            else {}
                        ),
                    }
                )
            return {"replies": [_response_message(result.response, metadata)], "meta": metadata}

        chunks: list[StreamingChunk] = []
        component_info = ComponentInfo.from_component(self)
        call_span = self._start_trace_span(
            metadata,
            span_name=f"haystack.llm.{self.route_id or canonical_protocol}",
            span_type="CHAT_MODEL",
            payload=payload,
        )
        try:
            for rendered in self.runtime.stream(canonical_protocol, payload, context=context):
                event = parse_stream(canonical_protocol, rendered)
                chunk = _streaming_chunk(event, component_info)
                chunks.append(chunk)
                callback(chunk)
        except Exception as error:
            if call_span is not None:
                call_span.finish(error={"exception_type": type(error).__name__, "error_code": "llm_stream_error"})
            raise
        if call_span is not None:
            call_span.finish(attributes={"status": "completed", "route": self.route_id, "streamed_chunks": len(chunks)})
        self.last_metadata = metadata
        return {"replies": [_convert_streaming_chunks_to_chat_message(chunks=chunks)], "meta": metadata}

    def _start_trace_span(
        self, metadata: Mapping[str, Any], *, span_name: str, span_type: str, payload: Mapping[str, Any] | None = None
    ) -> Any:
        if self.trace_span_factory is None:
            return None
        span_metadata = {
            **dict(metadata),
            "model": self.model or self.route_id,
            "model_binding": self.route_id,
            "model_target": self.route_id,
            "route": self.route_id,
            "protocol": self.protocol,
            "provider": "databricks-serving",
            "provider_kind": "databricks_serving",
            "span_name": span_name,
            "span_type": span_type,
            "payload": payload,
        }
        span = self.trace_span_factory(span_metadata)
        if span is not None:
            span.__enter__()
        return span

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    async def run_async(
        self,
        messages: list[ChatMessage] | str,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        """Run the synchronous backend without blocking the async Haystack loop."""

        return await asyncio.to_thread(self.run, messages, generation_kwargs, tools, streaming_callback)

    def warm_up(self) -> None:
        """Backends are initialized eagerly by ``build_gateway_runtime``."""

    def close(self) -> None:
        """Close HTTP clients owned by configured backends."""

        for backend in self.runtime.backends.values():
            close = getattr(backend, "close", None)
            if callable(close):
                close()


__all__ = ["DatabricksChatGenerator", "DatabricksChatGeneratorError", "build_gateway_runtime"]
