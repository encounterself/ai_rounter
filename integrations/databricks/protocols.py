from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

PROTOCOL_MLFLOW_CHAT = "mlflow_chat"
PROTOCOL_MLFLOW_RESPONSES = "mlflow_responses"
PROTOCOL_OPENAI_RESPONSES = "openai_responses"
PROTOCOL_ANTHROPIC_MESSAGES = "anthropic_messages"
PROTOCOL_CURSOR_CHAT = "cursor_chat"

# Contract-facing names.  The older names above remain canonical for
# backwards-compatible callers and endpoint paths.
PROTOCOL_CHAT_COMPLETIONS = "chat_completions"
PROTOCOL_RESPONSES = "responses"
PROTOCOL_CURSOR_CHAT_COMPLETIONS = "cursor_chat_completions"


@dataclass(frozen=True, slots=True)
class ProtocolCapability:
    """Provider-neutral capability metadata for one protocol."""

    protocol: str
    streaming: bool = True
    tools: bool = True
    claude: bool = False
    codex_compatible: bool = False
    provider_family: str = "generic"


class ProtocolAdapterError(ValueError):
    """Base error raised by protocol adapters."""


class UnsupportedProtocolError(ProtocolAdapterError):
    """Raised when no adapter is registered for a protocol name."""


class ProtocolPayloadError(ProtocolAdapterError):
    """Raised when a protocol payload is structurally invalid."""


@dataclass(frozen=True, slots=True)
class InternalToolCall:
    """A provider-neutral tool call emitted by an assistant."""

    id: str
    name: str
    arguments: Mapping[str, Any] | str = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InternalTool:
    """A provider-neutral tool definition accepted by a model."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True, slots=True)
class InternalMessage:
    """A provider-neutral chat message."""

    role: str
    content: str = ""
    tool_calls: tuple[InternalToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class InternalUsage:
    """Provider-neutral token usage."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class InternalError:
    """Provider-neutral error information."""

    message: str
    type: str = "provider_error"
    code: str | int | None = None
    param: str | None = None


@dataclass(frozen=True, slots=True)
class InternalRequest:
    """Provider-neutral model request."""

    messages: tuple[InternalMessage, ...] = ()
    model: str | None = None
    instructions: str | None = None
    tools: tuple[InternalTool, ...] = ()
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InternalResponse:
    """Provider-neutral model response."""

    id: str | None = None
    model: str | None = None
    messages: tuple[InternalMessage, ...] = ()
    usage: InternalUsage | None = None
    finish_reason: str | None = None
    error: InternalError | None = None
    status: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InternalStreamEnvelope:
    """Provider-neutral streaming event envelope."""

    event_type: str
    id: str | None = None
    text: str | None = None
    role: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None
    usage: InternalUsage | None = None
    finish_reason: str | None = None
    index: int | None = None
    done: bool = False
    error: InternalError | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProtocolAdapter(ABC):
    """Base interface for local request, response, and stream conversions."""

    protocol: ClassVar[str]
    endpoint: ClassVar[str]

    @abstractmethod
    def parse_request(self, payload: Mapping[str, Any]) -> InternalRequest:
        """Convert a provider request into the internal request model."""

    @abstractmethod
    def render_request(self, request: InternalRequest) -> dict[str, Any]:
        """Convert an internal request into a provider request."""

    @abstractmethod
    def parse_response(self, payload: Mapping[str, Any]) -> InternalResponse:
        """Convert a provider response into the internal response model."""

    @abstractmethod
    def render_response(self, response: InternalResponse) -> dict[str, Any]:
        """Convert an internal response into a provider response."""

    @abstractmethod
    def parse_stream(self, payload: Mapping[str, Any] | str) -> InternalStreamEnvelope:
        """Convert one provider stream event into the internal envelope."""

    @abstractmethod
    def render_stream(self, event: InternalStreamEnvelope) -> dict[str, Any] | str:
        """Convert one internal stream event into a provider envelope."""


class ChatCompletionsAdapter(ProtocolAdapter):
    """Adapter for OpenAI-compatible chat completions envelopes."""

    def parse_request(self, payload: Mapping[str, Any]) -> InternalRequest:
        """Parse a chat-completions request."""

        messages = tuple(_parse_chat_message(item) for item in _required_sequence(payload, "messages"))
        return InternalRequest(
            messages=messages,
            model=_string(payload.get("model")),
            tools=tuple(_parse_chat_tool(item) for item in _sequence(payload.get("tools"))),
            stream=bool(payload.get("stream", False)),
            max_tokens=_int(payload.get("max_tokens", payload.get("max_completion_tokens"))),
            temperature=_float(payload.get("temperature")),
            metadata=_metadata(
                payload, {"messages", "model", "tools", "stream", "max_tokens", "max_completion_tokens", "temperature"}
            ),
        )

    def render_request(self, request: InternalRequest) -> dict[str, Any]:
        """Render an internal request as a chat-completions request."""

        payload: dict[str, Any] = {
            "messages": [_render_chat_message(message) for message in request.messages],
            "stream": request.stream,
        }
        _set_if_not_none(payload, "model", request.model)
        if request.tools:
            payload["tools"] = [_render_chat_tool(tool) for tool in request.tools]
        _set_if_not_none(payload, "max_tokens", request.max_tokens)
        _set_if_not_none(payload, "temperature", request.temperature)
        payload.update(request.metadata)
        return payload

    def parse_response(self, payload: Mapping[str, Any]) -> InternalResponse:
        """Parse a chat-completions response."""

        error = _parse_error(payload)
        if error is not None:
            return InternalResponse(
                id=_string(payload.get("id")),
                model=_string(payload.get("model")),
                usage=_parse_chat_usage(payload.get("usage")),
                error=error,
                status="failed",
            )
        choices = _required_sequence(payload, "choices")
        messages = tuple(_parse_chat_message(_mapping(item, "choice").get("message", {})) for item in choices)
        finish_reason = _first_finish_reason(choices)
        return InternalResponse(
            id=_string(payload.get("id")),
            model=_string(payload.get("model")),
            messages=messages,
            usage=_parse_chat_usage(payload.get("usage")),
            finish_reason=finish_reason,
            status="completed",
            metadata=_metadata(payload, {"id", "object", "created", "model", "choices", "usage"}),
        )

    def render_response(self, response: InternalResponse) -> dict[str, Any]:
        """Render an internal response as a chat-completions response."""

        if response.error is not None:
            return {"error": _render_error(response.error)}
        choices = [
            {"index": index, "message": _render_chat_message(message), "finish_reason": response.finish_reason}
            for index, message in enumerate(response.messages)
        ]
        payload: dict[str, Any] = {
            "id": response.id or "chatcmpl-local",
            "object": "chat.completion",
            "choices": choices,
        }
        _set_if_not_none(payload, "model", response.model)
        if response.usage is not None:
            payload["usage"] = _render_chat_usage(response.usage)
        payload.update(response.metadata)
        return payload

    def parse_stream(self, payload: Mapping[str, Any] | str) -> InternalStreamEnvelope:
        """Parse one chat-completions stream chunk."""

        if payload == "[DONE]":
            return InternalStreamEnvelope(event_type="done", done=True)
        data = _mapping(payload, "stream event")
        error = _parse_error(data)
        if error is not None:
            return InternalStreamEnvelope(event_type="error", error=error, done=True)
        choices = _sequence(data.get("choices"))
        if not choices:
            return InternalStreamEnvelope(
                event_type="message", id=_string(data.get("id")), usage=_parse_chat_usage(data.get("usage"))
            )
        choice = _mapping(choices[0], "choice")
        delta = _mapping(choice.get("delta", {}), "delta")
        tool_calls = _sequence(delta.get("tool_calls"))
        tool_call = _mapping(tool_calls[0], "tool call") if tool_calls else {}
        function = _mapping(tool_call.get("function", {}), "function")
        finish_reason = _string(choice.get("finish_reason"))
        event_type = "finish" if finish_reason else "delta"
        return InternalStreamEnvelope(
            event_type=event_type,
            id=_string(data.get("id")),
            text=_string(delta.get("content")),
            role=_string(delta.get("role")),
            tool_call_id=_string(tool_call.get("id")),
            tool_name=_string(function.get("name")),
            arguments_delta=_string(function.get("arguments")),
            usage=_parse_chat_usage(data.get("usage")),
            finish_reason=finish_reason,
            index=_int(tool_call.get("index")),
            done=finish_reason is not None,
        )

    def render_stream(self, event: InternalStreamEnvelope) -> dict[str, Any] | str:
        """Render one internal event as a chat-completions stream chunk."""

        if event.done and event.event_type == "done":
            return "[DONE]"
        if event.error is not None:
            return {"error": _render_error(event.error)}
        delta: dict[str, Any] = {}
        _set_if_not_none(delta, "role", event.role)
        _set_if_not_none(delta, "content", event.text)
        if event.tool_call_id or event.tool_name or event.arguments_delta:
            function: dict[str, Any] = {}
            _set_if_not_none(function, "name", event.tool_name)
            _set_if_not_none(function, "arguments", event.arguments_delta)
            delta["tool_calls"] = [
                {
                    "index": event.index or 0,
                    **({"id": event.tool_call_id} if event.tool_call_id else {}),
                    "type": "function",
                    "function": function,
                }
            ]
        payload: dict[str, Any] = {
            "id": event.id or "chatcmpl-local",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": event.finish_reason}],
        }
        if event.usage is not None:
            payload["usage"] = _render_chat_usage(event.usage)
        return payload


class ResponsesAdapter(ProtocolAdapter):
    """Adapter for MLflow and OpenAI Responses-compatible envelopes."""

    def parse_request(self, payload: Mapping[str, Any]) -> InternalRequest:
        """Parse a Responses request."""

        messages: list[InternalMessage] = []
        raw_input = payload.get("input", ())
        if isinstance(raw_input, str):
            messages.append(InternalMessage(role="user", content=raw_input))
        else:
            messages.extend(_parse_response_input(item) for item in _sequence(raw_input))
        return InternalRequest(
            messages=tuple(messages),
            model=_string(payload.get("model")),
            instructions=_string(payload.get("instructions")),
            tools=tuple(_parse_response_tool(item) for item in _sequence(payload.get("tools"))),
            stream=bool(payload.get("stream", False)),
            max_tokens=_int(payload.get("max_output_tokens", payload.get("max_tokens"))),
            temperature=_float(payload.get("temperature")),
            metadata=_metadata(
                payload,
                {"input", "model", "instructions", "tools", "stream", "max_output_tokens", "max_tokens", "temperature"},
            ),
        )

    def render_request(self, request: InternalRequest) -> dict[str, Any]:
        """Render an internal request as a Responses request."""

        input_items: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id or "tool-call",
                        "output": message.content,
                    }
                )
                continue
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": _json_arguments(tool_call.arguments),
                        }
                    )
                continue
            input_items.append(
                {"type": "message", "role": message.role, "content": [{"type": "input_text", "text": message.content}]}
            )
        payload: dict[str, Any] = {"input": input_items, "stream": request.stream}
        _set_if_not_none(payload, "model", request.model)
        _set_if_not_none(payload, "instructions", request.instructions)
        if request.tools:
            payload["tools"] = [_render_response_tool(tool) for tool in request.tools]
        _set_if_not_none(payload, "max_output_tokens", request.max_tokens)
        _set_if_not_none(payload, "temperature", request.temperature)
        payload.update(request.metadata)
        return payload

    def parse_response(self, payload: Mapping[str, Any]) -> InternalResponse:
        """Parse a Responses response."""

        error = _parse_error(payload)
        if error is not None:
            return InternalResponse(
                id=_string(payload.get("id")),
                model=_string(payload.get("model")),
                usage=_parse_response_usage(payload.get("usage")),
                error=error,
                status="failed",
            )
        messages: list[InternalMessage] = []
        for item in _sequence(payload.get("output")):
            item_mapping = _mapping(item, "output item")
            item_type = _string(item_mapping.get("type"))
            if item_type == "function_call":
                messages.append(
                    InternalMessage(
                        role="assistant",
                        tool_calls=(
                            InternalToolCall(
                                id=(
                                    _string(item_mapping.get("call_id"))
                                    or _string(item_mapping.get("id"))
                                    or "tool-call"
                                ),
                                name=_string(item_mapping.get("name")) or "tool",
                                arguments=_parse_arguments(item_mapping.get("arguments", {})),
                            ),
                        ),
                    )
                )
            elif item_type in {"message", "output_text"}:
                messages.append(_parse_response_output_message(item_mapping))
        status = _string(payload.get("status")) or "completed"
        return InternalResponse(
            id=_string(payload.get("id")),
            model=_string(payload.get("model")),
            messages=tuple(messages),
            usage=_parse_response_usage(payload.get("usage")),
            finish_reason=_string(payload.get("finish_reason")) or status,
            status=status,
            metadata=_metadata(
                payload, {"id", "object", "created_at", "model", "output", "usage", "status", "finish_reason"}
            ),
        )

    def render_response(self, response: InternalResponse) -> dict[str, Any]:
        """Render an internal response as a Responses response."""

        if response.error is not None:
            return {"id": response.id or "resp_local", "status": "failed", "error": _render_error(response.error)}
        output: list[dict[str, Any]] = []
        message_index = 0
        for message in response.messages:
            if message.content:
                output.append(
                    {
                        "type": "message",
                        "id": f"msg_local_{message_index}",
                        "role": message.role,
                        "content": [{"type": "output_text", "text": message.content}],
                    }
                )
                message_index += 1
            for tool_call in message.tool_calls:
                output.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": _json_arguments(tool_call.arguments),
                    }
                )
        payload: dict[str, Any] = {
            "id": response.id or "resp_local",
            "object": "response",
            "status": response.status or "completed",
            "output": output,
        }
        _set_if_not_none(payload, "model", response.model)
        if response.usage is not None:
            payload["usage"] = _render_response_usage(response.usage)
        payload.update(response.metadata)
        return payload

    def parse_stream(self, payload: Mapping[str, Any] | str) -> InternalStreamEnvelope:
        """Parse one Responses stream event."""

        data = _mapping(payload, "stream event")
        event_type = _string(data.get("type")) or "message"
        error = _parse_error(data)
        if error is not None or event_type == "error":
            return InternalStreamEnvelope(
                event_type="error", id=_string(data.get("id")), error=error or InternalError("stream error"), done=True
            )
        response = _mapping(data.get("response", {}), "response")
        event_id = _string(data.get("response_id")) or _string(response.get("id")) or _string(data.get("id"))
        if event_type == "response.output_text.delta":
            return InternalStreamEnvelope(event_type="text_delta", id=event_id, text=_string(data.get("delta")))
        if event_type == "response.function_call_arguments.delta":
            return InternalStreamEnvelope(
                event_type="tool_call_delta",
                id=event_id,
                tool_call_id=_string(data.get("call_id")) or _string(data.get("item_id")),
                tool_name=_string(data.get("name")),
                arguments_delta=_string(data.get("delta")),
                index=_int(data.get("output_index")),
            )
        if event_type == "response.completed":
            return InternalStreamEnvelope(
                event_type="done",
                id=event_id,
                usage=_parse_response_usage(response.get("usage") or data.get("usage")),
                done=True,
            )
        return InternalStreamEnvelope(event_type=event_type, id=event_id, metadata=dict(data))

    def render_stream(self, event: InternalStreamEnvelope) -> dict[str, Any] | str:
        """Render one internal event as a Responses stream event."""

        if event.error is not None:
            return {"type": "error", "error": _render_error(event.error)}
        if event.event_type == "done" or event.done:
            payload: dict[str, Any] = {
                "type": "response.completed",
                "response": {"id": event.id or "resp_local", "status": "completed"},
            }
            if event.usage is not None:
                payload["response"]["usage"] = _render_response_usage(event.usage)
            return payload
        if event.event_type == "text_delta":
            return {"type": "response.output_text.delta", "response_id": event.id, "delta": event.text or ""}
        if event.event_type == "tool_call_delta":
            return {
                "type": "response.function_call_arguments.delta",
                "response_id": event.id,
                "item_id": event.tool_call_id,
                "name": event.tool_name,
                "delta": event.arguments_delta or "",
                "output_index": event.index or 0,
            }
        payload = dict(event.metadata)
        payload.setdefault("type", event.event_type)
        return payload


class AnthropicMessagesAdapter(ProtocolAdapter):
    """Adapter for Anthropic Messages envelopes."""

    protocol = PROTOCOL_ANTHROPIC_MESSAGES
    endpoint = "anthropic/v1/messages"

    def parse_request(self, payload: Mapping[str, Any]) -> InternalRequest:
        """Parse an Anthropic Messages request."""

        messages = tuple(_parse_anthropic_message(item) for item in _required_sequence(payload, "messages"))
        system = payload.get("system")
        instructions = _content_text(system) if system is not None else None
        return InternalRequest(
            messages=messages,
            model=_string(payload.get("model")),
            instructions=instructions,
            tools=tuple(_parse_anthropic_tool(item) for item in _sequence(payload.get("tools"))),
            stream=bool(payload.get("stream", False)),
            max_tokens=_int(payload.get("max_tokens")),
            temperature=_float(payload.get("temperature")),
            metadata=_metadata(
                payload, {"messages", "system", "model", "tools", "stream", "max_tokens", "temperature"}
            ),
        )

    def render_request(self, request: InternalRequest) -> dict[str, Any]:
        """Render an internal request as an Anthropic Messages request."""

        payload: dict[str, Any] = {
            "messages": [_render_anthropic_message(message) for message in request.messages],
            "stream": request.stream,
        }
        _set_if_not_none(payload, "model", request.model)
        _set_if_not_none(payload, "system", request.instructions)
        if request.tools:
            payload["tools"] = [_render_anthropic_tool(tool) for tool in request.tools]
        _set_if_not_none(payload, "max_tokens", request.max_tokens)
        _set_if_not_none(payload, "temperature", request.temperature)
        payload.update(request.metadata)
        return payload

    def parse_response(self, payload: Mapping[str, Any]) -> InternalResponse:
        """Parse an Anthropic Messages response."""

        error = _parse_error(payload)
        if error is not None or payload.get("type") == "error":
            return InternalResponse(
                id=_string(payload.get("id")), error=error or InternalError("provider error"), status="failed"
            )
        messages: list[InternalMessage] = []
        tool_calls: list[InternalToolCall] = []
        text_parts: list[str] = []
        for block in _sequence(payload.get("content")):
            item = _mapping(block, "content block")
            block_type = _string(item.get("type"))
            if block_type == "text":
                text_parts.append(_string(item.get("text")) or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    InternalToolCall(
                        id=_string(item.get("id")) or "tool-call",
                        name=_string(item.get("name")) or "tool",
                        arguments=_parse_arguments(item.get("input", {})),
                    )
                )
        if text_parts or tool_calls:
            messages.append(
                InternalMessage(role="assistant", content="".join(text_parts), tool_calls=tuple(tool_calls))
            )
        usage = _parse_anthropic_usage(payload.get("usage"))
        return InternalResponse(
            id=_string(payload.get("id")),
            model=_string(payload.get("model")),
            messages=tuple(messages),
            usage=usage,
            finish_reason=_string(payload.get("stop_reason")),
            status="completed",
        )

    def render_response(self, response: InternalResponse) -> dict[str, Any]:
        """Render an internal response as an Anthropic Messages response."""

        if response.error is not None:
            return {"type": "error", "error": _render_error(response.error)}
        content: list[dict[str, Any]] = []
        for message in response.messages:
            if message.content:
                content.append({"type": "text", "text": message.content})
            for tool_call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": _arguments_mapping(tool_call.arguments),
                    }
                )
        payload: dict[str, Any] = {
            "id": response.id or "msg_local",
            "type": "message",
            "role": "assistant",
            "content": content,
            "stop_reason": response.finish_reason,
        }
        _set_if_not_none(payload, "model", response.model)
        if response.usage is not None:
            payload["usage"] = _render_anthropic_usage(response.usage)
        return payload

    def parse_stream(self, payload: Mapping[str, Any] | str) -> InternalStreamEnvelope:
        """Parse one Anthropic Messages stream event."""

        data = _mapping(payload, "stream event")
        event_type = _string(data.get("type")) or "message"
        if event_type == "error":
            error = _parse_error(data) or InternalError("provider stream error")
            return InternalStreamEnvelope(event_type="error", error=error, done=True)
        if event_type == "message_start":
            message = _mapping(data.get("message", {}), "message")
            return InternalStreamEnvelope(
                event_type="message_start",
                id=_string(message.get("id")),
                role=_string(message.get("role")),
                usage=_parse_anthropic_usage(message.get("usage")),
            )
        if event_type == "content_block_start":
            block = _mapping(data.get("content_block", {}), "content block")
            return InternalStreamEnvelope(
                event_type="tool_call_start" if block.get("type") == "tool_use" else "content_start",
                tool_call_id=_string(block.get("id")),
                tool_name=_string(block.get("name")),
                index=_int(data.get("index")),
            )
        if event_type == "content_block_delta":
            delta = _mapping(data.get("delta", {}), "delta")
            delta_type = _string(delta.get("type"))
            return InternalStreamEnvelope(
                event_type="tool_call_delta" if delta_type == "input_json_delta" else "text_delta",
                text=_string(delta.get("text")),
                arguments_delta=_string(delta.get("partial_json")),
                index=_int(data.get("index")),
            )
        if event_type == "message_delta":
            return InternalStreamEnvelope(
                event_type="finish",
                finish_reason=_string(_mapping(data.get("delta", {}), "delta").get("stop_reason")),
                usage=_parse_anthropic_usage(data.get("usage")),
            )
        if event_type == "message_stop":
            return InternalStreamEnvelope(event_type="done", done=True)
        return InternalStreamEnvelope(event_type=event_type, metadata=dict(data))

    def render_stream(self, event: InternalStreamEnvelope) -> dict[str, Any] | str:
        """Render one internal event as an Anthropic Messages stream event."""

        if event.error is not None:
            return {"type": "error", "error": _render_error(event.error)}
        if event.event_type == "message_start":
            message: dict[str, Any] = {
                "id": event.id or "msg_local",
                "type": "message",
                "role": event.role or "assistant",
                "content": [],
            }
            if event.usage is not None:
                message["usage"] = _render_anthropic_usage(event.usage)
            return {"type": "message_start", "message": message}
        if event.event_type in {"text_delta", "tool_call_delta"}:
            delta = {"type": "text_delta", "text": event.text or ""}
            if event.event_type == "tool_call_delta":
                delta = {"type": "input_json_delta", "partial_json": event.arguments_delta or ""}
            return {"type": "content_block_delta", "index": event.index or 0, "delta": delta}
        if event.event_type == "tool_call_start":
            return {
                "type": "content_block_start",
                "index": event.index or 0,
                "content_block": {
                    "type": "tool_use",
                    "id": event.tool_call_id or "tool-call",
                    "name": event.tool_name or "tool",
                    "input": {},
                },
            }
        if event.event_type == "finish":
            payload: dict[str, Any] = {"type": "message_delta", "delta": {"stop_reason": event.finish_reason}}
            if event.usage is not None:
                payload["usage"] = _render_anthropic_usage(event.usage)
            return payload
        if event.event_type == "done" or event.done:
            return {"type": "message_stop"}
        return dict(event.metadata) | {"type": event.event_type}


class MLflowChatAdapter(ChatCompletionsAdapter):
    """Adapter for the MLflow chat completions endpoint."""

    protocol = PROTOCOL_MLFLOW_CHAT
    endpoint = "mlflow/v1/chat/completions"


class CursorChatAdapter(ChatCompletionsAdapter):
    """Adapter for the Cursor chat completions endpoint."""

    protocol = PROTOCOL_CURSOR_CHAT
    endpoint = "cursor/v1/chat/completions"


class MLflowResponsesAdapter(ResponsesAdapter):
    """Adapter for the MLflow Responses endpoint."""

    protocol = PROTOCOL_MLFLOW_RESPONSES
    endpoint = "mlflow/v1/responses"


class OpenAIResponsesAdapter(ResponsesAdapter):
    """Adapter for the OpenAI Responses endpoint."""

    protocol = PROTOCOL_OPENAI_RESPONSES
    endpoint = "openai/v1/responses"


_ADAPTERS: dict[str, ProtocolAdapter] = {
    PROTOCOL_MLFLOW_CHAT: MLflowChatAdapter(),
    PROTOCOL_MLFLOW_RESPONSES: MLflowResponsesAdapter(),
    PROTOCOL_OPENAI_RESPONSES: OpenAIResponsesAdapter(),
    PROTOCOL_ANTHROPIC_MESSAGES: AnthropicMessagesAdapter(),
    PROTOCOL_CURSOR_CHAT: CursorChatAdapter(),
}

_PROTOCOL_ALIASES: dict[str, str] = {
    PROTOCOL_CHAT_COMPLETIONS: PROTOCOL_MLFLOW_CHAT,
    PROTOCOL_RESPONSES: PROTOCOL_OPENAI_RESPONSES,
    PROTOCOL_CURSOR_CHAT_COMPLETIONS: PROTOCOL_CURSOR_CHAT,
}

_PROTOCOL_CAPABILITIES: dict[str, ProtocolCapability] = {
    PROTOCOL_MLFLOW_CHAT: ProtocolCapability(PROTOCOL_MLFLOW_CHAT, provider_family="databricks"),
    PROTOCOL_MLFLOW_RESPONSES: ProtocolCapability(PROTOCOL_MLFLOW_RESPONSES, provider_family="databricks"),
    PROTOCOL_OPENAI_RESPONSES: ProtocolCapability(
        PROTOCOL_OPENAI_RESPONSES, codex_compatible=True, provider_family="openai-compatible"
    ),
    PROTOCOL_ANTHROPIC_MESSAGES: ProtocolCapability(
        PROTOCOL_ANTHROPIC_MESSAGES, claude=True, provider_family="anthropic"
    ),
    PROTOCOL_CURSOR_CHAT: ProtocolCapability(PROTOCOL_CURSOR_CHAT, provider_family="cursor", codex_compatible=False),
}


def normalize_protocol(protocol: str) -> str:
    """Return the canonical adapter name for a contract or legacy value."""

    return _PROTOCOL_ALIASES.get(protocol, protocol)


def get_protocol_adapter(protocol: str) -> ProtocolAdapter:
    """Return the local adapter registered for ``protocol``."""

    try:
        return _ADAPTERS[normalize_protocol(protocol)]
    except KeyError as error:
        raise UnsupportedProtocolError(f"Unsupported protocol: {protocol}") from error


def get_protocol_capability(protocol: str) -> ProtocolCapability:
    """Return capability metadata for a canonical or contract protocol name."""

    canonical = normalize_protocol(protocol)
    try:
        return _PROTOCOL_CAPABILITIES[canonical]
    except KeyError as error:
        raise UnsupportedProtocolError(f"Unsupported protocol: {protocol}") from error


def parse_request(protocol: str, payload: Mapping[str, Any]) -> InternalRequest:
    """Parse a provider request using the selected protocol adapter."""

    return get_protocol_adapter(protocol).parse_request(payload)


def render_request(protocol: str, request: InternalRequest) -> dict[str, Any]:
    """Render an internal request using the selected protocol adapter."""

    return get_protocol_adapter(protocol).render_request(request)


def parse_response(protocol: str, payload: Mapping[str, Any]) -> InternalResponse:
    """Parse a provider response using the selected protocol adapter."""

    return get_protocol_adapter(protocol).parse_response(payload)


def render_response(protocol: str, response: InternalResponse) -> dict[str, Any]:
    """Render an internal response using the selected protocol adapter."""

    return get_protocol_adapter(protocol).render_response(response)


def parse_stream(protocol: str, payload: Mapping[str, Any] | str) -> InternalStreamEnvelope:
    """Parse one provider stream event using the selected protocol adapter."""

    return get_protocol_adapter(protocol).parse_stream(payload)


def render_stream(protocol: str, event: InternalStreamEnvelope) -> dict[str, Any] | str:
    """Render one internal stream event using the selected protocol adapter."""

    return get_protocol_adapter(protocol).render_stream(event)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolPayloadError(f"{label} must be an object")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    raise ProtocolPayloadError("expected an array")


def _required_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    if key not in payload:
        raise ProtocolPayloadError(f"missing required field: {key}")
    return _sequence(payload[key])


def _string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return int(value) if isinstance(value, str) and value.isdigit() else None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _metadata(payload: Mapping[str, Any], known: set[str]) -> Mapping[str, Any]:
    return {key: value for key, value in payload.items() if key not in known}


def _set_if_not_none(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _parse_arguments(value: Any) -> Mapping[str, Any] | str:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return dict(decoded) if isinstance(decoded, Mapping) else value
    return str(value)


def _arguments_mapping(value: Mapping[str, Any] | str) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {"raw": value}


def _json_arguments(value: Mapping[str, Any] | str) -> str:
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), sort_keys=True)


def _parse_tool_call(value: Any) -> InternalToolCall:
    payload = _mapping(value, "tool call")
    function = _mapping(payload.get("function", {}), "function")
    return InternalToolCall(
        id=_string(payload.get("id")) or _string(payload.get("call_id")) or "tool-call",
        name=_string(function.get("name")) or _string(payload.get("name")) or "tool",
        arguments=_parse_arguments(function.get("arguments", payload.get("arguments", {}))),
    )


def _parse_chat_message(value: Any) -> InternalMessage:
    payload = _mapping(value, "message")
    return InternalMessage(
        role=_string(payload.get("role")) or "user",
        content=_content_text(payload.get("content", "")),
        tool_calls=tuple(_parse_tool_call(item) for item in _sequence(payload.get("tool_calls"))),
        tool_call_id=_string(payload.get("tool_call_id")),
        name=_string(payload.get("name")),
    )


def _render_chat_message(message: InternalMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    _set_if_not_none(payload, "name", message.name)
    _set_if_not_none(payload, "tool_call_id", message.tool_call_id)
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": _json_arguments(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return payload


def _parse_chat_tool(value: Any) -> InternalTool:
    payload = _mapping(value, "tool")
    function = _mapping(payload.get("function", payload), "function")
    return InternalTool(
        name=_string(function.get("name")) or "tool",
        description=_string(function.get("description")),
        parameters=_mapping(function.get("parameters", {}), "tool parameters"),
    )


def _render_chat_tool(tool: InternalTool) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool.name, "parameters": dict(tool.parameters)}
    _set_if_not_none(function, "description", tool.description)
    return {"type": "function", "function": function}


def _parse_response_input(value: Any) -> InternalMessage:
    payload = _mapping(value, "response input item")
    item_type = _string(payload.get("type"))
    if item_type == "function_call_output":
        return InternalMessage(
            role="tool", content=_content_text(payload.get("output", "")), tool_call_id=_string(payload.get("call_id"))
        )
    if item_type == "function_call":
        call = InternalToolCall(
            id=_string(payload.get("call_id")) or _string(payload.get("id")) or "tool-call",
            name=_string(payload.get("name")) or "tool",
            arguments=_parse_arguments(payload.get("arguments", {})),
        )
        return InternalMessage(role="assistant", tool_calls=(call,))
    return InternalMessage(
        role=_string(payload.get("role")) or "user", content=_content_text(payload.get("content", ""))
    )


def _parse_response_output_message(payload: Mapping[str, Any]) -> InternalMessage:
    content = _content_text(payload.get("content", payload.get("text", "")))
    return InternalMessage(role=_string(payload.get("role")) or "assistant", content=content)


def _parse_response_tool(value: Any) -> InternalTool:
    payload = _mapping(value, "tool")
    function = _mapping(payload.get("function", payload), "function")
    return InternalTool(
        name=_string(function.get("name")) or "tool",
        description=_string(function.get("description")),
        parameters=_mapping(function.get("parameters", function.get("input_schema", {})), "tool parameters"),
    )


def _render_response_tool(tool: InternalTool) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "function", "name": tool.name, "parameters": dict(tool.parameters)}
    _set_if_not_none(payload, "description", tool.description)
    return payload


def _parse_anthropic_message(value: Any) -> InternalMessage:
    payload = _mapping(value, "message")
    if isinstance(payload.get("content"), str):
        return InternalMessage(role=_string(payload.get("role")) or "user", content=payload["content"])
    text_parts: list[str] = []
    tool_calls: list[InternalToolCall] = []
    tool_call_id: str | None = None
    for block in _sequence(payload.get("content", "")):
        item = _mapping(block, "content block")
        block_type = _string(item.get("type"))
        if block_type == "text":
            text_parts.append(_string(item.get("text")) or "")
        elif block_type == "tool_use":
            tool_calls.append(
                InternalToolCall(
                    id=_string(item.get("id")) or "tool-call",
                    name=_string(item.get("name")) or "tool",
                    arguments=_parse_arguments(item.get("input", {})),
                )
            )
        elif block_type == "tool_result":
            tool_call_id = _string(item.get("tool_use_id"))
            text_parts.append(_content_text(item.get("content", "")))
    content = (
        _content_text(payload.get("content", "")) if isinstance(payload.get("content"), str) else "".join(text_parts)
    )
    return InternalMessage(
        role="tool" if tool_call_id is not None else (_string(payload.get("role")) or "user"),
        content=content,
        tool_calls=tuple(tool_calls),
        tool_call_id=tool_call_id,
    )


def _render_anthropic_message(message: InternalMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": message.tool_call_id or "tool-call", "content": message.content}
            ],
        }
    content: list[dict[str, Any]] = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    for call in message.tool_calls:
        content.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": _arguments_mapping(call.arguments)}
        )
    return {"role": message.role, "content": content}


def _parse_anthropic_tool(value: Any) -> InternalTool:
    payload = _mapping(value, "tool")
    return InternalTool(
        name=_string(payload.get("name")) or "tool",
        description=_string(payload.get("description")),
        parameters=_mapping(payload.get("input_schema", payload.get("parameters", {})), "tool parameters"),
    )


def _render_anthropic_tool(tool: InternalTool) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": tool.name, "input_schema": dict(tool.parameters)}
    _set_if_not_none(payload, "description", tool.description)
    return payload


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                parts.append(_string(item.get("text")) or _string(item.get("content")) or "")
        return "".join(parts)
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _first_finish_reason(choices: Sequence[Any]) -> str | None:
    for choice in choices:
        value = _string(_mapping(choice, "choice").get("finish_reason"))
        if value is not None:
            return value
    return None


def _parse_usage(value: Any, *, input_key: str, output_key: str, total_key: str) -> InternalUsage | None:
    if not isinstance(value, Mapping):
        return None
    input_tokens = _int(value.get(input_key)) or 0
    output_tokens = _int(value.get(output_key)) or 0
    total_tokens = _int(value.get(total_key)) or input_tokens + output_tokens
    return InternalUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)


def _parse_chat_usage(value: Any) -> InternalUsage | None:
    return _parse_usage(value, input_key="prompt_tokens", output_key="completion_tokens", total_key="total_tokens")


def _render_chat_usage(usage: InternalUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _parse_response_usage(value: Any) -> InternalUsage | None:
    return _parse_usage(value, input_key="input_tokens", output_key="output_tokens", total_key="total_tokens")


def _render_response_usage(usage: InternalUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _parse_anthropic_usage(value: Any) -> InternalUsage | None:
    return _parse_usage(value, input_key="input_tokens", output_key="output_tokens", total_key="total_tokens")


def _render_anthropic_usage(usage: InternalUsage) -> dict[str, int]:
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def _parse_error(payload: Mapping[str, Any]) -> InternalError | None:
    error_value = payload.get("error")
    if not isinstance(error_value, Mapping):
        return None
    return InternalError(
        message=_string(error_value.get("message")) or "provider error",
        type=_string(error_value.get("type")) or "provider_error",
        code=error_value.get("code"),
        param=_string(error_value.get("param")),
    )


def _render_error(error: InternalError) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": error.message, "type": error.type}
    _set_if_not_none(payload, "code", error.code)
    _set_if_not_none(payload, "param", error.param)
    return payload


__all__ = [
    "AnthropicMessagesAdapter",
    "ChatCompletionsAdapter",
    "CursorChatAdapter",
    "InternalError",
    "InternalMessage",
    "InternalRequest",
    "InternalResponse",
    "InternalStreamEnvelope",
    "InternalTool",
    "InternalToolCall",
    "InternalUsage",
    "MLflowChatAdapter",
    "MLflowResponsesAdapter",
    "OpenAIResponsesAdapter",
    "PROTOCOL_ANTHROPIC_MESSAGES",
    "PROTOCOL_CHAT_COMPLETIONS",
    "PROTOCOL_CURSOR_CHAT",
    "PROTOCOL_CURSOR_CHAT_COMPLETIONS",
    "PROTOCOL_MLFLOW_CHAT",
    "PROTOCOL_MLFLOW_RESPONSES",
    "PROTOCOL_OPENAI_RESPONSES",
    "PROTOCOL_RESPONSES",
    "ProtocolCapability",
    "ProtocolAdapter",
    "ProtocolAdapterError",
    "ProtocolPayloadError",
    "UnsupportedProtocolError",
    "get_protocol_adapter",
    "get_protocol_capability",
    "normalize_protocol",
    "parse_request",
    "parse_response",
    "parse_stream",
    "render_request",
    "render_response",
    "render_stream",
]
