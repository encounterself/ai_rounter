from __future__ import annotations

import json

import httpx

from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from integrations.databricks.ai_gateway import GatewayConfig, ModelProvider, ModelRoute
from integrations.databricks.haystack_chat_generator import DatabricksChatGenerator, build_gateway_runtime
from integrations.databricks.protocols import PROTOCOL_ANTHROPIC_MESSAGES, PROTOCOL_RESPONSES


def _config() -> GatewayConfig:
    return GatewayConfig(
        providers=(ModelProvider("db", "databricks_serving", protocols=("chat_completions",)),),
        routes=(ModelRoute("writing", "db", "serving-endpoints/writing", protocols=("chat_completions",)),),
    )


def test_generator_maps_chat_messages_and_tool_specs_to_serving_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "model": "writing",
                "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            },
            request=request,
        )

    runtime = build_gateway_runtime(_config(), host="https://workspace.example", transport=httpx.MockTransport(handler))
    generator = DatabricksChatGenerator(runtime, protocol="chat_completions", route_id="writing")
    tool = Tool("lookup", "Look up a value", {"type": "object", "properties": {}}, function=lambda: "ok")

    result = generator.run([ChatMessage.from_user("hello")], tools=[tool])

    assert result["replies"][0].text == "done"
    assert result["replies"][0].meta["route"] == "writing"
    assert seen["path"] == "/serving-endpoints/writing/invocations"
    assert seen["body"]["messages"][0]["content"] == "hello"  # type: ignore[index]
    assert seen["body"]["tools"][0]["function"]["name"] == "lookup"  # type: ignore[index]


def test_generator_streams_chat_chunks_through_haystack_callback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"id":"chat-1","choices":[{"delta":{"role":"assistant","content":"hi"}}]}\n\n'
            'data: {"id":"chat-1","choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body, request=request)

    chunks = []
    runtime = build_gateway_runtime(_config(), host="https://workspace.example", transport=httpx.MockTransport(handler))
    generator = DatabricksChatGenerator(runtime, protocol="chat_completions", route_id="writing")
    result = generator.run([ChatMessage.from_user("hello")], streaming_callback=chunks.append)

    assert result["replies"][0].text == "hi!"
    assert "hi" in "".join(chunk.content for chunk in chunks)
    assert any(chunk.finish_reason == "stop" for chunk in chunks)


def test_generator_uses_non_streaming_tool_execution_when_callback_is_set() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "tool-result"}],
                    }
                ],
            },
            request=request,
        )

    config = GatewayConfig(
        providers=(ModelProvider("db", "databricks_serving", protocols=("responses",)),),
        routes=(ModelRoute("responses", "db", "serving-endpoints/responder", protocols=("responses",)),),
    )
    runtime = build_gateway_runtime(config, host="https://workspace.example", transport=httpx.MockTransport(handler))
    tool = Tool("lookup", "Look up a value", {"type": "object", "properties": {}}, function=lambda: "ok")
    chunks = []

    result = DatabricksChatGenerator(runtime, protocol=PROTOCOL_RESPONSES, route_id="responses").run(
        [ChatMessage.from_user("hello")], tools=[tool], streaming_callback=chunks.append
    )

    assert result["replies"][0].text == "tool-result"
    assert chunks == []
    assert seen["body"]["stream"] is False  # type: ignore[index]


def test_route_id_forces_the_configured_route() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}, request=request
        )

    config = GatewayConfig(
        providers=(ModelProvider("db", "databricks_serving", protocols=("chat_completions",)),),
        routes=(
            ModelRoute("first", "db", "serving-endpoints/first", protocols=("chat_completions",), priority=1),
            ModelRoute("second", "db", "serving-endpoints/second", protocols=("chat_completions",), priority=2),
        ),
    )
    runtime = build_gateway_runtime(config, host="https://workspace.example", transport=httpx.MockTransport(handler))
    DatabricksChatGenerator(runtime, protocol="chat_completions", route_id="second").run(
        [ChatMessage.from_user("hello")]
    )

    assert seen == ["/serving-endpoints/second/invocations"]


def test_generator_maps_responses_input_for_a_serving_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "responses-ok"}],
                    }
                ],
            },
            request=request,
        )

    config = GatewayConfig(
        providers=(ModelProvider("db", "databricks_serving", protocols=("responses",)),),
        routes=(ModelRoute("responses", "db", "serving-endpoint/responder", protocols=("responses",)),),
    )
    runtime = build_gateway_runtime(config, host="https://workspace.example", transport=httpx.MockTransport(handler))
    result = DatabricksChatGenerator(runtime, protocol=PROTOCOL_RESPONSES, route_id="responses").run(
        [ChatMessage.from_system("Be concise."), ChatMessage.from_user("hello")]
    )

    assert result["replies"][0].text == "responses-ok"
    assert seen["path"] == "/serving-endpoints/responses"
    body = seen["body"]
    assert "input" in body  # type: ignore[operator]
    assert "messages" not in body  # type: ignore[operator]
    assert body["model"] == "responder"  # type: ignore[index]
    assert body["input"][0]["type"] == "message"  # type: ignore[index]
    assert body["instructions"] == "Be concise."  # type: ignore[index]


def test_generator_maps_prior_assistant_messages_for_databricks_responses() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]
            },
            request=request,
        )

    config = GatewayConfig(
        providers=(ModelProvider("db", "databricks_serving", protocols=("responses",)),),
        routes=(ModelRoute("responses", "db", "serving-endpoints/responder", protocols=("responses",)),),
    )
    runtime = build_gateway_runtime(config, host="https://workspace.example", transport=httpx.MockTransport(handler))
    DatabricksChatGenerator(runtime, protocol=PROTOCOL_RESPONSES, route_id="responses").run(
        [ChatMessage.from_assistant(text="prior review"), ChatMessage.from_user("apply it")]
    )

    body = seen["body"]
    assert body["input"][0]["role"] == "developer"  # type: ignore[index]


def test_generator_renders_responses_function_tools_without_chat_wrapper() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]
            },
            request=request,
        )

    config = GatewayConfig(
        providers=(ModelProvider("db", "databricks_serving", protocols=("responses",)),),
        routes=(ModelRoute("responses", "db", "serving-endpoints/responder", protocols=("responses",)),),
    )
    runtime = build_gateway_runtime(config, host="https://workspace.example", transport=httpx.MockTransport(handler))
    tool = Tool("lookup", "Look up a value", {"type": "object", "properties": {}}, function=lambda: "ok")
    DatabricksChatGenerator(runtime, protocol=PROTOCOL_RESPONSES, route_id="responses").run(
        [ChatMessage.from_user("hello")], tools=[tool]
    )

    assert seen["body"]["tools"] == [  # type: ignore[index]
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_generator_renders_anthropic_tools_and_system_message() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}, request=request
        )

    config = GatewayConfig(
        providers=(ModelProvider("db", "databricks_serving", protocols=(PROTOCOL_ANTHROPIC_MESSAGES,)),),
        routes=(
            ModelRoute("anthropic", "db", "serving-endpoints/anthropic", protocols=(PROTOCOL_ANTHROPIC_MESSAGES,)),
        ),
    )
    runtime = build_gateway_runtime(config, host="https://workspace.example", transport=httpx.MockTransport(handler))
    tool = Tool("lookup", "Look up a value", {"type": "object", "properties": {}}, function=lambda: "ok")
    DatabricksChatGenerator(runtime, protocol=PROTOCOL_ANTHROPIC_MESSAGES, route_id="anthropic").run(
        [ChatMessage.from_system("Be concise."), ChatMessage.from_user("hello")], tools=[tool]
    )

    assert seen["body"]["system"] == "Be concise."  # type: ignore[index]
    assert seen["body"]["tools"] == [  # type: ignore[index]
        {"name": "lookup", "description": "Look up a value", "input_schema": {"type": "object", "properties": {}}}
    ]
