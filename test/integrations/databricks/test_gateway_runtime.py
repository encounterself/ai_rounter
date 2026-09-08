from __future__ import annotations

import httpx

from integrations.databricks.ai_gateway import GatewayConfig, ModelProvider, ModelRoute
from integrations.databricks.gateway import (
    AnthropicBackend,
    DatabricksFoundationModelBackend,
    GatewayRuntime,
    OpenAICompatibleBackend,
)
from integrations.databricks.protocols import InternalMessage, InternalRequest


def test_databricks_foundation_model_backend_uses_chat_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "model": "databricks-gpt-5-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "smoke-ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
            request=request,
        )

    backend = DatabricksFoundationModelBackend(
        "https://workspace.example",
        "databricks-gpt-5-mini",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = backend.invoke(
            request=InternalRequest(
                messages=(
                    InternalMessage(role="user", content="hello"),
                ),
                max_tokens=16,
            ),
            request_id="test-request",
        )
    finally:
        backend.close()

    assert backend.protocol == "chat_completions"
    assert requests[0].url.path == "/serving-endpoints/databricks-gpt-5-mini/invocations"
    assert response.messages[0].content == "smoke-ok"
    assert response.usage is not None
    assert response.usage.total_tokens == 4


def test_codex_compatible_provider_uses_explicit_http_path_without_shell() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    backend = OpenAICompatibleBackend(
        "https://codex.example",
        api_key="codex-secret",
        protocol="responses",
        path="/v1/responses",
        transport=transport,
    )
    config = GatewayConfig(
        providers=(
            ModelProvider(
                "codex", "openai_compatible", base_url="https://codex.example", protocols=("responses",)
            ),
        ),
        routes=(ModelRoute("codex-route", "codex", "provider/codex", protocols=("responses",)),),
    )

    try:
        response = GatewayRuntime(config, {"codex-route": backend}).handle(
            "responses", {"model": "gpt-5-3-codex", "input": "hello"}
        )
    finally:
        backend.close()

    assert response["output"][0]["content"] == [{"type": "output_text", "text": "done"}]
    assert requests[0].url.path == "/v1/responses"
    assert requests[0].headers["authorization"] == "Bearer codex-secret"


def test_anthropic_backend_preserves_tool_use_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "search", "input": {"q": "x"}}],
                "stop_reason": "tool_use",
            },
            request=request,
        )

    backend = AnthropicBackend(
        "https://anthropic.example", api_key="anthropic-secret", transport=httpx.MockTransport(handler)
    )
    config = GatewayConfig(
        providers=(
            ModelProvider(
                "claude", "anthropic", base_url="https://anthropic.example", protocols=("anthropic_messages",)
            ),
        ),
        routes=(ModelRoute("claude-route", "claude", "provider/claude", protocols=("anthropic_messages",)),),
    )

    try:
        response = GatewayRuntime(config, {"claude-route": backend}).handle(
            "anthropic_messages",
            {"model": "claude-sonnet", "max_tokens": 100, "messages": [{"role": "user", "content": "search"}]},
        )
    finally:
        backend.close()

    assert response["content"][0]["type"] == "tool_use"
    assert response["content"][0]["name"] == "search"


def test_anthropic_stream_events_are_forwarded_through_internal_envelopes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"type":"message_start","message":{"id":"msg-stream","role":"assistant"}}\n\n'
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n'
            'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
            request=request,
        )

    backend = AnthropicBackend("https://anthropic.example", transport=httpx.MockTransport(handler))
    config = GatewayConfig(
        providers=(
            ModelProvider(
                "claude", "anthropic", base_url="https://anthropic.example", protocols=("anthropic_messages",)
            ),
        ),
        routes=(ModelRoute("claude-route", "claude", "provider/claude", protocols=("anthropic_messages",)),),
    )

    try:
        events = list(
            GatewayRuntime(config, {"claude-route": backend}).stream(
                "anthropic_messages",
                {
                    "model": "claude-sonnet",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        )
    finally:
        backend.close()

    assert events[0]["type"] == "message_start"
    assert events[1]["delta"]["text"] == "Hi"
    assert events[2]["type"] == "message_stop"
