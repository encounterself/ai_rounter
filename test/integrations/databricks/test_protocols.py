import unittest
from collections.abc import Mapping

from integrations.databricks.protocols import (
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_CURSOR_CHAT,
    PROTOCOL_MLFLOW_CHAT,
    PROTOCOL_MLFLOW_RESPONSES,
    PROTOCOL_OPENAI_RESPONSES,
    InternalError,
    InternalMessage,
    InternalRequest,
    InternalUsage,
    ProtocolPayloadError,
    UnsupportedProtocolError,
    get_protocol_adapter,
    parse_request,
    parse_response,
    parse_stream,
    render_request,
    render_response,
    render_stream,
)


class ProtocolAdapterTests(unittest.TestCase):
    def test_all_protocols_are_registered_with_expected_endpoints(self) -> None:
        expected = {
            PROTOCOL_MLFLOW_CHAT: "mlflow/v1/chat/completions",
            PROTOCOL_MLFLOW_RESPONSES: "mlflow/v1/responses",
            PROTOCOL_OPENAI_RESPONSES: "openai/v1/responses",
            PROTOCOL_ANTHROPIC_MESSAGES: "anthropic/v1/messages",
            PROTOCOL_CURSOR_CHAT: "cursor/v1/chat/completions",
        }

        for protocol, endpoint in expected.items():
            self.assertEqual(get_protocol_adapter(protocol).endpoint, endpoint)

    def test_mlflow_chat_request_and_response_round_trip_with_tool_call(self) -> None:
        payload = {
            "model": "route-primary",
            "messages": [
                {"role": "user", "content": "Find the report"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"report"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "matched"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search documents",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    },
                }
            ],
            "stream": True,
            "max_tokens": 300,
        }

        request = parse_request(PROTOCOL_MLFLOW_CHAT, payload)
        self.assertEqual(request.messages[1].tool_calls[0].arguments, {"query": "report"})
        self.assertEqual(request.tools[0].name, "search")
        rendered_request = render_request(PROTOCOL_MLFLOW_CHAT, request)
        self.assertEqual(rendered_request["messages"][1]["tool_calls"][0]["function"]["name"], "search")

        response = parse_response(
            PROTOCOL_MLFLOW_CHAT,
            {
                "id": "chat-1",
                "model": "route-primary",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
        self.assertEqual(response.messages[0].content, "Done")
        self.assertEqual(response.usage, InternalUsage(10, 5, 15))
        rendered_response = render_response(PROTOCOL_MLFLOW_CHAT, response)
        self.assertEqual(rendered_response["choices"][0]["message"]["content"], "Done")
        self.assertEqual(rendered_response["usage"]["total_tokens"], 15)

    def test_mlflow_and_openai_responses_share_tool_and_usage_conversion(self) -> None:
        payload = {
            "model": "route-primary",
            "instructions": "Use tools when needed",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Search"}]}],
            "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
            "stream": True,
        }

        for protocol in (PROTOCOL_MLFLOW_RESPONSES, PROTOCOL_OPENAI_RESPONSES):
            request = parse_request(protocol, payload)
            self.assertEqual(request.messages[0].content, "Search")
            self.assertEqual(request.tools[0].name, "search")
            rendered = render_request(protocol, request)
            self.assertEqual(rendered["input"][0]["content"][0]["text"], "Search")

            response = parse_response(
                protocol,
                {
                    "id": "resp-1",
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {"type": "function_call", "call_id": "call-1", "name": "search", "arguments": '{"query":"x"}'}
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                },
            )
            self.assertEqual(response.messages[0].tool_calls[0].name, "search")
            self.assertEqual(response.usage, InternalUsage(11, 7, 18))
            rendered_response = render_response(protocol, response)
            self.assertEqual(rendered_response["output"][0]["type"], "function_call")
            self.assertEqual(rendered_response["usage"]["input_tokens"], 11)

    def test_anthropic_messages_converts_tool_use_and_tool_result(self) -> None:
        request = parse_request(
            PROTOCOL_ANTHROPIC_MESSAGES,
            {
                "model": "claude-route",
                "system": [{"type": "text", "text": "Be concise"}],
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Search"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "toolu-1", "name": "search", "input": {"q": "x"}}],
                    },
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu-1", "content": "hit"}]},
                ],
                "tools": [{"name": "search", "input_schema": {"type": "object"}}],
            },
        )

        self.assertEqual(request.instructions, "Be concise")
        self.assertEqual(request.messages[1].tool_calls[0].id, "toolu-1")
        self.assertEqual(request.messages[2].role, "tool")
        rendered_request = render_request(PROTOCOL_ANTHROPIC_MESSAGES, request)
        self.assertEqual(rendered_request["messages"][2]["content"][0]["type"], "tool_result")

        response = parse_response(
            PROTOCOL_ANTHROPIC_MESSAGES,
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I found it"},
                    {"type": "tool_use", "id": "toolu-2", "name": "search", "input": {"q": "y"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
        )
        self.assertEqual(response.messages[0].content, "I found it")
        self.assertEqual(response.messages[0].tool_calls[0].arguments, {"q": "y"})
        self.assertEqual(response.usage, InternalUsage(12, 8, 20))
        rendered_response = render_response(PROTOCOL_ANTHROPIC_MESSAGES, response)
        self.assertEqual(rendered_response["content"][1]["type"], "tool_use")

    def test_cursor_chat_uses_chat_envelope_and_endpoint(self) -> None:
        request = InternalRequest(
            model="cursor-route", messages=(InternalMessage(role="user", content="Hello"),), stream=True
        )

        rendered = render_request(PROTOCOL_CURSOR_CHAT, request)
        self.assertEqual(rendered["messages"], [{"role": "user", "content": "Hello"}])
        self.assertTrue(rendered["stream"])
        self.assertEqual(get_protocol_adapter(PROTOCOL_CURSOR_CHAT).endpoint, "cursor/v1/chat/completions")

    def test_stream_envelopes_cover_chat_responses_and_anthropic(self) -> None:
        chat_event = parse_stream(
            PROTOCOL_MLFLOW_CHAT, {"id": "chat-1", "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
        )
        self.assertEqual(chat_event.text, "Hi")
        chat_rendered = render_stream(PROTOCOL_MLFLOW_CHAT, chat_event)
        if not isinstance(chat_rendered, Mapping):
            self.fail("chat stream adapter returned a non-mapping payload")
        self.assertEqual(chat_rendered["choices"][0]["delta"]["content"], "Hi")

        response_event = parse_stream(
            PROTOCOL_OPENAI_RESPONSES,
            {"type": "response.function_call_arguments.delta", "item_id": "call-1", "delta": '{"q"'},
        )
        self.assertEqual(response_event.arguments_delta, '{"q"')
        response_rendered = render_stream(PROTOCOL_OPENAI_RESPONSES, response_event)
        if not isinstance(response_rendered, Mapping):
            self.fail("responses stream adapter returned a non-mapping payload")
        self.assertEqual(response_rendered["type"], "response.function_call_arguments.delta")

        anthropic_event = parse_stream(
            PROTOCOL_ANTHROPIC_MESSAGES,
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        )
        self.assertEqual(anthropic_event.text, "Hi")
        anthropic_rendered = render_stream(PROTOCOL_ANTHROPIC_MESSAGES, anthropic_event)
        if not isinstance(anthropic_rendered, Mapping):
            self.fail("Anthropic stream adapter returned a non-mapping payload")
        self.assertEqual(anthropic_rendered["type"], "content_block_delta")

        done = parse_stream(PROTOCOL_MLFLOW_CHAT, "[DONE]")
        self.assertTrue(done.done)
        self.assertEqual(render_stream(PROTOCOL_MLFLOW_CHAT, done), "[DONE]")

    def test_error_envelopes_are_normalized_and_rendered(self) -> None:
        error = parse_response(
            PROTOCOL_OPENAI_RESPONSES,
            {"id": "resp-1", "error": {"type": "invalid_request", "message": "bad input", "code": "bad"}},
        )
        self.assertEqual(error.error, InternalError("bad input", "invalid_request", "bad", None))
        self.assertEqual(render_response(PROTOCOL_OPENAI_RESPONSES, error)["error"]["message"], "bad input")

        stream_error = parse_stream(
            PROTOCOL_ANTHROPIC_MESSAGES, {"type": "error", "error": {"type": "overloaded", "message": "busy"}}
        )
        self.assertTrue(stream_error.done)
        error_rendered = render_stream(PROTOCOL_ANTHROPIC_MESSAGES, stream_error)
        if not isinstance(error_rendered, Mapping):
            self.fail("Anthropic error adapter returned a non-mapping payload")
        self.assertEqual(error_rendered["type"], "error")

    def test_unsupported_protocol_and_invalid_payload_fail_closed(self) -> None:
        with self.assertRaises(UnsupportedProtocolError):
            get_protocol_adapter("unknown")
        with self.assertRaises(ProtocolPayloadError):
            parse_request(PROTOCOL_MLFLOW_CHAT, {})
        with self.assertRaises(ProtocolPayloadError):
            parse_request(PROTOCOL_ANTHROPIC_MESSAGES, {"messages": "not-an-array"})


if __name__ == "__main__":
    unittest.main()
