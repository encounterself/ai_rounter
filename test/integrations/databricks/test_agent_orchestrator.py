from __future__ import annotations

import httpx

from contracts.runtime.haystack_agent_serving.loader import AgentServingSnapshotLoader
from contracts.runtime.haystack_agent_serving.tests.test_loader import valid_profile
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, Toolset
from integrations.databricks.agent_orchestrator import ConfiguredAgentOrchestrator
from integrations.databricks.ai_gateway import GatewayConfig


def test_profile_routes_prompt_skill_and_model_through_haystack_agent() -> None:
    profile = valid_profile()
    profile["model_bindings"][0]["protocol"] = "chat_completions"
    profile["prompt_sources"] = {"runtime-system": "Mode={{request.mode}}"}
    profile["skill_sources"] = {"retrieval-guidance": "Prefer precise retrieval."}
    result = AgentServingSnapshotLoader().load_mapping(profile)
    assert result.accepted and result.snapshot is not None

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ai-gateway/mlflow/v1/chat/completions"
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "orchestrated"}}]}, request=request
        )

    gateway_config = GatewayConfig.from_mapping(
        {
            "providers": [{"id": "db", "kind": "model_service", "protocols": ["chat_completions"]}],
            "routes": [
                {
                    "id": "primary",
                    "provider": "db",
                    "target": "model-services/simo_prod.ai_models.route_primary",
                    "protocols": ["chat_completions"],
                }
            ],
        }
    )
    orchestrator = ConfiguredAgentOrchestrator(
        result.snapshot,
        gateway_config,
        host="https://workspace.example",
        transport=httpx.MockTransport(handler),
        prompt_sources=profile["prompt_sources"],
        skill_sources=profile["skill_sources"],
    )

    output = orchestrator.run([ChatMessage.from_user("hello")], request={"mode": "safe", "scenario": "daily"})

    assert output.replies[0].text == "orchestrated"
    assert output.meta["profile"] == "default-runtime"
    assert output.meta["nodes"][0]["model_binding"] == "primary"


def test_model_node_trace_factory_receives_route_and_binding_dimensions() -> None:
    profile = valid_profile()
    profile["model_bindings"][0]["protocol"] = "chat_completions"
    result = AgentServingSnapshotLoader().load_mapping(profile)
    assert result.accepted and result.snapshot is not None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "traced"}}]}, request=request
        )

    gateway_config = GatewayConfig.from_mapping(
        {
            "providers": [{"id": "db", "kind": "model_service", "protocols": ["chat_completions"]}],
            "routes": [
                {
                    "id": "primary",
                    "provider": "db",
                    "target": "model-services/simo_prod.ai_models.route_primary",
                    "protocols": ["chat_completions"],
                }
            ],
        }
    )
    orchestrator = ConfiguredAgentOrchestrator(
        result.snapshot,
        gateway_config,
        host="https://workspace.example",
        transport=httpx.MockTransport(handler),
        prompt_sources={"runtime-system": "Mode={{request.mode}}"},
        skill_sources={"retrieval-guidance": "Prefer precise retrieval."},
    )
    spans: list[tuple[dict[str, object], dict[str, object]]] = []

    class Span:
        def __init__(self, metadata: dict[str, object]) -> None:
            self.metadata = metadata

        def __enter__(self) -> "Span":
            return self

        def finish(self, *, attributes=None, error=None) -> None:
            spans.append((self.metadata, {**(attributes or {}), **(error or {})}))

    output = orchestrator.run(
        [ChatMessage.from_user("hello")],
        request={"mode": "safe", "scenario": "daily"},
        trace_span_factory=lambda metadata: Span(dict(metadata)),
    )

    assert output.replies[0].text == "traced"
    assert spans[0][0]["model"] == "primary"
    assert spans[0][0]["protocol"] == "chat_completions"
    assert spans[0][1]["status"] == "completed"


def test_route_id_prefers_binding_id_when_targets_are_shared() -> None:
    profile = valid_profile()
    result = AgentServingSnapshotLoader().load_mapping(profile)
    assert result.accepted and result.snapshot is not None

    gateway_config = GatewayConfig.from_mapping(
        {
            "providers": [{"id": "db", "kind": "model_service", "protocols": ["chat_completions"]}],
            "routes": [
                {
                    "id": "primary",
                    "provider": "db",
                    "target": "model-services/simo_prod.ai_models.route_primary",
                    "protocols": ["chat_completions"],
                },
                {
                    "id": "review",
                    "provider": "db",
                    "target": "model-services/simo_prod.ai_models.route_primary",
                    "protocols": ["chat_completions"],
                },
            ],
        }
    )
    orchestrator = ConfiguredAgentOrchestrator.__new__(ConfiguredAgentOrchestrator)
    orchestrator.snapshot = result.snapshot
    orchestrator.gateway_config = gateway_config

    assert orchestrator._route_id_for_binding("primary") == "primary"


def test_policy_route_mcp_binding_overrides_node_binding() -> None:
    profile = valid_profile()
    profile["model_bindings"][0]["protocol"] = "chat_completions"
    profile["mcp_bindings"].append({"id": "search-tools", "ref": "knowledge-tools", "tools": ["search"]})
    profile["policy_router"]["routes"][0]["mcp_binding"] = "search-tools"
    result = AgentServingSnapshotLoader().load_mapping(profile)
    assert result.accepted and result.snapshot is not None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "orchestrated"}}]}, request=request
        )

    gateway_config = GatewayConfig.from_mapping(
        {
            "providers": [{"id": "db", "kind": "model_service", "protocols": ["chat_completions"]}],
            "routes": [
                {
                    "id": "primary",
                    "provider": "db",
                    "target": "model-services/simo_prod.ai_models.route_primary",
                    "protocols": ["chat_completions"],
                }
            ],
        }
    )
    search_tool = Tool(
        name="search",
        description="Search the approved fixture",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        function=lambda query: f"found:{query}",
    )
    orchestrator = ConfiguredAgentOrchestrator(
        result.snapshot,
        gateway_config,
        host="https://workspace.example",
        transport=httpx.MockTransport(handler),
        prompt_sources={"runtime-system": "Mode={{request.mode}}"},
        skill_sources={"retrieval-guidance": "Use the approved search tool."},
        toolsets={"search-tools": Toolset(tools=[search_tool])},
    )

    output = orchestrator.run([ChatMessage.from_user("hello")], request={"mode": "safe", "scenario": "daily"})

    assert output.meta["nodes"][0]["mcp_binding"] == "search-tools"
    assert output.meta["nodes"][0]["tool_count"] == 1


def test_graph_conditions_select_one_configured_branch() -> None:
    profile = valid_profile()
    profile["agent_graph"]["nodes"] = [
        {"id": "request", "kind": "input"},
        {
            "id": "safe",
            "kind": "model",
            "model_binding": "primary",
            "policy_route": "default",
            "prompt_bindings": ["runtime-system"],
        },
        {
            "id": "fallback",
            "kind": "model",
            "model_binding": "primary",
            "policy_route": "default",
            "prompt_bindings": ["runtime-system"],
        },
        {"id": "response", "kind": "output"},
        {"id": "router", "kind": "router"},
    ]
    profile["agent_graph"]["edges"] = [
        {"source": "request", "target": "router"},
        {"source": "router", "target": "safe", "condition": "request.mode == 'safe'"},
        {"source": "router", "target": "fallback", "condition": "request.mode != 'safe'"},
        {"source": "safe", "target": "response"},
        {"source": "fallback", "target": "response"},
    ]
    profile["model_bindings"][0]["protocol"] = "chat_completions"
    profile["prompt_sources"] = {"runtime-system": "Mode={{request.mode}}"}
    result = AgentServingSnapshotLoader().load_mapping(profile)
    assert result.accepted and result.snapshot is not None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "branched"}}]}, request=request
        )

    gateway_config = GatewayConfig.from_mapping(
        {
            "providers": [{"id": "db", "kind": "model_service", "protocols": ["chat_completions"]}],
            "routes": [
                {
                    "id": "primary",
                    "provider": "db",
                    "target": "model-services/simo_prod.ai_models.route_primary",
                    "protocols": ["chat_completions"],
                }
            ],
        }
    )
    orchestrator = ConfiguredAgentOrchestrator(
        result.snapshot,
        gateway_config,
        host="https://workspace.example",
        transport=httpx.MockTransport(handler),
        prompt_sources=profile["prompt_sources"],
    )

    output = orchestrator.run([ChatMessage.from_user("hello")], request={"mode": "safe"})

    assert output.meta["graph_path"] == ["request", "router", "safe", "response"]
    assert output.state["schema_version"] == 1
    assert output.state["graph_path"] == output.meta["graph_path"]


class _RecordingGenerator:
    def __init__(self, replies: list[ChatMessage], *, on_run=None) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[ChatMessage], object]] = []
        self.on_run = on_run

    def run(self, messages, tools=None, streaming_callback=None, **kwargs):
        self.calls.append((list(messages), tools))
        if self.on_run is not None:
            self.on_run(messages, tools)
        reply = self.replies.pop(0) if self.replies else ChatMessage.from_assistant("done")
        return {"replies": [reply]}


def _multi_agent_profile() -> dict:
    profile = valid_profile()
    profile["agent_graph"] = {
        "entry_node": "request",
        "terminal_nodes": ["response"],
        "nodes": [
            {"id": "request", "kind": "input"},
            {
                "id": "coordinator",
                "kind": "model",
                "model_binding": "coordinator",
                "prompt_bindings": ["coordinator-prompt"],
                "agent_tools": ["review-tool", "coding-tool"],
                "policy_route": "coding",
            },
            {
                "id": "review",
                "kind": "model",
                "model_binding": "review",
                "prompt_bindings": ["review-prompt"],
            },
            {
                "id": "coder",
                "kind": "model",
                "model_binding": "coder",
                "mcp_binding": "knowledge",
                "prompt_bindings": ["coder-prompt"],
            },
            {"id": "response", "kind": "output"},
        ],
        "edges": [
            {"source": "request", "target": "coordinator"},
            {"source": "coordinator", "target": "response"},
        ],
    }
    profile["agent_tools"] = [
        {
            "id": "review-tool",
            "node": "review",
            "name": "review_code",
            "description": "Review code.",
            "outputs_to_state": {
                "review_findings": {"source": "last_message", "handler": "message_text"}
            },
        },
        {
            "id": "coding-tool",
            "node": "coder",
            "name": "implement_code",
            "description": "Implement code using the review.",
            "inputs_from_state": {"review_findings": "review_context"},
        },
    ]
    profile["state_schema"] = {"review_findings": {"type": "string", "persist": True}}
    profile["model_bindings"] = [
        {
            "id": "coordinator",
            "ref": "serving-endpoints/coordinator",
            "protocol": "chat_completions",
            "tools": True,
        },
        {
            "id": "review",
            "ref": "serving-endpoints/reviewer",
            "protocol": "chat_completions",
            "tools": False,
        },
        {
            "id": "coder",
            "ref": "serving-endpoints/coder",
            "protocol": "chat_completions",
            "tools": True,
        },
    ]
    profile["policy_router"]["routes"] = [
        {
            "id": "coding",
            "priority": 1,
            "when": {"attribute": "request.scenario", "operator": "equals", "value": "coding"},
            "action": "allow",
            "model_binding": "review",
        }
    ]
    profile["prompt_bindings"] = [
        {
            "id": prompt_id,
            "role": "system",
            "ref": prompt_id,
            "allowed_variables": [],
            "user_override": False,
            "max_length": 4000,
        }
        for prompt_id in ("coordinator-prompt", "review-prompt", "coder-prompt")
    ]
    profile["prompt_sources"] = {
        "coordinator-prompt": "Delegate to the specialists.",
        "review-prompt": "You are the review specialist.",
        "coder-prompt": "You are the coding specialist.",
    }
    return profile


def test_coordinator_agent_tools_transfer_review_state_to_coder() -> None:
    profile = _multi_agent_profile()
    result = AgentServingSnapshotLoader().load_mapping(profile)
    assert result.accepted and result.snapshot is not None, result.issues

    review_generator = _RecordingGenerator([ChatMessage.from_assistant("REVIEW_FINDINGS")])
    coder_messages: list[list[ChatMessage]] = []
    coder_generator = _RecordingGenerator(
        [ChatMessage.from_assistant("IMPLEMENTED")],
        on_run=lambda messages, _tools: coder_messages.append(list(messages)),
    )
    coordinator_generator = _RecordingGenerator(
        [
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(
                        tool_name="review_code",
                        arguments={"messages": [{"role": "user", "content": "Review this change."}]},
                        id="review-call",
                    )
                ]
            ),
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(
                        tool_name="implement_code",
                        arguments={"messages": [{"role": "user", "content": "Implement this change."}]},
                        id="coding-call",
                    )
                ]
            ),
            ChatMessage.from_assistant("FINAL_RESULT"),
        ],
    )

    search_tool = Tool(
        name="search",
        description="Search approved context.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        function=lambda query: f"found:{query}",
    )
    orchestrator = ConfiguredAgentOrchestrator.__new__(ConfiguredAgentOrchestrator)
    orchestrator.snapshot = result.snapshot
    orchestrator.gateway_config = GatewayConfig.from_mapping({"providers": [], "routes": []})
    orchestrator.prompt_sources = profile["prompt_sources"]
    orchestrator.skill_sources = {}
    orchestrator.toolsets = {"knowledge": Toolset(tools=[search_tool])}
    orchestrator._generators = {
        "coordinator": coordinator_generator,
        "review": review_generator,
        "coder": coder_generator,
    }

    output = orchestrator.run([ChatMessage.from_user("Fix the code")], request={"scenario": "coding"})

    assert output.replies[0].text == "FINAL_RESULT"
    assert output.state["review_findings"] == "REVIEW_FINDINGS"
    assert coder_messages
    assert any("REVIEW_FINDINGS" in (message.text or "") for message in coder_messages[0])
    coordinator_tools = coordinator_generator.calls[0][1]
    assert {tool.name for tool in coordinator_tools} == {"review_code", "implement_code"}
    assert coder_generator.calls[0][1][0].name == "search"


def test_coordinator_keeps_specialist_model_and_mcp_bindings() -> None:
    profile = _multi_agent_profile()
    result = AgentServingSnapshotLoader().load_mapping(profile)
    assert result.accepted and result.snapshot is not None
    orchestrator = ConfiguredAgentOrchestrator.__new__(ConfiguredAgentOrchestrator)
    orchestrator.snapshot = result.snapshot
    orchestrator.gateway_config = GatewayConfig.from_mapping({"providers": [], "routes": []})
    orchestrator.prompt_sources = profile["prompt_sources"]
    orchestrator.skill_sources = {}
    orchestrator.toolsets = {}
    orchestrator._generators = {
        "coordinator": _RecordingGenerator([ChatMessage.from_assistant("done")]),
        "review": _RecordingGenerator([ChatMessage.from_assistant("review")]),
        "coder": _RecordingGenerator([ChatMessage.from_assistant("code")]),
    }

    tools = orchestrator._agent_tools_for_node(
        result.snapshot.agent_graph.nodes[1],
        {"scenario": "coding"},
        trace_span_factory=None,
        building=frozenset(),
    )

    assert [tool.name for tool in tools] == ["review_code", "implement_code"]
    assert orchestrator._generators["review"] is not orchestrator._generators["coordinator"]
    assert orchestrator._generators["coder"] is not orchestrator._generators["coordinator"]
