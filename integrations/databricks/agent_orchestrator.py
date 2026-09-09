"""Profile-driven Haystack Agent orchestration for Databricks routes."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from contracts.runtime.haystack_agent_serving.loader import (
    AgentServingSnapshot,
    AgentToolBinding,
    GraphNode,
    PolicyRoute,
)
from haystack.components.agents import Agent as HaystackAgent
from haystack.components.agents.state.state_utils import merge_lists
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import AgentTool, Tool, Toolset, ToolsType

from .ai_gateway import GatewayConfig
from .haystack_chat_generator import DatabricksChatGenerator, build_gateway_runtime


class AgentOrchestrationError(RuntimeError):
    """Raised when a profile cannot execute its configured graph."""


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    replies: tuple[ChatMessage, ...]
    meta: Mapping[str, Any]
    state: Mapping[str, Any] = field(default_factory=dict)


class ConfiguredAgentOrchestrator:
    """
    Execute a validated graph with route-, prompt-, skill-, and MCP bindings.

    The orchestrator intentionally receives resolved prompt bodies, skill
    bodies, and toolsets from the deployment boundary.  The serving profile
    selects *which* resources are allowed; it never contains Python imports or
    credentials and therefore cannot create arbitrary tools at runtime.
    """

    def __init__(
        self,
        snapshot: AgentServingSnapshot,
        gateway_config: GatewayConfig | Mapping[str, Any],
        *,
        host: str | None = None,
        token: str | None = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        prompt_sources: Mapping[str, str] | None = None,
        skill_sources: Mapping[str, str] | None = None,
        toolsets: Mapping[str, ToolsType] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.gateway_config = gateway_config
        self.runtime = build_gateway_runtime(
            gateway_config, host=host, token=token, headers=headers, transport=transport
        )
        self.prompt_sources = dict(prompt_sources or {})
        self.skill_sources = dict(skill_sources or {})
        self.toolsets = dict(toolsets or {})
        self._generators: dict[str, DatabricksChatGenerator] = {}

    def run(
        self,
        messages: list[ChatMessage],
        *,
        request: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        trace_span_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> OrchestrationResult:
        """Run the selected Haystack Agent nodes in graph order."""

        request = dict(request or {})
        policy = self._select_policy(request)
        if policy.action != "allow":
            raise AgentOrchestrationError(f"request blocked by policy: {policy.action}")

        current_messages = list(messages)
        session_state = _copy_state(state)
        node_results: list[dict[str, Any]] = []
        graph_nodes = {node.id: node for node in self.snapshot.agent_graph.nodes}
        current_node_id = self.snapshot.agent_graph.entry_node
        graph_path: list[str] = []
        max_graph_steps = max(16, len(graph_nodes) * 8)
        while len(graph_path) < max_graph_steps:
            node = graph_nodes.get(current_node_id)
            if node is None:
                raise AgentOrchestrationError(f"graph references unknown node: {current_node_id}")
            graph_path.append(node.id)
            context = {
                "request": request,
                "state": session_state,
                "node": {"id": node.id, "kind": node.kind},
                "last": session_state.get("last_result", {}),
            }
            if node.kind == "model":
                if not node.policy_route or node.policy_route == policy.id:
                    current_messages, node_result = self._run_model_node(
                        node,
                        policy,
                        request,
                        current_messages,
                        state=session_state,
                        streaming_callback=streaming_callback,
                        trace_span_factory=trace_span_factory,
                    )
                    node_results.append(node_result)
                    session_state["last_result"] = dict(node_result)
                    session_state["step_count"] = int(session_state.get("step_count", 0)) + int(
                        node_result.get("step_count") or 0
                    )
                    session_state["token_usage"] = _merge_usage(
                        session_state.get("token_usage", {}), node_result.get("token_usage", {})
                    )
                    session_state["tool_call_counts"] = _merge_counts(
                        session_state.get("tool_call_counts", {}), node_result.get("tool_call_counts", {})
                    )
                    session_state["exit_reason"] = node_result.get("exit_reason")
                    for key, value in node_result.get("state_updates", {}).items():
                        session_state[key] = value
            elif node.kind in {"input", "output", "router"}:
                pass
            else:
                raise AgentOrchestrationError(
                    f"graph node kind '{node.kind}' is not executable; use a model node with configured tools"
                )
            if node.id in self.snapshot.agent_graph.terminal_nodes:
                break
            next_node = self._next_graph_node(node.id, context={**context, "state": session_state})
            if next_node is None:
                raise AgentOrchestrationError(f"graph terminated before terminal node: {node.id}")
            current_node_id = next_node
        else:
            raise AgentOrchestrationError(f"graph exceeded {max_graph_steps} transitions")

        if not current_messages:
            raise AgentOrchestrationError("agent graph produced no messages")
        reply = current_messages[-1]
        persisted_state = {
            key: session_state[key]
            for key, definition in self.snapshot.state_schema.items()
            if definition.get("persist", True) and key in session_state
        }
        return OrchestrationResult(
            replies=(reply,),
            meta={
                "profile": self.snapshot.profile.name,
                "profile_version": self.snapshot.profile.version,
                "policy_route": policy.id,
                "scenario": request.get("scenario"),
                "nodes": node_results,
                "graph_path": graph_path,
                "terminal_node": graph_path[-1],
            },
            state={
                "schema_version": 1,
                "messages": [message.to_openai_dict_format() for message in current_messages],
                "graph_path": graph_path,
                "node_results": node_results,
                "step_count": session_state.get("step_count", 0),
                "token_usage": session_state.get("token_usage", {}),
                "tool_call_counts": session_state.get("tool_call_counts", {}),
                "exit_reason": session_state.get("exit_reason"),
                **persisted_state,
            },
        )

    def _run_model_node(
        self,
        node: GraphNode,
        policy: PolicyRoute,
        request: Mapping[str, Any],
        messages: list[ChatMessage],
        *,
        state: Mapping[str, Any],
        streaming_callback: StreamingCallbackT | None,
        trace_span_factory: Callable[[Mapping[str, Any]], Any] | None,
    ) -> tuple[list[ChatMessage], dict[str, Any]]:
        binding_id = node.model_binding or policy.model_binding
        if not binding_id:
            raise AgentOrchestrationError(f"model node has no model binding: {node.id}")
        binding = self.snapshot.model_bindings.get(binding_id)
        if binding is None:
            raise AgentOrchestrationError(f"unknown model binding: {binding_id}")
        generator = self._generator(binding_id, binding.protocol)
        mcp_binding_id = policy.mcp_binding or node.mcp_binding
        tools = self._tools_for_node(node, mcp_binding_id=mcp_binding_id)
        delegated_tools = self._agent_tools_for_node(
            node,
            request,
            trace_span_factory=trace_span_factory,
            building=frozenset(),
        )
        tools = _combine_tools(tools, delegated_tools)
        prompt = self._prompt_for_node(node, request)
        if delegated_tools:
            delegate_names = ", ".join(tool.name for tool in delegated_tools)
            prompt = (prompt + "\n\n" if prompt else "") + (
                f"You are a coordinator. Delegate the task to these specialist tools when appropriate: {delegate_names}."
            )
        agent = HaystackAgent(
            chat_generator=generator,
            tools=tools,
            system_prompt=prompt,
            max_agent_steps=policy.max_steps or 8,
            state_schema=self._haystack_state_schema(),
        )
        route_id = self._route_id_for_binding(binding_id)
        provider_metadata = self._provider_metadata(route_id, binding.ref)
        trace_metadata = {
            "node_id": node.id,
            "model": binding_id,
            "model_target": binding.ref,
            "route": route_id,
            "protocol": binding.protocol,
            **provider_metadata,
            "mcp_binding": mcp_binding_id,
            "skill_binding": list(node.skill_bindings),
            "prompt_binding_ids": list(node.prompt_bindings),
            "tool_count": len(self._tool_list(tools)),
            "tool_names": [tool.name for tool in self._tool_list(tools)],
        }
        node_span = trace_span_factory(trace_metadata) if trace_span_factory else None
        if node_span is not None:
            node_span.__enter__()
        try:
            state_kwargs = {
                key: state[key] for key in self.snapshot.state_schema if key in state and key != "messages"
            }
            result = agent.run(
                messages=messages,
                streaming_callback=streaming_callback,
                orchestration_context={"node_id": node.id, "scenario": request.get("scenario")},
                **state_kwargs,
            )
        except Exception as error:
            if node_span is not None:
                node_span.finish(error={"exception_type": type(error).__name__, "error_code": "node_error"})
            raise
        else:
            if node_span is not None:
                node_span.finish(
                    attributes={
                        "status": "completed",
                        "step_count": result.get("step_count"),
                        "token_usage": result.get("token_usage", {}),
                        "tool_call_counts": result.get("tool_call_counts", {}),
                        "exit_reason": result.get("exit_reason"),
                    }
                )
        current_messages = list(result.get("messages", messages))
        last_message = result.get("last_message")
        if isinstance(last_message, ChatMessage):
            current_messages = current_messages or [last_message]
        generator_meta = getattr(generator, "last_metadata", {})
        return current_messages, {
            **trace_metadata,
            "model_binding": binding_id,
            "skill_bindings": list(node.skill_bindings),
            "prompt_bindings": list(node.prompt_bindings),
            "step_count": result.get("step_count"),
            "token_usage": result.get("token_usage", {}),
            "tool_call_counts": result.get("tool_call_counts", {}),
            "exit_reason": result.get("exit_reason"),
            "state_updates": {
                key: result[key] for key in self.snapshot.state_schema if key in result and key != "messages"
            },
            "route_attempts": generator_meta.get("route_attempts", []),
        }

    def _agent_tools_for_node(
        self,
        node: GraphNode,
        request: Mapping[str, Any],
        *,
        trace_span_factory: Callable[[Mapping[str, Any]], Any] | None,
        building: frozenset[str],
    ) -> list[AgentTool]:
        tools: list[AgentTool] = []
        for binding_id in node.agent_tools:
            if binding_id in building:
                raise AgentOrchestrationError(f"agent tool delegation cycle detected at: {binding_id}")
            binding = self.snapshot.agent_tools.get(binding_id)
            if binding is None:
                raise AgentOrchestrationError(f"unknown agent tool binding: {binding_id}")
            delegate_node = next(
                (candidate for candidate in self.snapshot.agent_graph.nodes if candidate.id == binding.node), None
            )
            if delegate_node is None or delegate_node.kind != "model":
                raise AgentOrchestrationError(f"agent tool target must be a model node: {binding.node}")
            delegate_binding_id = delegate_node.model_binding
            if not delegate_binding_id:
                raise AgentOrchestrationError(f"agent tool target has no model binding: {binding.node}")
            delegate_binding = self.snapshot.model_bindings.get(delegate_binding_id)
            if delegate_binding is None:
                raise AgentOrchestrationError(f"unknown delegated model binding: {delegate_binding_id}")
            delegate_generator = self._generator(delegate_binding_id, delegate_binding.protocol)
            delegate_mcp = self._tools_for_node(delegate_node, mcp_binding_id=delegate_node.mcp_binding)
            nested = self._agent_tools_for_node(
                delegate_node,
                request,
                trace_span_factory=trace_span_factory,
                building=building | {binding_id},
            )
            delegate_tools = _combine_tools(delegate_mcp, nested)
            delegate_prompt = self._prompt_for_node(delegate_node, request)
            input_names = tuple(dict.fromkeys(binding.inputs_from_state.values()))
            delegate_user_prompt = self._state_input_prompt(input_names)
            delegate_agent = HaystackAgent(
                chat_generator=delegate_generator,
                tools=delegate_tools,
                system_prompt=delegate_prompt,
                user_prompt=delegate_user_prompt,
                required_variables=list(input_names) if input_names else None,
                max_agent_steps=8,
                state_schema=self._haystack_state_schema(exclude=set(input_names)),
            )
            outputs_to_state = self._outputs_to_state(binding)
            tools.append(
                AgentTool(
                    agent=delegate_agent,
                    name=binding.name,
                    description=binding.description,
                    inputs_from_state=dict(binding.inputs_from_state) or None,
                    outputs_to_state=outputs_to_state or None,
                )
            )
        return tools

    def _state_input_prompt(self, input_names: Sequence[str]) -> str | None:
        if not input_names:
            return None
        sections = [f"Specialist context ({name}): {{{{ {name} }}}}" for name in input_names]
        return "\n\n".join(sections)

    def _outputs_to_state(self, binding: AgentToolBinding) -> dict[str, dict[str, Any]]:
        outputs: dict[str, dict[str, Any]] = {}
        for state_key, raw_config in binding.outputs_to_state.items():
            config = dict(raw_config)
            handler_name = config.get("handler")
            source = config.get("source")
            if handler_name is None and source == "last_message":
                field = self.snapshot.state_schema.get(state_key, {})
                if field.get("type") in {"string", "str"}:
                    handler_name = "message_text"
            if handler_name is not None:
                config["handler"] = _state_output_handler(str(handler_name))
            outputs[state_key] = config
        return outputs

    def _haystack_state_schema(self, *, exclude: set[str] | None = None) -> dict[str, dict[str, Any]]:
        excluded = exclude or set()
        schema: dict[str, dict[str, Any]] = {
            "orchestration_context": {"type": dict},
        }
        for key, definition in self.snapshot.state_schema.items():
            if key in excluded:
                continue
            value_type = _state_type(definition.get("type"))
            item = {"type": value_type}
            if definition.get("handler") == "append":
                if value_type is not list:
                    raise AgentOrchestrationError(f"append state handler requires a list field: {key}")
                item["handler"] = merge_lists
            schema[key] = item
        return schema

    def _generator(self, binding_id: str, protocol: str) -> DatabricksChatGenerator:
        if binding_id not in self._generators:
            route_id = self._route_id_for_binding(binding_id)
            self._generators[binding_id] = DatabricksChatGenerator(self.runtime, protocol=protocol, route_id=route_id)
        return self._generators[binding_id]

    def _route_id_for_binding(self, binding_id: str) -> str | None:
        binding = self.snapshot.model_bindings[binding_id]
        config = self.gateway_config
        routes = config.routes if isinstance(config, GatewayConfig) else GatewayConfig.from_mapping(config).routes
        for route in routes:
            if route.id == binding_id:
                return route.id
        for route in routes:
            if route.target == binding.ref:
                return route.id
        return binding_id

    def _provider_metadata(self, route_id: str, target: str) -> dict[str, str]:
        config = self.gateway_config
        config = config if isinstance(config, GatewayConfig) else GatewayConfig.from_mapping(config)
        route = next((item for item in config.routes if item.id == route_id or item.target == target), None)
        if route is None:
            return {}
        provider = next((item for item in config.providers if item.id == route.provider), None)
        metadata = {"provider": route.provider}
        if provider is not None:
            metadata["provider_kind"] = provider.kind
        return metadata

    def _next_graph_node(self, source: str, *, context: Mapping[str, Any]) -> str | None:
        outgoing = [edge for edge in self.snapshot.agent_graph.edges if edge.source == source]
        for edge in outgoing:
            if _edge_condition_matches(edge.condition, context):
                return edge.target
        return None

    def _select_policy(self, request: Mapping[str, Any]) -> PolicyRoute:
        for route in self.snapshot.policy_router.routes:
            if _condition_matches(route.when, request):
                return route
        return PolicyRoute(
            id="implicit-default", priority=2**31 - 1, when={}, action=self.snapshot.policy_router.default_action
        )

    def _prompt_for_node(self, node: GraphNode, request: Mapping[str, Any]) -> str | None:
        sections: list[str] = []
        variables = {f"request.{key}": str(value) for key, value in request.items() if value is not None}
        for binding_id in node.prompt_bindings:
            binding = self.snapshot.prompt_bindings[binding_id]
            template = self.prompt_sources.get(binding_id) or self.prompt_sources.get(binding.ref)
            if template is None:
                raise AgentOrchestrationError(f"prompt body is not supplied for binding: {binding_id}")
            prompt_variables = {key: variables[key] for key in binding.allowed_variables if key in variables}
            sections.append(binding.render(template, prompt_variables))
        for binding_id in node.skill_bindings:
            binding = self.snapshot.skill_bindings[binding_id]
            skill = self.snapshot.skills_registry.skill(binding.ref)
            body = self.skill_sources.get(binding_id) or self.skill_sources.get(skill.id)
            if body is None:
                if binding.required:
                    raise AgentOrchestrationError(f"required skill body is not supplied: {binding_id}")
                continue
            sections.append(f"# Skill: {skill.id}\n{body}")
        return "\n\n".join(sections) or None

    def _tools_for_node(self, node: GraphNode, *, mcp_binding_id: str | None = None) -> ToolsType | None:
        selected_binding = mcp_binding_id or node.mcp_binding
        if not selected_binding:
            return None
        binding = self.snapshot.mcp_bindings[selected_binding]
        server = self.snapshot.mcp_registry.server(binding.ref)
        configured = self.toolsets.get(selected_binding)
        if configured is None:
            configured = self.toolsets.get(binding.ref)
        if configured is None:
            return None
        allowed = set(binding.tools) or set(server.allowed_tools)
        if isinstance(configured, Toolset):
            configured.warm_up()
            return [tool for tool in configured if tool.name in allowed]
        if isinstance(configured, list):
            return [tool for tool in configured if isinstance(tool, Tool) and tool.name in allowed]
        return configured

    @staticmethod
    def _tool_list(tools: ToolsType | None) -> list[Tool]:
        if tools is None:
            return []
        if isinstance(tools, Toolset):
            return [tool for tool in tools if isinstance(tool, Tool)]
        return [tool for tool in tools if isinstance(tool, Tool)] if isinstance(tools, list) else []


def _condition_matches(condition: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    if not condition:
        return True
    attribute = condition.get("attribute")
    if not isinstance(attribute, str):
        return False
    value: Any = request
    for part in attribute.split("."):
        if part == "request":
            continue
        if not isinstance(value, Mapping):
            return False
        value = value.get(part)
    operator = condition.get("operator", "exists")
    expected = condition.get("value")
    if operator == "exists":
        return value is not None
    if operator in {"equals", "eq"}:
        return value == expected
    if operator == "in":
        return isinstance(expected, Sequence) and value in expected
    return operator == "not_equals" and value != expected


_EDGE_CONDITION_RE = re.compile(
    r"^(?P<path>[A-Za-z_][A-Za-z0-9_.-]*)\s*(?P<operator>==|!=|in|contains|exists)\s*(?P<literal>.+)?$"
)


def _edge_condition_matches(condition: str | None, context: Mapping[str, Any]) -> bool:
    """Evaluate the intentionally small, safe edge-condition language."""

    if not condition or not condition.strip():
        return True
    match = _EDGE_CONDITION_RE.fullmatch(condition.strip())
    if match is None:
        raise AgentOrchestrationError(f"invalid graph edge condition: {condition}")
    value = _context_value(context, match.group("path"))
    operator = match.group("operator")
    if operator == "exists":
        return value is not None
    literal = match.group("literal")
    if literal is None:
        raise AgentOrchestrationError(f"graph edge condition requires a value: {condition}")
    expected = _parse_condition_literal(literal)
    if operator == "==":
        return value == expected
    if operator == "!=":
        return value != expected
    if operator == "in":
        return isinstance(expected, Sequence) and value in expected
    return isinstance(value, str) and str(expected) in value


def _context_value(context: Mapping[str, Any], path: str) -> Any:
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _parse_condition_literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip('"\'')


def _copy_state(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _state_output_handler(name: str) -> Callable[[Any, Any], Any]:
    handlers: dict[str, Callable[[Any, Any], Any]] = {
        "identity": _replace_state_value,
        "string": _string_state_value,
        "message_text": _message_text_state_value,
    }
    try:
        return handlers[name]
    except KeyError as error:
        raise AgentOrchestrationError(f"unsupported AgentTool state output handler: {name}") from error


def _replace_state_value(_current: Any, value: Any) -> Any:
    return value


def _string_state_value(_current: Any, value: Any) -> str:
    return str(value)


def _message_text_state_value(_current: Any, value: Any) -> str:
    if isinstance(value, ChatMessage):
        if value.text is not None:
            return value.text
        return json.dumps(value.to_dict(), ensure_ascii=False, default=str)
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)


def _combine_tools(primary: ToolsType | None, delegated: Sequence[Tool]) -> ToolsType | None:
    primary_tools = ConfiguredAgentOrchestrator._tool_list(primary)
    combined = [*primary_tools, *delegated]
    return combined or None


def _state_type(value: Any) -> type:
    types = {
        "string": str,
        "str": str,
        "integer": int,
        "int": int,
        "number": float,
        "float": float,
        "boolean": bool,
        "bool": bool,
        "object": dict,
        "dict": dict,
        "array": list,
        "list": list,
    }
    try:
        return types[str(value)]
    except KeyError as error:
        raise AgentOrchestrationError(f"unsupported state field type: {value}") from error


def _merge_usage(current: Any, update: Any) -> dict[str, int]:
    result = {str(key): int(value) for key, value in current.items()} if isinstance(current, Mapping) else {}
    if isinstance(update, Mapping):
        for key, value in update.items():
            if isinstance(value, (int, float)):
                result[str(key)] = result.get(str(key), 0) + int(value)
    return result


def _merge_counts(current: Any, update: Any) -> dict[str, int]:
    return _merge_usage(current, update)


__all__ = ["AgentOrchestrationError", "ConfiguredAgentOrchestrator", "OrchestrationResult"]
