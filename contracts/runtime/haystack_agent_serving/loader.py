"""Pure-local loading and validation for Haystack agent serving profiles."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator

from contracts.runtime.haystack_agent_serving.registries import (
    McpRegistrySnapshot,
    ObservabilityPolicy,
    PromptBinding,
    RegistryValidationError,
    SkillsRegistrySnapshot,
    build_registry_state,
    registry_state_to_dict,
)

_SCHEMA_PATH = Path(__file__).with_name("schema") / "agent-serving.schema.json"
_MODEL_REF_PATTERNS = (
    re.compile(r"^model-services/simo_prod\.ai_models\.route_[a-z0-9][a-z0-9_-]{0,127}$"),
    re.compile(r"^model-services/simo_prod\.ai_gateway\.[a-z0-9][a-z0-9_-]{0,127}$"),
    re.compile(r"^serving-endpoints?/[A-Za-z0-9][A-Za-z0-9_.-]{0,126}$"),
)
_LOGICAL_REF_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_FORBIDDEN_KEYS = {
    "callable",
    "entrypoint",
    "import",
    "module",
    "python",
    "python_import",
    "supervisor",
    "supervisor_agent",
}
_FORBIDDEN_RESOURCE_PREFIX = "agent_"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    message: str


class SnapshotValidationError(ValueError):
    """Raised when no valid snapshot exists for an invalid initial load."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        super().__init__(details or "snapshot validation failed")


@dataclass(frozen=True, slots=True)
class ModelBinding:
    id: str
    ref: str
    protocol: str
    streaming: bool = False
    tools: bool = False
    structured_output: bool = False


@dataclass(frozen=True, slots=True)
class McpBinding:
    id: str
    ref: str
    tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillBinding:
    id: str
    ref: str
    required: bool = False


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    kind: str
    model_binding: str | None = None
    mcp_binding: str | None = None
    skill_bindings: tuple[str, ...] = ()
    prompt_bindings: tuple[str, ...] = ()
    policy_route: str | None = None
    agent_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentToolBinding:
    id: str
    node: str
    name: str
    description: str
    inputs_from_state: Mapping[str, str] = MappingProxyType({})
    outputs_to_state: Mapping[str, Mapping[str, Any]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    condition: str | None = None


@dataclass(frozen=True, slots=True)
class AgentGraph:
    entry_node: str
    terminal_nodes: tuple[str, ...]
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


@dataclass(frozen=True, slots=True)
class PolicyRoute:
    id: str
    priority: int
    when: Mapping[str, Any]
    action: str
    model_binding: str | None = None
    mcp_binding: str | None = None
    max_steps: int | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class PolicyRouter:
    default_action: str
    routes: tuple[PolicyRoute, ...]
    router_model_binding: str | None = None


@dataclass(frozen=True, slots=True)
class AgentProfile:
    name: str
    version: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class AgentServingSnapshot:
    schema_version: str
    profile: AgentProfile
    agent_graph: AgentGraph
    policy_router: PolicyRouter
    model_bindings: Mapping[str, ModelBinding]
    mcp_bindings: Mapping[str, McpBinding]
    skill_bindings: Mapping[str, SkillBinding]
    mcp_registry: McpRegistrySnapshot
    skills_registry: SkillsRegistrySnapshot
    prompt_bindings: Mapping[str, PromptBinding]
    observability: ObservabilityPolicy
    content_sha256: str
    agent_tools: Mapping[str, AgentToolBinding] = MappingProxyType({})
    state_schema: Mapping[str, Mapping[str, Any]] = MappingProxyType({})

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the immutable snapshot."""

        return {
            "schema_version": self.schema_version,
            "profile": {
                "name": self.profile.name,
                "version": self.profile.version,
                **({"description": self.profile.description} if self.profile.description is not None else {}),
            },
            "agent_graph": {
                "entry_node": self.agent_graph.entry_node,
                "terminal_nodes": list(self.agent_graph.terminal_nodes),
                "nodes": [
                    {
                        "id": node.id,
                        "kind": node.kind,
                        **({"model_binding": node.model_binding} if node.model_binding is not None else {}),
                        **({"mcp_binding": node.mcp_binding} if node.mcp_binding is not None else {}),
                        **({"skill_bindings": list(node.skill_bindings)} if node.skill_bindings else {}),
                        **({"prompt_bindings": list(node.prompt_bindings)} if node.prompt_bindings else {}),
                        **({"policy_route": node.policy_route} if node.policy_route is not None else {}),
                        **({"agent_tools": list(node.agent_tools)} if node.agent_tools else {}),
                    }
                    for node in self.agent_graph.nodes
                ],
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        **({"condition": edge.condition} if edge.condition is not None else {}),
                    }
                    for edge in self.agent_graph.edges
                ],
            },
            "policy_router": {
                "default_action": self.policy_router.default_action,
                **(
                    {"router_model_binding": self.policy_router.router_model_binding}
                    if self.policy_router.router_model_binding is not None
                    else {}
                ),
                "routes": [
                    {
                        "id": route.id,
                        "priority": route.priority,
                        "when": _thaw(route.when),
                        "action": route.action,
                        **({"model_binding": route.model_binding} if route.model_binding is not None else {}),
                        **({"mcp_binding": route.mcp_binding} if route.mcp_binding is not None else {}),
                        **({"max_steps": route.max_steps} if route.max_steps is not None else {}),
                        **({"max_tokens": route.max_tokens} if route.max_tokens is not None else {}),
                    }
                    for route in self.policy_router.routes
                ],
            },
            "model_bindings": [
                {
                    "id": binding.id,
                    "ref": binding.ref,
                    "protocol": binding.protocol,
                    "streaming": binding.streaming,
                    "tools": binding.tools,
                    "structured_output": binding.structured_output,
                }
                for binding in self.model_bindings.values()
            ],
            "agent_tools": [
                {
                    "id": binding.id,
                    "node": binding.node,
                    "name": binding.name,
                    "description": binding.description,
                    "inputs_from_state": dict(binding.inputs_from_state),
                    "outputs_to_state": {key: dict(value) for key, value in binding.outputs_to_state.items()},
                }
                for binding in self.agent_tools.values()
            ],
            "state_schema": {key: dict(value) for key, value in self.state_schema.items()},
            "mcp_bindings": [
                {"id": binding.id, "ref": binding.ref, "tools": list(binding.tools)}
                for binding in self.mcp_bindings.values()
            ],
            "skill_bindings": [
                {"id": binding.id, "ref": binding.ref, "required": binding.required}
                for binding in self.skill_bindings.values()
            ],
            **registry_state_to_dict(self.mcp_registry, self.skills_registry, self.prompt_bindings, self.observability),
        }


@dataclass(frozen=True, slots=True)
class LoadResult:
    snapshot: AgentServingSnapshot | None
    accepted: bool
    used_last_known_good: bool
    issues: tuple[ValidationIssue, ...] = ()


class AgentServingSnapshotLoader:
    """Load local JSON profiles and retain the last valid immutable snapshot."""

    def __init__(self, schema_path: str | Path | None = None) -> None:
        selected_schema = Path(schema_path) if schema_path is not None else _SCHEMA_PATH
        self._schema = json.loads(selected_schema.read_text(encoding="utf-8"))
        self._validator = Draft202012Validator(self._schema)
        self._last_known_good: AgentServingSnapshot | None = None

    @property
    def last_known_good(self) -> AgentServingSnapshot | None:
        """Return the most recent accepted snapshot, if one exists."""

        return self._last_known_good

    def load_file(self, path: str | Path) -> LoadResult:
        """Load a local JSON file without contacting Databricks or importing user code."""

        source_path = Path(path)
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return self._reject((ValidationIssue("$", f"cannot read JSON: {error}"),))
        return self.load_mapping(raw)

    def load_mapping(self, raw: Mapping[str, Any]) -> LoadResult:
        """Validate and freeze a detached mapping supplied by local code or tests."""

        issues = self._validate(raw)
        if issues:
            return self._reject(issues)

        try:
            snapshot = _build_snapshot(raw)
        except RegistryValidationError as error:
            return self._reject((ValidationIssue("$", str(error)),))
        self._last_known_good = snapshot
        return LoadResult(snapshot=snapshot, accepted=True, used_last_known_good=False)

    def require_file(self, path: str | Path) -> AgentServingSnapshot:
        """Load a file or raise when there is no valid snapshot to serve."""

        result = self.load_file(path)
        if result.snapshot is None:
            raise SnapshotValidationError(result.issues)
        return result.snapshot

    def _validate(self, raw: Mapping[str, Any]) -> tuple[ValidationIssue, ...]:
        if not isinstance(raw, Mapping):
            return (ValidationIssue("$", "document must be an object"),)

        schema_issues = []
        for error in sorted(self._validator.iter_errors(raw), key=lambda item: list(item.absolute_path)):
            path = _json_path(error.absolute_path)
            if path.startswith("$/model_bindings/") and path.endswith("/ref"):
                schema_issues.append(ValidationIssue(path, _model_ref_error(raw, error.absolute_path)))
            else:
                schema_issues.append(ValidationIssue(path, error.message))
        if schema_issues:
            return _dedupe_issues(schema_issues)
        return _dedupe_issues(_semantic_issues(raw))

    def _reject(self, issues: Sequence[ValidationIssue]) -> LoadResult:
        frozen_issues = tuple(issues)
        if self._last_known_good is not None:
            return LoadResult(
                snapshot=self._last_known_good, accepted=False, used_last_known_good=True, issues=frozen_issues
            )
        return LoadResult(snapshot=None, accepted=False, used_last_known_good=False, issues=frozen_issues)


def _build_snapshot(raw: Mapping[str, Any]) -> AgentServingSnapshot:
    profile_data = raw["profile"]
    graph_data = raw["agent_graph"]
    router_data = raw["policy_router"]
    mcp_registry, skills_registry, prompt_bindings, observability = build_registry_state(raw)
    model_bindings = {
        item["id"]: ModelBinding(
            id=item["id"],
            ref=item["ref"],
            protocol=item["protocol"],
            streaming=item.get("streaming", False),
            tools=item.get("tools", False),
            structured_output=item.get("structured_output", False),
        )
        for item in raw["model_bindings"]
    }
    agent_tools = {
        item["id"]: AgentToolBinding(
            id=item["id"],
            node=item["node"],
            name=item.get("name", item["id"]),
            description=item["description"],
            inputs_from_state=MappingProxyType(dict(item.get("inputs_from_state", {}))),
            outputs_to_state=MappingProxyType(
                {key: MappingProxyType(dict(value)) for key, value in item.get("outputs_to_state", {}).items()}
            ),
        )
        for item in raw.get("agent_tools", ())
    }
    mcp_bindings = {
        item["id"]: McpBinding(id=item["id"], ref=item["ref"], tools=tuple(item.get("tools", ())))
        for item in raw["mcp_bindings"]
    }
    skill_bindings = {
        item["id"]: SkillBinding(id=item["id"], ref=item["ref"], required=item.get("required", False))
        for item in raw["skill_bindings"]
    }
    graph = AgentGraph(
        entry_node=graph_data["entry_node"],
        terminal_nodes=tuple(graph_data["terminal_nodes"]),
        nodes=tuple(
            GraphNode(
                id=node["id"],
                kind=node["kind"],
                model_binding=node.get("model_binding"),
                mcp_binding=node.get("mcp_binding"),
                skill_bindings=tuple(node.get("skill_bindings", ())),
                prompt_bindings=tuple(node.get("prompt_bindings", ())),
                policy_route=node.get("policy_route"),
                agent_tools=tuple(node.get("agent_tools", ())),
            )
            for node in graph_data["nodes"]
        ),
        edges=tuple(
            GraphEdge(source=edge["source"], target=edge["target"], condition=edge.get("condition"))
            for edge in graph_data["edges"]
        ),
    )
    router = PolicyRouter(
        default_action=router_data["default_action"],
        router_model_binding=router_data.get("router_model_binding"),
        routes=tuple(
            PolicyRoute(
                id=route["id"],
                priority=route["priority"],
                when=_freeze(route["when"]),
                action=route["action"],
                model_binding=route.get("model_binding"),
                mcp_binding=route.get("mcp_binding"),
                max_steps=route.get("max_steps"),
                max_tokens=route.get("max_tokens"),
            )
            for route in router_data["routes"]
        ),
    )
    detached = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return AgentServingSnapshot(
        schema_version=raw["schema_version"],
        profile=AgentProfile(
            name=profile_data["name"], version=profile_data["version"], description=profile_data.get("description")
        ),
        agent_graph=graph,
        policy_router=router,
        model_bindings=MappingProxyType(model_bindings),
        mcp_bindings=MappingProxyType(mcp_bindings),
        skill_bindings=MappingProxyType(skill_bindings),
        mcp_registry=mcp_registry,
        skills_registry=skills_registry,
        prompt_bindings=prompt_bindings,
        observability=observability,
        content_sha256=hashlib.sha256(detached).hexdigest(),
        agent_tools=MappingProxyType(agent_tools),
        state_schema=MappingProxyType(
            {key: MappingProxyType(dict(value)) for key, value in raw.get("state_schema", {}).items()}
        ),
    )


def _semantic_issues(raw: Mapping[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(_forbidden_content_issues(raw))

    def unique_ids(items: Sequence[Mapping[str, Any]], path: str) -> set[str]:
        seen: set[str] = set()
        for index, item in enumerate(items):
            identifier = item.get("id")
            if not isinstance(identifier, str):
                continue
            if identifier in seen:
                issues.append(ValidationIssue(f"{path}/{index}/id", f"duplicate id: {identifier}"))
            seen.add(identifier)
        return seen

    model_ids = unique_ids(raw["model_bindings"], "/model_bindings")
    mcp_ids = unique_ids(raw["mcp_bindings"], "/mcp_bindings")
    skill_ids = unique_ids(raw["skill_bindings"], "/skill_bindings")
    mcp_server_ids = unique_ids(raw["mcp_registry"]["servers"], "/mcp_registry/servers")
    skill_store_ids = unique_ids(raw["skills_registry"]["stores"], "/skills_registry/stores")
    skill_manifest_ids = unique_ids(raw["skills_registry"]["skills"], "/skills_registry/skills")
    prompt_ids = unique_ids(raw["prompt_bindings"], "/prompt_bindings")
    agent_tool_ids = unique_ids(raw.get("agent_tools", ()), "/agent_tools")
    state_schema = raw.get("state_schema", {})
    state_ids = set(state_schema)
    graph_data = raw["agent_graph"]
    router_data = raw["policy_router"]
    node_ids = unique_ids(graph_data["nodes"], "/agent_graph/nodes")
    route_ids = unique_ids(router_data["routes"], "/policy_router/routes")

    reserved_state_keys = {"messages", "step_count", "token_usage", "tool_call_counts", "exit_reason"}
    for state_key, definition in state_schema.items():
        if not _IDENTIFIER_RE.fullmatch(str(state_key)):
            issues.append(ValidationIssue(f"/state_schema/{state_key}", "state key must be a valid identifier"))
        if state_key in reserved_state_keys:
            issues.append(ValidationIssue(f"/state_schema/{state_key}", "state key is reserved by Haystack Agent"))
        if definition.get("handler") == "append" and definition.get("type") not in {"array", "list"}:
            issues.append(
                ValidationIssue(
                    f"/state_schema/{state_key}/handler", "append state handler requires an array/list state field"
                )
            )

    _require_ref(graph_data["entry_node"], node_ids, "/agent_graph/entry_node", issues)
    _require_ref(router_data.get("router_model_binding"), model_ids, "/policy_router/router_model_binding", issues)
    for index, terminal in enumerate(graph_data["terminal_nodes"]):
        _require_ref(terminal, node_ids, f"/agent_graph/terminal_nodes/{index}", issues)
    for index, edge in enumerate(graph_data["edges"]):
        _require_ref(edge["source"], node_ids, f"/agent_graph/edges/{index}/source", issues)
        _require_ref(edge["target"], node_ids, f"/agent_graph/edges/{index}/target", issues)
    for index, node in enumerate(graph_data["nodes"]):
        if node.get("agent_tools") and node.get("kind") != "model":
            issues.append(
                ValidationIssue(
                    f"/agent_graph/nodes/{index}/agent_tools", "AgentTool bindings may only be attached to model nodes"
                )
            )
        _require_ref(node.get("model_binding"), model_ids, f"/agent_graph/nodes/{index}/model_binding", issues)
        _require_ref(node.get("mcp_binding"), mcp_ids, f"/agent_graph/nodes/{index}/mcp_binding", issues)
        _require_refs(node.get("skill_bindings", ()), skill_ids, f"/agent_graph/nodes/{index}/skill_bindings", issues)
        _require_refs(
            node.get("prompt_bindings", ()), prompt_ids, f"/agent_graph/nodes/{index}/prompt_bindings", issues
        )
        _require_ref(node.get("policy_route"), route_ids, f"/agent_graph/nodes/{index}/policy_route", issues)
        _require_refs(node.get("agent_tools", ()), agent_tool_ids, f"/agent_graph/nodes/{index}/agent_tools", issues)
    for index, binding in enumerate(raw.get("agent_tools", ())):
        _require_ref(binding.get("node"), node_ids, f"/agent_tools/{index}/node", issues)
        for state_key, source in binding.get("inputs_from_state", {}).items():
            _require_ref(state_key, state_ids, f"/agent_tools/{index}/inputs_from_state/{state_key}", issues)
            if not isinstance(source, str) or not _IDENTIFIER_RE.fullmatch(source):
                issues.append(
                    ValidationIssue(
                        f"/agent_tools/{index}/inputs_from_state/{state_key}",
                        "mapped Agent input name must be a valid identifier",
                    )
                )
        for state_key in binding.get("outputs_to_state", {}):
            _require_ref(state_key, state_ids, f"/agent_tools/{index}/outputs_to_state/{state_key}", issues)

    _validate_agent_tool_targets(raw, node_ids, issues)
    for index, route in enumerate(raw["policy_router"]["routes"]):
        _require_ref(route.get("model_binding"), model_ids, f"/policy_router/routes/{index}/model_binding", issues)
        _require_ref(route.get("mcp_binding"), mcp_ids, f"/policy_router/routes/{index}/mcp_binding", issues)

    for index, binding in enumerate(raw["mcp_bindings"]):
        _require_ref(binding["ref"], mcp_server_ids, f"/mcp_bindings/{index}/ref", issues)
    for index, binding in enumerate(raw["skill_bindings"]):
        _require_ref(binding["ref"], skill_manifest_ids, f"/skill_bindings/{index}/ref", issues)
    for index, skill in enumerate(raw["skills_registry"]["skills"]):
        _require_ref(skill["store_ref"], skill_store_ids, f"/skills_registry/skills/{index}/store_ref", issues)

    priorities = [route["priority"] for route in raw["policy_router"]["routes"]]
    if priorities != sorted(priorities):
        issues.append(ValidationIssue("/policy_router/routes", "routes must be sorted by ascending priority"))
    return issues


def _validate_agent_tool_targets(raw: Mapping[str, Any], node_ids: set[str], issues: list[ValidationIssue]) -> None:
    """Validate AgentTool target kinds and reject recursive delegation graphs."""

    nodes = {node["id"]: node for node in raw["agent_graph"]["nodes"] if isinstance(node, Mapping)}
    bindings = {binding["id"]: binding for binding in raw.get("agent_tools", ()) if isinstance(binding, Mapping)}
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for index, binding in enumerate(raw.get("agent_tools", ())):
        target = binding.get("node")
        target_node = nodes.get(target)
        if target not in node_ids:
            continue
        if target_node is None or target_node.get("kind") != "model":
            issues.append(ValidationIssue(f"/agent_tools/{index}/node", "AgentTool target must reference a model node"))

    for node_index, node in enumerate(raw["agent_graph"]["nodes"]):
        source = node.get("id")
        for tool_index, binding_id in enumerate(node.get("agent_tools", ())):
            binding = bindings.get(binding_id)
            if binding is None:
                continue
            target = binding.get("node")
            if target in nodes and nodes[target].get("kind") == "model":
                outgoing.setdefault(source, []).append((target, binding_id))
            if len([item for item in node.get("agent_tools", ()) if item == binding_id]) > 1:
                issues.append(
                    ValidationIssue(
                        f"/agent_graph/nodes/{node_index}/agent_tools/{tool_index}",
                        f"duplicate AgentTool reference: {binding_id}",
                    )
                )

    visiting: list[str] = []
    visited: set[str] = set()
    reported: set[tuple[str, ...]] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            cycle = tuple(visiting[visiting.index(node_id) :] + [node_id])
            normalized = tuple(sorted(set(cycle)))
            if normalized not in reported:
                reported.add(normalized)
                issues.append(
                    ValidationIssue("/agent_tools", f"AgentTool delegation cycle detected: {' -> '.join(cycle)}")
                )
            return
        if node_id in visited:
            return
        visiting.append(node_id)
        for target, _binding_id in outgoing.get(node_id, ()):
            visit(target)
        visiting.pop()
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)


def _forbidden_content_issues(value: Any, path: str = "$") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in _FORBIDDEN_KEYS or "python_import" in key_text:
                issues.append(ValidationIssue(f"{path}/{key}", "arbitrary Python/import configuration is forbidden"))
            issues.extend(_forbidden_content_issues(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_forbidden_content_issues(child, f"{path}/{index}"))
    elif isinstance(value, str):
        normalized = value.strip().lower()
        is_mcp_endpoint = path.startswith("$/mcp_endpoints/")
        if (normalized.startswith("system.") or "system.ai." in normalized) and not is_mcp_endpoint:
            issues.append(
                ValidationIssue(path, "system.* references are forbidden; foundation models are not bindings")
            )
        if _FORBIDDEN_RESOURCE_PREFIX in normalized.split("/", maxsplit=1)[-1][: len(_FORBIDDEN_RESOURCE_PREFIX)]:
            issues.append(ValidationIssue(path, "resources with the agent_ prefix are forbidden"))
        if "supervisor agent" in normalized or normalized == "supervisor" or "supervisor_agent" in normalized:
            issues.append(ValidationIssue(path, "Supervisor Agent references are forbidden"))
        if re.search(r"(?:^|[\s.:])import(?:\s|$)", normalized):
            issues.append(ValidationIssue(path, "arbitrary Python import expressions are forbidden"))
    return issues


def _model_ref_error(raw: Mapping[str, Any], path: Sequence[Any]) -> str:
    value: Any = raw
    for part in path:
        value = value[part]
    if isinstance(value, str) and re.match(r"^system(?:\.|$)", value):
        return f"{value}: system.* foundation models cannot be declared as model bindings"
    if isinstance(value, str) and any(pattern.fullmatch(value) for pattern in _MODEL_REF_PATTERNS):
        return "model ref must be an approved Serving Endpoint or Model Services resource"
    return "model ref must be valid under an approved Serving Endpoint or Model Services resource pattern"


def _require_ref(value: str | None, available: set[str], path: str, issues: list[ValidationIssue]) -> None:
    if value is not None and value not in available:
        issues.append(ValidationIssue(path, f"unknown reference: {value}"))


def _require_refs(values: Sequence[str], available: set[str], path: str, issues: list[ValidationIssue]) -> None:
    for index, value in enumerate(values):
        _require_ref(value, available, f"{path}/{index}", issues)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return copy.deepcopy(value)


def _json_path(path: Sequence[Any]) -> str:
    result = "$"
    for item in path:
        result += f"/{item}"
    return result


def _dedupe_issues(issues: Sequence[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    seen: set[tuple[str, str]] = set()
    result: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.path, issue.message)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return tuple(result)
