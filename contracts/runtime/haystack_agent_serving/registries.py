"""Immutable MCP, skills, prompt, and observability registry contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

_PROMPT_VARIABLE_RE = re.compile(r"{{\s*([a-z][a-z0-9._-]{0,127})\s*}}")


class RegistryValidationError(ValueError):
    """Raised when a registry cannot be converted into a safe snapshot."""


@dataclass(frozen=True, slots=True)
class McpServer:
    id: str
    protocol: str
    source_kind: str
    source_ref: str
    auth_ref: str | None
    allowed_tools: tuple[str, ...]
    refresh_seconds: int | None

    def allows_tool(self, tool_name: str) -> bool:
        """Return whether a tool is explicitly allowlisted for this server."""

        return tool_name in self.allowed_tools


@dataclass(frozen=True, slots=True)
class McpRegistrySnapshot:
    registry_version: str
    servers: Mapping[str, McpServer]

    def server(self, server_id: str) -> McpServer:
        """Return a server or fail closed when it is not registered."""

        try:
            return self.servers[server_id]
        except KeyError as error:
            raise RegistryValidationError(f"unknown MCP server: {server_id}") from error


@dataclass(frozen=True, slots=True)
class SkillStore:
    id: str
    kind: str
    ref: str
    read_only: bool


@dataclass(frozen=True, slots=True)
class SkillManifest:
    id: str
    store_ref: str
    manifest_path: str
    version: str
    content_sha256: str
    load_mode: str


@dataclass(frozen=True, slots=True)
class SkillsRegistrySnapshot:
    registry_version: str
    stores: Mapping[str, SkillStore]
    skills: Mapping[str, SkillManifest]

    def skill(self, skill_id: str) -> SkillManifest:
        """Return a skill or fail closed when it is not registered."""

        try:
            return self.skills[skill_id]
        except KeyError as error:
            raise RegistryValidationError(f"unknown skill: {skill_id}") from error


@dataclass(frozen=True, slots=True)
class PromptBinding:
    id: str
    role: str
    ref: str
    allowed_variables: tuple[str, ...]
    user_override: bool
    max_length: int
    content_sha256: str | None

    def render(self, template: str, variables: Mapping[str, str]) -> str:
        """Render a stored prompt using only its declared variables."""

        if self.user_override:
            raise RegistryValidationError(f"user prompt override is forbidden: {self.id}")
        unknown = set(variables) - set(self.allowed_variables)
        if unknown:
            raise RegistryValidationError(f"undeclared prompt variables: {sorted(unknown)}")
        if len(template) > self.max_length:
            raise RegistryValidationError(f"prompt exceeds max_length: {self.id}")

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in self.allowed_variables:
                raise RegistryValidationError(f"undeclared prompt variable: {name}")
            return variables.get(name, match.group(0))

        return _PROMPT_VARIABLE_RE.sub(replace, template)


@dataclass(frozen=True, slots=True)
class ObservabilityPolicy:
    mode: str
    trace_enabled: bool
    sampling_ratio: float
    capture_content: bool
    redaction_policy: str
    metric_namespace: str
    emit_tool_events: bool
    emit_model_events: bool


def build_registry_state(
    raw: Mapping[str, Any],
) -> tuple[McpRegistrySnapshot, SkillsRegistrySnapshot, Mapping[str, PromptBinding], ObservabilityPolicy]:
    """Build immutable registry state from a schema-valid serving document."""

    mcp_data = raw["mcp_registry"]
    mcp_servers = {
        item["id"]: McpServer(
            id=item["id"],
            protocol=item["protocol"],
            source_kind=item["source_kind"],
            source_ref=item["source_ref"],
            auth_ref=item.get("auth_ref"),
            allowed_tools=tuple(item["allowed_tools"]),
            refresh_seconds=item.get("refresh_seconds"),
        )
        for item in mcp_data["servers"]
    }
    for server in mcp_servers.values():
        if server.protocol == "stdio" and server.source_kind != "local_fixture":
            raise RegistryValidationError(f"remote MCP server cannot use stdio: {server.id}")

    skills_data = raw["skills_registry"]
    stores = {
        item["id"]: SkillStore(id=item["id"], kind=item["kind"], ref=item["ref"], read_only=item.get("read_only", True))
        for item in skills_data["stores"]
    }
    skills = {
        item["id"]: SkillManifest(
            id=item["id"],
            store_ref=item["store_ref"],
            manifest_path=item["manifest_path"],
            version=item["version"],
            content_sha256=item["content_sha256"],
            load_mode=item["load_mode"],
        )
        for item in skills_data["skills"]
    }
    unknown_stores = {skill.store_ref for skill in skills.values()} - set(stores)
    if unknown_stores:
        raise RegistryValidationError(f"unknown skill stores: {sorted(unknown_stores)}")

    prompts = {
        item["id"]: PromptBinding(
            id=item["id"],
            role=item["role"],
            ref=item["ref"],
            allowed_variables=tuple(item["allowed_variables"]),
            user_override=item["user_override"],
            max_length=item["max_length"],
            content_sha256=item.get("content_sha256"),
        )
        for item in raw["prompt_bindings"]
    }
    observation_data = raw["observability"]
    observability = ObservabilityPolicy(
        mode=observation_data["mode"],
        trace_enabled=observation_data["trace_enabled"],
        sampling_ratio=observation_data["sampling_ratio"],
        capture_content=observation_data["capture_content"],
        redaction_policy=observation_data["redaction_policy"],
        metric_namespace=observation_data["metric_namespace"],
        emit_tool_events=observation_data["emit_tool_events"],
        emit_model_events=observation_data["emit_model_events"],
    )
    return (
        McpRegistrySnapshot(mcp_data["registry_version"], MappingProxyType(mcp_servers)),
        SkillsRegistrySnapshot(skills_data["registry_version"], MappingProxyType(stores), MappingProxyType(skills)),
        MappingProxyType(prompts),
        observability,
    )


def read_skill_file(root: str | Path, skill: SkillManifest, relative_path: str) -> str:
    """Read a skill file under an approved local root without path traversal."""

    candidate_path = PurePosixPath(relative_path)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise RegistryValidationError("skill file path must stay within the skill root")
    root_path = Path(root).resolve()
    file_path = (root_path / skill.manifest_path).resolve()
    if file_path.parent != root_path and root_path not in file_path.parents:
        raise RegistryValidationError("skill manifest escapes the skill root")
    requested_path = (file_path.parent / candidate_path).resolve()
    try:
        requested_path.relative_to(file_path.parent)
    except ValueError as error:
        raise RegistryValidationError("skill file path escapes the skill root") from error
    return requested_path.read_text(encoding="utf-8")


def registry_state_to_dict(
    mcp_registry: McpRegistrySnapshot,
    skills_registry: SkillsRegistrySnapshot,
    prompts: Mapping[str, PromptBinding],
    observability: ObservabilityPolicy,
) -> dict[str, Any]:
    """Return a detached JSON-compatible representation of registry state."""

    return {
        "mcp_registry": {
            "registry_version": mcp_registry.registry_version,
            "servers": [
                {
                    "id": server.id,
                    "protocol": server.protocol,
                    "source_kind": server.source_kind,
                    "source_ref": server.source_ref,
                    **({"auth_ref": server.auth_ref} if server.auth_ref else {}),
                    "allowed_tools": list(server.allowed_tools),
                    **({"refresh_seconds": server.refresh_seconds} if server.refresh_seconds is not None else {}),
                }
                for server in mcp_registry.servers.values()
            ],
        },
        "skills_registry": {
            "registry_version": skills_registry.registry_version,
            "stores": [
                {"id": store.id, "kind": store.kind, "ref": store.ref, "read_only": store.read_only}
                for store in skills_registry.stores.values()
            ],
            "skills": [
                {
                    "id": skill.id,
                    "store_ref": skill.store_ref,
                    "manifest_path": skill.manifest_path,
                    "version": skill.version,
                    "content_sha256": skill.content_sha256,
                    "load_mode": skill.load_mode,
                }
                for skill in skills_registry.skills.values()
            ],
        },
        "prompt_bindings": [
            {
                "id": prompt.id,
                "role": prompt.role,
                "ref": prompt.ref,
                "allowed_variables": list(prompt.allowed_variables),
                "user_override": prompt.user_override,
                "max_length": prompt.max_length,
                **({"content_sha256": prompt.content_sha256} if prompt.content_sha256 else {}),
            }
            for prompt in prompts.values()
        ],
        "observability": {
            "mode": observability.mode,
            "trace_enabled": observability.trace_enabled,
            "sampling_ratio": observability.sampling_ratio,
            "capture_content": observability.capture_content,
            "redaction_policy": observability.redaction_policy,
            "metric_namespace": observability.metric_namespace,
            "emit_tool_events": observability.emit_tool_events,
            "emit_model_events": observability.emit_model_events,
        },
    }
