"""Small streamable-HTTP MCP Toolset for profile-bound serving agents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool, Toolset

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.credentials_provider import ModelServingUserCredentials
    from databricks_mcp import DatabricksMCPClient
except ImportError:  # Keep the local HTTP fixture usable without Databricks extras.
    DatabricksMCPClient = None
    ModelServingUserCredentials = None
    WorkspaceClient = None


class McpToolsetError(RuntimeError):
    """Raised when an MCP server cannot list or execute an allowlisted tool."""


class HttpMcpToolset(Toolset):
    """Discover and invoke tools from one MCP streamable-HTTP server."""

    def __init__(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str] | None = None,
        allowed_tools: tuple[str, ...] | list[str] = (),
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        use_databricks_auth: bool | None = None,
        workspace_client: Any | None = None,
    ) -> None:
        if not endpoint.startswith("https://") and not endpoint.startswith("http://localhost"):
            raise ValueError("MCP endpoint must use HTTPS, except for localhost development")
        self.endpoint = endpoint
        self.headers = dict(headers or {})
        self.allowed_tools = tuple(allowed_tools)
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._request_id = 0
        self._session_id: str | None = None
        self._warmed = False
        self._databricks_client: Any | None = None
        self._workspace_client = workspace_client
        self._use_databricks_auth = (
            use_databricks_auth if use_databricks_auth is not None else "/ai-gateway/mcp-services/" in endpoint
        )
        if self._use_databricks_auth and (
            DatabricksMCPClient is None or WorkspaceClient is None or ModelServingUserCredentials is None
        ):
            raise McpToolsetError("databricks-mcp is required for Databricks MCP Service endpoints")
        super().__init__(tools=[])

    def warm_up(self) -> None:
        """Initialize the MCP session and discover allowlisted tools."""

        if self._warmed:
            return
        if self._use_databricks_auth:
            self._ensure_databricks_client()
            self._warm_databricks_tools()
            return
        initialize = self._request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "haystack-databricks", "version": "1.0"},
            },
        )
        self._session_id = initialize.get("sessionId") or self._session_id
        self._notify("notifications/initialized", {})
        listed = self._request("tools/list", {})
        tools = listed.get("tools", [])
        if not isinstance(tools, list):
            raise McpToolsetError("MCP tools/list result is invalid")
        selected: list[Tool] = []
        allowed = set(self.allowed_tools)
        for item in tools:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                continue
            if allowed and item["name"] not in allowed:
                continue
            selected.append(self._make_tool(item))
        self.tools = selected
        self._warmed = True

    def _ensure_databricks_client(self) -> None:
        if self._databricks_client is not None:
            return
        if DatabricksMCPClient is None or WorkspaceClient is None or ModelServingUserCredentials is None:
            raise McpToolsetError("databricks-mcp is required for Databricks MCP Service endpoints")
        authenticated_client = self._workspace_client or WorkspaceClient(
            credentials_strategy=ModelServingUserCredentials()
        )
        self._databricks_client = DatabricksMCPClient(self.endpoint, workspace_client=authenticated_client)

    def _warm_databricks_tools(self) -> None:
        try:
            descriptors = self._databricks_client.list_tools()
        except Exception as error:
            raise McpToolsetError("Databricks MCP tool discovery failed") from error
        allowed = set(self.allowed_tools)
        selected: list[Tool] = []
        for descriptor in descriptors:
            name = getattr(descriptor, "name", None)
            if not isinstance(name, str) or (allowed and name not in allowed):
                continue
            schema = getattr(descriptor, "inputSchema", None) or getattr(descriptor, "input_schema", None)
            if not isinstance(schema, Mapping):
                schema = {"type": "object", "properties": {}}
            selected.append(self._make_databricks_tool(name, descriptor, dict(schema)))
        self.tools = selected
        self._warmed = True

    def _make_databricks_tool(self, name: str, descriptor: Any, schema: dict[str, Any]) -> Tool:
        description = str(getattr(descriptor, "description", None) or f"MCP tool {name}")

        def invoke(**arguments: Any) -> Any:
            try:
                result = self._databricks_client.call_tool(name, arguments)
            except Exception as error:
                raise McpToolsetError(f"Databricks MCP tool call failed: {name}") from error
            if getattr(result, "isError", False):
                raise McpToolsetError(f"Databricks MCP tool returned an error: {name}")
            content = getattr(result, "content", result)
            if isinstance(content, list):
                text_parts = [getattr(item, "text", None) for item in content if getattr(item, "text", None)]
                if text_parts:
                    return "\n".join(str(item) for item in text_parts)
            return content

        return Tool(name=name, description=description, parameters=schema, function=invoke)

    def close(self) -> None:
        """Close the HTTP client when this toolset owns it."""

        if self._owns_client:
            self._client.close()

    def to_dict(self) -> dict[str, Any]:
        """Serialize endpoint configuration without headers or credentials."""

        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "endpoint": self.endpoint,
                "allowed_tools": list(self.allowed_tools),
                "timeout": self.timeout,
                "use_databricks_auth": self._use_databricks_auth,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HttpMcpToolset":
        """Recreate a toolset from its serialized endpoint configuration."""

        payload = data["data"]
        return cls(
            payload["endpoint"],
            allowed_tools=tuple(payload.get("allowed_tools", [])),
            timeout=float(payload.get("timeout", 30.0)),
            use_databricks_auth=payload.get("use_databricks_auth"),
        )

    def _make_tool(self, descriptor: Mapping[str, Any]) -> Tool:
        name = str(descriptor["name"])
        description = str(descriptor.get("description", f"MCP tool {name}"))
        schema = descriptor.get("inputSchema", {"type": "object", "properties": {}})

        def invoke(**arguments: Any) -> Any:
            result = self._request("tools/call", {"name": name, "arguments": arguments})
            content = result.get("content", result)
            if isinstance(content, list):
                text_parts = [item.get("text") for item in content if isinstance(item, Mapping) and item.get("text")]
                if text_parts:
                    return "\n".join(str(item) for item in text_parts)
            return content

        return Tool(name=name, description=description, parameters=dict(schema), function=invoke)

    def _request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": dict(params)}
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json", **self.headers}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            response = self._client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise McpToolsetError("MCP request failed") from error
        body = _decode_response(response)
        if isinstance(body.get("error"), Mapping):
            raise McpToolsetError(str(body["error"].get("message", "MCP server returned an error")))
        result = body.get("result")
        return dict(result) if isinstance(result, Mapping) else {}

    def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        headers = {"Content-Type": "application/json", **self.headers}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        payload = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        try:
            response = self._client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise McpToolsetError("MCP notification failed") from error


def _decode_response(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        body = None
    if isinstance(body, Mapping):
        return dict(body)
    for line in response.text.splitlines():
        if line.startswith("data:"):
            try:
                body = json.loads(line[5:].strip())
            except json.JSONDecodeError as error:
                raise McpToolsetError("MCP stream event is invalid JSON") from error
            if isinstance(body, Mapping):
                return dict(body)
    raise McpToolsetError("MCP response did not contain a JSON-RPC object")


__all__ = ["HttpMcpToolset", "McpToolsetError"]
