from __future__ import annotations

import httpx

from integrations.databricks.mcp_toolset import HttpMcpToolset


def test_mcp_toolset_discovers_and_allowlists_tools() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        method = payload.decode().split('"method":"', 1)[1].split('"', 1)[0]
        methods.append(method)
        if method == "initialize":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "mcp-1"}}, request=request
            )
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "tools": [
                            {"name": "search", "description": "Search", "inputSchema": {"type": "object"}},
                            {"name": "delete", "description": "Delete", "inputSchema": {"type": "object"}},
                        ]
                    },
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "found"}]}},
            request=request,
        )

    toolset = HttpMcpToolset(
        "https://mcp.example/tools",
        allowed_tools=["search"],
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    toolset.warm_up()

    assert [tool.name for tool in toolset] == ["search"]
    assert toolset[0].invoke(query="haystack") == "found"
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
