from __future__ import annotations

import httpx
import pytest

from integrations.databricks.serving_endpoints import (
    DatabricksServingEndpointsClient,
    ServingEndpointsHTTPError,
)


def test_list_get_and_openapi_parse_endpoint_payloads() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/2.0/serving-endpoints":
            if request.url.params.get("page_token") == "next":
                return httpx.Response(
                    200,
                    json={"endpoints": [{"name": "endpoint-two", "state": {"ready": "READY"}}]},
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "endpoints": [
                        {
                            "name": "endpoint-one",
                            "id": "ep-1",
                            "state": {"ready": "READY", "config_update": "NOT_UPDATING"},
                            "config": {
                                "served_entities": [
                                    {
                                        "name": "model-a",
                                        "entity_name": "catalog.schema.model",
                                        "entity_version": "3",
                                        "task": "llm/v1/chat",
                                    }
                                ],
                                "ai_gateway": {"enabled": True, "api_key": "not-for-display"},
                            },
                        }
                    ],
                    "next_page_token": "next",
                },
                request=request,
            )
        if request.url.path.endswith("/openapi"):
            return httpx.Response(200, json={"openapi": "3.0.0", "paths": {"/invocations": {}}}, request=request)
        return httpx.Response(
            200,
            json={"name": "endpoint-one", "state": {"ready": "READY"}, "config": {}},
            request=request,
        )

    with DatabricksServingEndpointsClient(
        "https://workspace.example", token="workspace-secret", profile="DEFAULT", transport=httpx.MockTransport(handler)
    ) as client:
        endpoints = client.list_endpoints()
        endpoint = client.get_endpoint("endpoint-one")
        openapi = client.get_openapi("endpoint-one")

    assert [item.name for item in endpoints] == ["endpoint-one", "endpoint-two"]
    assert endpoint.is_ready
    assert endpoint.served_entity_names == ()
    assert openapi.paths == {"/invocations": {}}
    assert requests[0].headers["authorization"] == "Bearer workspace-secret"
    assert endpoint.config == {}


def test_http_errors_redact_authentication_and_secret_values() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": "denied", "token": "workspace-secret", "api_key": "provider-secret"},
            request=request,
        )

    with DatabricksServingEndpointsClient(
        "https://workspace.example", token="workspace-secret", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ServingEndpointsHTTPError) as error:
            client.get_endpoint("protected")

    message = str(error.value)
    assert "workspace-secret" not in message
    assert "provider-secret" not in message
    assert "HTTP 403" in message
