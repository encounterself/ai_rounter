from datetime import datetime, timedelta, timezone

import httpx
import pytest

from integrations.databricks.model_services import (
    MODEL_SERVICES_PATH,
    DatabricksModelServiceRegistry,
    ModelServiceNotAllowedError,
)


@pytest.fixture
def model_services_payload() -> dict[str, object]:
    return {
        "model_services": [
            {
                "name": "model-services/simo_prod.ai_models.route_primary",
                "etag": "service-etag-1",
                "supported_api_types": ["chat", "embeddings"],
                "config": {
                    "routing": {
                        "primary": [
                            {"name": "primary-route", "model_name": "foundation-chat", "traffic_percentage": 100}
                        ],
                        "fallback": [{"name": "fallback-route", "model_name": "backup-chat"}],
                    },
                    "rate_limits": [
                        {
                            "key": "user",
                            "calls": 20,
                            "tokens": 4000,
                            "renewal_period": 60,
                            "renewal_period_unit": "seconds",
                        }
                    ],
                },
            },
            {
                "name": "model-services/simo_prod.ai_models.route_secondary",
                "supported_api_types": ["chat"],
                "routing": {
                    "routes": [
                        {"name": "secondary-primary", "model_name": "secondary-chat"},
                        {"name": "secondary-fallback", "model_name": "secondary-backup", "fallback": True},
                    ]
                },
            },
            {"name": "model-services/system.ai.gpt_5", "supported_api_types": ["chat"]},
            {"name": "model-services/other_prod.ai_models.route_external", "supported_api_types": ["chat"]},
        ],
        "next_page_token": "page-2",
    }


def test_refresh_filters_resources_and_parses_capabilities(model_services_payload):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("page_token") == "page-2":
            return httpx.Response(
                200,
                json={
                    "model_services": [
                        {
                            "name": "model-services/simo_prod.ai_models.route_tertiary",
                            "supportedApiTypes": ["chat"],
                            "rateLimits": [{"name": "workspace", "requests": 100, "period": 60}],
                        }
                    ]
                },
                headers={"ETag": "registry-etag"},
                request=request,
            )
        return httpx.Response(
            200,
            json=model_services_payload,
            headers={"ETag": "registry-etag", "Cache-Control": "max-age=60"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with DatabricksModelServiceRegistry("https://workspace.example", transport=transport) as registry:
        snapshot = registry.refresh()

    assert [service.name for service in snapshot.services] == [
        "model-services/simo_prod.ai_models.route_primary",
        "model-services/simo_prod.ai_models.route_secondary",
        "model-services/simo_prod.ai_models.route_tertiary",
    ]
    assert snapshot.etag == "registry-etag"
    assert snapshot.capability_etag.startswith("sha256:")
    assert snapshot.expires_at == snapshot.fetched_at + timedelta(seconds=60)
    assert snapshot.services[0].supported_api_types == ("chat", "embeddings")
    assert snapshot.services[0].primary_routes[0].destination == "foundation-chat"
    assert snapshot.services[0].fallback_routes[0].destination == "backup-chat"
    assert snapshot.services[0].rate_limits[0].calls == 20
    assert snapshot.services[1].primary_routes[0].name == "secondary-primary"
    assert snapshot.services[1].fallback_routes[0].name == "secondary-fallback"
    assert requests[0].url.path == MODEL_SERVICES_PATH
    assert requests[0].url.params["parent"] == "schemas/simo_prod.ai_models"
    assert requests[0].url.params["view"] == "FULL"
    assert requests[1].url.params["page_token"] == "page-2"


def test_system_ai_services_are_hidden_and_rejected(model_services_payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=model_services_payload, request=request)

    with DatabricksModelServiceRegistry(
        base_url="https://workspace.example", transport=httpx.MockTransport(handler)
    ) as registry:
        assert not registry.is_user_selectable("model-services/system.ai.gpt_5")
        assert not registry.is_user_selectable("system.ai.gpt_5")
        with pytest.raises(ModelServiceNotAllowedError):
            registry.get("model-services/system.ai.gpt_5")


def test_refresh_uses_ttl_and_renews_with_not_modified():
    current_time = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    requests: list[httpx.Request] = []

    def now() -> datetime:
        return current_time

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "model_services": [
                        {"name": "model-services/simo_prod.ai_models.route_primary", "etag": "service-etag"}
                    ]
                },
                headers={"ETag": "registry-etag", "Cache-Control": "max-age=30"},
                request=request,
            )
        return httpx.Response(304, headers={"ETag": "registry-etag", "Cache-Control": "max-age=30"}, request=request)

    with DatabricksModelServiceRegistry(
        "https://workspace.example", transport=httpx.MockTransport(handler), now=now
    ) as registry:
        first = registry.refresh()
        assert registry.refresh() is first
        current_time += timedelta(seconds=31)
        renewed = registry.refresh()

    assert len(requests) == 2
    assert requests[1].headers["if-none-match"] == "registry-etag"
    assert renewed.services == first.services
    assert renewed.fetched_at == current_time
    assert renewed.expires_at == current_time + timedelta(seconds=30)


def test_invalid_http_response_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden", request=request)

    with DatabricksModelServiceRegistry(
        "https://workspace.example", token="secret", transport=httpx.MockTransport(handler)
    ) as registry:
        with pytest.raises(Exception, match="HTTP 403"):
            registry.refresh()
