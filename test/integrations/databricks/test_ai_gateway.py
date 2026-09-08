from __future__ import annotations

from integrations.databricks.ai_gateway import (
    FallbackPolicy,
    GatewayConfig,
    GatewayRequest,
    InMemoryRateLimiter,
    ModelProvider,
    ModelRoute,
    RateLimitPolicy,
    RouteSelector,
    resolve_route_reference,
    validate_gateway_config,
)
from integrations.databricks.protocols import (
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_CURSOR_CHAT_COMPLETIONS,
    PROTOCOL_RESPONSES,
    get_protocol_adapter,
    get_protocol_capability,
)


def test_contract_protocol_aliases_resolve_without_mislabeling_cursor_as_codex() -> None:
    assert get_protocol_adapter(PROTOCOL_CHAT_COMPLETIONS).endpoint == "mlflow/v1/chat/completions"
    assert get_protocol_adapter(PROTOCOL_RESPONSES).endpoint == "openai/v1/responses"
    assert get_protocol_adapter(PROTOCOL_CURSOR_CHAT_COMPLETIONS).endpoint == "cursor/v1/chat/completions"
    assert get_protocol_capability(PROTOCOL_RESPONSES).codex_compatible
    assert not get_protocol_capability(PROTOCOL_CURSOR_CHAT_COMPLETIONS).codex_compatible


def test_route_selection_uses_contract_alias_and_ordered_fallbacks() -> None:
    config = GatewayConfig(
        providers=(ModelProvider("db", "databricks_serving", protocols=("chat_completions",)),),
        routes=(
            ModelRoute(
                "primary",
                "db",
                "model-services/simo_prod.ai_models.route_primary",
                protocols=("chat_completions",),
                priority=1,
                fallback_ids=("fallback",),
            ),
            ModelRoute("fallback", "db", "serving-endpoint/backup", protocols=("chat_completions",), priority=10),
        ),
        fallback=FallbackPolicy(),
    )

    result = validate_gateway_config(config)
    selection = RouteSelector(config).select(GatewayRequest(protocol="chat_completions"))

    assert result.valid
    assert [route.id for route in selection.candidates] == ["primary", "fallback"]
    assert resolve_route_reference(selection.primary.target).kind == "model_service"
    assert resolve_route_reference("serving-endpoint/backup").kind == "serving_endpoint"


def test_provider_selection_and_rate_limit_decision_are_deterministic() -> None:
    config = GatewayConfig(
        providers=(
            ModelProvider("db", "databricks_serving", protocols=("chat_completions",)),
            ModelProvider("codex", "openai_compatible", base_url="https://codex.example", protocols=("responses",)),
        ),
        routes=(
            ModelRoute("chat", "db", "serving-endpoint/chat", protocols=("chat_completions",), priority=1),
            ModelRoute(
                "code",
                "codex",
                "provider/codex",
                protocols=("responses",),
                model_names=("gpt-5-3-codex",),
                rate_limit=RateLimitPolicy(requests=1, window_seconds=60),
            ),
        ),
    )

    selector = RouteSelector(config)
    route = selector.select(GatewayRequest(protocol="responses", model="gpt-5-3-codex")).primary
    limiter = InMemoryRateLimiter(clock=lambda: 10.0)

    assert selector.provider_for(route).id == "codex"
    assert limiter.allow(route).allowed
    denied = limiter.allow(route)
    assert not denied.allowed
    assert denied.reason == "rate_limit"
