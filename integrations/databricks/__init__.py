"""Databricks integration helpers for route-based agent serving.

The public helpers are deliberately transport-neutral except for the explicit
read-only Serving Endpoint client and injected gateway backends.
"""

from .ai_gateway import (
    DraftValidationResult,
    FallbackPolicy,
    GatewayConfig,
    GatewayConfigurationError,
    GatewayRequest,
    InMemoryRateLimiter,
    ModelProvider,
    ModelRoute,
    RateLimitDecision,
    RateLimitPolicy,
    RouteSelection,
    RouteSelector,
    resolve_route_reference,
    validate_gateway_config,
)
from .gateway import (
    AnthropicBackend,
    DatabricksServingBackend,
    GatewayBackend,
    GatewayBackendError,
    GatewayResult,
    GatewayRuntime,
    GatewayRuntimeError,
    OpenAICompatibleBackend,
    RequestContext,
)
from .serving_endpoints import (
    DatabricksServingEndpointsClient,
    ServingEndpointSummary,
    ServingEndpointsError,
    ServingEndpointsHTTPError,
    ServingEndpointsPayloadError,
    ServingEndpointsTransportError,
    ServingEntitySummary,
    ServingOpenAPI,
    summarize_endpoint,
)

__all__ = [
    "DatabricksServingEndpointsClient",
    "DatabricksServingBackend",
    "DraftValidationResult",
    "FallbackPolicy",
    "GatewayConfig",
    "GatewayBackend",
    "GatewayBackendError",
    "GatewayConfigurationError",
    "GatewayRequest",
    "GatewayResult",
    "GatewayRuntime",
    "GatewayRuntimeError",
    "InMemoryRateLimiter",
    "ModelProvider",
    "ModelRoute",
    "OpenAICompatibleBackend",
    "AnthropicBackend",
    "RateLimitDecision",
    "RateLimitPolicy",
    "RouteSelection",
    "RouteSelector",
    "RequestContext",
    "ServingEndpointSummary",
    "ServingEndpointsError",
    "ServingEndpointsHTTPError",
    "ServingEndpointsPayloadError",
    "ServingEndpointsTransportError",
    "ServingEntitySummary",
    "ServingOpenAPI",
    "resolve_route_reference",
    "summarize_endpoint",
    "validate_gateway_config",
]
