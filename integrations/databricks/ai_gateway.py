"""Pure configuration and routing primitives for the gateway runtime.

The classes in this module are intentionally transport-free.  They validate a
draft, resolve references to Databricks Model Services or Serving Endpoints,
select a protocol-capable route, and make a deterministic rate-limit decision.
An application can persist the resulting configuration elsewhere and a
Serving handler can inject the selected route into a concrete backend.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from .protocols import UnsupportedProtocolError, get_protocol_adapter, normalize_protocol


class GatewayConfigurationError(ValueError):
    """Raised when a gateway draft cannot be converted to a valid config."""


@dataclass(frozen=True, slots=True)
class ModelProvider:
    """A provider declaration without secret material."""

    id: str
    kind: str
    base_url: str | None = None
    token_ref: str | None = None
    model: str | None = None
    protocols: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """A fixed-window request limit for one route."""

    requests: int
    window_seconds: float
    key: str = "tenant"


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """A routable model target, such as a Model Service or endpoint."""

    id: str
    provider: str
    target: str
    protocols: tuple[str, ...] = ()
    model_names: tuple[str, ...] = ()
    target_type: str | None = None
    priority: int = 100
    fallback_ids: tuple[str, ...] = ()
    timeout_seconds: float = 60.0
    rate_limit: RateLimitPolicy | None = None
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def resolved_target_type(self) -> str:
        """Infer a target kind when a draft omitted it."""

        if self.target_type:
            return self.target_type
        return resolve_route_reference(self.target).kind

    def supports_protocol(self, protocol: str, provider_protocols: Sequence[str] = ()) -> bool:
        """Return whether this route advertises the requested protocol."""

        try:
            adapter = get_protocol_adapter(protocol)
        except UnsupportedProtocolError:
            return False
        normalized = {normalize_protocol(value) for value in self.protocols}
        provider_normalized = {normalize_protocol(value) for value in provider_protocols}
        return (not self.protocols or adapter.protocol in normalized) and (
            not provider_protocols or adapter.protocol in provider_normalized
        )

    def matches_model(self, model: str | None) -> bool:
        """Return whether this route can serve a requested model alias."""

        if not model or not self.model_names:
            return True
        return model in self.model_names or model == self.id or model == self.target

    def supports_protocols_declared_by_provider(self, provider: ModelProvider) -> bool:
        """Return whether route capabilities are a subset of provider capabilities."""

        if not self.protocols or not provider.protocols:
            return True
        route_protocols = {normalize_protocol(value) for value in self.protocols}
        provider_protocols = {normalize_protocol(value) for value in provider.protocols}
        return route_protocols.issubset(provider_protocols)


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    """Ordered fallback references used after a primary route fails."""

    route_ids: tuple[str, ...] = ()
    max_attempts: int | None = None


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """Immutable gateway draft or published configuration."""

    providers: tuple[ModelProvider, ...] = ()
    routes: tuple[ModelRoute, ...] = ()
    fallback: FallbackPolicy = field(default_factory=FallbackPolicy)
    default_protocols: tuple[str, ...] = ()
    revision: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GatewayConfig":
        """Build a config from a JSON-compatible draft mapping."""

        providers = tuple(_provider_from_mapping(item) for item in _items(payload.get("providers")))
        routes = tuple(_route_from_mapping(item) for item in _items(payload.get("routes")))
        fallback_payload = payload.get("fallback", {})
        fallback = (
            _fallback_from_mapping(fallback_payload) if isinstance(fallback_payload, Mapping) else FallbackPolicy()
        )
        return cls(
            providers=providers,
            routes=routes,
            fallback=fallback,
            default_protocols=tuple(_strings(payload.get("default_protocols"))),
            revision=_optional_string(payload.get("revision")),
            metadata=_mapping(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config without resolving or injecting secrets."""

        return {
            "providers": [
                {
                    "id": provider.id,
                    "kind": provider.kind,
                    "base_url": provider.base_url,
                    "token_ref": provider.token_ref,
                    "model": provider.model,
                    "protocols": list(provider.protocols),
                    "metadata": dict(provider.metadata),
                }
                for provider in self.providers
            ],
            "routes": [
                {
                    "id": route.id,
                    "provider": route.provider,
                    "target": route.target,
                    "target_type": route.target_type,
                    "protocols": list(route.protocols),
                    "model_names": list(route.model_names),
                    "priority": route.priority,
                    "fallback_ids": list(route.fallback_ids),
                    "timeout_seconds": route.timeout_seconds,
                    "rate_limit": (
                        {
                            "requests": route.rate_limit.requests,
                            "window_seconds": route.rate_limit.window_seconds,
                            "key": route.rate_limit.key,
                        }
                        if route.rate_limit
                        else None
                    ),
                    "enabled": route.enabled,
                    "metadata": dict(route.metadata),
                }
                for route in self.routes
            ],
            "fallback": {
                "route_ids": list(self.fallback.route_ids),
                "max_attempts": self.fallback.max_attempts,
            },
            "default_protocols": list(self.default_protocols),
            "revision": self.revision,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DraftValidationResult:
    """Validation result that is safe to show in a control-plane UI."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether the draft can be published."""

        return not self.errors


@dataclass(frozen=True, slots=True)
class GatewayRequest:
    """Inputs used by deterministic route selection."""

    protocol: str
    model: str | None = None
    tenant: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouteSelection:
    """Primary route and ordered fallback candidates."""

    primary: ModelRoute
    candidates: tuple[ModelRoute, ...]


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Result of a request admission decision."""

    allowed: bool
    remaining: int
    retry_after_seconds: float = 0.0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedRouteReference:
    """Normalized kind and name for a route target reference."""

    kind: str
    name: str


class InMemoryRateLimiter:
    """Small process-local fixed-window limiter for tests and single workers."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def allow(self, route: ModelRoute, *, key: str = "default") -> RateLimitDecision:
        """Admit or reject a request for ``route``."""

        policy = route.rate_limit
        if policy is None:
            return RateLimitDecision(True, remaining=2**31 - 1)
        now = self._clock()
        bucket = self._events[(route.id, key)]
        cutoff = now - policy.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= policy.requests:
            retry_after = max(0.0, bucket[0] + policy.window_seconds - now)
            return RateLimitDecision(False, remaining=0, retry_after_seconds=retry_after, reason="rate_limit")
        bucket.append(now)
        return RateLimitDecision(True, remaining=policy.requests - len(bucket))


class RouteSelector:
    """Validate and select protocol/model-capable routes."""

    def __init__(self, config: GatewayConfig) -> None:
        result = validate_gateway_config(config)
        if not result.valid:
            raise GatewayConfigurationError("; ".join(result.errors))
        self.config = config
        self._providers = {provider.id: provider for provider in config.providers}
        self._routes = {route.id: route for route in config.routes}

    def select(self, request: GatewayRequest) -> RouteSelection:
        """Return the best primary and deterministic fallback candidates."""

        try:
            get_protocol_adapter(request.protocol)
        except UnsupportedProtocolError as error:
            raise GatewayConfigurationError(str(error)) from error
        eligible = [
            route
            for route in self.config.routes
            if route.enabled
            and route.supports_protocol(request.protocol, self._providers[route.provider].protocols)
            and route.matches_model(request.model)
        ]
        if not eligible:
            raise GatewayConfigurationError(
                f"No enabled route supports protocol={request.protocol!r} model={request.model!r}."
            )
        eligible.sort(key=lambda route: (route.priority, route.id))
        primary = eligible[0]
        ordered: list[ModelRoute] = [primary]
        explicit_ids = list(primary.fallback_ids) + list(self.config.fallback.route_ids)
        seen = {primary.id}
        for route_id in explicit_ids:
            route = self._routes.get(route_id)
            if route is None or route.id in seen or not route.enabled:
                continue
            provider_protocols = self._providers[route.provider].protocols
            if route.supports_protocol(request.protocol, provider_protocols) and route.matches_model(request.model):
                ordered.append(route)
                seen.add(route.id)
        for route in eligible:
            if route.id not in seen:
                ordered.append(route)
                seen.add(route.id)
        if self.config.fallback.max_attempts is not None:
            ordered = ordered[: max(1, self.config.fallback.max_attempts)]
        return RouteSelection(primary=primary, candidates=tuple(ordered))

    def provider_for(self, route: ModelRoute) -> ModelProvider:
        """Return the provider bound to a route."""

        return self._providers[route.provider]


def validate_gateway_config(config: GatewayConfig) -> DraftValidationResult:
    """Validate a gateway draft without making network calls."""

    errors: list[str] = []
    warnings: list[str] = []
    provider_ids: set[str] = set()
    providers_by_id: dict[str, ModelProvider] = {}
    route_ids: set[str] = set()
    for provider in config.providers:
        if not provider.id:
            errors.append("provider id must not be empty")
        if provider.id in provider_ids:
            errors.append(f"duplicate provider id: {provider.id}")
        provider_ids.add(provider.id)
        providers_by_id[provider.id] = provider
        if provider.kind not in {"databricks_serving", "model_service", "openai_compatible", "anthropic"}:
            errors.append(f"provider {provider.id} has unsupported kind: {provider.kind}")
        if provider.kind in {"openai_compatible", "anthropic"} and not provider.base_url:
            errors.append(f"provider {provider.id} requires an explicit base_url")
        if not provider.protocols:
            warnings.append(f"provider {provider.id} does not advertise protocol capabilities")
        errors.extend(_protocol_errors(provider.protocols, f"provider {provider.id}"))

    for route in config.routes:
        if not route.id:
            errors.append("route id must not be empty")
        if route.id in route_ids:
            errors.append(f"duplicate route id: {route.id}")
        route_ids.add(route.id)
        if route.provider not in provider_ids:
            errors.append(f"route {route.id} references unknown provider: {route.provider}")
        elif not route.supports_protocols_declared_by_provider(providers_by_id[route.provider]):
            errors.append(f"route {route.id} advertises protocols not supported by provider {route.provider}")
        if not route.target:
            errors.append(f"route {route.id} requires a target")
        if route.priority < 0:
            errors.append(f"route {route.id} priority must be non-negative")
        if route.timeout_seconds <= 0:
            errors.append(f"route {route.id} timeout_seconds must be greater than zero")
        if route.target_type and route.target_type not in {"model_service", "serving_endpoint", "provider_target"}:
            errors.append(f"route {route.id} has unsupported target_type: {route.target_type}")
        errors.extend(_protocol_errors(route.protocols, f"route {route.id}"))
        if route.rate_limit and (route.rate_limit.requests <= 0 or route.rate_limit.window_seconds <= 0):
            errors.append(f"route {route.id} rate limit must use positive requests and window_seconds")

    for route in config.routes:
        for fallback_id in route.fallback_ids:
            if fallback_id not in route_ids:
                errors.append(f"route {route.id} references unknown fallback route: {fallback_id}")
    for route_id in config.fallback.route_ids:
        if route_id not in route_ids:
            errors.append(f"fallback policy references unknown route: {route_id}")
    if config.fallback.max_attempts is not None and config.fallback.max_attempts <= 0:
        errors.append("fallback max_attempts must be positive")
    errors.extend(_protocol_errors(config.default_protocols, "default_protocols"))
    if not config.routes:
        warnings.append("gateway has no routes")
    return DraftValidationResult(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)))


def resolve_route_reference(reference: str) -> ResolvedRouteReference:
    """Resolve a Model Services or Serving Endpoint reference."""

    if reference.startswith("model-services/"):
        return ResolvedRouteReference("model_service", reference.removeprefix("model-services/"))
    if reference.startswith("serving-endpoint/"):
        return ResolvedRouteReference("serving_endpoint", reference.removeprefix("serving-endpoint/"))
    if reference.startswith("serving-endpoints/"):
        return ResolvedRouteReference("serving_endpoint", reference.removeprefix("serving-endpoints/"))
    if reference.startswith("endpoint/"):
        return ResolvedRouteReference("serving_endpoint", reference.removeprefix("endpoint/"))
    return ResolvedRouteReference("provider_target", reference)


def _protocol_errors(protocols: Sequence[str], label: str) -> list[str]:
    errors: list[str] = []
    for protocol in protocols:
        try:
            get_protocol_adapter(protocol)
        except UnsupportedProtocolError:
            errors.append(f"{label} references unsupported protocol: {protocol}")
    return errors


def _provider_from_mapping(value: Any) -> ModelProvider:
    payload = _mapping(value)
    return ModelProvider(
        id=_required_string(payload, "id"),
        kind=_required_string(payload, "kind"),
        base_url=_optional_string(payload.get("base_url")),
        token_ref=_optional_string(payload.get("token_ref")),
        model=_optional_string(payload.get("model")),
        protocols=tuple(_strings(payload.get("protocols"))),
        metadata=_mapping(payload.get("metadata")),
    )


def _route_from_mapping(value: Any) -> ModelRoute:
    payload = _mapping(value)
    rate_limit_payload = payload.get("rate_limit")
    rate_limit = _rate_limit_from_mapping(rate_limit_payload) if isinstance(rate_limit_payload, Mapping) else None
    return ModelRoute(
        id=_required_string(payload, "id"),
        provider=_required_string(payload, "provider"),
        target=_required_string(payload, "target"),
        protocols=tuple(_strings(payload.get("protocols"))),
        model_names=tuple(_strings(payload.get("model_names"))),
        target_type=_optional_string(payload.get("target_type")),
        priority=int(payload.get("priority", 100)),
        fallback_ids=tuple(_strings(payload.get("fallback_ids"))),
        timeout_seconds=float(payload.get("timeout_seconds", 60.0)),
        rate_limit=rate_limit,
        enabled=bool(payload.get("enabled", True)),
        metadata=_mapping(payload.get("metadata")),
    )


def _rate_limit_from_mapping(value: Mapping[str, Any]) -> RateLimitPolicy:
    return RateLimitPolicy(
        requests=int(value.get("requests", 0)),
        window_seconds=float(value.get("window_seconds", 0)),
        key=str(value.get("key", "tenant")),
    )


def _fallback_from_mapping(value: Mapping[str, Any]) -> FallbackPolicy:
    max_attempts = value.get("max_attempts")
    return FallbackPolicy(
        route_ids=tuple(_strings(value.get("route_ids"))),
        max_attempts=int(max_attempts) if max_attempts is not None else None,
    )


def _items(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [item for item in _items(value) if isinstance(item, str)]


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GatewayConfigurationError(f"{key} must be a non-empty string")
    return value


__all__ = [
    "DraftValidationResult",
    "FallbackPolicy",
    "GatewayConfig",
    "GatewayConfigurationError",
    "GatewayRequest",
    "InMemoryRateLimiter",
    "ModelProvider",
    "ModelRoute",
    "RateLimitDecision",
    "RateLimitPolicy",
    "ResolvedRouteReference",
    "RouteSelection",
    "RouteSelector",
    "resolve_route_reference",
    "validate_gateway_config",
]
