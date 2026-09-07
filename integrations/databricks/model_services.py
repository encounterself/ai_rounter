from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from types import MappingProxyType
from typing import Any

import httpx

MODEL_SERVICES_PATH = "/api/2.1/unity-catalog/model-services"
DEFAULT_ALLOWED_RESOURCE_PREFIX = "model-services/simo_prod.ai_models.route_"
DEFAULT_ALLOWED_MODEL_SERVICE_PREFIX = DEFAULT_ALLOWED_RESOURCE_PREFIX
DEFAULT_PARENT = "schemas/simo_prod.ai_models"
DEFAULT_CACHE_TTL = timedelta(minutes=5)
SYSTEM_AI_PREFIX = "system.ai."


class ModelServicesError(RuntimeError):
    """Base error for the Databricks Model Services registry."""


class ModelServicesHTTPError(ModelServicesError):
    """Raised when the Model Services API returns an unsuccessful response."""

    def __init__(self, status_code: int, url: str, body: str) -> None:
        self.status_code = status_code
        self.url = url
        self.body = body
        super().__init__(f"Databricks Model Services API returned HTTP {status_code} for {url}: {body}")


class ModelServicesPayloadError(ModelServicesError):
    """Raised when the Model Services API response cannot be parsed."""


class ModelServiceNotAllowedError(ModelServicesError):
    """Raised when a caller tries to resolve a non-selectable model service."""


@dataclass(frozen=True)
class ModelServiceRoute:
    """A normalized primary or fallback route exposed by a model service."""

    name: str | None = None
    model_name: str | None = None
    route_type: str | None = None
    traffic_percentage: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def destination(self) -> str | None:
        """Return the most useful destination identifier for the route."""

        return self.model_name or self.name

    def to_dict(self) -> dict[str, Any]:
        """Serialize the route into a JSON-compatible dictionary."""

        return {
            "name": self.name,
            "model_name": self.model_name,
            "route_type": self.route_type,
            "traffic_percentage": self.traffic_percentage,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelServiceRateLimit:
    """A normalized rate-limit rule attached to a model service."""

    key: str | None = None
    calls: int | None = None
    tokens: int | None = None
    renewal_period: int | None = None
    renewal_period_unit: str | None = None
    principal: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def requests(self) -> int | None:
        """Return the request count using the common ``requests`` name."""

        return self.calls

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rate limit into a JSON-compatible dictionary."""

        return {
            "key": self.key,
            "calls": self.calls,
            "tokens": self.tokens,
            "renewal_period": self.renewal_period,
            "renewal_period_unit": self.renewal_period_unit,
            "principal": self.principal,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelServiceCapabilitySnapshot:
    """The capability view of one model service at registry refresh time."""

    supported_api_types: tuple[str, ...] = ()
    primary_routes: tuple[ModelServiceRoute, ...] = ()
    fallback_routes: tuple[ModelServiceRoute, ...] = ()
    rate_limits: tuple[ModelServiceRateLimit, ...] = ()
    etag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the capability snapshot into a JSON-compatible dictionary."""

        return {
            "supported_api_types": list(self.supported_api_types),
            "primary_routes": [route.to_dict() for route in self.primary_routes],
            "fallback_routes": [route.to_dict() for route in self.fallback_routes],
            "rate_limits": [rate_limit.to_dict() for rate_limit in self.rate_limits],
            "etag": self.etag,
        }


@dataclass(frozen=True)
class ModelService:
    """A user-selectable Databricks Unity Catalog model service."""

    name: str
    supported_api_types: tuple[str, ...] = ()
    primary_routes: tuple[ModelServiceRoute, ...] = ()
    fallback_routes: tuple[ModelServiceRoute, ...] = ()
    rate_limits: tuple[ModelServiceRateLimit, ...] = ()
    etag: str | None = None
    expires_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def capability_snapshot(self) -> ModelServiceCapabilitySnapshot:
        """Return the normalized capabilities associated with this service."""

        return ModelServiceCapabilitySnapshot(
            supported_api_types=self.supported_api_types,
            primary_routes=self.primary_routes,
            fallback_routes=self.fallback_routes,
            rate_limits=self.rate_limits,
            etag=self.etag,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model service into a JSON-compatible dictionary."""

        return {
            "name": self.name,
            "supported_api_types": list(self.supported_api_types),
            "primary_routes": [route.to_dict() for route in self.primary_routes],
            "fallback_routes": [route.to_dict() for route in self.fallback_routes],
            "rate_limits": [rate_limit.to_dict() for rate_limit in self.rate_limits],
            "etag": self.etag,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelServiceSnapshot:
    """An immutable, time-bounded registry snapshot."""

    services: tuple[ModelService, ...]
    etag: str
    capability_etag: str
    fetched_at: datetime
    expires_at: datetime
    capability_snapshot: Mapping[str, ModelServiceCapabilitySnapshot]

    @property
    def is_expired(self) -> bool:
        """Return whether the snapshot has expired relative to the current UTC time."""

        return self.is_expired_at(datetime.now(timezone.utc))

    def is_expired_at(self, now: datetime) -> bool:
        """Return whether the snapshot has expired relative to ``now``."""

        return _ensure_aware(now) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry snapshot into a JSON-compatible dictionary."""

        return {
            "services": [service.to_dict() for service in self.services],
            "etag": self.etag,
            "capability_etag": self.capability_etag,
            "fetched_at": self.fetched_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "capability_snapshot": {
                name: capability.to_dict() for name, capability in self.capability_snapshot.items()
            },
        }


class DatabricksModelServiceRegistry:
    """
    Dynamically discover and cache selectable Databricks model services.

    The registry uses the Unity Catalog Model Services REST API directly. A caller may inject an ``httpx`` transport
    for deterministic tests or custom network behavior; no Databricks SDK is required.
    """

    def __init__(
        self,
        host: str | None = None,
        *,
        base_url: str | None = None,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
        http_client: httpx.Client | None = None,
        headers: Mapping[str, str] | None = None,
        allowed_resource_prefix: str = DEFAULT_ALLOWED_RESOURCE_PREFIX,
        parent: str = DEFAULT_PARENT,
        cache_ttl: timedelta | float = DEFAULT_CACHE_TTL,
        page_size: int = 100,
        timeout: float | httpx.Timeout = 30.0,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        resolved_base_url = base_url or host
        if not resolved_base_url:
            raise ValueError("Either 'host' or 'base_url' must be provided.")
        if client is not None and http_client is not None:
            raise ValueError("Provide only one of 'client' and 'http_client'.")
        if (client is not None or http_client is not None) and transport is not None:
            raise ValueError("'transport' cannot be combined with an injected client.")
        if page_size < 1:
            raise ValueError("'page_size' must be greater than zero.")

        if isinstance(cache_ttl, timedelta):
            resolved_cache_ttl = cache_ttl
        else:
            resolved_cache_ttl = timedelta(seconds=cache_ttl)
        if resolved_cache_ttl.total_seconds() < 0:
            raise ValueError("'cache_ttl' cannot be negative.")

        default_headers = {"Accept": "application/json"}
        if token:
            default_headers["Authorization"] = f"Bearer {token}"
        if headers:
            default_headers.update(headers)

        self.base_url = resolved_base_url.rstrip("/")
        self.allowed_resource_prefix = allowed_resource_prefix
        self.parent = parent
        self.cache_ttl = resolved_cache_ttl
        self.page_size = page_size
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._snapshot: ModelServiceSnapshot | None = None
        self._owns_client = client is None and http_client is None
        self._client = (
            client
            or http_client
            or httpx.Client(base_url=self.base_url, headers=default_headers, timeout=timeout, transport=transport)
        )
        if not self._owns_client:
            self._client.headers.update(default_headers)

    @property
    def snapshot(self) -> ModelServiceSnapshot | None:
        """Return the most recently loaded snapshot, if one exists."""

        return self._snapshot

    def close(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "DatabricksModelServiceRegistry":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def refresh(self, *, force: bool = False) -> ModelServiceSnapshot:
        """Refresh the registry unless a non-expired snapshot can be reused."""

        now = _ensure_aware(self._now())
        current = self._snapshot
        if current is not None and not force and not current.is_expired_at(now):
            return current

        request_headers: dict[str, str] = {}
        if current is not None and current.etag:
            request_headers["If-None-Match"] = current.etag

        raw_services: list[Mapping[str, Any]] = []
        page_token: str | None = None
        response_etag: str | None = None
        response_expiry: datetime | None = None
        response_headers: Mapping[str, str] | None = None
        first_page = True

        while True:
            params = {"parent": self.parent, "view": "FULL", "page_size": str(self.page_size)}
            if page_token:
                params["page_token"] = page_token
            response = self._client.get(
                MODEL_SERVICES_PATH, params=params, headers=request_headers if first_page else None
            )

            if first_page and response.status_code == httpx.codes.NOT_MODIFIED:
                if current is None:
                    raise ModelServicesPayloadError("Received HTTP 304 without an existing snapshot.")
                renewed = self._renew_snapshot(current, now, response)
                self._snapshot = renewed
                return renewed

            if response.status_code >= 400:
                raise ModelServicesHTTPError(response.status_code, str(response.url), response.text)

            response_headers = response_headers or response.headers
            try:
                payload = response.json()
            except ValueError as error:
                raise ModelServicesPayloadError("The Model Services API returned invalid JSON.") from error

            raw_services.extend(_extract_service_items(payload))
            response_etag = response_etag or response.headers.get("etag")
            response_expiry = response_expiry or _extract_expiry(payload)
            page_token = _extract_page_token(payload)
            if not page_token:
                break
            if not first_page and page_token == params.get("page_token"):
                raise ModelServicesPayloadError("The Model Services API returned a repeated page token.")
            first_page = False
            request_headers = {}

        services = tuple(
            service
            for raw_service in raw_services
            if (service := _parse_model_service(raw_service)) is not None and self.is_user_selectable(service.name)
        )
        header_expiry = _expiry_from_headers(response_headers or {}, now, self.cache_ttl)
        expires_at = _earliest_datetime([header_expiry, response_expiry, *(service.expires_at for service in services)])
        if expires_at is None:
            expires_at = now + self.cache_ttl
        expires_at = max(expires_at, now)

        capability_snapshot = MappingProxyType({service.name: service.capability_snapshot for service in services})
        capability_etag = _digest({name: capability.to_dict() for name, capability in capability_snapshot.items()})
        etag = response_etag or _digest([service.to_dict() for service in services])
        snapshot = ModelServiceSnapshot(
            services=services,
            etag=etag,
            capability_etag=capability_etag,
            fetched_at=now,
            expires_at=expires_at,
            capability_snapshot=capability_snapshot,
        )
        self._snapshot = snapshot
        return snapshot

    def list_services(self, *, force_refresh: bool = False) -> tuple[ModelService, ...]:
        """Return only services allowed for user selection."""

        return self.refresh(force=force_refresh).services

    def get(self, name: str, *, force_refresh: bool = False) -> ModelService:
        """Resolve one selectable model service by its resource name."""

        if not self.is_user_selectable(name):
            raise ModelServiceNotAllowedError(f"Model service '{name}' is not user-selectable.")
        for service in self.list_services(force_refresh=force_refresh):
            if service.name == name:
                return service
        raise KeyError(name)

    resolve = get

    def is_user_selectable(self, resource_name: str) -> bool:
        """Return whether a resource is visible and selectable by users."""

        normalized_name = resource_name.strip().lower()
        if not normalized_name or SYSTEM_AI_PREFIX in normalized_name:
            return False
        return normalized_name.startswith(self.allowed_resource_prefix.lower())

    def _renew_snapshot(
        self, current: ModelServiceSnapshot, now: datetime, response: httpx.Response
    ) -> ModelServiceSnapshot:
        expires_at = _expiry_from_headers(response.headers, now, self.cache_ttl)
        if expires_at is None:
            expires_at = now + self.cache_ttl
        return ModelServiceSnapshot(
            services=current.services,
            etag=response.headers.get("etag") or current.etag,
            capability_etag=current.capability_etag,
            fetched_at=now,
            expires_at=expires_at,
            capability_snapshot=current.capability_snapshot,
        )


ModelServicesRegistry = DatabricksModelServiceRegistry


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return (value,)


def _first(value: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return None


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _number(value: Any, integer: bool = False) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if integer else float(value)
    if isinstance(value, str):
        try:
            return int(value) if integer else float(value)
        except ValueError:
            return None
    return None


def _integer(value: Any) -> int | None:
    result = _number(value, integer=True)
    return result if isinstance(result, int) else None


def _strings(value: Any) -> tuple[str, ...]:
    values = [value] if isinstance(value, str) else value if isinstance(value, Sequence) else ()
    return tuple(dict.fromkeys(item.strip() for item in values if isinstance(item, str) and item.strip()))


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return _ensure_aware(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        try:
            return _ensure_aware(parsedate_to_datetime(text))
        except (TypeError, ValueError, OverflowError):
            return None


def _extract_expiry(value: Any) -> datetime | None:
    mapping = _as_mapping(value)
    candidates = [
        _parse_datetime(
            _first(mapping, "expires_at", "expiresAt", "expiration_time", "expirationTime", "expire_at", "expireAt")
        )
    ]
    services = _extract_service_items(value)
    candidates.extend(_extract_expiry(service) for service in services)
    return _earliest_datetime(candidates)


def _extract_page_token(payload: Any) -> str | None:
    return _string(_first(_as_mapping(payload), "next_page_token", "nextPageToken", "next_token"))


def _extract_service_items(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        return [item for item in payload if isinstance(item, Mapping)]
    mapping = _as_mapping(payload)
    for key in ("model_services", "modelServices", "services", "items"):
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _parse_model_service(raw: Mapping[str, Any]) -> ModelService | None:
    nested = _as_mapping(_first(raw, "model_service", "modelService"))
    payload = {**nested, **raw}
    name = _string(_first(payload, "name", "resource_name", "resourceName", "full_name", "fullName"))
    if name is None:
        return None

    config = _as_mapping(_first(payload, "config", "configuration"))
    capabilities = _as_mapping(_first(payload, "capabilities", "capability"))
    routing = _as_mapping(_first(payload, "routing", "routes"))
    config_routing = _as_mapping(_first(config, "routing", "routes"))
    routing = {**config_routing, **routing}

    supported_api_types = _strings(
        _first(payload, "supported_api_types", "supportedApiTypes")
        or _first(capabilities, "supported_api_types", "supportedApiTypes")
        or _first(config, "supported_api_types", "supportedApiTypes")
    )
    primary_raw = _first(payload, "primary_routes", "primaryRoutes", "primary_route", "primaryRoute")
    primary_raw = primary_raw or _first(routing, "primary_routes", "primaryRoutes", "primary")
    fallback_raw = _first(payload, "fallback_routes", "fallbackRoutes", "fallback_route", "fallbackRoute")
    fallback_raw = fallback_raw or _first(routing, "fallback_routes", "fallbackRoutes", "fallback")

    all_routes = _first(routing, "routes", "destinations")
    if not primary_raw and not fallback_raw and all_routes:
        primary_items: list[Any] = []
        fallback_items: list[Any] = []
        for item in _as_sequence(all_routes):
            if isinstance(item, Mapping) and item.get("fallback") is True:
                fallback_items.append(item)
            else:
                primary_items.append(item)
        primary_raw = primary_items
        fallback_raw = fallback_items

    rate_limits_raw = _first(payload, "rate_limits", "rateLimits")
    rate_limits_raw = rate_limits_raw or _first(config, "rate_limits", "rateLimits")
    service_expiry = _extract_expiry(payload)
    etag = _string(_first(payload, "etag", "e_tag", "eTag"))
    ignored_keys = {
        "name",
        "resource_name",
        "resourceName",
        "full_name",
        "fullName",
        "model_service",
        "modelService",
        "config",
        "configuration",
        "capabilities",
        "capability",
        "routing",
        "routes",
        "primary_routes",
        "primaryRoutes",
        "primary_route",
        "primaryRoute",
        "fallback_routes",
        "fallbackRoutes",
        "fallback_route",
        "fallbackRoute",
        "rate_limits",
        "rateLimits",
        "supported_api_types",
        "supportedApiTypes",
        "etag",
        "e_tag",
        "eTag",
    }
    return ModelService(
        name=name,
        supported_api_types=supported_api_types,
        primary_routes=tuple(_parse_route(item) for item in _as_sequence(primary_raw)),
        fallback_routes=tuple(_parse_route(item) for item in _as_sequence(fallback_raw)),
        rate_limits=tuple(_parse_rate_limit(item) for item in _as_sequence(rate_limits_raw)),
        etag=etag,
        expires_at=service_expiry,
        metadata={key: value for key, value in payload.items() if key not in ignored_keys},
    )


def _parse_route(value: Any) -> ModelServiceRoute:
    if isinstance(value, str):
        return ModelServiceRoute(name=value)
    payload = _as_mapping(value)
    return ModelServiceRoute(
        name=_string(_first(payload, "name", "route_name", "routeName", "id")),
        model_name=_string(_first(payload, "model_name", "modelName", "served_model_name", "endpoint_name")),
        route_type=_string(_first(payload, "type", "route_type", "routeType", "destination_type")),
        traffic_percentage=_number(
            _first(payload, "traffic_percentage", "trafficPercentage", "traffic_percent", "weight")
        ),
        metadata=dict(payload),
    )


def _parse_rate_limit(value: Any) -> ModelServiceRateLimit:
    if isinstance(value, str):
        return ModelServiceRateLimit(key=value)
    payload = _as_mapping(value)
    return ModelServiceRateLimit(
        key=_string(_first(payload, "key", "name", "scope")),
        calls=_integer(_first(payload, "calls", "requests", "max_requests", "limit", "value")),
        tokens=_integer(_first(payload, "tokens", "max_tokens", "token_limit")),
        renewal_period=_integer(_first(payload, "renewal_period", "renewalPeriod", "period", "window")),
        renewal_period_unit=_string(_first(payload, "renewal_period_unit", "renewalPeriodUnit", "period_unit", "unit")),
        principal=_string(_first(payload, "principal", "principal_type", "principalType")),
        metadata=dict(payload),
    )


def _expiry_from_headers(headers: Mapping[str, str], now: datetime, default_ttl: timedelta) -> datetime | None:
    cache_control = headers.get("cache-control", "")
    if "no-cache" in cache_control.lower() or "no-store" in cache_control.lower():
        return now
    match = re.search(r"(?:^|,)\s*max-age\s*=\s*(\d+)", cache_control, flags=re.IGNORECASE)
    if match:
        return now + timedelta(seconds=int(match.group(1)))
    expires = _parse_datetime(headers.get("expires"))
    if expires is not None:
        return expires
    return now + default_ttl


def _earliest_datetime(values: Sequence[datetime | None]) -> datetime | None:
    valid_values = [_ensure_aware(value) for value in values if value is not None]
    return min(valid_values) if valid_values else None


def _digest(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"
