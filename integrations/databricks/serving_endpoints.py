"""Read-only helpers for Databricks Model Serving endpoints.

This module deliberately does not create or update endpoints.  It provides the
small REST surface needed by a control plane and keeps endpoint payload parsing
separate from the gateway runtime.  Callers must inject a workspace ``host``
and, when required, a ``token``.  ``profile`` is only descriptive metadata; it
is never resolved implicitly from a local Databricks configuration file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

SERVING_ENDPOINTS_PATH = "/api/2.0/serving-endpoints"


class ServingEndpointsError(RuntimeError):
    """Base error raised by the read-only Serving Endpoints client."""


class ServingEndpointsHTTPError(ServingEndpointsError):
    """Raised for an unsuccessful Databricks Serving API response."""

    def __init__(
        self,
        status_code: int,
        url: str,
        body: str = "",
        *,
        operation: str = "request",
        profile: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.operation = operation
        self.profile = profile
        self.body = _redact_text(body)
        detail = f": {self.body}" if self.body else ""
        profile_detail = f" (profile={profile})" if profile else ""
        super().__init__(
            f"Databricks Serving Endpoints {operation} failed with HTTP {status_code} "
            f"for {_redact_text(url)}{profile_detail}{detail}"
        )


class ServingEndpointsTransportError(ServingEndpointsError):
    """Raised when an endpoint request cannot reach the workspace."""


class ServingEndpointsPayloadError(ServingEndpointsError):
    """Raised when a Databricks endpoint payload is malformed."""


@dataclass(frozen=True, slots=True)
class ServingEntitySummary:
    """Non-sensitive information about one served entity."""

    name: str | None = None
    entity_name: str | None = None
    entity_version: str | None = None
    external_model_name: str | None = None
    task: str | None = None
    workload_size: str | None = None
    scale_to_zero_enabled: bool | None = None


@dataclass(frozen=True, slots=True)
class ServingEndpointSummary:
    """A normalized, safe-to-display endpoint summary."""

    name: str
    endpoint_id: str | None = None
    state: str | None = None
    config_update: str | None = None
    ready: bool | None = None
    state_message: str | None = None
    served_entities: tuple[ServingEntitySummary, ...] = ()
    ai_gateway: Mapping[str, Any] | None = None
    route_optimization: Mapping[str, Any] | None = None
    config: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_ready(self) -> bool:
        """Return whether the endpoint reports a ready state."""

        return self.ready is True or (self.ready is None and self.state == "READY")

    @property
    def served_entity_names(self) -> tuple[str, ...]:
        """Return the names of entities that can be shown in a UI."""

        return tuple(entity.name for entity in self.served_entities if entity.name)


@dataclass(frozen=True, slots=True)
class ServingOpenAPI:
    """OpenAPI document returned by the endpoint metadata API."""

    document: Mapping[str, Any]

    @property
    def paths(self) -> Mapping[str, Any]:
        """Return the endpoint paths from the OpenAPI document."""

        paths = self.document.get("paths", {})
        return paths if isinstance(paths, Mapping) else {}


def summarize_endpoint(payload: Mapping[str, Any]) -> ServingEndpointSummary:
    """Parse a Databricks endpoint payload into a safe summary."""

    name = _string(payload.get("name"))
    if not name:
        raise ServingEndpointsPayloadError("Serving endpoint payload is missing 'name'.")

    state_payload = _mapping(payload.get("state"))
    config_payload = _mapping(payload.get("config"))
    if not config_payload:
        config_payload = _mapping(payload.get("pending_config"))
    served_entities = _sequence(config_payload.get("served_entities"))
    if not served_entities:
        served_entities = _sequence(config_payload.get("served_models"))

    entities = tuple(_parse_served_entity(item) for item in served_entities if isinstance(item, Mapping))
    state = _string(state_payload.get("ready")) or _string(payload.get("state"))
    config_update = _string(state_payload.get("config_update")) or _string(payload.get("config_update"))
    state_message = _string(state_payload.get("message")) or _string(payload.get("state_message"))
    ready = _parse_ready(state_payload.get("ready"), payload.get("ready"))

    ai_gateway = _safe_mapping(config_payload.get("ai_gateway") or payload.get("ai_gateway"))
    route_optimization = _safe_mapping(
        config_payload.get("route_optimization_config") or payload.get("route_optimization_config")
    )
    return ServingEndpointSummary(
        name=name,
        endpoint_id=_string(payload.get("id")) or _string(payload.get("endpoint_id")),
        state=state,
        config_update=config_update,
        ready=ready,
        state_message=state_message,
        served_entities=entities,
        ai_gateway=ai_gateway,
        route_optimization=route_optimization,
        config=_safe_mapping(config_payload) or {},
        raw=_safe_mapping(payload) or {},
    )


class DatabricksServingEndpointsClient:
    """Read-only Databricks Serving Endpoints REST client.

    ``profile`` is intentionally not used to discover credentials.  It is
    retained on errors and summaries so an application can report which
    explicitly selected profile it used.  Supplying an ``httpx.Client`` is the
    preferred way to integrate custom auth, retry, or mock transports.
    """

    def __init__(
        self,
        host: str | None = None,
        *,
        token: str | None = None,
        profile: str | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | httpx.Timeout = 30.0,
    ) -> None:
        if not host:
            raise ValueError("A Databricks workspace host must be injected explicitly.")
        self.host = host.rstrip("/")
        self.profile = profile
        self._owns_client = client is None
        headers: dict[str, str] = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.Client(base_url=self.host, headers=headers, timeout=timeout, transport=transport)
        if client is not None:
            self._client.headers.update(headers)

    def close(self) -> None:
        """Close the HTTP client when it was created by this instance."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "DatabricksServingEndpointsClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def list_endpoints(self, *, page_size: int | None = None) -> tuple[ServingEndpointSummary, ...]:
        """List endpoints, following Databricks pagination tokens."""

        endpoints: list[ServingEndpointSummary] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, str] = {}
            if page_size is not None:
                params["page_size"] = str(page_size)
            if page_token:
                params["page_token"] = page_token
            payload = self._request_json("GET", SERVING_ENDPOINTS_PATH, params=params, operation="list")
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, Mapping):
                items = payload.get("endpoints", payload.get("serving_endpoints", []))
            else:
                items = []
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
                raise ServingEndpointsPayloadError("Serving endpoint list payload is missing an endpoint array.")
            endpoints.extend(summarize_endpoint(item) for item in items if isinstance(item, Mapping))
            next_token = _string(payload.get("next_page_token")) if isinstance(payload, Mapping) else None
            if not next_token:
                break
            if next_token in seen_tokens:
                raise ServingEndpointsPayloadError("Serving endpoint list returned a repeated page token.")
            seen_tokens.add(next_token)
            page_token = next_token
        return tuple(endpoints)

    def get_endpoint(self, name: str) -> ServingEndpointSummary:
        """Fetch and normalize one endpoint by name."""

        if not name:
            raise ValueError("Endpoint name must not be empty.")
        payload = self._request_json("GET", f"{SERVING_ENDPOINTS_PATH}/{quote(name, safe='')}", operation="get")
        if not isinstance(payload, Mapping):
            raise ServingEndpointsPayloadError("Serving endpoint get payload must be an object.")
        return summarize_endpoint(payload)

    def get_openapi(self, name: str) -> ServingOpenAPI:
        """Fetch the read-only OpenAPI document exposed by an endpoint."""

        payload = self._request_json(
            "GET", f"{SERVING_ENDPOINTS_PATH}/{quote(name, safe='')}/openapi", operation="get-openapi"
        )
        if not isinstance(payload, Mapping):
            raise ServingEndpointsPayloadError("Serving endpoint OpenAPI payload must be an object.")
        return ServingOpenAPI(document=payload)

    def get_config_summary(self, name: str) -> Mapping[str, Any]:
        """Return the non-sensitive configuration summary for an endpoint."""

        return self.get_endpoint(name).config

    def _request_json(self, method: str, path: str, *, operation: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as error:
            raise ServingEndpointsTransportError(
                f"Databricks Serving Endpoints {operation} could not reach the workspace: "
                f"{_redact_text(str(error))}"
            ) from error
        if response.status_code >= 400:
            raise ServingEndpointsHTTPError(
                response.status_code,
                str(response.url),
                response.text,
                operation=operation,
                profile=self.profile,
            )
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError) as error:
            raise ServingEndpointsPayloadError(
                f"Databricks Serving Endpoints {operation} returned invalid JSON."
            ) from error


def _parse_served_entity(value: Mapping[str, Any]) -> ServingEntitySummary:
    return ServingEntitySummary(
        name=_string(value.get("name")),
        entity_name=_string(value.get("entity_name")),
        entity_version=_string(value.get("entity_version")),
        external_model_name=_string(value.get("external_model_name")),
        task=_string(value.get("task")),
        workload_size=_string(value.get("workload_size")),
        scale_to_zero_enabled=(
            value.get("scale_to_zero_enabled") if isinstance(value.get("scale_to_zero_enabled"), bool) else None
        ),
    )


def _parse_ready(state: Any, payload_ready: Any) -> bool | None:
    for value in (payload_ready, state):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.upper() in {"READY", "NOT_READY"}:
            return value.upper() == "READY"
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_mapping(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): _safe_value(item, str(key)) for key, item in value.items()}


def _safe_value(value: Any, key: str = "") -> Any:
    if _is_sensitive_key(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): _safe_value(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_safe_value(item, key) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_value(item, key) for item in value)
    return value


def _is_sensitive_key(key: str) -> bool:
    return bool(re.search(r"(?:token|secret|password|authorization|api[_-]?key|credential)", key, re.IGNORECASE))


def _redact_text(value: str) -> str:
    redacted = re.sub(r"Bearer\s+[^\s,;]+", "Bearer [REDACTED]", value, flags=re.IGNORECASE)
    redacted = re.sub(
        r"([\"']?(?:token|secret|password|authorization|api[_-]?key|credential)[\"']?\s*[:=]\s*[\"']?)([^\"'\s,;}]+)",
        r"\1[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "SERVING_ENDPOINTS_PATH",
    "DatabricksServingEndpointsClient",
    "ServingEndpointSummary",
    "ServingEndpointsError",
    "ServingEndpointsHTTPError",
    "ServingEndpointsPayloadError",
    "ServingEndpointsTransportError",
    "ServingEntitySummary",
    "ServingOpenAPI",
    "summarize_endpoint",
]
