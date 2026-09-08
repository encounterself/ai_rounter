"""Provider-neutral request routing runtime for a future Serving handler."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..ai_gateway import (
    GatewayConfig,
    GatewayRequest,
    InMemoryRateLimiter,
    ModelRoute,
    RouteSelection,
    RouteSelector,
)
from ..protocols import (
    InternalError,
    InternalRequest,
    InternalResponse,
    InternalStreamEnvelope,
    ProtocolAdapterError,
    get_protocol_adapter,
    parse_request,
    render_response,
    render_stream,
)
from .backends import GatewayBackend, GatewayBackendError


class GatewayRuntimeError(RuntimeError):
    """Raised when no configured route can complete a gateway request."""


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Request-scoped metadata shared by routing and backend calls."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str | None = None
    timeout_seconds: float | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class GatewayResult:
    """Internal result with the selected route and attempted route IDs."""

    response: InternalResponse
    selection: RouteSelection
    attempts: tuple[str, ...]


class GatewayRuntime:
    """Parse, route, invoke, and render protocol requests.

    The runtime has no web framework dependency.  A future Databricks Serving
    handler can pass decoded JSON into :meth:`handle` or :meth:`stream`.
    """

    def __init__(
        self,
        config: GatewayConfig,
        backends: Mapping[str, GatewayBackend],
        *,
        rate_limiter: InMemoryRateLimiter | None = None,
    ) -> None:
        self.selector = RouteSelector(config)
        self.backends = backends
        self.rate_limiter = rate_limiter or InMemoryRateLimiter()

    def handle(
        self,
        protocol: str,
        payload: Mapping[str, Any],
        *,
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        """Handle a JSON request and render a protocol-specific response."""

        result = self.handle_internal(protocol, payload, context=context)
        return render_response(protocol, result.response)

    def handle_internal(
        self,
        protocol: str,
        payload: Mapping[str, Any],
        *,
        context: RequestContext | None = None,
    ) -> GatewayResult:
        """Handle a request while retaining routing and attempt metadata."""

        request_context = context or RequestContext()
        try:
            request = parse_request(protocol, payload)
            selection = self.selector.select(
                GatewayRequest(
                    protocol=protocol,
                    model=request.model,
                    tenant=request_context.tenant_id,
                    metadata=request_context.metadata,
                )
            )
        except ProtocolAdapterError as error:
            selection = self._fallback_selection(protocol)
            response = InternalResponse(
                error=InternalError(message=str(error), type="invalid_request"), status="failed"
            )
            return GatewayResult(response, selection, ())
        except GatewayRuntimeError:
            raise
        except Exception as error:
            selection = self._fallback_selection(protocol)
            response = InternalResponse(
                error=InternalError(message=str(error), type="routing_error"), status="failed"
            )
            return GatewayResult(response, selection, ())

        attempts: list[str] = []
        last_error: GatewayBackendError | None = None
        for route in selection.candidates:
            attempts.append(route.id)
            decision = self._admit(route, request_context)
            if not decision.allowed:
                last_error = GatewayBackendError(
                    f"route {route.id} is rate limited; retry after {decision.retry_after_seconds:.3f}s",
                    retryable=True,
                    status_code=429,
                )
                continue
            backend = self._backend_for(route)
            if backend is None:
                last_error = GatewayBackendError(f"no backend registered for route {route.id}", retryable=False)
                continue
            try:
                response = backend.invoke(
                    request,
                    request_id=request_context.request_id,
                    timeout=_route_timeout(route, request_context),
                )
                return GatewayResult(response, selection, tuple(attempts))
            except GatewayBackendError as error:
                last_error = error
                if not error.retryable:
                    break

        if last_error is None:
            last_error = GatewayBackendError("no route candidate completed the request", retryable=False)
        response = InternalResponse(
            error=InternalError(message=str(last_error), type="upstream_error", code=last_error.status_code),
            status="failed",
        )
        return GatewayResult(response, selection, tuple(attempts))

    def stream(
        self,
        protocol: str,
        payload: Mapping[str, Any],
        *,
        context: RequestContext | None = None,
    ) -> Iterator[dict[str, Any] | str]:
        """Stream rendered protocol events, falling back before first output."""

        request_context = context or RequestContext()
        try:
            request = parse_request(protocol, payload)
            selection = self.selector.select(
                GatewayRequest(
                    protocol=protocol,
                    model=request.model,
                    tenant=request_context.tenant_id,
                    metadata=request_context.metadata,
                )
            )
        except ProtocolAdapterError as error:
            error_event = InternalStreamEnvelope(
                event_type="error", done=True, error=InternalError(str(error), "invalid_request")
            )
            yield render_stream(protocol, error_event)
            return

        last_error: GatewayBackendError | None = None
        for route in selection.candidates:
            decision = self._admit(route, request_context)
            if not decision.allowed:
                last_error = GatewayBackendError(f"route {route.id} is rate limited", retryable=True, status_code=429)
                continue
            backend = self._backend_for(route)
            if backend is None:
                last_error = GatewayBackendError(f"no backend registered for route {route.id}", retryable=False)
                continue
            emitted = False
            try:
                events = backend.stream(
                    request,
                    request_id=request_context.request_id,
                    timeout=_route_timeout(route, request_context),
                )
                first = next(events, None)
                if first is None:
                    return
                emitted = True
                yield render_stream(protocol, first)
                for event in events:
                    emitted = True
                    yield render_stream(protocol, event)
                return
            except GatewayBackendError as error:
                last_error = error
                if emitted or not error.retryable:
                    break

        message = str(last_error or GatewayRuntimeError("no route candidate completed the stream"))
        yield render_stream(
            protocol,
            InternalStreamEnvelope(event_type="error", done=True, error=InternalError(message, "upstream_error")),
        )

    def _backend_for(self, route: ModelRoute) -> GatewayBackend | None:
        return self.backends.get(route.id) or self.backends.get(route.provider)

    def _admit(self, route: ModelRoute, context: RequestContext):
        key = context.tenant_id or context.request_id
        if route.rate_limit and route.rate_limit.key == "request":
            key = context.request_id
        return self.rate_limiter.allow(route, key=key)

    def _fallback_selection(self, protocol: str) -> RouteSelection:
        candidates = tuple(route for route in self.selector.config.routes if route.enabled)
        if not candidates:
            raise GatewayRuntimeError(f"no routes configured for protocol {protocol}")
        return RouteSelection(primary=candidates[0], candidates=candidates)


def _route_timeout(route: ModelRoute, context: RequestContext) -> float:
    if context.timeout_seconds is None:
        return route.timeout_seconds
    return min(route.timeout_seconds, context.timeout_seconds)


__all__ = ["GatewayResult", "GatewayRuntime", "GatewayRuntimeError", "RequestContext"]
