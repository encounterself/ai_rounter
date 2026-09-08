"""Runtime boundary that preserves safe diagnostics instead of swallowing errors."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

try:
    from .gateway_client import GatewayClient, GatewayError, new_correlation_id, safe_error
    from .tracing import TracingConfig, safe_error_attributes, start_span
except ImportError:  # MLflow code_path loads modules without the package prefix.
    from gateway_client import GatewayClient, GatewayError, new_correlation_id, safe_error
    from tracing import TracingConfig, safe_error_attributes, start_span


class LocalServingRuntime:
    """Execute one normalized request and return a response-shaped mapping."""

    def __init__(self, gateway: GatewayClient, tracing: TracingConfig | None = None):
        self.gateway = gateway
        self.tracing = tracing or TracingConfig.from_env()

    def _execute(self, request: Mapping[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
        correlation_id = str(request.get("correlation_id") or new_correlation_id())
        request_id = str(request.get("request_id") or new_correlation_id())
        route = request.get("route") or self.gateway.settings.route
        metadata = request.get("route_metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        span = start_span(
            self.tracing,
            correlation_id=correlation_id,
            request_id=request_id,
            model=request.get("model") or metadata.get("model"),
            route=str(route),
            provider=metadata.get("provider"),
            metadata=metadata,
            payload=request.get("payload"),
        )
        span.__enter__()
        try:
            result = self.gateway.invoke(
                request["payload"],
                route=str(route),
                metadata=metadata,
                stream=bool(request.get("stream", False)),
                correlation_id=correlation_id,
            )
            if isinstance(result, Iterator):
                return self._trace_stream(result, span, correlation_id, route)
            if isinstance(result, Mapping) and result.get("status") == "failed":
                span.finish(error=result.get("error", {}))
            else:
                span.finish()
            return dict(result)
        except GatewayError as exc:
            diagnostic = safe_error(
                exc,
                correlation_id=correlation_id,
                route=str(route),
                base_url=self.gateway.settings.base_url,
                path=(
                    "/mlflow/v1/responses"
                    if self.gateway.settings.protocol == "responses"
                    else "/mlflow/v1/chat/completions"
                ),
                http_status=exc.diagnostic.get("http_status"),
                error_code=exc.diagnostic.get("error_code"),
            )
            diagnostic.update(exc.diagnostic)
            span.finish(error=diagnostic)
            return {"status": "failed", "error": diagnostic}
        except Exception as exc:
            diagnostic = safe_error(
                exc,
                correlation_id=correlation_id,
                route=str(route),
                base_url=self.gateway.settings.base_url,
                path=(
                    "/mlflow/v1/responses"
                    if self.gateway.settings.protocol == "responses"
                    else "/mlflow/v1/chat/completions"
                ),
                error_code="runtime_error",
            )
            span.finish(error=diagnostic)
            return {"status": "failed", "error": diagnostic}

    def _trace_stream(
        self,
        result: Iterator[dict[str, Any]],
        span: Any,
        correlation_id: str,
        route: Any,
    ) -> Iterator[dict[str, Any]]:
        error: Mapping[str, Any] | None = None
        try:
            yield from result
        except GatewayError as exc:
            error = exc.diagnostic
            raise
        except Exception as exc:
            error = safe_error_attributes(
                correlation_id=correlation_id,
                route=str(route),
                host=self.gateway.settings.base_url.split("://", 1)[-1].split("/", 1)[0],
                path=(
                    "/mlflow/v1/responses"
                    if self.gateway.settings.protocol == "responses"
                    else "/mlflow/v1/chat/completions"
                ),
                exception_type=type(exc).__name__,
                error_code="stream_runtime_error",
            )
            raise
        finally:
            span.finish(error=error)
