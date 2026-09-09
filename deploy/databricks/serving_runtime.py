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

    def __init__(self, gateway: GatewayClient, tracing: TracingConfig | None = None) -> None:
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
            session_id=request.get("session_id"),
            turn_id=request.get("turn_id"),
            user_id=request.get("user_id"),
            tenant_id=request.get("tenant_id"),
            scenario=request.get("scenario"),
            agent_profile=request.get("agent_profile"),
            protocol=metadata.get("protocol") or self.gateway.settings.protocol,
            endpoint_name=metadata.get("endpoint"),
            streamed=bool(request.get("stream", False)),
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
                span.finish(error=result.get("error", {}), attributes=_result_trace_attributes(result))
            else:
                span.finish(attributes=_result_trace_attributes(result))
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
            span.finish(error=diagnostic, attributes={"http_status": diagnostic.get("http_status")})
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
        self, result: Iterator[dict[str, Any]], span: Any, correlation_id: str, route: Any
    ) -> Iterator[dict[str, Any]]:
        error: Mapping[str, Any] | None = None
        try:
            for item in result:
                span.mark_first_token()
                yield item
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


def _result_trace_attributes(result: Mapping[str, Any]) -> dict[str, Any]:
    usage = result.get("usage") if isinstance(result, Mapping) else None
    if not isinstance(usage, Mapping):
        choices = result.get("choices") if isinstance(result, Mapping) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            usage = choices[0].get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    choices = result.get("choices") if isinstance(result, Mapping) else None
    first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, Mapping):
        input_details = {}
    if not isinstance(output_details, Mapping):
        output_details = {}
    return {
        "response_id": result.get("id") if isinstance(result, Mapping) else None,
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens", output_details.get("reasoning_tokens")),
        "cached_tokens": usage.get("cached_tokens", input_details.get("cached_tokens")),
        "finish_reason": first_choice.get("finish_reason"),
        "status": result.get("status") if isinstance(result, Mapping) else None,
    }
