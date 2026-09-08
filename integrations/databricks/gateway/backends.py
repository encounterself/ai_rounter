"""HTTP backend implementations used by the provider-neutral gateway."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

import httpx

from ..protocols import (
    InternalRequest,
    InternalResponse,
    InternalStreamEnvelope,
    ProtocolAdapterError,
    get_protocol_adapter,
    parse_stream,
)


class GatewayBackendError(RuntimeError):
    """Base error raised by a gateway backend."""

    def __init__(self, message: str, *, retryable: bool = True, status_code: int | None = None) -> None:
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(_redact(message))


class GatewayBackendHTTPError(GatewayBackendError):
    """Raised for an unsuccessful upstream provider response."""

    def __init__(self, status_code: int, url: str, body: str) -> None:
        super().__init__(
            f"upstream gateway request failed with HTTP {status_code} for {_redact(url)}: {_redact(body)}",
            retryable=status_code >= 500 or status_code == 429,
            status_code=status_code,
        )


class GatewayBackendPayloadError(GatewayBackendError):
    """Raised when an upstream response cannot be parsed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


@dataclass(frozen=True, slots=True)
class BackendRequestOptions:
    """Provider request options shared by the HTTP backends."""

    protocol: str
    path: str | None = None
    model: str | None = None


@runtime_checkable
class GatewayBackend(Protocol):
    """Synchronous backend interface for the gateway runtime."""

    protocol: str

    def invoke(self, request: InternalRequest, *, request_id: str, timeout: float | None = None) -> InternalResponse:
        """Send one non-streaming request and normalize its response."""

        raise NotImplementedError

    def stream(
        self, request: InternalRequest, *, request_id: str, timeout: float | None = None
    ) -> Iterator[InternalStreamEnvelope]:
        """Send a streaming request and yield normalized events."""

        raise NotImplementedError


class _HTTPGatewayBackend:
    """Common HTTP request and SSE parsing behavior."""

    def __init__(
        self,
        base_url: str,
        *,
        protocol: str,
        path: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | httpx.Timeout = 60.0,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.protocol = protocol
        self.options = BackendRequestOptions(protocol=protocol, path=path, model=model)
        self._owns_client = client is None
        default_headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)
        self._client = client or httpx.Client(
            base_url=self.base_url, headers=default_headers, timeout=timeout, transport=transport
        )
        if client is not None:
            self._client.headers.update(default_headers)

    def close(self) -> None:
        """Close an internally created HTTP client."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "_HTTPGatewayBackend":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def invoke(self, request: InternalRequest, *, request_id: str, timeout: float | None = None) -> InternalResponse:
        """Render, send, and parse a provider-neutral request."""

        payload = self._render_request(request)
        try:
            response = self._client.post(
                self._request_path(),
                json=payload,
                headers=self._request_headers(request_id),
                timeout=timeout,
            )
        except httpx.RequestError as error:
            raise GatewayBackendError(f"upstream gateway request failed: {error}") from error
        self._check_response(response)
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as error:
            raise GatewayBackendPayloadError("upstream gateway response was not valid JSON") from error
        try:
            return get_protocol_adapter(self.protocol).parse_response(body)
        except ProtocolAdapterError as error:
            raise GatewayBackendPayloadError(f"upstream {self.protocol} response was invalid") from error

    def stream(
        self, request: InternalRequest, *, request_id: str, timeout: float | None = None
    ) -> Iterator[InternalStreamEnvelope]:
        """Render a streaming request and normalize SSE or JSON-line events."""

        payload = self._render_request(replace(request, stream=True))
        try:
            response_context = self._client.stream(
                "POST",
                self._request_path(),
                json=payload,
                headers=self._request_headers(request_id),
                timeout=timeout,
            )
            with response_context as response:
                self._check_response(response)
                for line in response.iter_lines():
                    event_payload = _stream_payload(line)
                    if event_payload is None:
                        continue
                    try:
                        yield parse_stream(self.protocol, event_payload)
                    except ProtocolAdapterError as error:
                        raise GatewayBackendPayloadError(
                            f"upstream {self.protocol} stream event was invalid"
                        ) from error
        except httpx.RequestError as error:
            raise GatewayBackendError(f"upstream gateway stream failed: {error}") from error

    def _render_request(self, request: InternalRequest) -> dict[str, Any]:
        if self.options.model and request.model != self.options.model:
            request = replace(request, model=self.options.model)
        return get_protocol_adapter(self.protocol).render_request(request)

    def _request_path(self) -> str:
        path = self.options.path or get_protocol_adapter(self.protocol).endpoint
        return "/" + path.lstrip("/")

    @staticmethod
    def _request_headers(request_id: str) -> dict[str, str]:
        return {"X-Request-ID": request_id}

    @staticmethod
    def _check_response(response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise GatewayBackendHTTPError(response.status_code, str(response.url), response.text)


class DatabricksServingBackend(_HTTPGatewayBackend):
    """Backend for a Databricks Serving Endpoint invocation path."""

    def __init__(
        self,
        host: str,
        endpoint_name: str,
        *,
        protocol: str = "mlflow_chat",
        token: str | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | httpx.Timeout = 60.0,
        model: str | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        path = f"/serving-endpoints/{quote(endpoint_name, safe='')}/invocations"
        super().__init__(
            host,
            protocol=protocol,
            path=path,
            headers=headers,
            client=client,
            transport=transport,
            timeout=timeout,
            model=model,
        )
        self.endpoint_name = endpoint_name


class DatabricksFoundationModelBackend(DatabricksServingBackend):
    """Backend for a Databricks ``system.ai`` chat endpoint.

    Foundation Model API endpoints use the OpenAI-compatible ``llm/v1/chat``
    contract.  The invocation URL is still the standard Serving Endpoint path,
    but naming the backend explicitly prevents callers from accidentally using
    an MLflow Responses protocol for a chat endpoint.
    """

    def __init__(
        self,
        host: str,
        endpoint_name: str,
        *,
        token: str | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | httpx.Timeout = 60.0,
        model: str | None = None,
    ) -> None:
        super().__init__(
            host,
            endpoint_name,
            protocol="chat_completions",
            token=token,
            client=client,
            transport=transport,
            timeout=timeout,
            model=model,
        )


class OpenAICompatibleBackend(_HTTPGatewayBackend):
    """Backend for an explicitly configured OpenAI/Codex-compatible HTTP API.

    Codex integration is intentionally HTTP-only: this class never invokes a
    shell command or a local CLI.  The caller supplies the provider base URL,
    optional API key, and protocol/path explicitly.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        protocol: str = "openai_responses",
        path: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | httpx.Timeout = 60.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(
            base_url,
            protocol=protocol,
            path=path,
            headers=headers,
            client=client,
            transport=transport,
            timeout=timeout,
            model=model,
        )


class AnthropicBackend(_HTTPGatewayBackend):
    """Backend for the Anthropic Messages HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        protocol: str = "anthropic_messages",
        path: str = "/v1/messages",
        model: str | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | httpx.Timeout = 60.0,
    ) -> None:
        headers = {"anthropic-version": "2023-06-01"}
        if api_key:
            headers["x-api-key"] = api_key
        super().__init__(
            base_url,
            protocol=protocol,
            path=path,
            headers=headers,
            client=client,
            transport=transport,
            timeout=timeout,
            model=model,
        )


def _stream_payload(line: str) -> Mapping[str, Any] | str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("event:"):
        return None
    if stripped.startswith("data:"):
        stripped = stripped[5:].strip()
    if not stripped:
        return None
    if stripped == "[DONE]":
        return stripped
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise GatewayBackendPayloadError("upstream stream line was not valid JSON") from error
    if not isinstance(value, Mapping):
        raise GatewayBackendPayloadError("upstream stream event must be an object")
    return value


def _redact(value: str) -> str:
    result = re.sub(r"Bearer\s+[^\s,;]+", "Bearer [REDACTED]", value, flags=re.IGNORECASE)
    result = re.sub(
        r"([\"']?(?:token|secret|password|authorization|api[_-]?key|credential)[\"']?\s*[:=]\s*[\"']?)([^\"'\s,;}]+)",
        r"\1[REDACTED]",
        result,
        flags=re.IGNORECASE,
    )
    return result


__all__ = [
    "AnthropicBackend",
    "DatabricksFoundationModelBackend",
    "DatabricksServingBackend",
    "GatewayBackend",
    "GatewayBackendError",
    "GatewayBackendHTTPError",
    "GatewayBackendPayloadError",
    "OpenAICompatibleBackend",
]
