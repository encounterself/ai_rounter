"""A deliberately small, diagnosable client for Databricks AI Gateway routes."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

DEFAULT_ROUTE = "model-services/simo_prod.ai_models.route_claude-haiku-4-5"
ROUTE_PREFIX = "model-services/"
SYSTEM_AI_ROUTE_PREFIX = "system-ai/"
SERVING_ENDPOINT_ROUTE_PREFIX = "serving-endpoints/"
_ROUTE_RE = re.compile(r"^model-services/[A-Za-z0-9_.-]+$")
_SYSTEM_AI_ROUTE_RE = re.compile(r"^system-ai/[A-Za-z0-9_.-]+$")
_SERVING_ENDPOINT_ROUTE_RE = re.compile(r"^serving-endpoints/[A-Za-z0-9_.-]+$")
_SECRET_KEYS = {"authorization", "token", "access_token", "api_key", "secret"}
_MODEL_SERVING_OAUTH_FILE = "/var/credentials-secret/model-dependencies-oauth-token"


class GatewayError(RuntimeError):
    """Base error carrying a safe, serializable diagnostic."""

    def __init__(self, message: str, diagnostic: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostic = dict(diagnostic)


class ConfigurationError(GatewayError):
    """The local deployment configuration is incomplete or unsafe."""


class ManagedAuthError(GatewayError):
    """Databricks SDK managed authentication failed."""


class RouteValidationError(GatewayError):
    """A model-service route is not approved or lacks metadata."""


class GatewayHTTPError(GatewayError):
    """The gateway returned a non-success HTTP response."""


class GatewayResponseError(GatewayError):
    """The gateway returned a response that cannot be interpreted."""


@dataclass(frozen=True)
class GatewaySettings:
    """Non-secret settings resolved from environment and profile data."""

    workspace_id: str | None
    base_url: str
    route: str
    workspace_host: str | None = None
    allowed_routes: tuple[str, ...] = ()
    protocol: str = "responses"
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls, profile: Mapping[str, Any] | None = None) -> "GatewaySettings":
        profile = profile or {}
        gateway = profile.get("gateway", {})
        if not isinstance(gateway, Mapping):
            raise ConfigurationError("gateway profile must be an object", {"error_code": "invalid_profile"})
        workspace_id = os.getenv("DATABRICKS_WORKSPACE_ID") or _optional_string(gateway.get("workspace_id"))
        configured_base_url = os.getenv("DATABRICKS_AI_GATEWAY_BASE_URL") or gateway.get("base_url")
        base_url = str(configured_base_url or "").rstrip("/")
        workspace_host = (
            os.getenv("DATABRICKS_HOST")
            or _optional_string(gateway.get("workspace_host"))
            or _optional_string(profile.get("workspace_host"))
        )
        if not base_url and workspace_host:
            base_url = f"{workspace_host.rstrip('/')}/ai-gateway/mlflow/v1"
        if not base_url:
            raise ConfigurationError(
                "DATABRICKS_AI_GATEWAY_BASE_URL or DATABRICKS_HOST is required",
                {
                    "error_code": "missing_gateway_base_url",
                    "missing": ["DATABRICKS_AI_GATEWAY_BASE_URL", "DATABRICKS_HOST"],
                },
            )
        routes = gateway.get("allowed_routes", profile.get("allowed_routes", ()))
        allowed = tuple(str(route) for route in routes) if isinstance(routes, (list, tuple)) else ()
        route = str(os.getenv("DATABRICKS_MODEL_ROUTE") or gateway.get("route") or DEFAULT_ROUTE)
        return cls(
            workspace_id=workspace_id,
            base_url=base_url,
            route=route,
            workspace_host=workspace_host.rstrip("/") if workspace_host else None,
            allowed_routes=allowed,
            protocol=str(gateway.get("protocol", "responses")),
            timeout_seconds=float(gateway.get("timeout_seconds", 60.0)),
        )


@dataclass(frozen=True)
class SafeError:
    """Fields safe to expose in a serving response."""

    correlation_id: str
    route: str | None
    provider_host: str | None
    provider_path: str | None
    exception_type: str
    http_status: int | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "correlation_id": self.correlation_id,
                "route": self.route,
                "provider_host": self.provider_host,
                "provider_path": self.provider_path,
                "exception_type": self.exception_type,
                "http_status": self.http_status,
                "error_code": self.error_code,
            }.items()
            if value is not None
        }


def new_correlation_id() -> str:
    return str(uuid.uuid4())


def safe_error(
    exc: BaseException,
    *,
    correlation_id: str,
    route: str | None = None,
    base_url: str | None = None,
    path: str | None = None,
    http_status: int | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    """Build a redacted error without copying exception text or request data."""

    host = None
    if base_url:
        try:
            host = httpx.URL(base_url).host
        except Exception:
            host = None
    return SafeError(
        correlation_id=correlation_id,
        route=route,
        provider_host=host,
        provider_path=path,
        exception_type=type(exc).__name__,
        http_status=http_status,
        error_code=error_code,
    ).to_dict()


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _redact_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if str(key).lower() in _SECRET_KEYS else _redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


class ManagedAuth:
    """Resolve Databricks SDK managed auth without requiring a token env var."""

    def __init__(self, provider: Callable[[], Mapping[str, str]] | None = None):
        self._provider = provider

    def headers(self) -> dict[str, str]:
        if self._provider is not None:
            try:
                return dict(self._provider())
            except Exception as exc:
                raise ManagedAuthError(
                    "managed authentication failed",
                    {
                        "error_code": "managed_auth_failed",
                        "auth_stage": "headers",
                        "auth_strategy": "test_provider",
                        "exception_type": type(exc).__name__,
                    },
                ) from exc
        serving_environment = _is_model_serving_environment()
        m2m_configured = serving_environment and _has_m2m_environment()
        workload_identity = serving_environment and _as_bool(
            os.getenv("DATABRICKS_SERVING_USE_WORKLOAD_IDENTITY"), False
        )
        auth_strategy = (
            "m2m"
            if m2m_configured
            else ("workload_identity" if workload_identity else ("obo" if serving_environment else "default"))
        )
        try:
            from databricks.sdk import WorkspaceClient

            if m2m_configured:
                client = WorkspaceClient(auth_type="oauth-m2m")
            elif serving_environment and not workload_identity:
                from databricks_ai_bridge import ModelServingUserCredentials

                client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
            else:
                client = WorkspaceClient()
        except Exception as exc:
            raise ManagedAuthError(
                "managed authentication initialization failed",
                {
                    "error_code": "managed_auth_failed",
                    "auth_stage": "initialization",
                    "auth_strategy": auth_strategy,
                    "exception_type": type(exc).__name__,
                },
            ) from exc
        try:
            config = client.config
            if hasattr(config, "authenticate"):
                result = config.authenticate()
                if not isinstance(result, Mapping):
                    raise RuntimeError("Databricks SDK returned invalid authentication headers")
                return {str(key): str(value) for key, value in result.items()}
            raise RuntimeError("Databricks SDK config has no authenticate method")
        except Exception as exc:
            raise ManagedAuthError(
                "managed authentication failed",
                {
                    "error_code": "managed_auth_failed",
                    "auth_stage": "headers",
                    "auth_strategy": auth_strategy,
                    "exception_type": type(exc).__name__,
                },
            ) from exc


def _is_model_serving_environment() -> bool:
    serving_flag = (
        os.getenv("IS_IN_DB_MODEL_SERVING_ENV", "").lower() == "true"
        or os.getenv("IS_IN_DATABRICKS_MODEL_SERVING_ENV", "").lower() == "true"
    )
    return serving_flag or os.path.isfile(_MODEL_SERVING_OAUTH_FILE)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _has_m2m_environment() -> bool:
    """Detect injected M2M references without logging or exposing their values."""

    return bool(os.getenv("DATABRICKS_CLIENT_ID")) and bool(os.getenv("DATABRICKS_CLIENT_SECRET"))


class GatewayClient:
    """Call a configured model-services or system.ai Serving route."""

    def __init__(
        self, settings: GatewaySettings, *, auth: ManagedAuth | None = None, client: httpx.Client | None = None
    ):
        self.settings = settings
        self.auth = auth or ManagedAuth()
        self.client = client or httpx.Client(timeout=settings.timeout_seconds)

    def validate_route(self, route: str, metadata: Mapping[str, Any] | None = None) -> None:
        metadata = metadata if isinstance(metadata, Mapping) else None
        direct_serving = _is_direct_serving_route(route, metadata)
        if metadata is None and direct_serving:
            metadata = {}
        explicitly_approved = route in self.settings.allowed_routes
        approved = explicitly_approved or route.startswith(ROUTE_PREFIX) or direct_serving
        valid_route = (
            _ROUTE_RE.fullmatch(route)
            or _SYSTEM_AI_ROUTE_RE.fullmatch(route)
            or _SERVING_ENDPOINT_ROUTE_RE.fullmatch(route)
        )
        if not approved or (not explicitly_approved and not valid_route):
            raise RouteValidationError("route is not approved", {"error_code": "route_not_allowed", "route": route})
        if metadata is None:
            raise RouteValidationError(
                "route metadata is required", {"error_code": "route_metadata_missing", "route": route}
            )
        if metadata.get("enabled") is False:
            raise RouteValidationError("route is unavailable", {"error_code": "route_unavailable", "route": route})
        if direct_serving:
            endpoint = _direct_endpoint_name(route, metadata)
            if not isinstance(endpoint, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", endpoint):
                raise RouteValidationError(
                    "system.ai Serving endpoint is invalid", {"error_code": "serving_endpoint_invalid", "route": route}
                )
        for binding in ("mcp_binding", "skill_binding", "prompt_binding"):
            if metadata.get(binding) in (False, "", None) and binding in metadata:
                raise RouteValidationError(
                    "route binding is invalid", {"error_code": f"{binding}_invalid", "route": route}
                )

    def invoke(
        self,
        payload: Mapping[str, Any],
        *,
        route: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        stream: bool = False,
        correlation_id: str | None = None,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        route = route or self.settings.route
        self.validate_route(route, metadata)
        correlation_id = correlation_id or new_correlation_id()
        metadata = metadata if isinstance(metadata, Mapping) else {}
        direct_serving = _is_direct_serving_route(route, metadata)
        protocol = _route_protocol("chat" if direct_serving else self.settings.protocol, metadata)
        use_responses = protocol == "responses"
        base_url, path = _route_target(self.settings, route, metadata)
        headers = {"content-type": "application/json", "x-correlation-id": correlation_id}
        try:
            headers.update(self.auth.headers())
        except ManagedAuthError as exc:
            raise GatewayError(
                str(exc),
                {
                    **safe_error(
                        exc,
                        correlation_id=correlation_id,
                        route=route,
                        base_url=base_url,
                        path=path,
                        error_code="managed_auth_failed",
                    ),
                    **{key: value for key, value in exc.diagnostic.items() if key in {"auth_stage", "auth_strategy"}},
                },
            ) from exc
        body = _responses_to_chat(payload) if not use_responses else dict(payload)
        if not direct_serving:
            body.setdefault("model", route.removeprefix(ROUTE_PREFIX))
        body["stream"] = stream
        body.setdefault("input" if use_responses else "messages", [])
        if use_responses:
            body.pop("messages", None)
        try:
            if stream:
                return self._stream_request(
                    body, headers, correlation_id=correlation_id, route=route, base_url=base_url, path=path
                )
            response = self.client.post(base_url + path, json=body, headers=headers)
        except Exception as exc:
            raise GatewayError(
                str(exc),
                safe_error(
                    exc,
                    correlation_id=correlation_id,
                    route=route,
                    base_url=base_url,
                    path=path,
                    error_code="gateway_transport_error",
                ),
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            code = _error_code(response)
            error = safe_error(
                GatewayHTTPError("gateway HTTP error", {}),
                correlation_id=correlation_id,
                route=route,
                base_url=base_url,
                path=path,
                http_status=response.status_code,
                error_code=code,
            )
            raise GatewayHTTPError("gateway HTTP error", error)
        try:
            decoded = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise GatewayResponseError(
                "gateway returned non-JSON",
                safe_error(
                    exc,
                    correlation_id=correlation_id,
                    route=route,
                    base_url=base_url,
                    path=path,
                    http_status=response.status_code,
                    error_code="invalid_json",
                ),
            ) from exc
        if not isinstance(decoded, Mapping):
            raise GatewayResponseError(
                "gateway returned an invalid JSON object",
                safe_error(
                    ValueError("invalid object"),
                    correlation_id=correlation_id,
                    route=route,
                    base_url=base_url,
                    path=path,
                    http_status=response.status_code,
                    error_code="invalid_json_shape",
                ),
            )
        if decoded.get("error"):
            raise GatewayResponseError(
                "upstream returned an error",
                safe_error(
                    GatewayResponseError("upstream error", {}),
                    correlation_id=correlation_id,
                    route=route,
                    base_url=base_url,
                    path=path,
                    http_status=response.status_code,
                    error_code="upstream_error",
                ),
            )
        return dict(decoded)

    def _stream_request(
        self,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
        *,
        correlation_id: str,
        route: str,
        base_url: str,
        path: str,
    ) -> Iterator[dict[str, Any]]:
        try:
            with self.client.stream("POST", base_url + path, json=body, headers=headers) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise GatewayHTTPError(
                        "gateway HTTP error",
                        safe_error(
                            GatewayHTTPError("gateway HTTP error", {}),
                            correlation_id=correlation_id,
                            route=route,
                            base_url=base_url,
                            path=path,
                            http_status=response.status_code,
                            error_code=_error_code(response),
                        ),
                    )
                yield from self._stream_json(response, correlation_id, route, base_url, path)
        except GatewayError:
            raise
        except Exception as exc:
            raise GatewayError(
                "gateway stream transport failed",
                safe_error(
                    exc,
                    correlation_id=correlation_id,
                    route=route,
                    base_url=base_url,
                    path=path,
                    error_code="gateway_transport_error",
                ),
            ) from exc

    def _stream_json(
        self, response: httpx.Response, correlation_id: str, route: str, base_url: str, path: str
    ) -> Iterator[dict[str, Any]]:
        try:
            for line in response.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                raw = line[5:].strip() if line.startswith("data:") else line
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise GatewayResponseError(
                        "stream contained non-JSON",
                        safe_error(
                            exc,
                            correlation_id=correlation_id,
                            route=route,
                            base_url=base_url,
                            path=path,
                            error_code="invalid_stream_json",
                        ),
                    ) from exc
                if isinstance(value, Mapping) and value.get("error"):
                    raise GatewayResponseError(
                        "stream contained an upstream error",
                        safe_error(
                            GatewayResponseError("upstream error", {}),
                            correlation_id=correlation_id,
                            route=route,
                            base_url=base_url,
                            path=path,
                            error_code="upstream_stream_error",
                        ),
                    )
                if not isinstance(value, Mapping):
                    raise GatewayResponseError(
                        "stream item is not an object",
                        safe_error(
                            ValueError("invalid stream item"),
                            correlation_id=correlation_id,
                            route=route,
                            base_url=base_url,
                            path=path,
                            error_code="invalid_stream_shape",
                        ),
                    )
                yield dict(value)
        finally:
            response.close()


def _error_code(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"http_{response.status_code}"
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping) and error.get("code"):
            return str(error["code"])
        if body.get("error_code"):
            return str(body["error_code"])
    return f"http_{response.status_code}"


def _is_direct_serving_route(route: str, metadata: Mapping[str, Any] | None) -> bool:
    """Return whether a route explicitly opts into a workspace Serving endpoint."""

    return bool(
        _SERVING_ENDPOINT_ROUTE_RE.fullmatch(route)
        or (
            _SYSTEM_AI_ROUTE_RE.fullmatch(route)
            and isinstance(metadata, Mapping)
            and metadata.get("backend") == "databricks_serving"
        )
    )


def _direct_endpoint_name(route: str, metadata: Mapping[str, Any]) -> str | None:
    if _SERVING_ENDPOINT_ROUTE_RE.fullmatch(route):
        return route.removeprefix(SERVING_ENDPOINT_ROUTE_PREFIX)
    endpoint = metadata.get("endpoint")
    return endpoint if isinstance(endpoint, str) else None


def _route_protocol(default: str, metadata: Mapping[str, Any]) -> str:
    """Normalize profile protocol names used by the deployment runtime."""

    protocol = str(metadata.get("protocol") or default).lower()
    if protocol in {"chat", "chat_completions", "mlflow_chat"}:
        return "chat"
    if protocol in {"responses", "openai_responses", "mlflow_responses"}:
        return "responses"
    return protocol


def _route_target(settings: GatewaySettings, route: str, metadata: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve the URL base and invocation path for one selected route."""

    if _is_direct_serving_route(route, metadata):
        if not settings.workspace_host:
            raise ConfigurationError(
                "DATABRICKS_HOST is required for system.ai Serving routes", {"error_code": "missing_workspace_host"}
            )
        endpoint = _direct_endpoint_name(route, metadata)
        if endpoint is None:
            raise ConfigurationError(
                "system.ai Serving endpoint metadata is required", {"error_code": "serving_endpoint_invalid"}
            )
        return settings.workspace_host, f"/serving-endpoints/{quote(endpoint, safe='')}/invocations"
    protocol = _route_protocol(settings.protocol, metadata)
    path = "/responses" if protocol == "responses" else "/chat/completions"
    return settings.base_url, path


def _responses_to_chat(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the public Responses input into an explicit chat payload."""

    input_items = payload.get("input", [])
    if isinstance(input_items, str):
        input_items = [{"role": "user", "content": input_items}]
    messages = []
    for item in input_items:
        if not isinstance(item, Mapping):
            raise ValueError("Responses input items must be objects")
        role = item.get("role", "user")
        content = item.get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) if isinstance(part, Mapping) else str(part) for part in content)
        messages.append({"role": role, "content": content})
    return {"messages": messages}
