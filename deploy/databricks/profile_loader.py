"""Profile loading and validation for deploy-time model routes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


class ProfileError(ValueError):
    """A profile cannot produce a diagnosable route."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_SYSTEM_AI_ROUTE_RE = re.compile(r"^system-ai/[A-Za-z0-9_.-]+$")
_SERVING_ENDPOINT_ROUTE_RE = re.compile(r"^serving-endpoints/[A-Za-z0-9_.-]+$")
_ENDPOINT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_profile(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            profile = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError("profile_invalid", "profile could not be read") from exc
    if not isinstance(profile, dict):
        raise ProfileError("profile_invalid", "profile must be an object")
    validate_profile(profile)
    return profile


def validate_profile(profile: Mapping[str, Any], *, available_routes: set[str] | None = None) -> None:
    route = profile.get("default_route")
    if not isinstance(route, str) or not route:
        raise ProfileError("default_route_missing", "default route is missing")
    if available_routes is not None and route not in available_routes:
        raise ProfileError("default_route_unavailable", "default route is unavailable")
    metadata = profile.get("route_metadata")
    if not isinstance(metadata, Mapping):
        raise ProfileError("route_metadata_missing", "route metadata is missing")
    _validate_route_metadata(route, metadata)
    if metadata.get("mcp_binding") in (None, False, "") and "mcp_binding" in metadata:
        raise ProfileError("mcp_binding_invalid", "MCP binding is invalid")
    if metadata.get("skill_binding") in (None, False, "") and "skill_binding" in metadata:
        raise ProfileError("skill_binding_invalid", "skill binding is invalid")
    if metadata.get("prompt_binding") in (None, False, "") and "prompt_binding" in metadata:
        raise ProfileError("prompt_binding_invalid", "prompt binding is invalid")
    route_metadata_by_route = profile.get("route_metadata_by_route")
    if route_metadata_by_route is not None and not isinstance(route_metadata_by_route, Mapping):
        raise ProfileError("route_metadata_invalid", "route_metadata_by_route must be an object")
    if isinstance(route_metadata_by_route, Mapping):
        for route_name, route_metadata in route_metadata_by_route.items():
            if not isinstance(route_name, str) or not isinstance(route_metadata, Mapping):
                raise ProfileError("route_metadata_invalid", "route metadata entries must be objects")
            _validate_route_metadata(route_name, route_metadata)


def _validate_route_metadata(route: str, metadata: Mapping[str, Any]) -> None:
    if _SERVING_ENDPOINT_ROUTE_RE.fullmatch(route):
        endpoint = route.removeprefix("serving-endpoints/")
        configured_endpoint = metadata.get("endpoint", endpoint)
        if configured_endpoint != endpoint:
            raise ProfileError("serving_endpoint_invalid", "serving endpoint metadata does not match route")
        protocol = str(metadata.get("protocol", "chat")).lower()
        if protocol not in {"chat", "chat_completions", "mlflow_chat"}:
            raise ProfileError("serving_protocol_invalid", "Serving endpoint routes must use a chat protocol")
        if metadata.get("backend") not in (None, "databricks_serving"):
            raise ProfileError("serving_backend_invalid", "Serving endpoint routes must use databricks_serving")
        return
    if not _SYSTEM_AI_ROUTE_RE.fullmatch(route):
        return
    if metadata.get("backend") != "databricks_serving":
        raise ProfileError("serving_backend_invalid", "system.ai routes must use databricks_serving")
    endpoint = metadata.get("endpoint")
    if not isinstance(endpoint, str) or not _ENDPOINT_RE.fullmatch(endpoint):
        raise ProfileError("serving_endpoint_invalid", "system.ai route endpoint is invalid")
    protocol = str(metadata.get("protocol", "chat")).lower()
    if protocol not in {"chat", "chat_completions", "mlflow_chat"}:
        raise ProfileError("serving_protocol_invalid", "system.ai routes must use a chat protocol")
