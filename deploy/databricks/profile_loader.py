"""Profile loading and validation for model-service routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class ProfileError(ValueError):
    """A profile cannot produce a diagnosable route."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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
    if metadata.get("mcp_binding") in (None, False, "") and "mcp_binding" in metadata:
        raise ProfileError("mcp_binding_invalid", "MCP binding is invalid")
    if metadata.get("skill_binding") in (None, False, "") and "skill_binding" in metadata:
        raise ProfileError("skill_binding_invalid", "skill binding is invalid")
    if metadata.get("prompt_binding") in (None, False, "") and "prompt_binding" in metadata:
        raise ProfileError("prompt_binding_invalid", "prompt binding is invalid")
