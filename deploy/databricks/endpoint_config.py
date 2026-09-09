"""Build safe Databricks Model Serving endpoint configurations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_REF_RE = re.compile(r"^\{\{secrets/[^/{}\s]+/[^/{}\s]+\}\}$")
_LAKEBASE_ENDPOINT_RE = re.compile(r"^projects/[a-z0-9-]+/branches/[a-z0-9-]+/endpoints/[a-z0-9-]+$")


def _validate_name(value: str, field: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ValueError(f"{field} must contain only letters, numbers, '.', '_' or '-'.")
    return value


def _validate_secret_ref(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SECRET_REF_RE.fullmatch(value):
        raise ValueError(f"{field} must be a Databricks secret reference: {{{{secrets/<scope>/<key>}}}}.")
    return value


def build_session_environment(
    *,
    lakebase_endpoint: str | None,
    database: str = "test_ontos",
    database_user: str | None = None,
    table: str = "haystack_agent_sessions",
    auto_migrate: bool = True,
    ttl_seconds: int | float | None = None,
    use_user_auth: bool = False,
) -> dict[str, str]:
    """Build non-secret environment values for Lakebase-backed sessions."""

    if lakebase_endpoint is None:
        return {}
    if not _LAKEBASE_ENDPOINT_RE.fullmatch(lakebase_endpoint):
        raise ValueError("lakebase_endpoint must be a Lakebase endpoint resource name")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", database):
        raise ValueError("database must be a simple identifier")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("table must be a simple identifier")
    if database_user is not None and not database_user.strip():
        raise ValueError("database_user must not be empty")
    if ttl_seconds is not None and ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    environment = {
        "HAYSTACK_SESSION_LAKEBASE_ENDPOINT": lakebase_endpoint,
        "HAYSTACK_SESSION_DATABASE": database,
        "HAYSTACK_SESSION_TABLE": table,
        "HAYSTACK_SESSION_AUTO_MIGRATE": str(auto_migrate).lower(),
        "HAYSTACK_SESSION_USE_USER_AUTH": str(use_user_auth).lower(),
    }
    if ttl_seconds is not None:
        environment["HAYSTACK_SESSION_TTL_SECONDS"] = (
            str(int(ttl_seconds)) if float(ttl_seconds).is_integer() else str(ttl_seconds)
        )
    if database_user is not None:
        environment["HAYSTACK_SESSION_DATABASE_USER"] = database_user
    return environment


def build_served_entity(
    *,
    served_entity_name: str,
    model_name: str,
    model_version: str | int,
    workspace_id: str,
    serving_endpoint_name: str | None = None,
    workspace_host: str | None = None,
    client_id_secret_ref: str | None = None,
    client_secret_secret_ref: str | None = None,
    tracing_environment: dict[str, str] | None = None,
    session_environment: Mapping[str, str] | None = None,
    workload_size: str = "Small",
    scale_to_zero_enabled: bool = True,
) -> dict[str, Any]:
    """Build the served entity payload for a registered UC model version."""

    _validate_name(served_entity_name, "served_entity_name")
    if not model_name or model_name.count(".") < 2:
        raise ValueError("model_name must be a fully qualified Unity Catalog model name.")
    if not str(model_version).isdigit() or int(model_version) < 1:
        raise ValueError("model_version must be a positive integer.")
    if not workspace_id or not workspace_id.isdigit():
        raise ValueError("workspace_id must be a numeric Databricks workspace ID.")
    if serving_endpoint_name is not None:
        _validate_name(serving_endpoint_name, "serving_endpoint_name")
    if workspace_host is not None:
        parsed_host = urlparse(workspace_host)
        if parsed_host.scheme != "https" or not parsed_host.netloc or parsed_host.path not in {"", "/"}:
            raise ValueError("workspace_host must be an HTTPS Databricks workspace URL.")
    if workload_size not in {"Small", "Medium", "Large"}:
        raise ValueError("workload_size must be Small, Medium, or Large.")
    client_id_secret_ref = _validate_secret_ref(client_id_secret_ref, "client_id_secret_ref")
    client_secret_secret_ref = _validate_secret_ref(client_secret_secret_ref, "client_secret_secret_ref")
    if (client_id_secret_ref is None) != (client_secret_secret_ref is None):
        raise ValueError("client_id_secret_ref and client_secret_secret_ref must be provided together.")
    environment_vars = {"DATABRICKS_WORKSPACE_ID": workspace_id}
    if serving_endpoint_name:
        environment_vars.update(
            {
                "HAYSTACK_SERVING_ENDPOINT_NAME": serving_endpoint_name,
                "HAYSTACK_SERVED_ENTITY_NAME": served_entity_name,
                "HAYSTACK_SERVED_MODEL_VERSION": str(model_version),
            }
        )
    if workspace_host:
        normalized_host = workspace_host.rstrip("/")
        environment_vars.update(
            {
                "DATABRICKS_HOST": normalized_host,
                "DATABRICKS_MODEL_SERVING_HOST_URL": normalized_host,
                "IS_IN_DB_MODEL_SERVING_ENV": "true",
                "DATABRICKS_SERVING_USE_WORKLOAD_IDENTITY": "true",
            }
        )
    if client_id_secret_ref:
        environment_vars.update(
            {"DATABRICKS_CLIENT_ID": client_id_secret_ref, "DATABRICKS_CLIENT_SECRET": client_secret_secret_ref}
        )
    if tracing_environment:
        environment_vars.update({str(key): str(value) for key, value in tracing_environment.items()})
    if session_environment:
        environment_vars.update({str(key): str(value) for key, value in session_environment.items()})

    return {
        "name": served_entity_name,
        "entity_name": model_name,
        "entity_version": str(model_version),
        "workload_size": workload_size,
        "scale_to_zero_enabled": scale_to_zero_enabled,
        "environment_vars": environment_vars,
    }


def build_endpoint_config(
    *, endpoint_name: str, served_entity: dict[str, Any], task: str = "agent/v1/responses", owner: str = ""
) -> dict[str, Any]:
    """Build a create payload with explicit routing and deployment metadata."""

    _validate_name(endpoint_name, "endpoint_name")
    served_name = served_entity.get("name")
    if not isinstance(served_name, str) or not served_name:
        raise ValueError("served_entity must contain a non-empty name.")
    tags = [
        {"key": "managed_by", "value": "haystack-control-plane"},
        {"key": "source_model", "value": str(served_entity["entity_name"])},
        {"key": "source_version", "value": str(served_entity["entity_version"])},
        {"key": "task", "value": task},
    ]
    if owner:
        tags.append({"key": "owner", "value": owner})
    return {
        "name": endpoint_name,
        "served_entities": [served_entity],
        "traffic_config": {
            "routes": [{"served_entity_name": served_name, "served_model_name": served_name, "traffic_percentage": 100}]
        },
        "tags": tags,
    }


def build_update_config(served_entity: dict[str, Any]) -> dict[str, Any]:
    """Build the complete config body accepted by endpoint update-config."""

    served_name = served_entity.get("name")
    if not isinstance(served_name, str) or not served_name:
        raise ValueError("served_entity must contain a non-empty name.")
    return {
        "served_entities": [served_entity],
        "traffic_config": {
            "routes": [{"served_entity_name": served_name, "served_model_name": served_name, "traffic_percentage": 100}]
        },
    }
