"""Build safe Databricks Model Serving endpoint configurations."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_REF_RE = re.compile(r"^\{\{secrets/[^/{}\s]+/[^/{}\s]+\}\}$")


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


def build_served_entity(
    *,
    served_entity_name: str,
    model_name: str,
    model_version: str | int,
    workspace_id: str,
    workspace_host: str | None = None,
    client_id_secret_ref: str | None = None,
    client_secret_secret_ref: str | None = None,
    tracing_environment: dict[str, str] | None = None,
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
    if workspace_host:
        normalized_host = workspace_host.rstrip("/")
        environment_vars.update(
            {
                "DATABRICKS_HOST": normalized_host,
                "DATABRICKS_MODEL_SERVING_HOST_URL": normalized_host,
                "IS_IN_DB_MODEL_SERVING_ENV": "true",
            }
        )
    if client_id_secret_ref:
        environment_vars.update(
            {
                "DATABRICKS_CLIENT_ID": client_id_secret_ref,
                "DATABRICKS_CLIENT_SECRET": client_secret_secret_ref,
            }
        )
    if tracing_environment:
        environment_vars.update({str(key): str(value) for key, value in tracing_environment.items()})

    return {
        "name": served_entity_name,
        "entity_name": model_name,
        "entity_version": str(model_version),
        "workload_size": workload_size,
        "scale_to_zero_enabled": scale_to_zero_enabled,
        "environment_vars": environment_vars,
    }


def build_endpoint_config(
    *,
    endpoint_name: str,
    served_entity: dict[str, Any],
    task: str = "agent/v1/responses",
    owner: str = "",
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
