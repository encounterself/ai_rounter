"""Register the current Haystack runtime and publish a Serving endpoint.

Run this from a Databricks notebook or deployment job with Managed MLflow
authentication. It never places a PAT or provider credential in the model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import mlflow
from mlflow.models.auth_policy import AuthPolicy, UserAuthPolicy

try:
    from .endpoint_config import build_endpoint_config, build_served_entity, build_update_config
except ImportError:  # MLflow loads this file as a top-level script.
    from endpoint_config import build_endpoint_config, build_served_entity, build_update_config


def obo_auth_policy() -> AuthPolicy:
    """Authorize model-serving calls with the invoking user's identity."""

    return AuthPolicy(user_auth_policy=UserAuthPolicy(api_scopes=["model-serving"]))


def profile_secret_refs(profile_path: Path) -> tuple[str | None, str | None]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    gateway = profile.get("gateway", {})
    if not isinstance(gateway, dict):
        return None, None
    return gateway.get("client_id_secret_ref"), gateway.get("client_secret_secret_ref")


def profile_tracing_environment(profile_path: Path) -> dict[str, str]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    tracing = profile.get("tracing", {})
    if not isinstance(tracing, dict) or not tracing.get("enabled"):
        return {}
    environment = {
        "ENABLE_MLFLOW_TRACING": "true",
        "MLFLOW_TRACKING_URI": str(tracing.get("tracking_uri") or "databricks"),
    }
    mapping = {
        "experiment_id": "MLFLOW_EXPERIMENT_ID",
        "sampling_ratio": "MLFLOW_TRACE_SAMPLING_RATIO",
        "capture_content": "MLFLOW_TRACE_CAPTURE_CONTENT",
        "beta": "MLFLOW_TRACE_BETA",
        "destination": "MLFLOW_TRACING_DESTINATION",
        "sql_warehouse_id": "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
    }
    for profile_key, environment_key in mapping.items():
        value = tracing.get(profile_key)
        if value not in (None, ""):
            environment[environment_key] = str(value).lower() if isinstance(value, bool) else str(value)
    return environment


def register_model(
    *,
    deploy_dir: Path,
    profile_path: Path,
    model_name: str,
    experiment_name: str | None = None,
) -> str:
    """Register the current code model and return its exact registered version."""

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="router",
            python_model=str(deploy_dir / "agent.py"),
            code_paths=[str(deploy_dir)],
            artifacts={"profile": str(profile_path)},
            pip_requirements=str(deploy_dir / "requirements.txt"),
            input_example={"input": [{"role": "user", "content": "health check"}], "stream": False},
            auth_policy=obo_auth_policy(),
            registered_model_name=model_name,
        )
        print(f"registered run: {run.info.run_id}")
    registered_version = getattr(model_info, "registered_model_version", None)
    if registered_version is None:
        raise RuntimeError("MLflow did not return the registered model version for this run")
    return str(registered_version)


def write_endpoint_config(
    *,
    endpoint_name: str,
    model_name: str,
    model_version: str,
    workspace_id: str,
    workspace_host: str | None = None,
    client_id_secret_ref: str | None = None,
    client_secret_secret_ref: str | None = None,
    tracing_environment: dict[str, str] | None = None,
    owner: str,
) -> dict[str, Any]:
    served_entity = build_served_entity(
        served_entity_name=f"{endpoint_name}-{model_version}",
        model_name=model_name,
        model_version=model_version,
        workspace_id=workspace_id,
        workspace_host=workspace_host,
        client_id_secret_ref=client_id_secret_ref,
        client_secret_secret_ref=client_secret_secret_ref,
        tracing_environment=tracing_environment,
    )
    return build_update_config(served_entity)


def write_create_endpoint_config(
    *,
    endpoint_name: str,
    model_name: str,
    model_version: str,
    workspace_id: str,
    workspace_host: str | None = None,
    client_id_secret_ref: str | None = None,
    client_secret_secret_ref: str | None = None,
    tracing_environment: dict[str, str] | None = None,
    owner: str,
) -> dict[str, Any]:
    served_entity = build_served_entity(
        served_entity_name=f"{endpoint_name}-{model_version}",
        model_name=model_name,
        model_version=model_version,
        workspace_id=workspace_id,
        workspace_host=workspace_host,
        client_id_secret_ref=client_id_secret_ref,
        client_secret_secret_ref=client_secret_secret_ref,
        tracing_environment=tracing_environment,
    )
    return build_endpoint_config(
        endpoint_name=endpoint_name,
        served_entity=served_entity,
        owner=owner,
    )


def update_endpoint_with_cli(*, endpoint_name: str, config_path: Path, cli_profile: str) -> None:
    """Apply a config through the supported Databricks CLI command."""

    command = [
        "databricks",
        "serving-endpoints",
        "update-config",
        endpoint_name,
        "--json",
        f"@{config_path}",
        "--profile",
        cli_profile,
        "--no-wait",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-dir", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--workspace-host", default=os.getenv("DATABRICKS_HOST"))
    parser.add_argument("--client-id-secret-ref")
    parser.add_argument("--client-secret-secret-ref")
    parser.add_argument("--owner", default="")
    parser.add_argument("--config-out", type=Path, default=Path("/tmp/haystack-agent-router-update.json"))
    parser.add_argument("--cli-profile", default="DEFAULT")
    parser.add_argument("--update-endpoint", action="store_true")
    args = parser.parse_args()
    if args.deploy_dir is None:
        args.deploy_dir = Path.cwd()
    os.environ.setdefault("DATABRICKS_WORKSPACE_ID", args.workspace_id)
    if args.workspace_host:
        os.environ.setdefault("DATABRICKS_HOST", args.workspace_host)
    profile_client_id_ref, profile_client_secret_ref = profile_secret_refs(args.profile)
    tracing_environment = profile_tracing_environment(args.profile)
    client_id_secret_ref = args.client_id_secret_ref or profile_client_id_ref
    client_secret_secret_ref = args.client_secret_secret_ref or profile_client_secret_ref

    version = register_model(
        deploy_dir=args.deploy_dir,
        profile_path=args.profile,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
    )
    result = write_endpoint_config(
        endpoint_name=args.endpoint_name,
        model_name=args.model_name,
        model_version=version,
        workspace_id=args.workspace_id,
        workspace_host=args.workspace_host,
        client_id_secret_ref=client_id_secret_ref,
        client_secret_secret_ref=client_secret_secret_ref,
        tracing_environment=tracing_environment,
        owner=args.owner,
    )
    args.config_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.update_endpoint:
        update_endpoint_with_cli(
            endpoint_name=args.endpoint_name,
            config_path=args.config_out,
            cli_profile=args.cli_profile,
        )
    print(
        json.dumps(
            {
                "endpoint": args.endpoint_name,
                "model": args.model_name,
                "version": version,
                "config": str(args.config_out),
                "updated_with_cli": args.update_endpoint,
            }
        )
    )


if __name__ == "__main__":
    main()
