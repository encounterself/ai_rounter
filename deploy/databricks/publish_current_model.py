"""
Register the current Haystack runtime and publish a Serving endpoint.

Run this from a Databricks notebook or deployment job with Managed MLflow
authentication. It never places a PAT or provider credential in the model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from mlflow.models.resources import DatabricksServingEndpoint, Resource, ResourceType


class DatabricksAutoscalingPostgres(Resource):
    """Declare a modern Lakebase Autoscaling project for system auth."""

    def __init__(self, project_name: str, database_name: str = "databricks_postgres") -> None:
        self.project_name = project_name
        self.database_name = database_name

    @property
    def target_uri(self) -> str:
        return "databricks"

    @property
    def type(self) -> ResourceType:
        return ResourceType.LAKEBASE

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {"postgres": [{"name": self.project_name, "database_name": self.database_name}]}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "DatabricksAutoscalingPostgres":
        return cls(data["name"], data.get("database_name", "databricks_postgres"))


try:
    from .endpoint_config import (
        build_endpoint_config,
        build_served_entity,
        build_session_environment,
        build_update_config,
    )
except ImportError:  # MLflow loads this file as a top-level script.
    from endpoint_config import (
        build_endpoint_config,
        build_served_entity,
        build_session_environment,
        build_update_config,
    )


def lakebase_resource_name(endpoint: str | None) -> str | None:
    """Return the autoscaling Lakebase project resource for a session endpoint."""

    if not endpoint:
        return None
    prefix, separator, _ = endpoint.partition("/branches/")
    if not separator or not prefix.startswith("projects/"):
        raise ValueError("session Lakebase endpoint must identify a projects/<project> resource")
    return prefix


def profile_serving_endpoint_names(profile: dict[str, Any]) -> tuple[str, ...]:
    """Collect the Databricks Serving endpoints referenced by an orchestration profile."""

    names: set[str] = set()

    def add_route(route: Any, metadata: Any = None) -> None:
        if isinstance(route, str) and route.startswith("serving-endpoints/"):
            endpoint = route.removeprefix("serving-endpoints/")
            if endpoint:
                names.add(endpoint)
        if isinstance(metadata, dict) and metadata.get("backend") == "databricks_serving":
            endpoint = metadata.get("endpoint")
            if isinstance(endpoint, str) and endpoint:
                names.add(endpoint)

    add_route(profile.get("default_route"), profile.get("route_metadata"))
    route_metadata_by_route = profile.get("route_metadata_by_route")
    if isinstance(route_metadata_by_route, dict):
        for route, metadata in route_metadata_by_route.items():
            add_route(route, metadata)
    model_bindings = profile.get("model_bindings")
    if isinstance(model_bindings, list):
        for binding in model_bindings:
            if isinstance(binding, dict):
                add_route(binding.get("ref"))
    return tuple(sorted(names))


def obo_auth_policy(
    *, lakebase_resource: str | None = None, serving_endpoint_names: tuple[str, ...] = ()
) -> AuthPolicy:
    """Authorize serving dependencies and optional Lakebase system access."""

    resources: list[Resource] = []
    if lakebase_resource:
        resources.append(DatabricksAutoscalingPostgres(lakebase_resource))
    resources.extend(DatabricksServingEndpoint(name) for name in serving_endpoint_names)
    system_auth_policy = SystemAuthPolicy(resources=resources) if resources else None
    return AuthPolicy(
        user_auth_policy=UserAuthPolicy(api_scopes=["model-serving", "ai-gateway"]),
        system_auth_policy=system_auth_policy,
    )


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
    lakebase_resource: str | None = None,
) -> str:
    """Register the current code model and return its exact registered version."""

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if experiment_name:
        mlflow.set_experiment(experiment_name=experiment_name)
    else:
        tracing = profile.get("tracing") if isinstance(profile, dict) else None
        experiment_id = tracing.get("experiment_id") if isinstance(tracing, dict) else None
        if experiment_id not in (None, ""):
            mlflow.set_experiment(experiment_id=str(experiment_id))
    log_model_options: dict[str, Any] = {}
    if not (isinstance(profile, dict) and "schema_version" in profile and "agent_graph" in profile):
        log_model_options["input_example"] = {"input": [{"role": "user", "content": "health check"}], "stream": False}
    repository_root = deploy_dir.parent.parent
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="router",
            python_model=str(deploy_dir / "agent.py"),
            code_paths=[str(deploy_dir), str(repository_root / "integrations"), str(repository_root / "contracts")],
            artifacts={"profile": str(profile_path)},
            pip_requirements=str(deploy_dir / "requirements.txt"),
            auth_policy=obo_auth_policy(
                lakebase_resource=lakebase_resource, serving_endpoint_names=profile_serving_endpoint_names(profile)
            ),
            registered_model_name=model_name,
            **log_model_options,
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
    session_environment: dict[str, str] | None = None,
    owner: str,
) -> dict[str, Any]:
    served_entity = build_served_entity(
        served_entity_name=f"{endpoint_name}-{model_version}",
        model_name=model_name,
        model_version=model_version,
        workspace_id=workspace_id,
        serving_endpoint_name=endpoint_name,
        workspace_host=workspace_host,
        client_id_secret_ref=client_id_secret_ref,
        client_secret_secret_ref=client_secret_secret_ref,
        tracing_environment=tracing_environment,
        session_environment=session_environment,
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
    session_environment: dict[str, str] | None = None,
    owner: str,
) -> dict[str, Any]:
    served_entity = build_served_entity(
        served_entity_name=f"{endpoint_name}-{model_version}",
        model_name=model_name,
        model_version=model_version,
        workspace_id=workspace_id,
        serving_endpoint_name=endpoint_name,
        workspace_host=workspace_host,
        client_id_secret_ref=client_id_secret_ref,
        client_secret_secret_ref=client_secret_secret_ref,
        tracing_environment=tracing_environment,
        session_environment=session_environment,
    )
    return build_endpoint_config(endpoint_name=endpoint_name, served_entity=served_entity, owner=owner)


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
    parser.add_argument("--session-lakebase-endpoint")
    parser.add_argument("--session-database", default="test_ontos")
    parser.add_argument("--session-database-user")
    parser.add_argument(
        "--session-use-user-auth",
        action="store_true",
        help="Use the invoking user's OBO identity for Lakebase instead of the endpoint workload identity.",
    )
    parser.add_argument("--session-table", default="haystack_agent_sessions")
    parser.add_argument("--session-ttl-seconds", type=float)
    parser.add_argument("--no-session-auto-migrate", action="store_true")
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
    session_environment = build_session_environment(
        lakebase_endpoint=args.session_lakebase_endpoint,
        database=args.session_database,
        database_user=args.session_database_user,
        table=args.session_table,
        auto_migrate=not args.no_session_auto_migrate,
        ttl_seconds=args.session_ttl_seconds,
        use_user_auth=args.session_use_user_auth,
    )

    version = register_model(
        deploy_dir=args.deploy_dir,
        profile_path=args.profile,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        lakebase_resource=lakebase_resource_name(args.session_lakebase_endpoint),
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
        session_environment=session_environment,
        owner=args.owner,
    )
    args.config_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.update_endpoint:
        update_endpoint_with_cli(
            endpoint_name=args.endpoint_name, config_path=args.config_out, cli_profile=args.cli_profile
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
