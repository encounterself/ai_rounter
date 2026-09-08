from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from deploy.databricks import publish_current_model as publisher
from deploy.databricks.publish_current_model import (
    obo_auth_policy,
    profile_secret_refs,
    profile_tracing_environment,
    register_model,
    update_endpoint_with_cli,
    write_endpoint_config,
)


def test_register_model_returns_this_registration_version_not_latest(monkeypatch, tmp_path: Path):
    class Run:
        info = SimpleNamespace(run_id="run-123")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    model_info = SimpleNamespace(
        model_uri="models:/simo_prod.mysql_cdc_sync.haystack_agent_router/23",
        registered_model_version="23",
    )
    (tmp_path / "profile.json").write_text(
        json.dumps({"default_route": "model-services/catalog.schema.default_route"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        publisher,
        "mlflow",
        SimpleNamespace(
            set_tracking_uri=lambda *_args: None,
            set_registry_uri=lambda *_args: None,
            start_run=lambda: Run(),
            pyfunc=SimpleNamespace(log_model=lambda **_kwargs: model_info),
        ),
    )

    version = register_model(
        deploy_dir=tmp_path,
        profile_path=tmp_path / "profile.json",
        model_name="simo_prod.mysql_cdc_sync.haystack_agent_router",
    )

    assert version == "23"


def test_obo_auth_policy_requests_model_serving_scope():
    assert obo_auth_policy().to_dict() == {
        "system_auth_policy": {},
        "user_auth_policy": {"api_scopes": ["model-serving"]},
    }


def test_profile_secret_refs_only_reads_reference_strings(tmp_path: Path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "gateway": {
                    "client_id_secret_ref": "{{secrets/oauth/client-id}}",
                    "client_secret_secret_ref": "{{secrets/oauth/client-secret}}",
                }
            }
        ),
        encoding="utf-8",
    )
    assert profile_secret_refs(profile) == (
        "{{secrets/oauth/client-id}}",
        "{{secrets/oauth/client-secret}}",
    )


def test_profile_tracing_environment_uses_databricks_tracking_uri(tmp_path: Path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "tracing": {
                    "enabled": True,
                    "experiment_name": "/Users/test@example.com/router",
                    "experiment_id": "123456789",
                    "sampling_ratio": 1.0,
                    "capture_content": False,
                    "beta": True,
                }
            }
        ),
        encoding="utf-8",
    )
    assert profile_tracing_environment(profile) == {
        "ENABLE_MLFLOW_TRACING": "true",
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_EXPERIMENT_ID": "123456789",
        "MLFLOW_TRACE_SAMPLING_RATIO": "1.0",
        "MLFLOW_TRACE_CAPTURE_CONTENT": "false",
        "MLFLOW_TRACE_BETA": "true",
    }


def test_write_endpoint_config_targets_the_new_model_version():
    config = write_endpoint_config(
        endpoint_name="haystack-agent-router",
        model_name="simo_prod.mysql_cdc_sync.haystack_agent_router",
        model_version="7",
        workspace_id="3761760657895016",
        workspace_host="https://dbc-b21ff395-726b.cloud.databricks.com",
        owner="dean.he@simo.co",
    )

    assert config["served_entities"][0]["entity_version"] == "7"
    assert config["traffic_config"]["routes"][0]["served_model_name"] == "haystack-agent-router-7"


def test_update_endpoint_uses_official_cli(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        calls.append(command)
        assert check is True

    monkeypatch.setattr("deploy.databricks.publish_current_model.subprocess.run", fake_run)
    config_path = tmp_path / "update.json"
    config_path.write_text(json.dumps({"served_entities": []}), encoding="utf-8")

    update_endpoint_with_cli(
        endpoint_name="haystack-agent-router",
        config_path=config_path,
        cli_profile="DEFAULT",
    )

    assert calls == [
        [
            "databricks",
            "serving-endpoints",
            "update-config",
            "haystack-agent-router",
            "--json",
            f"@{config_path}",
            "--profile",
            "DEFAULT",
            "--no-wait",
        ]
    ]
