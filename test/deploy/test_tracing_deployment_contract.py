"""Static deployment contract checks; no runtime imports or network access."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
README = (ROOT / "deploy/README.md").read_text(encoding="utf-8")
MONITORING = json.loads(
    (ROOT / "deploy/databricks/monitoring.example.json").read_text(encoding="utf-8")
)
REQUIREMENTS = (ROOT / "deploy/databricks/requirements.txt").read_text(encoding="utf-8")


def test_serving_and_operator_mlflow_version_boundaries_are_explicit():
    assert "mlflow>=3.1.0,<4" in REQUIREMENTS
    assert "databricks-agents>=1.2.0" in REQUIREMENTS
    assert MONITORING["serving_runtime_mlflow"] == ">=3.1.0,<4"
    assert MONITORING["minimum_operator_mlflow"] == ">=3.14.0"
    assert MONITORING["operator_requirements"] == "mlflow[databricks]>=3.14.0"
    assert "serving image remains mlflow>=3.1.0,<4" in MONITORING["requirements_boundary"]


def test_trace_defaults_and_preview_marker_are_not_enablement():
    assert MONITORING["enabled"] is False
    assert MONITORING["trace_enabled"] is False
    assert MONITORING["preview_or_beta"] is True
    assert "not proof" in README
    assert "cloud_confirmation_required" in MONITORING
    assert MONITORING["cloud_confirmation_required"] is True


def test_current_uc_trace_api_and_four_table_contract_are_documented():
    required_readme = (
        "from mlflow.entities.trace_location import UnityCatalog",
        'mlflow.set_tracking_uri("databricks")',
        "trace_location=UnityCatalog(",
        "table_prefix=",
        "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
        "mlflow_experiment_trace_otel_annotations",
        "_otel_logs",
        "_otel_metrics",
        "_otel_spans",
        "USE_CATALOG",
        "USE_SCHEMA",
        "MODIFY, SELECT",
        "ALL_PRIVILEGES is not sufficient",
    )
    for marker in required_readme:
        assert marker in README

    prefix = MONITORING["table_prefix"]
    expected_tables = {
        f"{prefix}_otel_annotations",
        f"{prefix}_otel_logs",
        f"{prefix}_otel_metrics",
        f"{prefix}_otel_spans",
    }
    assert set(MONITORING["tables"]) == expected_tables
    assert set(MONITORING["table_privileges"]) == expected_tables
    assert all(privileges == ["MODIFY", "SELECT"] for privileges in MONITORING["table_privileges"].values())


def test_cloud_gates_and_rollback_are_explicit():
    required_readme = (
        "Unity Catalog-enabled",
        "supported region",
        "CAN USE",
        "CAN EDIT",
        "Managed MLflow experiment",
        "monitoring job permission",
        "MLFLOW_TRACING_DESTINATION",
        "MLFLOW_TRACING_ENABLED",
        "beta=true",
        "preview_or_beta=true",
        "trace_enabled=false",
        "ENABLE_MLFLOW_TRACING=false",
        "stop",
        "Do not delete UC trace tables",
        "create a new experiment",
        "administrator",
        "does not enable, validate, or roll back any cloud capability",
        "mlflow.monitoring.sqlWarehouseId",
        "MLFLOW_TRACING_SQL_WAREHOUSE_ID alone is not a substitute",
    )
    for marker in required_readme:
        assert marker in README

    assert MONITORING["cloud_confirmations"]["managed_mlflow_experiment_can_edit_confirmed"] is False
    assert MONITORING["cloud_confirmations"]["monitoring_job_permission_confirmed"] is False


def test_monitoring_template_contains_no_token_or_secret_values():
    serialized = json.dumps(MONITORING).lower()
    assert "token" not in serialized
    assert "secret" not in serialized
