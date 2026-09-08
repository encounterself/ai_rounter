# Databricks agent deployment

This directory contains the reproducible source for an MLflow 3.1+ `python_function`/`ResponsesAgent` code model. Its public contract is `agent/v1/responses`, for example `{"input":[{"role":"user","content":"Hello!"}],"stream":false}`. `deploy/databricks/agent.py` is packaged as `code_path`, registers an unconfigured `Agent` with `mlflow.models.set_model`, and receives the non-secret profile during `load_context`. Missing route metadata, unavailable routes, and broken MCP/skill/prompt bindings fail with an explicit diagnostic.

## Configuration

Copy `deploy/databricks/profile.example.json` and provide a real profile outside source control. Set either non-secret `DATABRICKS_AI_GATEWAY_BASE_URL` or `DATABRICKS_WORKSPACE_ID`; an explicit base URL takes precedence, otherwise the client constructs `https://<workspace_id>.ai-gateway.cloud.databricks.com`. Authentication comes from Databricks SDK managed auth; do not add `DATABRICKS_TOKEN`, Claude tokens, Codex tokens, or old endpoint credentials.

The default route is `model-services/simo_prod.ai_models.route_claude-haiku-4-5`. Routes with the `model-services/` prefix are validated, and an explicitly configured `allowed_routes` list may add another approved route. Provider/model metadata is carried by the profile so Claude-compatible, Codex-compatible, and other backends can be selected by route without claiming that any backend is already connected.

## Register a new version

1. Use MLflow 3.1+ Managed MLflow and install `deploy/databricks/requirements.txt` (`mlflow>=3.1.0,<4`, `databricks-agents>=1.2.0`) in the build image. Managed MLflow access and the endpoint identity's experiment permissions are operator prerequisites.
2. Register `Agent` as the MLflow `python_function`/`ResponsesAgent` code-model entrypoint, package `deploy/databricks` as `code_path`, and register a new model version through the operator's normal MLflow flow. The model module calls `mlflow.models.set_model`; this repository does not perform that registration remotely.
3. Inject `profile.json` as a model artifact named `profile` (or provide equivalent model config). `load_context` validates `default_route` and `route_metadata`; it never falls back to an empty profile.
4. Configure only non-secret endpoint environment values: `DATABRICKS_WORKSPACE_ID` or `DATABRICKS_AI_GATEWAY_BASE_URL` (the latter wins). Do not configure `DATABRICKS_TOKEN`; authentication uses Databricks SDK managed auth.
5. Validate with the Responses request above and inspect `error` when `status` is `failed`. Diagnostics never include authorization, tokens, complete prompts, or MCP/skill secrets.

### Copyable notebook registration skeleton

Run the following in a Databricks notebook after placing this repository at `<WORKSPACE_REPO_ROOT>` or another operator-controlled workspace path. It is a registration skeleton only; this repository does not execute it or create a model/endpoint. The MLflow models-from-code entrypoint is `agent.py`, and the module itself calls `mlflow.models.set_model(Agent())` with an unconfigured instance that receives the profile in `load_context`.

```python
from pathlib import Path
import mlflow

DEPLOY_DIR = Path("<WORKSPACE_REPO_ROOT>/deploy/databricks")
PROFILE_PATH = DEPLOY_DIR / "profile.json"  # operator-managed copy of profile.example.json
MODEL_NAME = "<CATALOG>.<SCHEMA>.<REGISTERED_MODEL_NAME>"
INPUT_EXAMPLE = {
    "input": [{"role": "user", "content": "Hello!"}],
    "stream": False,
}

mlflow.set_tracking_uri("databricks")
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        artifact_path="router",
        python_model=str(DEPLOY_DIR / "agent.py"),
        code_paths=[str(DEPLOY_DIR)],
        artifacts={"profile": str(PROFILE_PATH)},
        pip_requirements=str(DEPLOY_DIR / "requirements.txt"),
        input_example=INPUT_EXAMPLE,
        registered_model_name=MODEL_NAME,
    )

print(model_info.model_uri)
```

The endpoint environment for this model contains only one of these non-secret values: `DATABRICKS_WORKSPACE_ID=<WORKSPACE_ID>` or `DATABRICKS_AI_GATEWAY_BASE_URL=https://<AI_GATEWAY_HOST>`. Do not put `DATABRICKS_TOKEN` in the endpoint environment. If the serving integration injects model configuration instead of an artifact, omit `artifacts={"profile": ...}` and inject equivalent non-secret `model_config`; `Agent.load_context` accepts either the `profile` artifact or validated model config.

The `mlflow.models.set_model` call is the code-model registration hook, not an endpoint deployment call. Keep `deploy/databricks` in `code_paths` so top-level code-path imports work, and keep the profile outside source control. Register a new version only after the model URI and profile have been reviewed.

## Deployment runbook and incident notes

### Why an endpoint can stay on the old model

The production endpoint was not updated because the new model version never finished registering. MLflow loaded `agent.py` as a top-level code model, where `__package__` is empty; the old fallback import therefore raised `No module named 'gateway_client'`. No new version existed for `update-config` to reference, so the endpoint correctly remained on the last registered version. Keep all sibling modules in the `code_paths` directory and use the deterministic package bootstrap in `agent.py`; a `READY` endpoint alone does not prove that the new model was registered or that a request succeeded.

### Safe promotion order

Run these stages in order and stop at the first failure:

1. Build and register a new model version with Managed MLflow, `mlflow>=3.1.0,<4`, and `databricks-agents>=1.2.0`.
2. Read back the model version, `MLmodel`, requirements, signature, and registration task logs. Confirm that `agent.py`, `gateway_client.py`, `profile_loader.py`, `serving_runtime.py`, and `tracing.py` are all present in the packaged code path.
3. Create or use an isolated test endpoint. Poll `state.ready == "READY"` and `state.config_update == "NOT_UPDATING"`, then run both non-streaming and streaming `agent/v1/responses` smoke requests and inspect the returned `error` field.
4. Only after the test endpoint is healthy, update the production endpoint with a complete `served_entities` and `traffic_config` body. Always pass `--profile DEFAULT`, preserve the current workload/scaling settings, and poll the endpoint again before declaring success.
5. If any stage fails, collect the current `config_version`, served entity name, build logs, and runtime logs; do not guess a version or retry an endpoint update against an unregistered model.

Never copy a PAT, provider API key, Claude/Codex credential, or old endpoint token into the model artifact, profile, endpoint environment, logs, or CI. The serving runtime uses managed Databricks authentication; endpoint configuration should contain only non-secret values such as `DATABRICKS_WORKSPACE_ID` or `DATABRICKS_AI_GATEWAY_BASE_URL`.

### Tracing beta gate

Tracing is an operator-controlled feature, not a deployment-side switch. Before enabling it, confirm the workspace preview/beta flag, a Managed MLflow experiment with `CAN EDIT`, a SQL warehouse with `CAN USE`, the monitoring job permission, the endpoint identity, and the Unity Catalog trace catalog/schema privileges. The serving image uses `mlflow>=3.1.0,<4`; the separate UC trace-location/operator environment currently requires `mlflow[databricks]>=3.14.0`. `MLFLOW_TRACING_ENABLED`, `beta=true`, and `preview_or_beta=true` are only configuration markers. If any cloud prerequisite is unconfirmed, leave tracing disabled and report the exact missing gate rather than treating the template as enabled.

## Operational runbook

### Why an endpoint may not update

The first failed endpoint update was not caused by the endpoint configuration. The new model version failed during MLflow code-model loading because a sibling module import from the top-level `deploy/databricks` code path was not available in the serving loader. Keep the complete `deploy/databricks` directory in `code_paths`, use the package-aware imports in `agent.py`, and validate model loading before attempting an endpoint update. A registered version is not evidence that the serving runtime can import it.

Use this order for every change:

1. Register the model version with MLflow `>=3.1.0` and `databricks-agents>=1.2.0`.
2. Validate the model URI, packaged code path, profile, and a non-secret Responses request in an isolated test endpoint.
3. Inspect the serving endpoint state and logs; only then update the intended endpoint to the validated version.

All Databricks CLI examples must select the profile explicitly, for example `databricks --profile DEFAULT serving-endpoints list`. Never rely on the implicit profile, paste or copy a PAT, or put a token in a notebook, model artifact, endpoint environment, issue, or log. Use managed authentication and server-side secret references.

### GenAI tracing beta

Tracing is deliberately disabled by default. The `monitoring.example.json` file is a checklist, not an activation command. The operator environment needs MLflow `>=3.14.0` with the `databricks` extra, a supported workspace/region with the beta available, an editable Managed MLflow experiment, a Unity Catalog destination, a SQL warehouse, and the required catalog/schema, trace-table, experiment, and monitoring-job permissions. The serving runtime remains pinned to `mlflow>=3.1.0,<4`; do not solve operator-side tracing prerequisites by changing the serving image.

Known blockers are the beta rollout/region gate, missing Managed MLflow experiment edit permission, unavailable SQL warehouse or monitoring-job permission, and missing Unity Catalog table privileges. Until each confirmation in `monitoring.example.json` is true, keep `enabled` and `trace_enabled` false and treat tracing as a non-blocking diagnostic capability rather than a deployment requirement.

## Independent test endpoint

`deploy/databricks/serving-endpoint.test.json` is a placeholder-only test endpoint configuration. Replace every `<...>` value, use a new endpoint name, and confirm the registered model version before executing any command. The served entity name and the traffic route value are intentionally identical. The template uses `served_entities`, `traffic_config.routes`, scaling fields, endpoint environment variables, and string key/value tags.

Create the independent test endpoint; this command is documented and is **not** executed by this repository:

```bash
databricks serving-endpoints create <TEST_ENDPOINT> \
  --json @deploy/databricks/serving-endpoint.test.json \
  --profile DEFAULT \
  --no-wait
```

Poll both readiness fields before querying:

```bash
databricks serving-endpoints get <TEST_ENDPOINT> --profile DEFAULT \
  | jq '{ready: .state.ready, config_update: .state.config_update, config_version: .config.config_version, entities: [.config.served_entities[]?.name]}'
```

Proceed only when `state.ready == "READY"` and `state.config_update == "NOT_UPDATING"`. The served model/entity name for logs comes from `served_entities[].name` in this `get` response:

```bash
databricks serving-endpoints build-logs <TEST_ENDPOINT> <SERVED_ENTITY_NAME_FROM_GET> --profile DEFAULT
databricks serving-endpoints logs <TEST_ENDPOINT> <SERVED_ENTITY_NAME_FROM_GET> --profile DEFAULT
```

### Responses smoke requests

The CLI `serving-endpoints query` shortcut is shaped around chat `messages`; this model's public contract is `agent/v1/responses`. Use the API command below with a Responses body when the CLI query shortcut cannot express `input` and `stream`:

```bash
databricks api post /api/2.0/serving-endpoints/<TEST_ENDPOINT>/invocations \
  --json '{"input":[{"role":"user","content":"Hello from a test endpoint"}],"stream":false}' \
  --profile DEFAULT
```

For streaming, send the same Responses body with `"stream":true`:

```bash
databricks api post /api/2.0/serving-endpoints/<TEST_ENDPOINT>/invocations \
  --json '{"input":[{"role":"user","content":"Stream from a test endpoint"}],"stream":true}' \
  --profile DEFAULT
```

If an operator uses the chat-compatible gateway path internally, the runtime explicitly converts Responses `input` items to chat `messages`; never send `{"input": ...}` directly to `/mlflow/v1/chat/completions`. Wait for the endpoint readiness condition above before either smoke request.

### Test endpoint update and rollback

Use `update-config` with a complete `served_entities` and `traffic_config` configuration and a route matching the served entity name. Endpoint `tags` are top-level endpoint configuration, as shown in `serving-endpoint.test.json`; `update-config` replaces only `served_entities` and `traffic_config`, so do not invent or add a `tags` field to its JSON body. After updating, use `get` to verify that the endpoint-level tags are still present:

```bash
databricks serving-endpoints update-config <TEST_ENDPOINT> \
  --json '{
    "served_entities": [{
      "name": "<TEST_ENTITY_NAME>-<NEW_VERSION>",
      "entity_name": "<CATALOG>.<SCHEMA>.<REGISTERED_MODEL_NAME>",
      "entity_version": "<NEW_VERSION>",
      "workload_size": "Small",
      "scale_to_zero_enabled": true,
      "environment_vars": {
        "DATABRICKS_WORKSPACE_ID": "<WORKSPACE_ID>"
      }
    }],
    "traffic_config": {
      "routes": [{
        "served_model_name": "<TEST_ENTITY_NAME>-<NEW_VERSION>",
        "traffic_percentage": 100
      }]
    }
  }' \
  --profile DEFAULT
```

```bash
databricks serving-endpoints get <TEST_ENDPOINT> --profile DEFAULT \
  | jq '{tags: .tags, config_version: .config.config_version, entities: [.config.served_entities[]?.name]}'
```

### Configuration failure diagnostics

Do not guess a model version, served entity name, or undocumented CLI flag. Save the endpoint response first, use `config_version` only to identify which failed configuration to correlate with the logs, and pass the current `served_entities[].name` to the documented log commands:

```bash
databricks serving-endpoints get <TEST_ENDPOINT> --profile DEFAULT \
  | tee endpoint-state.json
CONFIG_VERSION=$(jq -r '.config.config_version' endpoint-state.json)
SERVED_ENTITY_NAME=$(jq -r '.config.served_entities[0].name' endpoint-state.json)
printf 'failed config_version=%s served_entity=%s\n' "$CONFIG_VERSION" "$SERVED_ENTITY_NAME"
databricks serving-endpoints build-logs <TEST_ENDPOINT> "$SERVED_ENTITY_NAME" --profile DEFAULT
databricks serving-endpoints logs <TEST_ENDPOINT> "$SERVED_ENTITY_NAME" --profile DEFAULT
```

The `config_version` is for checking that the logs correspond to the failed configuration; `build-logs` and `logs` receive the served entity name, not a guessed version or an extra `config_version` flag.

After the update, poll `state.ready` and `state.config_update` again and re-check the top-level tags with `get`. To roll back, first confirm the old model version is registered and previously validated, then use `update-config` to replace only the old `served_entities` and `traffic_config`; do not add tags to the rollback body:

```bash
databricks serving-endpoints update-config <TEST_ENDPOINT> \
  --json '{
    "served_entities": [{
      "name": "<TEST_ENTITY_NAME>-<OLD_VERSION>",
      "entity_name": "<CATALOG>.<SCHEMA>.<REGISTERED_MODEL_NAME>",
      "entity_version": "<OLD_VERSION>",
      "workload_size": "Small",
      "scale_to_zero_enabled": true,
      "environment_vars": {
        "DATABRICKS_WORKSPACE_ID": "<WORKSPACE_ID>"
      }
    }],
    "traffic_config": {
      "routes": [{
        "served_model_name": "<TEST_ENTITY_NAME>-<OLD_VERSION>",
        "traffic_percentage": 100
      }]
    }
  }' \
  --profile DEFAULT
```

If tags must change, first inspect the documented patch shape rather than guessing fields:

```bash
databricks serving-endpoints patch -h --profile DEFAULT
databricks serving-endpoints patch <TEST_ENDPOINT> ... --profile DEFAULT
```

Use the fields shown by that local `patch -h` output, then run `get <TEST_ENDPOINT> --profile DEFAULT` and verify the top-level `tags`. This repository does not execute tag updates.

## Optional tracing and monitoring

Tracing is disabled by default: `trace_enabled=false` in the profile/runtime and `MLFLOW_TRACING_ENABLED=false` are the safe defaults. `MLFLOW_TRACE_SAMPLING_RATIO` controls sampling, `MLFLOW_TRACE_CAPTURE_CONTENT` defaults to `false`, `MLFLOW_TRACING_DESTINATION` accepts `<catalog>.<schema>`, and `MLFLOW_TRACING_SQL_WAREHOUSE_ID` is a non-secret warehouse ID. Profile keys under `tracing` map to these values. `MLFLOW_TRACING_ENABLED`, `ENABLE_MLFLOW_TRACING`, `beta=true`, and `preview_or_beta=true` are configuration markers, not proof that a workspace feature, Managed MLflow experiment, SQL warehouse, UC destination, or permissions are enabled. Trace attributes contain correlation/request IDs, model, route, provider, and tool/MCP/skill identifiers or counts; content remains omitted by default and is recursively redacted and length-limited when explicitly enabled.

`mlflow.genai` is used only for GenAI evaluation and scorer/trace APIs; it is not required to serve an invocation. The serving image requirement remains exactly `mlflow>=3.1.0,<4` plus `databricks-agents>=1.2.0`. UC managed trace ingestion is a separate operator environment requirement: independently pin and validate `mlflow[databricks]>=3.14.0` (MLflow Python 3.14+) before enabling a UC trace location. Do not change the serving image requirement to 3.14 merely because the operator notebook needs the newer UC ingestion APIs.

### UC trace operator setup

The operator performs these steps in a Databricks notebook or approved deployment job; this repository does not execute them or create resources:

1. Confirm in the workspace UI and with the workspace administrator that the workspace is Unity Catalog-enabled, in a supported region, and exposes the required OpenTelemetry/trace preview or feature flag. Confirm a Managed MLflow experiment where the operator has `CAN EDIT`, a SQL warehouse with `CAN USE`, monitoring job permission, and the endpoint managed identity/service principal. These cloud gates must be confirmed by the operator; environment variables and beta markers do not enable them.
2. Use a separate operator notebook environment with `mlflow[databricks]>=3.14.0` (MLflow Python 3.14+), then create or retrieve the Managed MLflow experiment and bind its trace location:

```python
import os
import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = "<SQL_WAREHOUSE_ID>"
experiment = mlflow.set_experiment(
    experiment_name="/Shared/<EXPERIMENT_NAME>",
    trace_location=UnityCatalog(
        catalog_name="<CATALOG>",
        schema_name="<SCHEMA>",
        table_prefix="<TABLE_PREFIX>",
    ),
)
```

   `mlflow.set_experiment(..., trace_location=UnityCatalog(...))` binds the experiment to the trace location. A `create_experiment` call with the same trace location is the equivalent creation path; do not bind an existing experiment to a second location. The location automatically creates four tables: `<catalog>.<schema>.<prefix>_otel_annotations`, `<prefix>_otel_logs`, `<prefix>_otel_metrics`, and `<prefix>_otel_spans`.
3. Grant the endpoint identity explicit privileges on the destination and confirm the operator/monitoring job permissions; `ALL_PRIVILEGES` is not sufficient:

```sql
GRANT USE_CATALOG ON CATALOG <CATALOG> TO `<SERVING_IDENTITY>`;
GRANT USE_SCHEMA ON SCHEMA <CATALOG>.<SCHEMA> TO `<SERVING_IDENTITY>`;
GRANT MODIFY, SELECT ON TABLE <CATALOG>.<SCHEMA>.<PREFIX>_otel_annotations TO `<SERVING_IDENTITY>`;
GRANT MODIFY, SELECT ON TABLE <CATALOG>.<SCHEMA>.<PREFIX>_otel_logs TO `<SERVING_IDENTITY>`;
GRANT MODIFY, SELECT ON TABLE <CATALOG>.<SCHEMA>.<PREFIX>_otel_metrics TO `<SERVING_IDENTITY>`;
GRANT MODIFY, SELECT ON TABLE <CATALOG>.<SCHEMA>.<PREFIX>_otel_spans TO `<SERVING_IDENTITY>`;
```

4. Configure the destination in the operator-managed application or endpoint environment, without creating resources from this repository:

```bash
export MLFLOW_TRACING_DESTINATION=<CATALOG>.<SCHEMA>
export MLFLOW_TRACING_SQL_WAREHOUSE_ID=<SQL_WAREHOUSE_ID>
```

   `MLFLOW_TRACING_DESTINATION`, `MLFLOW_TRACING_SQL_WAREHOUSE_ID`, `MLFLOW_TRACING_ENABLED`, and `beta`/`preview_or_beta` values do not create or prove any cloud setup. `deploy/databricks/monitoring.example.json` contains non-sensitive placeholders and requires explicit operator confirmation fields.

### Production scorer setup and rollback

In a Databricks notebook, after the trace destination is working, configure production monitoring with the exact warehouse and scorer APIs:

```python
from mlflow.tracing import set_databricks_monitoring_sql_warehouse_id
from mlflow.genai.scorers import Safety, ScorerSamplingConfig

set_databricks_monitoring_sql_warehouse_id(
    sql_warehouse_id="<SQL_WAREHOUSE_ID>",
    experiment_id="<EXPERIMENT_ID>",
)
safety = Safety().register(name="safety_monitor")
safety = safety.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))

# Stop it during rollback or shutdown:
safety.stop()
```

`set_databricks_monitoring_sql_warehouse_id` writes the warehouse ID to the `mlflow.monitoring.sqlWarehouseId` tag on the specified experiment; the monitoring job reads that tag. Setting `MLFLOW_TRACING_SQL_WAREHOUSE_ID` alone is not a substitute for this monitoring helper and does not enable monitoring. Operator sequence: confirm the experiment `CAN EDIT` and monitoring job permission, validate one independent test endpoint, then start the scorer through the approved workspace workflow. To disable, set profile `tracing.enabled` or `MLFLOW_TRACING_ENABLED` to `false`, stop each scorer with `.stop()`, and restore the previous verified model version. The current production `simo_prod.mysql_cdc_sync.haystack_agent_router:2` is reported `READY` but its business invocation is failing; any new version must be deployed to an independent test endpoint and must not overwrite production.

### Disable and rollback tracing

To stop tracing, set profile/runtime `trace_enabled=false` and the deployment control `ENABLE_MLFLOW_TRACING=false` (also set `MLFLOW_TRACING_ENABLED=false` for this runtime), stop registered scorers with `.stop()`, and remove or restore the trace destination and SQL warehouse environment values. Keep the old verified model available. Do not delete UC trace tables, and do not rebind an existing experiment to another location; create a new experiment for a new location. Workspace preview/feature flags must be rolled back by an administrator through the official workspace UI/process. This repository does not enable, validate, or roll back any cloud capability.

## Rollback and current production note

Promote the prior known-good model version through the Serving Endpoint's normal configuration workflow to roll back; keep the new model version registered for investigation.

No Databricks cloud write, endpoint update, endpoint creation, online route verification, or token retrieval is performed by these files.
