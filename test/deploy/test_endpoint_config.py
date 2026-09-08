from deploy.databricks.endpoint_config import build_endpoint_config, build_served_entity, build_update_config


def test_endpoint_config_routes_to_the_same_served_entity():
    entity = build_served_entity(
        served_entity_name="haystack-agent-router-3",
        model_name="simo_prod.mysql_cdc_sync.haystack_agent_router",
        model_version=3,
        workspace_id="3761760657895016",
        workspace_host="https://dbc-b21ff395-726b.cloud.databricks.com",
    )
    config = build_endpoint_config(endpoint_name="haystack-agent-router", served_entity=entity)
    assert config["traffic_config"]["routes"] == [
        {
            "served_entity_name": "haystack-agent-router-3",
            "served_model_name": "haystack-agent-router-3",
            "traffic_percentage": 100,
        }
    ]
    assert entity["environment_vars"] == {
        "DATABRICKS_WORKSPACE_ID": "3761760657895016",
        "DATABRICKS_HOST": "https://dbc-b21ff395-726b.cloud.databricks.com",
        "DATABRICKS_MODEL_SERVING_HOST_URL": "https://dbc-b21ff395-726b.cloud.databricks.com",
        "IS_IN_DB_MODEL_SERVING_ENV": "true",
    }
    assert config["tags"][-1] == {"key": "task", "value": "agent/v1/responses"}


def test_update_config_is_minimal_for_official_cli():
    entity = build_served_entity(
        served_entity_name="haystack-agent-router-7",
        model_name="simo_prod.mysql_cdc_sync.haystack_agent_router",
        model_version="7",
        workspace_id="3761760657895016",
    )

    config = build_update_config(entity)

    assert set(config) == {"served_entities", "traffic_config"}
    assert config["served_entities"] == [entity]
    assert config["traffic_config"]["routes"][0]["served_entity_name"] == "haystack-agent-router-7"


def test_update_config_does_not_replace_endpoint_tags():
    entity = build_served_entity(
        served_entity_name="router-4",
        model_name="catalog.schema.model",
        model_version="4",
        workspace_id="123",
    )
    update = build_update_config(entity)
    assert set(update) == {"served_entities", "traffic_config"}


def test_served_entity_rejects_invalid_model_version():
    try:
        build_served_entity(
            served_entity_name="router-1",
            model_name="catalog.schema.model",
            model_version="latest",
            workspace_id="123",
        )
    except ValueError as error:
        assert "model_version" in str(error)
    else:
        raise AssertionError("invalid model version should fail")


def test_served_entity_passes_secret_references_without_secret_values():
    entity = build_served_entity(
        served_entity_name="router-5",
        model_name="catalog.schema.model",
        model_version=5,
        workspace_id="123",
        client_id_secret_ref="{{secrets/oauth/client-id}}",
        client_secret_secret_ref="{{secrets/oauth/client-secret}}",
    )
    assert entity["environment_vars"]["DATABRICKS_CLIENT_ID"] == "{{secrets/oauth/client-id}}"
    assert entity["environment_vars"]["DATABRICKS_CLIENT_SECRET"] == "{{secrets/oauth/client-secret}}"


def test_served_entity_includes_safe_tracing_environment():
    entity = build_served_entity(
        served_entity_name="router-5",
        model_name="catalog.schema.model",
        model_version=5,
        workspace_id="123",
        tracing_environment={
            "MLFLOW_TRACKING_URI": "databricks",
            "MLFLOW_TRACE_SAMPLING_RATIO": "1.0",
        },
    )
    assert entity["environment_vars"]["MLFLOW_TRACKING_URI"] == "databricks"
    assert entity["environment_vars"]["MLFLOW_TRACE_SAMPLING_RATIO"] == "1.0"


def test_served_entity_rejects_partial_or_literal_secret_configuration():
    try:
        build_served_entity(
            served_entity_name="router-5",
            model_name="catalog.schema.model",
            model_version=5,
            workspace_id="123",
            client_id_secret_ref="client-id-value",
        )
    except ValueError as error:
        assert "secret" in str(error)
    else:
        raise AssertionError("literal or partial secret configuration should fail")
