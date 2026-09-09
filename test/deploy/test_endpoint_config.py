from deploy.databricks.endpoint_config import (
    build_endpoint_config,
    build_served_entity,
    build_session_environment,
    build_update_config,
)


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
        "DATABRICKS_SERVING_USE_WORKLOAD_IDENTITY": "true",
    }
    assert config["tags"][-1] == {"key": "task", "value": "agent/v1/responses"}


def test_endpoint_config_exposes_serving_identity_to_tracing():
    entity = build_served_entity(
        served_entity_name="haystack-agent-router-3",
        model_name="simo_prod.mysql_cdc_sync.haystack_agent_router",
        model_version=3,
        workspace_id="3761760657895016",
        serving_endpoint_name="haystack-agent-router",
    )

    assert entity["environment_vars"] == {
        "DATABRICKS_WORKSPACE_ID": "3761760657895016",
        "HAYSTACK_SERVING_ENDPOINT_NAME": "haystack-agent-router",
        "HAYSTACK_SERVED_ENTITY_NAME": "haystack-agent-router-3",
        "HAYSTACK_SERVED_MODEL_VERSION": "3",
    }


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
        served_entity_name="router-4", model_name="catalog.schema.model", model_version="4", workspace_id="123"
    )
    update = build_update_config(entity)
    assert set(update) == {"served_entities", "traffic_config"}


def test_served_entity_rejects_invalid_model_version():
    try:
        build_served_entity(
            served_entity_name="router-1", model_name="catalog.schema.model", model_version="latest", workspace_id="123"
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
        tracing_environment={"MLFLOW_TRACKING_URI": "databricks", "MLFLOW_TRACE_SAMPLING_RATIO": "1.0"},
    )
    assert entity["environment_vars"]["MLFLOW_TRACKING_URI"] == "databricks"
    assert entity["environment_vars"]["MLFLOW_TRACE_SAMPLING_RATIO"] == "1.0"


def test_served_entity_configures_lakebase_session_persistence():
    entity = build_served_entity(
        served_entity_name="router-6",
        model_name="catalog.schema.model",
        model_version=6,
        workspace_id="123",
        session_environment=build_session_environment(
            lakebase_endpoint="projects/test-ontos/branches/production/endpoints/primary",
            database_user="9d049dd7-047a-41fc-9000-02a3baf155ce",
            ttl_seconds=3600,
        ),
    )

    assert entity["environment_vars"]["HAYSTACK_SESSION_LAKEBASE_ENDPOINT"].endswith("/primary")
    assert entity["environment_vars"]["HAYSTACK_SESSION_DATABASE"] == "test_ontos"
    assert entity["environment_vars"]["HAYSTACK_SESSION_AUTO_MIGRATE"] == "true"
    assert entity["environment_vars"]["HAYSTACK_SESSION_USE_USER_AUTH"] == "false"
    assert entity["environment_vars"]["HAYSTACK_SESSION_TTL_SECONDS"] == "3600"
    assert entity["environment_vars"]["HAYSTACK_SESSION_DATABASE_USER"] == "9d049dd7-047a-41fc-9000-02a3baf155ce"


def test_session_environment_can_select_invoking_user_obo_auth():
    environment = build_session_environment(
        lakebase_endpoint="projects/test-ontos/branches/production/endpoints/primary", use_user_auth=True
    )

    assert environment["HAYSTACK_SESSION_USE_USER_AUTH"] == "true"
    assert environment["HAYSTACK_SESSION_DATABASE"] == "test_ontos"


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
