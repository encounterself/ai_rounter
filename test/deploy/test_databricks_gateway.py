from __future__ import annotations

import json
import importlib.util
import os
import sys
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace

import httpx
import pytest

import deploy.databricks.agent as agent_module
from deploy.databricks.agent import Agent, normalize_request
from deploy.databricks.gateway_client import (
    ConfigurationError,
    GatewayClient,
    GatewayError,
    GatewayHTTPError,
    GatewayResponseError,
    GatewaySettings,
    ManagedAuth,
    ManagedAuthError,
    RouteValidationError,
)
from deploy.databricks.profile_loader import ProfileError, validate_profile
from deploy.databricks.serving_runtime import LocalServingRuntime
from deploy.databricks.tracing import TracingConfig, redact


ROUTE = "model-services/simo_prod.ai_models.route_test"
METADATA = {"provider": "test", "model": "test-model", "enabled": True}


def test_gateway_settings_use_workspace_host_when_base_url_is_absent(monkeypatch):
    monkeypatch.delenv("DATABRICKS_AI_GATEWAY_BASE_URL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc-b21ff395-726b.cloud.databricks.com/")
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "3761760657895016")
    settings = GatewaySettings.from_env()
    assert settings.workspace_id == "3761760657895016"
    assert settings.base_url == "https://dbc-b21ff395-726b.cloud.databricks.com/ai-gateway/mlflow/v1"
    assert settings.workspace_host == "https://dbc-b21ff395-726b.cloud.databricks.com"


def test_gateway_settings_explicit_base_url_wins(monkeypatch):
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "3761760657895016")
    monkeypatch.setenv("DATABRICKS_AI_GATEWAY_BASE_URL", "https://explicit.example/")
    settings = GatewaySettings.from_env()
    assert settings.base_url == "https://explicit.example"


def test_gateway_settings_require_base_url_or_workspace_id(monkeypatch):
    monkeypatch.delenv("DATABRICKS_AI_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    with pytest.raises(ConfigurationError) as raised:
        GatewaySettings.from_env()
    assert raised.value.diagnostic["error_code"] == "missing_gateway_base_url"
    assert raised.value.diagnostic["missing"] == [
        "DATABRICKS_AI_GATEWAY_BASE_URL",
        "DATABRICKS_HOST",
    ]


def make_client(handler, *, auth=None, allowed_routes=()):
    settings = GatewaySettings(
        workspace_id="workspace-id",
        base_url="https://gateway.example",
        route=ROUTE,
        allowed_routes=tuple(allowed_routes),
        protocol="chat",
    )
    return GatewayClient(
        settings,
        auth=auth or ManagedAuth(lambda: {"Authorization": "managed"}),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_route_allowlist_and_metadata_are_enforced():
    client = make_client(lambda request: httpx.Response(200, json={"output": []}, request=request))
    with pytest.raises(RouteValidationError) as missing:
        client.invoke({"input": []}, route="other-route", metadata=METADATA)
    assert missing.value.diagnostic["error_code"] == "route_not_allowed"
    with pytest.raises(RouteValidationError) as no_metadata:
        client.invoke({"input": []}, metadata=None)
    assert no_metadata.value.diagnostic["error_code"] == "route_metadata_missing"
    approved = make_client(
        lambda request: httpx.Response(200, json={"output": []}, request=request),
        allowed_routes=("approved",),
    )
    approved.validate_route("approved", METADATA)


def test_system_ai_route_uses_workspace_serving_endpoint_without_model_service_gateway():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=request)

    settings = GatewaySettings(
        workspace_id="workspace-id",
        base_url="https://gateway.example/ai-gateway/mlflow/v1",
        workspace_host="https://workspace.example",
        route="system-ai/databricks-gpt-5-mini",
        protocol="responses",
    )
    client = GatewayClient(
        settings,
        auth=ManagedAuth(lambda: {}),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    metadata = {
        "backend": "databricks_serving",
        "endpoint": "databricks-gpt-5-mini",
        "protocol": "chat",
        "model": "databricks-gpt-5-mini",
    }

    response = client.invoke(
        {"input": [{"role": "user", "content": "hello"}]},
        route="system-ai/databricks-gpt-5-mini",
        metadata=metadata,
    )

    assert seen["url"] == "https://workspace.example/serving-endpoints/databricks-gpt-5-mini/invocations"
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    assert response["choices"][0]["message"]["content"] == "ok"


def test_serving_endpoint_route_derives_endpoint_and_defaults_to_chat():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=request)

    settings = GatewaySettings(
        workspace_id="workspace-id",
        base_url="https://gateway.example/ai-gateway/mlflow/v1",
        workspace_host="https://workspace.example",
        route="serving-endpoints/databricks-kimi-k3",
    )
    client = GatewayClient(
        settings,
        auth=ManagedAuth(lambda: {}),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.invoke(
        {"input": [{"role": "user", "content": "hello"}]},
        route="serving-endpoints/databricks-kimi-k3",
        metadata={},
    )

    assert seen["url"] == "https://workspace.example/serving-endpoints/databricks-kimi-k3/invocations"
    assert seen["body"] == {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    assert response["choices"][0]["message"]["content"] == "ok"


def test_system_ai_profile_metadata_requires_serving_endpoint_and_chat_protocol():
    valid = {
        "default_route": "system-ai/databricks-gpt-5-mini",
        "route_metadata": {
            "backend": "databricks_serving",
            "endpoint": "databricks-gpt-5-mini",
            "protocol": "chat",
        },
    }
    validate_profile(valid)

    invalid = {
        **valid,
        "route_metadata": {"backend": "databricks_serving", "protocol": "responses"},
    }
    with pytest.raises(ProfileError) as raised:
        validate_profile(invalid)
    assert raised.value.code == "serving_endpoint_invalid"

    serving_route_profile = {
        "default_route": "serving-endpoints/databricks-kimi-k3",
        "route_metadata": {"endpoint": "databricks-kimi-k3", "protocol": "chat"},
    }
    validate_profile(serving_route_profile)


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 502, 503])
def test_http_errors_preserve_safe_diagnostics(status):
    def handler(request):
        return httpx.Response(
            status,
            json={"error": {"code": "upstream_code", "message": "do not copy"}},
            request=request,
        )

    client = make_client(handler)
    with pytest.raises(GatewayHTTPError) as raised:
        client.invoke({"input": []}, metadata=METADATA)
    diagnostic = raised.value.diagnostic
    assert diagnostic["http_status"] == status
    assert diagnostic["error_code"] == "upstream_code"
    assert "message" not in json.dumps(diagnostic)
    assert "Authorization" not in json.dumps(diagnostic)


def test_non_json_and_upstream_error_are_distinct():
    client = make_client(lambda request: httpx.Response(200, text="not-json", request=request))
    with pytest.raises(GatewayResponseError) as raised:
        client.invoke({"input": []}, metadata=METADATA)
    assert raised.value.diagnostic["error_code"] == "invalid_json"

    client = make_client(lambda request: httpx.Response(200, json={"error": {"code": "bad"}}, request=request))
    with pytest.raises(GatewayResponseError) as raised:
        client.invoke({"input": []}, metadata=METADATA)
    assert raised.value.diagnostic["error_code"] == "upstream_error"


def test_managed_auth_failure_is_not_silent():
    def broken_auth():
        raise RuntimeError("token should not be exposed")

    client = make_client(
        lambda request: httpx.Response(200, json={"output": []}, request=request),
        auth=ManagedAuth(broken_auth),
    )
    with pytest.raises(GatewayError) as raised:
        client.invoke({"input": []}, metadata=METADATA)
    assert raised.value.diagnostic["error_code"] == "managed_auth_failed"
    assert "token" not in json.dumps(raised.value.diagnostic).lower()


def test_managed_auth_uses_current_databricks_sdk_authenticate_contract(monkeypatch):
    monkeypatch.delenv("IS_IN_DB_MODEL_SERVING_ENV", raising=False)
    monkeypatch.delenv("IS_IN_DATABRICKS_MODEL_SERVING_ENV", raising=False)
    calls = []

    class Config:
        def authenticate(self):
            calls.append("authenticate")
            return {"Authorization": "Bearer managed"}

    class WorkspaceClient:
        config = Config()

        def __init__(self, **kwargs):
            assert kwargs == {}

    sdk_module = ModuleType("databricks.sdk")
    sdk_module.WorkspaceClient = WorkspaceClient
    credentials_module = ModuleType("databricks_ai_bridge")

    class ModelServingUserCredentials:
        pass

    credentials_module.ModelServingUserCredentials = ModelServingUserCredentials
    databricks_module = ModuleType("databricks")
    databricks_module.sdk = sdk_module
    monkeypatch.setitem(sys.modules, "databricks", databricks_module)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_module)
    monkeypatch.setitem(sys.modules, "databricks_ai_bridge", credentials_module)

    assert ManagedAuth().headers() == {"Authorization": "Bearer managed"}
    assert calls == ["authenticate"]


def test_managed_auth_uses_model_serving_strategy(monkeypatch):
    monkeypatch.setenv("IS_IN_DATABRICKS_MODEL_SERVING_ENV", "true")
    calls = []

    class Config:
        def authenticate(self):
            calls.append("authenticate")
            return {"Authorization": "Bearer managed"}

    class WorkspaceClient:
        config = Config()

        def __init__(self, **kwargs):
            calls.append(kwargs)

    sdk_module = ModuleType("databricks.sdk")
    sdk_module.WorkspaceClient = WorkspaceClient
    credentials_module = ModuleType("databricks_ai_bridge")

    class ModelServingUserCredentials:
        pass

    credentials_module.ModelServingUserCredentials = ModelServingUserCredentials
    databricks_module = ModuleType("databricks")
    databricks_module.sdk = sdk_module
    monkeypatch.setitem(sys.modules, "databricks", databricks_module)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_module)
    monkeypatch.setitem(sys.modules, "databricks_ai_bridge", credentials_module)

    assert ManagedAuth().headers() == {"Authorization": "Bearer managed"}
    assert calls[0].keys() == {"credentials_strategy"}
    assert calls[0]["credentials_strategy"].__class__.__name__ == "ModelServingUserCredentials"
    assert calls[1] == "authenticate"


def test_managed_auth_prefers_serving_m2m_environment(monkeypatch):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "injected-client-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "injected-client-secret")
    calls = []

    class Config:
        def authenticate(self):
            calls.append("authenticate")
            return {"Authorization": "Bearer managed"}

    class WorkspaceClient:
        config = Config()

        def __init__(self, **kwargs):
            calls.append(kwargs)

    sdk_module = ModuleType("databricks.sdk")
    sdk_module.WorkspaceClient = WorkspaceClient
    databricks_module = ModuleType("databricks")
    databricks_module.sdk = sdk_module
    monkeypatch.setitem(sys.modules, "databricks", databricks_module)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_module)

    assert ManagedAuth().headers() == {"Authorization": "Bearer managed"}
    assert calls == [{"auth_type": "oauth-m2m"}, "authenticate"]


def test_managed_auth_initialization_failure_has_safe_stage(monkeypatch):
    monkeypatch.setenv("IS_IN_DATABRICKS_MODEL_SERVING_ENV", "true")

    class BrokenWorkspaceClient:
        def __init__(self, **_kwargs):
            raise RuntimeError("secret details")

    sdk_module = ModuleType("databricks.sdk")
    sdk_module.WorkspaceClient = BrokenWorkspaceClient
    databricks_module = ModuleType("databricks")
    databricks_module.sdk = sdk_module
    monkeypatch.setitem(sys.modules, "databricks", databricks_module)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_module)

    with pytest.raises(ManagedAuthError) as raised:
        ManagedAuth().headers()
    assert raised.value.diagnostic["auth_stage"] == "initialization"
    assert raised.value.diagnostic["auth_strategy"] == "obo"
    assert "secret details" not in json.dumps(raised.value.diagnostic)


def test_runtime_and_agent_copy_error_fields_without_prompt_or_secret(monkeypatch):
    monkeypatch.setenv("DATABRICKS_AI_GATEWAY_BASE_URL", "https://gateway.example")
    client = make_client(lambda request: httpx.Response(429, json={"error": {"code": "rate_limited"}}, request=request))
    runtime = LocalServingRuntime(client)
    result = runtime._execute({
        "payload": {"input": [{"role": "user", "content": "secret prompt"}]},
        "route": ROUTE,
        "route_metadata": METADATA,
        "correlation_id": "corr-1",
    })
    assert result["status"] == "failed"
    assert result["error"]["correlation_id"] == "corr-1"
    assert result["error"]["http_status"] == 429
    assert "secret prompt" not in json.dumps(result)

    agent = Agent({
        "default_route": ROUTE,
        "route_metadata": METADATA,
        "gateway": {"base_url": "https://gateway.example"},
    })
    agent.runtime = runtime
    response = agent.predict({"input": [{"role": "user", "content": "secret prompt"}], "stream": False})
    response_data = response if isinstance(response, Mapping) else response.model_dump()
    error = response_data.get("error") or response_data.get("custom_outputs", {}).get("error", {})
    assert response_data["status"] == "failed"
    assert error["error_code"] == "rate_limited"
    assert "secret prompt" not in json.dumps(response_data)


def test_responses_normalization_and_streaming_error():
    assert normalize_request({"input": "Hello!", "stream": False})["input"] == [
        {"role": "user", "content": "Hello!"}
    ]
    client = make_client(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hi"}}]},
            request=request,
        )
    )
    response = client.invoke({"input": [{"role": "user", "content": "Hello!"}]}, metadata=METADATA)
    assert response["choices"][0]["message"]["content"] == "Hi"

    def stream_handler(request):
        return httpx.Response(200, text='data: {"error":{"code":"stream_failed"}}\n\n', request=request)

    stream_client = make_client(stream_handler)
    stream = stream_client.invoke({"input": []}, metadata=METADATA, stream=True)
    with pytest.raises(GatewayResponseError) as raised:
        list(stream)
    assert raised.value.diagnostic["error_code"] == "upstream_stream_error"


def test_chat_gateway_explicitly_converts_responses_input():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi"}}]}, request=request)

    settings = GatewaySettings("workspace-id", "https://gateway.example", ROUTE, protocol="chat")
    client = GatewayClient(
        settings,
        auth=ManagedAuth(lambda: {}),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.invoke({"input": [{"role": "user", "content": "Hello"}]}, metadata=METADATA)
    assert seen["path"] == "/chat/completions"
    assert seen["body"]["model"] == "simo_prod.ai_models.route_test"
    assert seen["body"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert "input" not in seen["body"]


def test_responses_gateway_sends_model_service_name_without_route_prefix():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"object": "response"}, request=request)

    settings = GatewaySettings("workspace-id", "https://gateway.example/ai-gateway/mlflow/v1", ROUTE)
    client = GatewayClient(
        settings,
        auth=ManagedAuth(lambda: {}),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.invoke({"input": [{"role": "user", "content": "Hello"}]}, metadata=METADATA)
    assert seen["path"] == "/ai-gateway/mlflow/v1/responses"
    assert seen["body"]["model"] == "simo_prod.ai_models.route_test"


def test_agent_load_context_requires_profile_and_loads_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "workspace-id")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"default_route": ROUTE, "route_metadata": METADATA}), encoding="utf-8")
    agent = Agent()
    with pytest.raises(ProfileError):
        agent.load_context(SimpleNamespace(artifacts={}, model_config=None))
    agent.load_context(SimpleNamespace(artifacts={"profile": str(profile_path)}, model_config=None))
    assert agent.profile["default_route"] == ROUTE


def test_responses_request_object_is_supported():
    request = SimpleNamespace(
        input="Hello",
        stream=False,
        model="route-model",
        correlation_id="corr-object",
    )
    normalized = normalize_request(request)
    assert normalized == {
        "input": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "model": "route-model",
        "metadata": {},
        "correlation_id": "corr-object",
        "request_id": None,
    }


def test_agent_uses_requested_route_metadata(monkeypatch):
    monkeypatch.setenv("DATABRICKS_AI_GATEWAY_BASE_URL", "https://gateway.example")
    profile = {
        "default_route": ROUTE,
        "route_metadata": METADATA,
        "route_metadata_by_route": {
            "model-services/simo_prod.ai_models.route_gpt-5-5": {"enabled": True, "provider": "openai"}
        },
        "gateway": {"base_url": "https://gateway.example"},
    }
    agent = Agent(profile)
    seen = {}

    def invoke(payload, **kwargs):
        seen["route"] = payload["route"]
        seen["metadata"] = payload["route_metadata"]
        return {"output": []}

    agent.runtime = SimpleNamespace(
        _execute=invoke,
        gateway=SimpleNamespace(settings=SimpleNamespace(protocol="responses", base_url="https://gateway.example")),
    )
    agent.predict({
        "input": [{"role": "user", "content": "Hello"}],
        "custom_inputs": {"model": "model-services/simo_prod.ai_models.route_gpt-5-5"},
        "stream": False,
    })
    assert seen == {
        "route": "model-services/simo_prod.ai_models.route_gpt-5-5",
        "metadata": {"enabled": True, "provider": "openai"},
    }


def test_set_model_registration_can_be_stubbed(monkeypatch):
    registered = []
    monkeypatch.setattr(agent_module, "_set_model", registered.append)
    agent_module._register_model()
    assert registered == [agent_module.MODEL]


def test_top_level_agent_loading_imports_siblings_without_sys_path(monkeypatch):
    module_name = "top_level_agent_under_test"
    spec = importlib.util.spec_from_file_location(module_name, "deploy/databricks/agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    original_path = list(sys.path)

    spec.loader.exec_module(module)

    assert module.Agent.__module__ == module_name
    assert module.GatewayClient.__module__ == "_haystack_databricks_code.gateway_client"
    assert sys.path == original_path


def test_offline_stream_events_are_responses_events():
    agent = Agent({"default_route": ROUTE, "route_metadata": METADATA, "gateway": {"base_url": "https://gateway.example"}})

    class Runtime:
        gateway = SimpleNamespace(settings=SimpleNamespace(protocol="responses", base_url="https://gateway.example"))

        def _execute(self, request):
            return iter([{"type": "response.output_text.delta", "delta": "Hi"}])

    agent.runtime = Runtime()
    events = list(agent.predict_stream(SimpleNamespace(input=[], stream=True, correlation_id="corr-stream")))
    assert events
    event_types = [event.get("type", "") if isinstance(event, dict) else getattr(event, "type", "") for event in events]
    assert all(event_type.startswith("response.") for event_type in event_types)


def test_requirements_pin_managed_mlflow_and_agents():
    requirements = open("deploy/databricks/requirements.txt", encoding="utf-8").read()
    assert "mlflow>=3.16.0,<4" in requirements
    assert "databricks-agents>=1.12.0" in requirements
    assert "databricks-ai-bridge>=0.21.0" in requirements


def test_recursive_trace_redaction_and_length_limit():
    value = redact({"nested": {"Authorization": "secret", "text": "x" * 700}})
    assert value["nested"]["Authorization"] == "[REDACTED]"
    assert len(value["nested"]["text"]) == 512


def test_tracing_destination_and_warehouse_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "catalog.schema")
    monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "warehouse-1")
    config = TracingConfig.from_env()
    assert config.enabled is True
    assert config.destination == "catalog.schema"
    assert config.sql_warehouse_id == "warehouse-1"


def test_tracing_profile_destination_maps_to_runtime_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_DESTINATION", "")
    monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "")
    fake_span = _FakeSpan()
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(start_span=lambda **_kwargs: fake_span))
    from deploy.databricks.tracing import start_span

    config = TracingConfig.from_env({
        "tracing": {
            "enabled": True,
            "sampling_ratio": 1.0,
            "destination": "profile_catalog.profile_schema",
            "sql_warehouse_id": "profile-warehouse",
        }
    })
    with start_span(
        config,
        correlation_id="corr-profile",
        request_id="req-profile",
        model="model",
        route=ROUTE,
        provider="provider",
    ):
        pass
    assert os.environ["MLFLOW_TRACING_DESTINATION"] == "profile_catalog.profile_schema"
    assert os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "profile-warehouse"


def test_tracing_disabled_is_a_complete_noop(monkeypatch):
    def fail_if_imported(**_kwargs):
        raise AssertionError("disabled tracing must not start a span")

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(start_span=fail_if_imported))
    from deploy.databricks.tracing import start_span

    span = start_span(
        TracingConfig(enabled=False),
        correlation_id="corr-disabled",
        request_id="req-disabled",
        model="model",
        route=ROUTE,
        provider="provider",
    )
    with span:
        assert not span.active


class _FakeSpan:
    def __init__(self):
        self.attributes = {}
        self.ended = False
        self.record_exception_called = False

    def set_attributes(self, attributes):
        self.attributes.update(attributes)

    def end(self):
        self.ended = True

    def record_exception(self, _exception):
        self.record_exception_called = True


def test_tracing_enabled_records_safe_dimensions_and_error(monkeypatch):
    fake_span = _FakeSpan()
    fake_mlflow = SimpleNamespace(start_span=lambda **_kwargs: fake_span)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    config = TracingConfig(enabled=True, sampling_ratio=1.0, capture_content=False)
    from deploy.databricks.tracing import start_span

    with start_span(
        config,
        correlation_id="corr-1",
        request_id="req-1",
        model="model-1",
        route=ROUTE,
        provider="provider-1",
        metadata={
            "mcp_binding": "mcp-1",
            "skill_binding": "skill-1",
            "tool_count": 2,
            "tool_names": ["search", "retrieve"],
        },
        payload={"input": "do not trace"},
    ) as span:
        assert span.active
        span.finish(error={"exception_type": "GatewayError", "error_code": "bad"})
    assert fake_span.ended
    assert fake_span.attributes["correlation_id"] == "corr-1"
    assert fake_span.attributes["request_id"] == "req-1"
    assert fake_span.attributes["route"] == ROUTE
    assert fake_span.attributes["provider"] == "provider-1"
    assert fake_span.attributes["tool_names"].startswith('["sha256:')
    assert "mcp-1" not in json.dumps(fake_span.attributes)
    assert "skill-1" not in json.dumps(fake_span.attributes)
    assert "input" not in fake_span.attributes
    assert fake_span.attributes["error.error_code"] == "bad"
    assert all(not isinstance(value, (dict, list)) for value in fake_span.attributes.values())
    assert not fake_span.record_exception_called


def test_span_ends_when_attribute_write_fails(monkeypatch):
    fake_span = _FakeSpan()

    def broken_attributes(_attributes):
        raise RuntimeError("attribute backend failure")

    fake_span.set_attributes = broken_attributes
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(start_span=lambda **_kwargs: fake_span))
    from deploy.databricks.tracing import start_span

    with start_span(
        TracingConfig(enabled=True, sampling_ratio=1.0),
        correlation_id="corr-error",
        request_id="req-error",
        model="model",
        route=ROUTE,
        provider="provider",
    ):
        pass
    assert fake_span.ended


def test_span_prefers_mlflow_agent_span_type(monkeypatch):
    fake_span = _FakeSpan()
    fake_mlflow = ModuleType("mlflow")
    fake_entities = ModuleType("mlflow.entities")
    fake_entities.SpanType = SimpleNamespace(AGENT="typed-agent")
    calls = []

    def start_span(**kwargs):
        calls.append(kwargs)
        return fake_span

    fake_mlflow.start_span = start_span
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.entities", fake_entities)
    from deploy.databricks.tracing import start_span as create_span

    with create_span(
        TracingConfig(enabled=True, sampling_ratio=1.0),
        correlation_id="corr-type",
        request_id="req-type",
        model="model",
        route=ROUTE,
        provider="provider",
    ):
        pass
    assert calls[0]["span_type"] == "typed-agent"


def test_runtime_closes_enabled_spans_for_success_and_stream_error(monkeypatch):
    spans = []

    class Span(_FakeSpan):
        pass

    def start_span(**_kwargs):
        span = Span()
        spans.append(span)
        return span

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(start_span=start_span))
    settings = GatewaySettings("workspace-id", "https://gateway.example", ROUTE, protocol="responses")

    gateway_settings = settings

    class Gateway:
        settings = gateway_settings

        def invoke(self, _payload, **kwargs):
            if kwargs["stream"]:
                def stream():
                    yield {"type": "response.output_text.delta", "delta": "Hi"}
                    raise RuntimeError("secret prompt must not be traced")

                return iter(stream())
            return {"output": []}

    runtime = LocalServingRuntime(Gateway(), TracingConfig(enabled=True, sampling_ratio=1.0))
    assert runtime._execute({"payload": {"input": "secret"}, "route": ROUTE, "route_metadata": METADATA})
    stream = runtime._execute({"payload": {"input": "secret"}, "stream": True, "route": ROUTE, "route_metadata": METADATA})
    with pytest.raises(RuntimeError):
        list(stream)
    assert len(spans) == 2
    assert all(span.ended for span in spans)
    assert "secret prompt" not in json.dumps(spans[1].attributes)


def test_profile_diagnostics():
    with pytest.raises(ProfileError) as raised:
        validate_profile({"default_route": ROUTE, "route_metadata": {}}, available_routes=set())
    assert raised.value.code == "default_route_unavailable"
    with pytest.raises(ProfileError) as raised:
        validate_profile({"default_route": ROUTE})
    assert raised.value.code == "route_metadata_missing"
