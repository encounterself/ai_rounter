from __future__ import annotations

import sys

from deploy.databricks import tracing
from deploy.databricks.tracing import TraceSpan, TracingConfig
from haystack import tracing as haystack_tracing


def test_configure_haystack_mlflow_tracing_uses_official_mlflow_adapter(monkeypatch):
    calls = []

    class HaystackMlflow:
        @staticmethod
        def autolog(**kwargs):
            calls.append(kwargs)

    class FakeMlflow:
        haystack = HaystackMlflow()

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    monkeypatch.setattr(haystack_tracing.tracer, "is_content_tracing_enabled", True)
    monkeypatch.setattr(tracing, "_HAYSTACK_MLFLOW_AUTOLOG_CONFIGURED", False)

    config = TracingConfig(enabled=True, sampling_ratio=1.0, capture_content=False)

    assert tracing.configure_haystack_mlflow_tracing(config) is True
    assert calls == [{"log_traces": True, "silent": True}]
    assert haystack_tracing.tracer.is_content_tracing_enabled is False


def test_tracing_config_reads_experiment_id_from_environment(monkeypatch):
    monkeypatch.setenv("ENABLE_MLFLOW_TRACING", "true")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "1575579281444652")
    monkeypatch.setenv("MLFLOW_TRACE_SAMPLING_RATIO", "1.0")

    config = TracingConfig.from_env()

    assert config.experiment_id == "1575579281444652"


def test_trace_span_binds_to_configured_experiment(monkeypatch):
    set_experiment_calls = []

    class FakeSpan:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def set_attributes(self, _attributes):
            return None

    fake_mlflow = type(
        "FakeMlflow",
        (),
        {
            "set_experiment": staticmethod(lambda **kwargs: set_experiment_calls.append(kwargs)),
            "start_span": staticmethod(lambda **_kwargs: FakeSpan()),
            "get_current_active_span": staticmethod(lambda: FakeSpan()),
            "update_current_trace": staticmethod(lambda **_kwargs: None),
            "flush_trace_async_logging": staticmethod(lambda: None),
        },
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with TraceSpan(TracingConfig(enabled=True, sampling_ratio=1.0, experiment_id="1575579281444652"), {}):
        pass

    assert set_experiment_calls == [{"experiment_id": "1575579281444652"}]


def test_trace_metadata_contains_session_and_orchestration_dimensions():
    span = TraceSpan(
        TracingConfig(),
        {
            "trace_id": "trace-1",
            "span_id": "span-1",
            "correlation_id": "correlation-1",
            "request_id": "request-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "component": "haystack-agent-router",
            "agent_profile": "simo-agent-router",
            "model_version": "51",
            "route": "daily",
            "provider": "databricks-serving",
            "tool_name": "review_code",
        },
    )

    metadata = span._trace_metadata()

    assert metadata["haystack.user_id"] == "user-1"
    assert metadata["haystack.session_id"] == "session-1"
    assert metadata["haystack.turn_id"] == "turn-1"
    assert metadata["haystack.agent_profile"] == "simo-agent-router"
    assert metadata["haystack.tool_name"] == "review_code"


def test_trace_metadata_stays_below_mlflow_request_metadata_limit():
    span = TraceSpan(
        TracingConfig(),
        {
            "trace_id": "trace-1",
            "span_id": "span-1",
            "correlation_id": "correlation-1",
            "request_id": "request-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "tenant_id": "default",
            "scenario": "daily",
            "agent_profile": "simo-agent-router",
            "endpoint_name": "haystack-agent-router",
            "served_entity_name": "haystack-agent-router-52",
            "model_version": "52",
            "streamed": False,
        },
    )

    metadata = span._trace_metadata(
        {
            "status": "completed",
            "policy_route": "daily",
            "models": ["daily"],
            "model_targets": ["serving-endpoints/databricks-claude-haiku-4-5"],
            "routes": ["daily"],
            "protocols": ["chat_completions"],
            "providers": ["databricks-serving"],
            "provider_kinds": ["databricks_serving"],
            "route_attempts": ["daily"],
            "mcp_bindings": [],
            "skill_bindings": [],
            "prompt_binding_ids": ["daily-system"],
            "tool_count": 0,
            "tool_names": [],
            "node_count": 1,
            "nodes": [{"node_id": "daily"}],
        }
    )

    assert len(metadata) <= 25
    assert metadata["mlflow.trace.session"] == "session-1"
    assert metadata["mlflow.trace.user"] == "user-1"
    assert metadata["haystack.session_id"] == "session-1"
    assert metadata["haystack.model_version"] == "52"
    assert metadata["haystack.status"] == "completed"
