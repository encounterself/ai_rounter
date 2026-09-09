from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from integrations.databricks.session_store import (
    InMemorySessionStore,
    LakebaseSessionStore,
    SessionConflictError,
    SessionStoreError,
    session_store_from_env,
)


def test_session_store_round_trips_and_uses_compare_and_swap() -> None:
    store = InMemorySessionStore()
    first = store.save(
        "session-1", "tenant-1", {"messages": [{"role": "user", "content": "hi"}]}, expected_revision=None
    )
    assert first.revision == 1
    second = store.save(
        "session-1",
        "tenant-1",
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        expected_revision=1,
        last_turn_id="turn-2",
    )
    assert second.revision == 2
    assert store.load("session-1", "tenant-1").last_turn_id == "turn-2"  # type: ignore[union-attr]

    with pytest.raises(SessionConflictError):
        store.save("session-1", "tenant-1", {"messages": []}, expected_revision=1)


def test_session_store_redacts_credentials_before_persistence() -> None:
    store = InMemorySessionStore()
    snapshot = store.save(
        "session-1",
        "tenant-1",
        {"messages": [], "authorization": "Bearer secret", "nested": {"api_key": "value"}},
        expected_revision=None,
    )

    assert snapshot.state["authorization"] == "[REDACTED]"
    assert snapshot.state["nested"]["api_key"] == "[REDACTED]"  # type: ignore[index]


def test_session_store_does_not_restore_expired_snapshots() -> None:
    store = InMemorySessionStore()
    store.save(
        "session-expired", "tenant-1", {"messages": []}, expected_revision=None, expires_at="2000-01-01T00:00:00+00:00"
    )

    assert store.load("session-expired", "tenant-1") is None


def test_lakebase_session_store_refreshes_oauth_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    connections: list[dict[str, object]] = []

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakePsycopg:
        @staticmethod
        def connect(**kwargs: object) -> FakeConnection:
            connections.append(kwargs)
            return FakeConnection()

    class FakeWorkspace:
        class Postgres:
            @staticmethod
            def get_endpoint(*, name: str) -> object:
                assert name == "projects/test-ontos/branches/production/endpoints/primary"
                return SimpleNamespace(status=SimpleNamespace(hosts=SimpleNamespace(host="lakebase.example")))

            @staticmethod
            def generate_database_credential(*, endpoint: str) -> object:
                assert endpoint.endswith("/primary")
                return SimpleNamespace(token="oauth-token")

        postgres = Postgres()
        current_user = SimpleNamespace(me=lambda: SimpleNamespace(user_name="agent@example.com"))

    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_SERVING_USE_WORKLOAD_IDENTITY", "true")
    monkeypatch.setitem(sys.modules, "psycopg", FakePsycopg())
    monkeypatch.setitem(sys.modules, "databricks.sdk", SimpleNamespace(WorkspaceClient=lambda **_: FakeWorkspace()))
    monkeypatch.setitem(sys.modules, "databricks_ai_bridge", SimpleNamespace(ModelServingUserCredentials=object))

    LakebaseSessionStore("projects/test-ontos/branches/production/endpoints/primary")._connect()

    assert connections == [
        {
            "host": "lakebase.example",
            "dbname": "test_ontos",
            "user": "agent@example.com",
            "password": "oauth-token",
            "sslmode": "require",
        }
    ]


def test_session_store_from_env_selects_lakebase(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = "projects/test-ontos/branches/production/endpoints/primary"
    monkeypatch.setenv("HAYSTACK_SESSION_LAKEBASE_ENDPOINT", endpoint)
    monkeypatch.setenv("HAYSTACK_SESSION_AUTO_MIGRATE", "false")

    store = session_store_from_env()

    assert isinstance(store, LakebaseSessionStore)
    assert store.database == "test_ontos"


def test_lakebase_session_store_uses_obo_when_workload_identity_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeWorkspace:
        class Postgres:
            @staticmethod
            def get_endpoint(*, name: str) -> object:
                return SimpleNamespace(status=SimpleNamespace(hosts=SimpleNamespace(host="lakebase.example")))

            @staticmethod
            def generate_database_credential(*, endpoint: str) -> object:
                return SimpleNamespace(token="oauth-token")

        postgres = Postgres()
        current_user = SimpleNamespace(me=lambda: SimpleNamespace(user_name="caller@example.com"))

        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    class ModelServingUserCredentials:
        pass

    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.delenv("DATABRICKS_SERVING_USE_WORKLOAD_IDENTITY", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda **_: FakeConnection()))
    monkeypatch.setitem(sys.modules, "databricks.sdk", SimpleNamespace(WorkspaceClient=FakeWorkspace))
    monkeypatch.setitem(
        sys.modules, "databricks_ai_bridge", SimpleNamespace(ModelServingUserCredentials=ModelServingUserCredentials)
    )

    LakebaseSessionStore("projects/test-ontos/branches/production/endpoints/primary")._connect()

    assert calls[0]["credentials_strategy"].__class__ is ModelServingUserCredentials


def test_lakebase_session_store_rejects_database_user_identity_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWorkspace:
        class Postgres:
            @staticmethod
            def get_endpoint(*, name: str) -> object:
                return SimpleNamespace(status=SimpleNamespace(hosts=SimpleNamespace(host="lakebase.example")))

            @staticmethod
            def generate_database_credential(*, endpoint: str) -> object:
                return SimpleNamespace(token="oauth-token")

        postgres = Postgres()
        current_user = SimpleNamespace(me=lambda: SimpleNamespace(user_name="84d1afd7-endpoint"))

    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("DATABRICKS_SERVING_USE_WORKLOAD_IDENTITY", "true")
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda **_: None))
    monkeypatch.setitem(sys.modules, "databricks.sdk", SimpleNamespace(WorkspaceClient=FakeWorkspace))

    with pytest.raises(SessionStoreError, match="does not match the OAuth token subject") as error:
        LakebaseSessionStore("projects/test-ontos/branches/production/endpoints/primary", user="9d049-user")._connect()

    assert error.value.code == "lakebase_identity_mismatch"
    assert error.value.diagnostic["auth_strategy"] == "workload_identity"


def test_lakebase_session_store_rejects_invalid_endpoint() -> None:
    with pytest.raises(ValueError):
        LakebaseSessionStore("https://database.example")


def test_lakebase_session_store_wraps_authentication_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda **_: None))

    def raise_authentication_error(**_: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "databricks.sdk", SimpleNamespace(WorkspaceClient=raise_authentication_error))

    with pytest.raises(SessionStoreError, match="OAuth credential initialization failed"):
        LakebaseSessionStore("projects/test-ontos/branches/production/endpoints/primary")._connect()
