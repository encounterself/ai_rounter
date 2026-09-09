"""
Session persistence primitives for Databricks-served Haystack agents.

The store is intentionally independent from Haystack ``State`` so the serving
wrapper can persist a request-safe snapshot before and after an Agent run.  A
revision is required for every write, which prevents two serving workers from
silently overwriting a conversation turn.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Protocol


class SessionConflictError(RuntimeError):
    """Raised when a session write uses a stale revision."""


class SessionStoreError(RuntimeError):
    """Raised when a session cannot be loaded or persisted."""

    def __init__(
        self, message: str, *, code: str = "session_store_error", diagnostic: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostic = dict(diagnostic or {})


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    session_id: str
    tenant_id: str
    revision: int
    state: Mapping[str, Any]
    agent_profile: str | None = None
    route: str | None = None
    model_alias: str | None = None
    last_turn_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    expires_at: str | None = None
    snapshot_hash: str = ""

    @classmethod
    def create(
        cls,
        session_id: str,
        tenant_id: str,
        state: Mapping[str, Any],
        *,
        revision: int,
        previous: "SessionSnapshot | None" = None,
        agent_profile: str | None = None,
        route: str | None = None,
        model_alias: str | None = None,
        last_turn_id: str | None = None,
        expires_at: str | None = None,
    ) -> "SessionSnapshot":
        """Create a redacted, hashed snapshot with revision metadata."""

        now = datetime.now(timezone.utc).isoformat()
        created_at = previous.created_at if previous else now
        resolved_expiry = expires_at or (previous.expires_at if previous else None) or _default_expiry(now)
        safe_state = _safe_state(state)
        digest = hashlib.sha256(
            json.dumps(safe_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            session_id=session_id,
            tenant_id=tenant_id,
            revision=revision,
            state=safe_state,
            agent_profile=agent_profile or (previous.agent_profile if previous else None),
            route=route or (previous.route if previous else None),
            model_alias=model_alias or (previous.model_alias if previous else None),
            last_turn_id=last_turn_id or (previous.last_turn_id if previous else None),
            created_at=created_at,
            updated_at=now,
            expires_at=resolved_expiry,
            snapshot_hash=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot mapping."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SessionSnapshot":
        """Restore a snapshot from a persisted mapping."""

        return cls(
            session_id=str(value["session_id"]),
            tenant_id=str(value.get("tenant_id", "default")),
            revision=int(value["revision"]),
            state=value.get("state", {}),
            agent_profile=value.get("agent_profile"),
            route=value.get("route"),
            model_alias=value.get("model_alias"),
            last_turn_id=value.get("last_turn_id"),
            created_at=str(value.get("created_at", "")),
            updated_at=str(value.get("updated_at", "")),
            expires_at=value.get("expires_at"),
            snapshot_hash=str(value.get("snapshot_hash", "")),
        )


class SessionStore(Protocol):
    def load(self, session_id: str, tenant_id: str = "default") -> SessionSnapshot | None:
        """Load one tenant-scoped session snapshot."""

        ...

    def save(
        self,
        session_id: str,
        tenant_id: str,
        state: Mapping[str, Any],
        *,
        expected_revision: int | None,
        agent_profile: str | None = None,
        route: str | None = None,
        model_alias: str | None = None,
        last_turn_id: str | None = None,
        expires_at: str | None = None,
    ) -> SessionSnapshot:
        """Persist one snapshot with compare-and-swap revision control."""

        ...


class InMemorySessionStore:
    """Process-local store for tests and explicitly degraded local mode."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], SessionSnapshot] = {}
        self._lock = RLock()

    def load(self, session_id: str, tenant_id: str = "default") -> SessionSnapshot | None:
        """Load one tenant-scoped session snapshot."""

        with self._lock:
            key = (tenant_id, session_id)
            snapshot = self._sessions.get(key)
            if snapshot is not None and _is_expired(snapshot):
                del self._sessions[key]
                return None
            return snapshot

    def save(self, session_id: str, tenant_id: str, state: Mapping[str, Any], **kwargs: Any) -> SessionSnapshot:
        """Persist one snapshot using an in-process compare-and-swap check."""

        expected_revision = kwargs.pop("expected_revision")
        with self._lock:
            previous = self._sessions.get((tenant_id, session_id))
            _check_revision(previous, expected_revision)
            snapshot = SessionSnapshot.create(
                session_id,
                tenant_id,
                state,
                revision=(previous.revision + 1 if previous else 1),
                previous=previous,
                **kwargs,
            )
            self._sessions[(tenant_id, session_id)] = snapshot
            return snapshot


class JsonFileSessionStore:
    """Atomic file-backed store for local development and single-worker smoke tests."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, session_id: str, tenant_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{tenant_id}__{session_id}")
        return self.directory / f"{safe}.json"

    def load(self, session_id: str, tenant_id: str = "default") -> SessionSnapshot | None:
        """Load one JSON-backed tenant-scoped session snapshot."""

        path = self._path(session_id, tenant_id)
        try:
            snapshot = SessionSnapshot.from_mapping(json.loads(path.read_text(encoding="utf-8")))
            if _is_expired(snapshot):
                path.unlink(missing_ok=True)
                return None
            return snapshot
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise SessionStoreError("session snapshot could not be read") from error

    def save(self, session_id: str, tenant_id: str, state: Mapping[str, Any], **kwargs: Any) -> SessionSnapshot:
        """Persist one JSON-backed snapshot atomically."""

        expected_revision = kwargs.pop("expected_revision")
        with self._lock:
            previous = self.load(session_id, tenant_id)
            _check_revision(previous, expected_revision)
            snapshot = SessionSnapshot.create(
                session_id,
                tenant_id,
                state,
                revision=(previous.revision + 1 if previous else 1),
                previous=previous,
                **kwargs,
            )
            path = self._path(session_id, tenant_id)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot.to_dict(), sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
            return snapshot


class PostgresSessionStore:
    """
    Revisioned store for Databricks Lakebase/Postgres.

    ``psycopg`` is imported lazily so local tests do not need a database driver.
    The caller supplies a Postgres connection URL through the deployment
    secret, never through the public serving profile.
    """

    def __init__(self, dsn: str, *, table: str = "haystack_agent_sessions", auto_migrate: bool = False) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("session table must be a simple identifier")
        self.dsn = dsn
        self.table = table
        self.auto_migrate = auto_migrate
        self._schema_ready = False
        self._schema_lock = RLock()

    def _connect(self) -> Any:
        try:
            import psycopg
        except ImportError as error:
            raise SessionStoreError(
                "psycopg is required for Postgres session persistence", code="session_dependency_missing"
            ) from error
        return psycopg.connect(self.dsn)

    def ensure_schema(self) -> None:
        """Create the Postgres session table when it does not exist."""

        with self._schema_lock:
            if self._schema_ready:
                return
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        tenant_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        revision BIGINT NOT NULL,
                        payload JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (tenant_id, session_id)
                    )
                    """
                )
            self._schema_ready = True

    def _ensure_schema_if_needed(self) -> None:
        if self.auto_migrate:
            self.ensure_schema()

    def load(self, session_id: str, tenant_id: str = "default") -> SessionSnapshot | None:
        """Load one tenant-scoped snapshot from Postgres."""

        self._ensure_schema_if_needed()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {self.table} WHERE tenant_id = %s AND session_id = %s", (tenant_id, session_id)
            )
            row = cursor.fetchone()
        snapshot = SessionSnapshot.from_mapping(row[0]) if row else None
        return None if snapshot is not None and _is_expired(snapshot) else snapshot

    def save(self, session_id: str, tenant_id: str, state: Mapping[str, Any], **kwargs: Any) -> SessionSnapshot:
        """Persist one Postgres snapshot with revision compare-and-swap."""

        self._ensure_schema_if_needed()
        expected_revision = kwargs.pop("expected_revision")
        previous = self.load(session_id, tenant_id)
        _check_revision(previous, expected_revision)
        snapshot = SessionSnapshot.create(
            session_id,
            tenant_id,
            state,
            revision=(previous.revision + 1 if previous else 1),
            previous=previous,
            **kwargs,
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.table} (tenant_id, session_id, revision, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (tenant_id, session_id) DO UPDATE SET
                    revision = EXCLUDED.revision,
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                WHERE {self.table}.revision = %s
                """,
                (tenant_id, session_id, snapshot.revision, json.dumps(snapshot.to_dict()), expected_revision or 0),
            )
            if cursor.rowcount != 1:
                raise SessionConflictError("session revision changed during write")
        return snapshot


class LakebaseSessionStore(PostgresSessionStore):
    """Revisioned store using a Databricks Lakebase OAuth credential per connection."""

    def __init__(
        self,
        endpoint: str,
        *,
        database: str = "test_ontos",
        user: str | None = None,
        table: str = "haystack_agent_sessions",
        auto_migrate: bool = False,
    ) -> None:
        super().__init__("", table=table, auto_migrate=auto_migrate)
        if not re.fullmatch(r"projects/[a-z0-9-]+/branches/[a-z0-9-]+/endpoints/[a-z0-9-]+", endpoint):
            raise ValueError(
                "Lakebase endpoint must be a projects/<project>/branches/<branch>/endpoints/<endpoint> resource"
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", database):
            raise ValueError("Lakebase database must be a simple identifier")
        if user is not None and not user.strip():
            raise ValueError("Lakebase database user must not be empty")
        self.endpoint = endpoint
        self.database = database
        self.user = user

    def _connect(self) -> Any:
        try:
            import psycopg
        except ImportError as error:
            raise SessionStoreError(
                "psycopg is required for Postgres session persistence",
                code="session_dependency_missing",
                diagnostic={"dependency": "psycopg"},
            ) from error
        try:
            from databricks.sdk import WorkspaceClient

            auth_strategy = _lakebase_auth_strategy()
            if auth_strategy == "obo":
                from databricks_ai_bridge import ModelServingUserCredentials

                workspace = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
            elif auth_strategy == "m2m":
                workspace = WorkspaceClient(auth_type="oauth-m2m")
            else:
                workspace = WorkspaceClient()
            endpoint = workspace.postgres.get_endpoint(name=self.endpoint)
            host = endpoint.status.hosts.host
            authenticated_user = workspace.current_user.me().user_name
            _validate_lakebase_database_user(
                configured_user=self.user,
                authenticated_user=authenticated_user,
                auth_strategy=auth_strategy,
                endpoint=self.endpoint,
                database=self.database,
            )
            token = workspace.postgres.generate_database_credential(endpoint=self.endpoint).token
            user = authenticated_user
        except SessionStoreError:
            raise
        except Exception as error:
            raise SessionStoreError(
                "Lakebase OAuth credential initialization failed",
                code="lakebase_auth_failed",
                diagnostic=_lakebase_diagnostic(
                    error,
                    stage="credential_initialization",
                    endpoint=self.endpoint,
                    database=self.database,
                    user=self.user,
                    auth_strategy=auth_strategy if "auth_strategy" in locals() else None,
                ),
            ) from error
        try:
            return psycopg.connect(host=host, dbname=self.database, user=user, password=token, sslmode="require")
        except Exception as error:
            raise SessionStoreError(
                "Lakebase Postgres connection failed",
                code="lakebase_connection_failed",
                diagnostic=_lakebase_diagnostic(
                    error,
                    stage="postgres_connect",
                    endpoint=self.endpoint,
                    database=self.database,
                    user=user,
                    host=host,
                    auth_strategy=auth_strategy if "auth_strategy" in locals() else None,
                ),
            ) from error


def _check_revision(previous: SessionSnapshot | None, expected_revision: int | None) -> None:
    actual = previous.revision if previous else None
    if actual != expected_revision:
        raise SessionConflictError(f"session revision conflict: expected {expected_revision}, actual {actual}")


def _default_expiry(now: str) -> str | None:
    """Resolve the optional deployment-wide session TTL."""

    raw_ttl = os.getenv("HAYSTACK_SESSION_TTL_SECONDS")
    if not raw_ttl:
        return None
    try:
        ttl_seconds = float(raw_ttl)
    except ValueError:
        return None
    if ttl_seconds <= 0:
        return None
    current = datetime.fromisoformat(now)
    return (current + timedelta(seconds=ttl_seconds)).isoformat()


def _is_expired(snapshot: SessionSnapshot) -> bool:
    if not snapshot.expires_at:
        return False
    try:
        expiry = datetime.fromisoformat(snapshot.expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return expiry <= datetime.now(timezone.utc)


def _safe_state(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): "[REDACTED]" if _sensitive(str(key)) else _safe_state(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_state(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:2048]


def _sensitive(key: str) -> bool:
    return bool(re.search(r"(?:authorization|token|secret|password|api[_-]?key|credential)", key, re.IGNORECASE))


def _is_model_serving_environment() -> bool:
    return (
        os.getenv("IS_IN_DB_MODEL_SERVING_ENV", "").lower() == "true"
        or os.getenv("IS_IN_DATABRICKS_MODEL_SERVING_ENV", "").lower() == "true"
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _has_m2m_environment() -> bool:
    return bool(os.getenv("DATABRICKS_CLIENT_ID")) and bool(os.getenv("DATABRICKS_CLIENT_SECRET"))


def _lakebase_auth_strategy() -> str:
    """Select the identity that must own the Lakebase OAuth database role."""

    if _as_bool(os.getenv("HAYSTACK_SESSION_USE_USER_AUTH"), False):
        return "obo"
    if _is_model_serving_environment():
        if _has_m2m_environment():
            return "m2m"
        if _as_bool(os.getenv("DATABRICKS_SERVING_USE_WORKLOAD_IDENTITY"), False):
            return "workload_identity"
        return "obo"
    return "default"


def _validate_lakebase_database_user(
    *, configured_user: str | None, authenticated_user: str, auth_strategy: str, endpoint: str, database: str
) -> None:
    if configured_user is None or configured_user == authenticated_user:
        return
    raise SessionStoreError(
        "configured Lakebase database user does not match the OAuth token subject",
        code="lakebase_identity_mismatch",
        diagnostic={
            "stage": "identity_binding",
            "auth_strategy": auth_strategy,
            "endpoint": endpoint,
            "database": database,
            "configured_user": configured_user,
            "authenticated_user": authenticated_user,
        },
    )


def _lakebase_diagnostic(
    error: Exception,
    *,
    stage: str,
    endpoint: str,
    database: str,
    user: str | None,
    host: str | None = None,
    auth_strategy: str | None = None,
) -> dict[str, Any]:
    """Return connection diagnostics without exposing OAuth credentials."""

    message = re.sub(
        r"(?i)(password|token|secret|authorization|credential)\s*[=:]\s*[^\s,;]+", r"\1=[REDACTED]", str(error)
    )
    return {
        "stage": stage,
        "endpoint": endpoint,
        "database": database,
        "database_user": user,
        "database_host": host,
        "auth_strategy": auth_strategy,
        "exception_type": type(error).__name__,
        "exception_message": message[:512],
    }


def session_store_from_env() -> SessionStore:
    """Build the configured Postgres, JSON-file, or in-memory session store."""

    lakebase_endpoint = os.getenv("HAYSTACK_SESSION_LAKEBASE_ENDPOINT")
    if lakebase_endpoint:
        auto_migrate = os.getenv("HAYSTACK_SESSION_AUTO_MIGRATE", "false").lower() in {"1", "true", "yes"}
        return LakebaseSessionStore(
            lakebase_endpoint,
            database=os.getenv("HAYSTACK_SESSION_DATABASE", "test_ontos"),
            user=os.getenv("HAYSTACK_SESSION_DATABASE_USER"),
            table=os.getenv("HAYSTACK_SESSION_TABLE", "haystack_agent_sessions"),
            auto_migrate=auto_migrate,
        )
    dsn = os.getenv("HAYSTACK_SESSION_DATABASE_URL") or os.getenv("DATABASE_URL")
    if dsn:
        auto_migrate = os.getenv("HAYSTACK_SESSION_AUTO_MIGRATE", "false").lower() in {"1", "true", "yes"}
        return PostgresSessionStore(
            dsn, table=os.getenv("HAYSTACK_SESSION_TABLE", "haystack_agent_sessions"), auto_migrate=auto_migrate
        )
    directory = os.getenv("HAYSTACK_SESSION_STORE_PATH")
    if directory:
        return JsonFileSessionStore(directory)
    return InMemorySessionStore()


__all__ = [
    "InMemorySessionStore",
    "JsonFileSessionStore",
    "LakebaseSessionStore",
    "PostgresSessionStore",
    "SessionConflictError",
    "SessionSnapshot",
    "SessionStore",
    "SessionStoreError",
    "session_store_from_env",
]
