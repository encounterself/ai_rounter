# Haystack Control Plane

This directory contains a small Databricks App control plane for drafting and publishing Haystack runtime configuration. It is intentionally server-first: the browser talks to same-origin `/api` routes, while the Node server owns Databricks authentication and any future provider credentials.

## Current capabilities

- React/Vite UI for model services and routes, protocol settings, agent graph, MCP bindings, skills, prompts, releases, rollback, and audit events.
- Server-side health, Databricks resource discovery, model-service discovery, permissions identity summary, and serving-endpoint invocation proxy.
- Draft → publish → rollback state transitions through a store interface.
- A `ControlPlaneStore` provider contract with `getState`, `saveDraft`, `publish`, and `rollback` methods; memory and JSON-file implementations are included, while Postgres/Lakebase is an explicit degraded provider until a database adapter is injected.
- File-backed control-plane storage when `CONTROL_PLANE_STORE=file:/absolute/path.json` is configured.
- Explicit degraded mode when Databricks credentials or a production store provider are missing. Empty state is not a local fixture and is never presented as a connected resource.
- Secret rejection and response redaction. PATs, provider tokens, API keys, passwords, and authorization headers must not be placed in draft JSON.

## Not included yet

This implementation does **not** claim to provide production Claude, Codex, MCP, Skills, prompt execution, or AI Gateway routing. The UI stores references and policy JSON; it does not execute agent graphs or create Databricks resources. A production deployment still needs:

- An App resource binding that injects `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and a durable `CONTROL_PLANE_STORE` provider.
- A real Postgres/Lakebase repository adapter for the `test_ontos` project. A `postgres://` store URL currently reports a visible degraded state rather than silently mocking persistence.
- Serving endpoint configuration and an authenticated runtime that implements the desired protocol adapters, model routing, fallback, streaming, and provider-specific semantics.
- Runtime adapters for Claude, Codex-compatible APIs, MCP transports, skill loading, prompt resolution, permissions, and audit/trace emission.

## Data-source identifiers

The default read-only source identifier is `simo_prod.mysql_cdc_sync`. It is displayed as a Unity Catalog schema identifier; the server does not assume that the identifier is a catalog. The default Lakebase project identifier is `test_ontos`, displayed as a Lakebase PostgreSQL project. Actual connectivity is reported by server-side configuration and Databricks API responses.

## Environment

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABRICKS_HOST` | Databricks workspace URL, server-side only | unset; resource calls are degraded |
| `DATABRICKS_TOKEN` | App-injected Databricks credential, server-side only | unset; never sent to the browser |
| `MODEL_SERVICE_PARENT` | Model service name filter | `simo_prod.ai_models` |
| `MODEL_SERVICE_PREFIX` | Model service name prefix/filter | `route_` |
| `CONTROL_PLANE_STORE` | `file:/absolute/path.json` or future provider URL | unset; empty in-memory degraded mode |
| `UC_READ_ONLY_SOURCE` | Read-only Unity Catalog source identifier | `simo_prod.mysql_cdc_sync` |
| `LAKEBASE_PROJECT` | Lakebase project identifier | `test_ontos` |
| `PORT` | HTTP port | `8000` |

The browser never receives `DATABRICKS_TOKEN`. Provider secrets should be represented by a server-side secret reference, not copied into draft configuration.

## Local development

The project uses only the existing React/Vite dependencies and Node built-ins for the server.

```bash
npm install
npm run dev
```

`npm run dev` starts the API on `127.0.0.1:8787` and Vite on `0.0.0.0:8000`; Vite proxies `/api` to the API server. Without Databricks environment variables the UI remains usable for draft editing but clearly shows degraded discovery and persistence status.

## Build and App startup

```bash
npm run build
npm run start
```

`npm run start` builds before starting the production static server, so the Databricks App command in `app.yaml` follows the same path. The server serves `dist/` and the API from one process. The app listens on `PORT` (default `8000`).

## API surface

- `GET /api/health` — readiness and degraded-mode summary.
- `GET /api/config` — draft, published release, release list, and audit history.
- `PUT /api/config` — replace the draft after secret validation.
- `POST /api/config/publish` — publish a server-side snapshot.
- `POST /api/config/rollback` — restore a published revision.
- `GET /api/resources` — read-only Apps, serving endpoints, model services, Unity Catalog schema, and Lakebase project discovery.
- `GET /api/model-services` — filtered model-service discovery.
- `GET /api/permissions` — server identity summary; resource-specific permissions still require explicit checks.
- `POST /api/gateway/request` — server-side proxy to a named Databricks serving endpoint invocation.

## Tests

```bash
npm test
```

The Node tests cover secret non-disclosure, explicit degraded resource errors, and draft/publish/rollback transitions. They do not replace an end-to-end test against a deployed Databricks App and Serving Endpoint.
