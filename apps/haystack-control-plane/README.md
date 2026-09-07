# Haystack Control Plane

Minimal read-only Databricks AppKit/React control-plane skeleton for local contract and model-service discovery.

## Scope

- Configuration views are backed by local fixtures only.
- Model discovery displays only `model-services/simo_prod.ai_models.route_*` references.
- Agent profiles expose the contract snapshot status and last-known-good state.
- MCP, Skills, and Prompts expose the logical registry bindings, allowlists, lazy-load state, and override policy.
- Releases, Runs, and Audit are intentionally read-only placeholders.
- Protocol adapters and observability policy are defined in the runtime contract; this shell does not emit MLflow traces.
- No Catalog, Lakebase, Volume, Serving endpoint, MCP connection, or Databricks resource is created or bound.

## Local development

```bash
npm install
npm run dev
```

The package versions follow the local AppKit template version reported by the installed Databricks CLI. The
dependency lockfile is intentionally not committed until the package registry is available.
