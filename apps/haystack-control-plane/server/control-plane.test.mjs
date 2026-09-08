import test from "node:test";
import assert from "node:assert/strict";
import { createApiHandler } from "./api.mjs";
import { DatabricksError } from "./databricks.mjs";
import { loadConfig } from "./config.mjs";
import { MemoryStore } from "./store.mjs";

function makeHandler(databricks = {}) {
  const config = loadConfig({
    DATABRICKS_HOST: "https://workspace.example",
    DATABRICKS_TOKEN: "server-token-must-not-leak",
    CONTROL_PLANE_STORE: "",
  });
  return createApiHandler({ config, store: new MemoryStore(), databricks });
}

test("health and rejected drafts never expose server secrets", async () => {
  const handler = makeHandler({});
  const health = await handler({ method: "GET", url: "/api/health", body: "" });
  assert.equal(health.status, 200);
  assert.equal(JSON.stringify(health.body).includes("server-token-must-not-leak"), false);

  const rejected = await handler({
    method: "PUT",
    url: "/api/config",
    body: JSON.stringify({ routes: [{ apiKey: "provider-secret" }] }),
  });
  assert.equal(rejected.status, 400);
  assert.equal(JSON.stringify(rejected.body).includes("provider-secret"), false);
});

test("resource failures are returned as explicit degraded API errors", async () => {
  const handler = makeHandler({
    discoverResources: async () => {
      throw new DatabricksError("DATABRICKS_API_ERROR", "workspace resource discovery failed", 503);
    },
  });
  const response = await handler({ method: "GET", url: "/api/resources", body: "" });
  assert.equal(response.status, 503);
  assert.deepEqual(response.body.error, {
    code: "DATABRICKS_API_ERROR",
    message: "workspace resource discovery failed",
    status: 503,
  });
});

test("draft, publish, and rollback use the server-side store", async () => {
  const handler = makeHandler({});
  const draft = {
    profileId: "demo",
    displayName: "Demo profile",
    revision: 0,
    status: "draft",
    routes: [{ id: "primary", endpoint: "model-endpoint" }],
    protocols: [{ id: "openai-responses", label: "OpenAI Responses", enabled: true }],
    agentGraph: { nodes: [{ id: "start", type: "model" }], edges: [] },
    mcpBindings: [],
    skills: [],
    prompts: [],
  };

  const saved = await handler({ method: "PUT", url: "/api/config", body: JSON.stringify(draft) });
  assert.equal(saved.status, 200);
  const published = await handler({ method: "POST", url: "/api/config/publish", body: "{}" });
  assert.equal(published.status, 200);
  assert.equal(published.body.releases.length, 1);
  assert.equal(published.body.published.revision, 1);

  const rolledBack = await handler({
    method: "POST",
    url: "/api/config/rollback",
    body: JSON.stringify({ revision: 1 }),
  });
  assert.equal(rolledBack.status, 200);
  assert.equal(rolledBack.body.draft.profileId, "demo");
  assert.equal(rolledBack.body.audit[0].action, "rollback");
});
