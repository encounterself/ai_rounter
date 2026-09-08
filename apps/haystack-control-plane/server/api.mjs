import { DatabricksError, safeDatabricksError } from "./databricks.mjs";
import { StoreError, redactSecrets } from "./store.mjs";

const JSON_LIMIT = 1024 * 1024;

function jsonResponse(status, body) {
  return {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
    body: redactSecrets(body),
  };
}

function errorResponse(error) {
  if (error instanceof StoreError || error instanceof DatabricksError) {
    return jsonResponse(error.status, {
      error: {
        code: error.code,
        message: error.message,
        degraded: error.status >= 500,
      },
    });
  }

  return jsonResponse(500, {
    error: {
      code: "CONTROL_PLANE_ERROR",
      message: "The control-plane request failed",
      degraded: true,
    },
  });
}

function parseBody(request) {
  if (!request.body) {
    return {};
  }

  if (request.body.length > JSON_LIMIT) {
    throw new StoreError("BODY_TOO_LARGE", "Request body exceeds 1 MiB", 413);
  }

  try {
    return JSON.parse(request.body);
  } catch {
    throw new StoreError("JSON_INVALID", "Request body must be valid JSON", 400);
  }
}

export function createApiHandler({ config, store, databricks }) {
  return async function handle(request) {
    const url = new URL(request.url, "http://control-plane.local");
    if (!url.pathname.startsWith("/api/")) {
      return jsonResponse(404, { error: { code: "NOT_FOUND", message: "API route not found" } });
    }

    try {
      if (request.method === "GET" && url.pathname === "/api/health") {
        const storeStatus = store.status;
        const databricksConfigured = Boolean(config.host && config.token);
        const degraded = storeStatus.degraded || !databricksConfigured;
        return jsonResponse(200, {
          status: degraded ? "degraded" : "ready",
          checks: {
            databricks: databricksConfigured ? "configured" : "not-configured",
            store: storeStatus.degraded ? "degraded" : "ready",
          },
          config: {
            ...configSummary(config),
            store: storeStatus,
          },
          message: degraded
            ? "The app is running, but one or more external resources are not configured or available."
            : "The control plane is ready.",
        });
      }

      if (request.method === "GET" && url.pathname === "/api/config") {
        return jsonResponse(200, await store.getState());
      }

      if (request.method === "PUT" && url.pathname === "/api/config") {
        return jsonResponse(200, await store.saveDraft(parseBody(request)));
      }

      if (request.method === "POST" && url.pathname === "/api/config/publish") {
        return jsonResponse(200, await store.publish());
      }

      if (request.method === "POST" && url.pathname === "/api/config/rollback") {
        const payload = parseBody(request);
        const revision = Number(payload.revision);
        if (!Number.isInteger(revision) || revision < 1) {
          throw new StoreError("REVISION_INVALID", "revision must be a positive integer", 400);
        }
        return jsonResponse(200, await store.rollback(revision));
      }

      if (request.method === "GET" && url.pathname === "/api/resources") {
        return jsonResponse(200, {
          source: configSummary(config),
          resources: await databricks.discoverResources(),
        });
      }

      if (request.method === "GET" && url.pathname === "/api/model-services") {
        return jsonResponse(200, {
          source: {
            parent: config.modelServiceParent,
            prefix: config.modelServicePrefix,
          },
          services: await databricks.listModelServices(),
        });
      }

      if (request.method === "GET" && url.pathname === "/api/permissions") {
        return jsonResponse(200, await databricks.permissionsSummary());
      }

      if (request.method === "POST" && url.pathname === "/api/gateway/request") {
        const payload = parseBody(request);
        return jsonResponse(200, {
          response: await databricks.gatewayRequest(payload),
        });
      }

      return jsonResponse(404, { error: { code: "NOT_FOUND", message: "API route not found" } });
    } catch (error) {
      if (error instanceof DatabricksError && error.status >= 500) {
        return jsonResponse(error.status, { error: safeDatabricksError(error) });
      }
      return errorResponse(error);
    }
  };
}

function configSummary(config) {
  return {
    databricksHost: config.host || null,
    databricksConfigured: Boolean(config.host && config.token),
    readOnlySource: {
      identifier: config.readOnlySource,
      type: "unity-catalog-schema",
      status: "configured-identifier",
    },
    lakebase: {
      identifier: config.lakebaseProject,
      type: "lakebase-postgres-project",
      status: "provider-dependent",
    },
  };
}
