const DEFAULT_MODEL_SERVICE_PARENT = "simo_prod.ai_models";
const DEFAULT_MODEL_SERVICE_PREFIX = "route_";
const DEFAULT_UC_SOURCE = "simo_prod.mysql_cdc_sync";
const DEFAULT_LAKEBASE_PROJECT = "test_ontos";
const DEFAULT_SERVING_ENDPOINT = "haystack-agent-router";

function normalizeHost(value) {
  if (!value) return "";
  return value.replace(/\/+$/, "");
}

function splitList(value) {
  return (value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function loadConfig(environment = process.env) {
  const store = environment.CONTROL_PLANE_STORE?.trim() ?? "";
  const servingEndpoint =
    environment.DATABRICKS_SERVING_ENDPOINT_NAME?.trim() ||
    environment.SERVING_ENDPOINT_NAME?.trim() ||
    DEFAULT_SERVING_ENDPOINT;

  return {
    host: normalizeHost(environment.DATABRICKS_HOST?.trim()),
    token: environment.DATABRICKS_TOKEN?.trim() ?? "",
    clientId: environment.DATABRICKS_CLIENT_ID?.trim() ?? "",
    clientSecret: environment.DATABRICKS_CLIENT_SECRET?.trim() ?? "",
    servingEndpoint,
    servingEndpointAllowlist: splitList(environment.SERVING_ENDPOINT_ALLOWLIST),
    modelServiceParent:
      environment.MODEL_SERVICE_PARENT?.trim() || DEFAULT_MODEL_SERVICE_PARENT,
    modelServicePrefix:
      environment.MODEL_SERVICE_PREFIX?.trim() || DEFAULT_MODEL_SERVICE_PREFIX,
    readOnlySource:
      environment.UC_READ_ONLY_SOURCE?.trim() || DEFAULT_UC_SOURCE,
    lakebaseProject:
      environment.LAKEBASE_PROJECT?.trim() || DEFAULT_LAKEBASE_PROJECT,
    store,
    port: Number.parseInt(environment.DATABRICKS_APP_PORT || environment.PORT || "8000", 10) || 8000,
  };
}

export function publicConfig(config) {
  return {
    databricks: {
      configured: Boolean(config.host && (config.token || (config.clientId && config.clientSecret))),
      host: config.host || null,
      tokenPresent: Boolean(config.token),
      oauthConfigured: Boolean(config.clientId && config.clientSecret),
    },
    serving: {
      boundEndpoint: config.servingEndpoint || null,
      allowlist: config.servingEndpointAllowlist,
      protocol: "agent/v1/responses",
    },
    modelServices: {
      parent: config.modelServiceParent,
      prefix: config.modelServicePrefix,
    },
    dataSources: {
      unityCatalog: {
        identifier: config.readOnlySource,
        type: "unity-catalog-schema",
        access: "read-only-intent",
      },
      lakebase: {
        identifier: config.lakebaseProject,
        type: "lakebase-postgres-project",
        access: "provider-dependent",
      },
    },
    store: {
      configured: Boolean(config.store),
      kind: config.store ? storeKind(config.store) : "memory-degraded",
    },
  };
}

function storeKind(value) {
  if (value.startsWith("postgres://") || value.startsWith("postgresql://")) return "postgres-provider";
  if (value.startsWith("file:")) return "file";
  return "file";
}

export const defaults = {
  modelServiceParent: DEFAULT_MODEL_SERVICE_PARENT,
  modelServicePrefix: DEFAULT_MODEL_SERVICE_PREFIX,
  readOnlySource: DEFAULT_UC_SOURCE,
  lakebaseProject: DEFAULT_LAKEBASE_PROJECT,
  servingEndpoint: DEFAULT_SERVING_ENDPOINT,
};
