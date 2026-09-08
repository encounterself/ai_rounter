import { redactSecrets } from "./store.mjs";

const MODEL_SERVICES_PATH = "/api/2.1/unity-catalog/model-services";

export class DatabricksError extends Error {
  constructor(code, message, status = 503, details = undefined) {
    super(message);
    this.name = "DatabricksError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

export class DatabricksClient {
  constructor(config, fetcher = globalThis.fetch) {
    this.config = config;
    this.fetcher = fetcher;
  }

  ensureConfigured() {
    if (!this.config.host || !this.config.token) {
      throw new DatabricksError(
        "DATABRICKS_CONFIGURATION_REQUIRED",
        "DATABRICKS_HOST and DATABRICKS_TOKEN must be configured on the server",
      );
    }
  }

  async request(path, options = {}) {
    this.ensureConfigured();
    const response = await this.fetcher(`${this.config.host}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
        Authorization: `Bearer ${this.config.token}`,
      },
    });
    const text = await response.text();
    let body = null;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        body = { message: text.slice(0, 500) };
      }
    }

    if (!response.ok) {
      const message = body?.message || body?.error_code || `Databricks request failed (${response.status})`;
      throw new DatabricksError("DATABRICKS_API_ERROR", message, response.status, redactSecrets(body));
    }

    return redactSecrets(body);
  }

  async listModelServices() {
    const body = await this.request(MODEL_SERVICES_PATH);
    const values = body?.model_services || body?.services || body?.items || (Array.isArray(body) ? body : []);
    const services = Array.isArray(values) ? values : [];
    const parent = this.config.modelServiceParent;
    const prefix = this.config.modelServicePrefix;

    return services.filter((service) => {
      const name = service.full_name || service.fullName || service.name || service.id || "";
      return (!parent || name.includes(parent)) && (!prefix || name.includes(prefix));
    });
  }

  async discoverResources() {
    const catalogSchema = this.config.readOnlySource;
    const paths = {
      apps: "/api/2.0/apps",
      servingEndpoints: "/api/2.0/serving-endpoints",
      modelServices: MODEL_SERVICES_PATH,
      unityCatalogSchema: `/api/2.1/unity-catalog/schemas/${encodeURIComponent(catalogSchema)}`,
      lakebaseProjects: "/api/2.0/postgres/projects",
    };

    const entries = await Promise.all(
      Object.entries(paths).map(async ([key, path]) => {
        try {
          return [key, { status: "ready", path, data: await this.request(path) }];
        } catch (error) {
          return [
            key,
            {
              status: "error",
              path,
              error: safeDatabricksError(error),
            },
          ];
        }
      }),
    );

    return Object.fromEntries(entries);
  }

  async permissionsSummary() {
    const currentUser = await this.request("/api/2.0/current-user/me");
    return {
      identity: {
        userName: currentUser?.userName || currentUser?.user_name || null,
        displayName: currentUser?.displayName || currentUser?.display_name || null,
      },
      note: "This summary confirms server identity only; resource-specific permissions require explicit resource IDs.",
    };
  }

  async gatewayRequest({ endpoint, body }) {
    if (!/^[A-Za-z0-9._-]+$/.test(endpoint || "")) {
      throw new DatabricksError("ENDPOINT_INVALID", "endpoint must be a simple serving endpoint name", 400);
    }

    if (!body || typeof body !== "object" || Array.isArray(body)) {
      throw new DatabricksError("REQUEST_INVALID", "gateway body must be a JSON object", 400);
    }

    return this.request(`/serving-endpoints/${encodeURIComponent(endpoint)}/invocations`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  }
}

export function safeDatabricksError(error) {
  if (error instanceof DatabricksError) {
    return {
      code: error.code,
      message: error.message,
      status: error.status,
    };
  }

  return { code: "DATABRICKS_REQUEST_FAILED", message: "Databricks request failed", status: 503 };
}
