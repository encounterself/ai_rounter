import { redactSecrets } from "./store.mjs";

const MODEL_SERVICES_PATH = "/api/2.1/unity-catalog/model-services";
const SERVING_ENDPOINTS_PATH = "/api/2.0/serving-endpoints";
const SUPPORTED_PROTOCOLS = new Set([
  "agent/v1/responses",
  "openai/v1/responses",
  "mlflow/v1/responses",
  "openai/v1/chat/completions",
  "mlflow/v1/chat/completions",
  "anthropic/v1/messages",
  "cursor/v1/chat/completions",
]);

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
    this.oauthToken = null;
  }

  ensureConfigured(requestHeaders = {}) {
    const forwardedToken = requestHeaders["x-forwarded-access-token"] || requestHeaders["X-Forwarded-Access-Token"];
    if (!this.config.host || !(this.config.token || forwardedToken || (this.config.clientId && this.config.clientSecret))) {
      throw new DatabricksError(
        "DATABRICKS_CONFIGURATION_REQUIRED",
        "Databricks App authentication is not configured on the server",
      );
    }
  }

  async accessToken(requestHeaders = {}) {
    const forwardedToken = requestHeaders["x-forwarded-access-token"] || requestHeaders["X-Forwarded-Access-Token"];
    if (forwardedToken) return forwardedToken;
    if (this.config.token) return this.config.token;
    if (!this.oauthToken || this.oauthToken.expiresAt <= Date.now()) {
      const credentials = Buffer.from(`${this.config.clientId}:${this.config.clientSecret}`).toString("base64");
      const response = await this.fetcher(`${this.config.host}/oidc/v1/token`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          Authorization: `Basic ${credentials}`,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: "grant_type=client_credentials&scope=all-apis",
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok || !body.access_token) {
        throw new DatabricksError("DATABRICKS_AUTH_FAILED", "Unable to obtain the App OAuth token", response.status || 503);
      }
      this.oauthToken = {
        value: body.access_token,
        expiresAt: Date.now() + Math.max(30, Number(body.expires_in || 3600) - 30) * 1000,
      };
    }
    return this.oauthToken.value;
  }

  async request(path, options = {}, requestHeaders = {}) {
    this.ensureConfigured(requestHeaders);
    const response = await this.fetcher(`${this.config.host}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
        Authorization: `Bearer ${await this.accessToken(requestHeaders)}`,
      },
    });
    const text = await response.text();
    let body = null;
    if (text) {
      try { body = JSON.parse(text); } catch { body = { message: text.slice(0, 500) }; }
    }
    if (!response.ok) {
      const message = body?.message || body?.error_code || `Databricks request failed (${response.status})`;
      throw new DatabricksError("DATABRICKS_API_ERROR", message, response.status, redactSecrets(body));
    }
    return redactSecrets(body);
  }

  async listModelServices(requestHeaders = {}) {
    const body = await this.request(MODEL_SERVICES_PATH, {}, requestHeaders);
    const values = body?.model_services || body?.services || body?.items || (Array.isArray(body) ? body : []);
    const parent = this.config.modelServiceParent;
    const prefix = this.config.modelServicePrefix;
    return (Array.isArray(values) ? values : []).filter((service) => {
      const name = service.full_name || service.fullName || service.name || service.id || "";
      return (!parent || name.includes(parent)) && (!prefix || name.includes(prefix));
    });
  }

  async listServingEndpoints(requestHeaders = {}) {
    const body = await this.request(SERVING_ENDPOINTS_PATH, {}, requestHeaders);
    const values = body?.endpoints || body?.serving_endpoints || body?.items || (Array.isArray(body) ? body : []);
    return Array.isArray(values) ? values : [];
  }

  async discoverResources(requestHeaders = {}) {
    const catalogSchema = this.config.readOnlySource;
    const paths = {
      apps: "/api/2.0/apps",
      servingEndpoints: SERVING_ENDPOINTS_PATH,
      modelServices: MODEL_SERVICES_PATH,
      unityCatalogSchema: `/api/2.1/unity-catalog/schemas/${encodeURIComponent(catalogSchema)}`,
      lakebaseProjects: "/api/2.0/postgres/projects",
    };
    const entries = await Promise.all(Object.entries(paths).map(async ([key, path]) => {
      try { return [key, { status: "ready", path, data: await this.request(path, {}, requestHeaders) }]; }
      catch (error) { return [key, { status: "error", path, error: safeDatabricksError(error) }]; }
    }));
    return Object.fromEntries(entries);
  }

  async permissionsSummary(requestHeaders = {}) {
    const currentUser = await this.request("/api/2.0/current-user/me", {}, requestHeaders);
    return {
      identity: {
        userName: currentUser?.userName || currentUser?.user_name || null,
        displayName: currentUser?.displayName || currentUser?.display_name || null,
      },
      note: "The App calls Databricks server-side; credentials are never sent to the browser.",
    };
  }

  async gatewayRequest({ endpoint, protocol = "agent/v1/responses", route, body, ...payload }, requestHeaders = {}) {
    const selectedEndpoint = endpoint || this.config.servingEndpoint;
    if (!/^[A-Za-z0-9._-]+$/.test(selectedEndpoint || "")) {
      throw new DatabricksError("ENDPOINT_INVALID", "endpoint must be a simple serving endpoint name", 400);
    }
    if (this.config.servingEndpointAllowlist.length && !this.config.servingEndpointAllowlist.includes(selectedEndpoint)) {
      throw new DatabricksError("ENDPOINT_NOT_ALLOWED", "endpoint is not in the configured allowlist", 403);
    }
    if (!SUPPORTED_PROTOCOLS.has(protocol)) {
      throw new DatabricksError("PROTOCOL_UNSUPPORTED", `Unsupported protocol: ${protocol}`, 400);
    }
    const sourceBody = body && typeof body === "object" && !Array.isArray(body) ? { ...body } : { ...payload };
    const requestBody = toResponsesRequest(sourceBody, protocol, route);
    if (!requestBody.input?.length) {
      throw new DatabricksError("REQUEST_INVALID", "gateway request requires input, messages, prompt, or inputs", 400);
    }
    return this.request(`/serving-endpoints/${encodeURIComponent(selectedEndpoint)}/invocations`, {
      method: "POST",
      body: JSON.stringify(requestBody),
    }, requestHeaders);
  }
}

function toResponsesRequest(body, protocol, route) {
  const input = body.input || body.messages || body.inputs || (body.prompt ? [{ role: "user", content: body.prompt }] : []);
  const normalizedInput = Array.isArray(input) ? input : [{ role: "user", content: String(input) }];
  const request = { ...body, input: normalizedInput };
  delete request.messages;
  delete request.inputs;
  delete request.prompt;
  if (route) request.model = route;
  delete request.route;
  if (protocol === "anthropic/v1/messages") {
    request.input = normalizedInput.map((item) => ({
      role: item.role || "user",
      content: Array.isArray(item.content)
        ? item.content.map((part) => part.text || part.content || "").join("")
        : item.content || "",
    }));
  }
  return request;
}

export function safeDatabricksError(error) {
  if (error instanceof DatabricksError) return { code: error.code, message: error.message, status: error.status };
  return { code: "DATABRICKS_REQUEST_FAILED", message: "Databricks control-plane request failed", degraded: true };
}
