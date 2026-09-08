import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { randomUUID } from "node:crypto";

const SECRET_KEY = /^(token|accessToken|apiKey|api_key|secret|password|authorization|clientSecret|privateKey)$/i;

export function createEmptyConfig() {
  return {
    profileId: "default",
    displayName: "Default control-plane profile",
    revision: 0,
    status: "draft",
    routes: [],
    protocols: [
      { id: "openai-responses", label: "OpenAI Responses", enabled: true },
      { id: "anthropic-messages", label: "Anthropic Messages", enabled: true },
      { id: "mlflow-chat", label: "MLflow Chat", enabled: true },
      { id: "cursor-chat", label: "Cursor Chat", enabled: true },
    ],
    agentGraph: { nodes: [], edges: [] },
    mcpBindings: [],
    skills: [],
    prompts: [],
    updatedAt: null,
  };
}

function createInitialState() {
  return {
    draft: createEmptyConfig(),
    published: null,
    releases: [],
    audit: [],
  };
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function now() {
  return new Date().toISOString();
}

function validateConfig(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new StoreError("CONFIG_INVALID", "Configuration must be a JSON object", 400);
  }

  const next = createEmptyConfig();
  for (const key of Object.keys(next)) {
    if (key in value) {
      next[key] = value[key];
    }
  }

  if (!Array.isArray(next.routes) || !Array.isArray(next.protocols)) {
    throw new StoreError(
      "CONFIG_INVALID",
      "routes and protocols must be arrays",
      400,
    );
  }

  if (!next.agentGraph || typeof next.agentGraph !== "object") {
    throw new StoreError("CONFIG_INVALID", "agentGraph must be an object", 400);
  }

  next.updatedAt = now();
  return next;
}

export function redactSecrets(value) {
  if (Array.isArray(value)) {
    return value.map(redactSecrets);
  }

  if (!value || typeof value !== "object") {
    return value;
  }

  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => !SECRET_KEY.test(key))
      .map(([key, entry]) => [key, redactSecrets(entry)]),
  );
}

function rejectSecrets(value, path = "config") {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => rejectSecrets(entry, `${path}[${index}]`));
    return;
  }

  if (!value || typeof value !== "object") {
    return;
  }

  for (const [key, entry] of Object.entries(value)) {
    if (SECRET_KEY.test(key)) {
      throw new StoreError(
        "SECRET_NOT_ALLOWED",
        `${path}.${key} cannot be stored in the control-plane configuration; use a secret reference`,
        400,
      );
    }
    rejectSecrets(entry, `${path}.${key}`);
  }
}

export class StoreError extends Error {
  constructor(code, message, status = 500) {
    super(message);
    this.name = "StoreError";
    this.code = code;
    this.status = status;
  }
}

export class ControlPlaneStore {
  constructor({ kind, configured, degraded, message }) {
    this.status = { kind, configured, degraded, message };
    this.state = createInitialState();
  }

  async getState() {
    return clone(this.state);
  }

  async saveDraft(value) {
    rejectSecrets(value);
    this.state.draft = validateConfig(value);
    await this.persist();
    return this.getState();
  }

  async publish() {
    const draft = validateConfig(this.state.draft);
    const release = {
      id: randomUUID(),
      revision: this.state.releases.length + 1,
      createdAt: now(),
      config: clone(draft),
    };
    this.state.releases.push(release);
    this.state.published = release;
    this.state.draft = { ...draft, revision: release.revision, status: "published" };
    this.state.audit.unshift({
      id: randomUUID(),
      action: "publish",
      revision: release.revision,
      at: release.createdAt,
    });
    await this.persist();
    return this.getState();
  }

  async rollback(revision) {
    const target = this.state.releases.find((release) => release.revision === revision);
    if (!target) {
      throw new StoreError(
        "RELEASE_NOT_FOUND",
        `No published release exists for revision ${revision}`,
        404,
      );
    }

    this.state.published = target;
    this.state.draft = {
      ...clone(target.config),
      revision: target.revision,
      status: "published",
      updatedAt: now(),
    };
    this.state.audit.unshift({
      id: randomUUID(),
      action: "rollback",
      revision: target.revision,
      at: now(),
    });
    await this.persist();
    return this.getState();
  }

  async persist() {}
}

export class MemoryStore extends ControlPlaneStore {
  constructor(message = "CONTROL_PLANE_STORE is not configured; using empty in-memory degraded mode") {
    super({
      kind: "memory-degraded",
      configured: false,
      degraded: true,
      message,
    });
  }
}

export class FileStore extends ControlPlaneStore {
  constructor(filePath) {
    super({
      kind: "file",
      configured: true,
      degraded: false,
      message: `Persisting control-plane state to ${filePath}`,
    });
    this.filePath = filePath;
    this.loaded = false;
  }

  async ensureLoaded() {
    if (this.loaded) {
      return;
    }

    try {
      const content = await readFile(this.filePath, "utf8");
      const parsed = JSON.parse(content);
      if (parsed?.draft && Array.isArray(parsed.releases) && Array.isArray(parsed.audit)) {
        this.state = parsed;
      }
    } catch (error) {
      if (error.code !== "ENOENT") {
        throw new StoreError("STORE_READ_FAILED", "Unable to read control-plane store", 500);
      }
    }
    this.loaded = true;
  }

  async getState() {
    await this.ensureLoaded();
    return super.getState();
  }

  async saveDraft(value) {
    await this.ensureLoaded();
    return super.saveDraft(value);
  }

  async publish() {
    await this.ensureLoaded();
    return super.publish();
  }

  async rollback(revision) {
    await this.ensureLoaded();
    return super.rollback(revision);
  }

  async persist() {
    await mkdir(dirname(this.filePath), { recursive: true });
    const temporaryPath = `${this.filePath}.tmp`;
    await writeFile(temporaryPath, JSON.stringify(this.state, null, 2), "utf8");
    await rename(temporaryPath, this.filePath);
  }
}

export class UnsupportedPostgresStore extends MemoryStore {
  constructor() {
    super(
      "A Postgres/Lakebase store URL was provided, but the Postgres provider is not bundled in this lightweight App; configure a provider adapter before production use",
    );
    this.status = {
      kind: "postgres-provider-unavailable",
      configured: true,
      degraded: true,
      message: this.status.message,
    };
  }
}

export function createStore(config) {
  if (!config.store) {
    return new MemoryStore();
  }

  if (config.store.startsWith("postgres://") || config.store.startsWith("postgresql://")) {
    return new UnsupportedPostgresStore();
  }

  const filePath = config.store.startsWith("file:")
    ? config.store.slice("file:".length)
    : config.store;
  return new FileStore(resolve(filePath));
}
