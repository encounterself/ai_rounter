import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createApiHandler } from "./api.mjs";
import { loadConfig } from "./config.mjs";
import { DatabricksClient } from "./databricks.mjs";
import { createStore } from "./store.mjs";

const appRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".tsx": "text/plain; charset=utf-8",
};

export function createControlPlaneServer(environment = process.env) {
  const config = loadConfig(environment);
  const store = createStore(config);
  const databricks = new DatabricksClient(config);
  const api = createApiHandler({ config, store, databricks });

  return createServer(async (request, response) => {
    if (request.method === "OPTIONS") {
      response.writeHead(204, {
        "access-control-allow-headers": "content-type",
        "access-control-allow-methods": "GET,PUT,POST,OPTIONS",
      });
      response.end();
      return;
    }

    if (request.url?.startsWith("/api/")) {
      const body = await readRequestBody(request);
      const result = await api({ method: request.method || "GET", url: request.url, body, headers: request.headers });
      response.writeHead(result.status, {
        ...result.headers,
        "cache-control": "no-store",
      });
      response.end(JSON.stringify(result.body));
      return;
    }

    await serveStatic(request, response);
  });
}

async function readRequestBody(request) {
  const chunks = [];
  let length = 0;
  for await (const chunk of request) {
    length += chunk.length;
    if (length > 1024 * 1024) {
      request.destroy();
      return "";
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function serveStatic(request, response) {
  const requestPath = decodeURIComponent(new URL(request.url || "/", "http://control-plane.local").pathname);
  const relativePath = requestPath === "/" ? "index.html" : requestPath.slice(1);
  const distRoot = resolve(appRoot, "dist");
  const candidate = resolve(normalize(join(distRoot, relativePath)));

  if (!candidate.startsWith(`${distRoot}/`)) {
    response.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
    response.end("Forbidden");
    return;
  }

  try {
    const body = await readFile(candidate);
    response.writeHead(200, {
      "content-type": contentTypes[extname(candidate)] || "application/octet-stream",
      "cache-control": candidate.endsWith("index.html") ? "no-store" : "public, max-age=300",
    });
    response.end(body);
  } catch (error) {
    if (error.code === "ENOENT" && requestPath !== "/") {
      const index = await readFile(join(distRoot, "index.html")).catch(() => null);
      if (index) {
        response.writeHead(200, { "content-type": contentTypes[".html"], "cache-control": "no-store" });
        response.end(index);
        return;
      }
    }
    response.writeHead(503, { "content-type": "text/plain; charset=utf-8" });
    response.end("Built frontend is unavailable. Run npm run build first.");
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const config = loadConfig();
  const server = createControlPlaneServer();
  server.listen(config.port, "0.0.0.0", () => {
    process.stdout.write(`Haystack Control Plane listening on port ${config.port}\n`);
  });
}
