import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createControlPlaneServer } from "./index.mjs";

const appRoot = fileURLToPath(new URL("..", import.meta.url));
const apiServer = createControlPlaneServer({ ...process.env, PORT: "8787" });
apiServer.listen(8787, "127.0.0.1", () => {
  process.stdout.write("Control-plane API listening on port 8787\n");
});

const viteBinary = fileURLToPath(new URL("../node_modules/vite/bin/vite.js", import.meta.url));
const vite = spawn(process.execPath, [viteBinary, "--host", "0.0.0.0", "--port", "8000"], {
  cwd: appRoot,
  env: process.env,
  stdio: "inherit",
});

function shutdown() {
  apiServer.close();
  vite.kill("SIGTERM");
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
vite.on("exit", (code) => {
  apiServer.close();
  process.exit(code ?? 0);
});
