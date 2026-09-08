from __future__ import annotations

import contextlib
import io
import json
import runpy
import sys
import tempfile
from pathlib import Path

source_root = "dbfs:/tmp/haystack-agent-router-src"
local_root = Path(tempfile.mkdtemp(prefix="haystack-agent-router-src-"))
for source in dbutils.fs.ls(source_root):
    if source.path.endswith("/") or source.name.startswith("__"):
        continue
    target = local_root / Path(source.path).name
    target.write_text(dbutils.fs.head(source.path, 10_000_000), encoding="utf-8")

sys.path.insert(0, str(local_root))
sys.argv = [
    str(local_root / "publish_current_model.py"),
    "--deploy-dir", str(local_root),
    "--profile", str(local_root / "profile.example.json"),
    "--model-name", "simo_prod.mysql_cdc_sync.haystack_agent_router",
    "--endpoint-name", "haystack-agent-router",
    "--workspace-id", "3761760657895016",
    "--workspace-host", "https://dbc-b21ff395-726b.cloud.databricks.com",
    "--owner", "dean.he@simo.co",
]
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    runpy.run_path(str(local_root / "publish_current_model.py"), run_name="__main__")
output = captured.getvalue()
print(output, end="")
result = None
for line in reversed(output.splitlines()):
    try:
        candidate = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict) and "version" in candidate:
        result = candidate
        break
if result is None:
    raise RuntimeError("publisher did not return a registered model version")
dbutils.notebook.exit(json.dumps(result))
