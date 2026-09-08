from __future__ import annotations

import runpy
import sys

sys.path.insert(0, "/dbfs/tmp/haystack-agent-router-src")
runpy.run_path("/dbfs/tmp/haystack-agent-router-src/publish_current_model.py", run_name="__main__")
