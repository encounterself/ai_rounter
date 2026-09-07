import tempfile
import unittest
from pathlib import Path

from contracts.runtime.haystack_agent_serving.registries import (
    RegistryValidationError,
    build_registry_state,
    read_skill_file,
)


def registry_document() -> dict:
    """Return a minimal valid registry document."""

    return {
        "mcp_registry": {
            "registry_version": "1.0",
            "servers": [
                {
                    "id": "tools",
                    "protocol": "streamable_http",
                    "source_kind": "databricks_mcp",
                    "source_ref": "approved-mcp",
                    "allowed_tools": ["search"],
                }
            ],
        },
        "skills_registry": {
            "registry_version": "1.0",
            "stores": [{"id": "skills", "kind": "local_fixture", "ref": "approved-skills", "read_only": True}],
            "skills": [
                {
                    "id": "retrieval",
                    "store_ref": "skills",
                    "manifest_path": "retrieval/SKILL.md",
                    "version": "1.0.0",
                    "content_sha256": "0" * 64,
                    "load_mode": "lazy_body",
                }
            ],
        },
        "prompt_bindings": [
            {
                "id": "system",
                "role": "system",
                "ref": "system-prompt",
                "allowed_variables": ["mode"],
                "user_override": False,
                "max_length": 100,
            }
        ],
        "observability": {
            "mode": "databricks_ui",
            "trace_enabled": False,
            "sampling_ratio": 0.0,
            "capture_content": False,
            "redaction_policy": "strict",
            "metric_namespace": "haystack.agent",
            "emit_tool_events": True,
            "emit_model_events": True,
        },
    }


class RegistryTests(unittest.TestCase):
    def test_tool_allowlist_and_prompt_variables_are_fail_closed(self) -> None:
        """Allow only registered MCP tools and prompt variables."""

        mcp, _, prompts, _ = build_registry_state(registry_document())
        self.assertTrue(mcp.server("tools").allows_tool("search"))
        self.assertFalse(mcp.server("tools").allows_tool("delete"))
        with self.assertRaises(RegistryValidationError):
            prompts["system"].render("{{unknown}}", {})

    def test_skill_file_loader_rejects_path_traversal(self) -> None:
        """Read skill files only below the declared manifest directory."""

        _, skills, _, _ = build_registry_state(registry_document())
        skill = skills.skill("retrieval")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill_dir = root / "retrieval"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Retrieval", encoding="utf-8")
            self.assertEqual(read_skill_file(root, skill, "SKILL.md"), "# Retrieval")
            with self.assertRaises(RegistryValidationError):
                read_skill_file(root, skill, "../secret.txt")

    def test_remote_stdio_is_rejected(self) -> None:
        """Require HTTP-style transport for non-local MCP sources."""

        document = registry_document()
        document["mcp_registry"]["servers"][0]["protocol"] = "stdio"
        with self.assertRaises(RegistryValidationError):
            build_registry_state(document)


if __name__ == "__main__":
    unittest.main()
