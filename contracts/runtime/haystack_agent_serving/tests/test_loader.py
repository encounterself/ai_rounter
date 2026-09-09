import json
import tempfile
import unittest
from pathlib import Path

from contracts.runtime.haystack_agent_serving.loader import AgentServingSnapshotLoader, SnapshotValidationError


def valid_profile() -> dict:
    """Return a valid fixture for runtime snapshot tests."""

    return {
        "schema_version": "1.0",
        "profile": {"name": "default-runtime", "version": "1.0.0", "description": "local test"},
        "agent_graph": {
            "entry_node": "request",
            "terminal_nodes": ["response"],
            "nodes": [
                {"id": "request", "kind": "input"},
                {
                    "id": "planner",
                    "kind": "model",
                    "model_binding": "primary",
                    "mcp_binding": "knowledge",
                    "skill_bindings": ["retrieval"],
                    "prompt_bindings": ["runtime-system"],
                    "policy_route": "default",
                },
                {"id": "response", "kind": "output"},
            ],
            "edges": [{"source": "request", "target": "planner"}, {"source": "planner", "target": "response"}],
        },
        "policy_router": {
            "default_action": "allow",
            "routes": [
                {
                    "id": "default",
                    "priority": 10,
                    "when": {"attribute": "request.mode", "operator": "exists"},
                    "action": "allow",
                    "model_binding": "primary",
                    "mcp_binding": "knowledge",
                    "max_steps": 12,
                    "max_tokens": 6000,
                }
            ],
        },
        "model_bindings": [
            {
                "id": "primary",
                "ref": "model-services/simo_prod.ai_models.route_primary",
                "protocol": "responses",
                "streaming": True,
                "tools": True,
            }
        ],
        "mcp_bindings": [
            {"id": "knowledge", "ref": "knowledge-tools", "tools": ["search"], "transport": "streamable_http"}
        ],
        "skill_bindings": [
            {
                "id": "retrieval",
                "ref": "retrieval-guidance",
                "required": True,
                "store_ref": "approved-skills",
                "manifest_path": "retrieval/SKILL.md",
                "load_mode": "lazy_body",
            }
        ],
        "mcp_registry": {
            "registry_version": "1.0",
            "servers": [
                {
                    "id": "knowledge-tools",
                    "protocol": "streamable_http",
                    "source_kind": "databricks_mcp",
                    "source_ref": "approved-mcp",
                    "allowed_tools": ["search"],
                }
            ],
        },
        "skills_registry": {
            "registry_version": "1.0",
            "stores": [
                {"id": "approved-skills", "kind": "local_fixture", "ref": "approved-skill-store", "read_only": True}
            ],
            "skills": [
                {
                    "id": "retrieval-guidance",
                    "store_ref": "approved-skills",
                    "manifest_path": "retrieval/SKILL.md",
                    "version": "1.0.0",
                    "content_sha256": "0" * 64,
                    "load_mode": "lazy_body",
                }
            ],
        },
        "prompt_bindings": [
            {
                "id": "runtime-system",
                "role": "system",
                "ref": "runtime-system-prompt",
                "allowed_variables": ["request.mode"],
                "user_override": False,
                "max_length": 4000,
                "content_sha256": "1" * 64,
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


class SnapshotLoaderTests(unittest.TestCase):
    def test_builds_immutable_snapshot_and_digest(self) -> None:
        """Build a detached immutable snapshot with a content digest."""

        loader = AgentServingSnapshotLoader()
        result = loader.load_mapping(valid_profile())

        self.assertTrue(result.accepted)
        self.assertFalse(result.used_last_known_good)
        self.assertIsNotNone(result.snapshot)
        assert result.snapshot is not None
        self.assertEqual(result.snapshot.profile.name, "default-runtime")
        self.assertTrue(result.snapshot.content_sha256)
        self.assertIsInstance(result.snapshot.agent_graph.nodes, tuple)
        self.assertTrue(result.snapshot.mcp_registry.server("knowledge-tools").allows_tool("search"))
        self.assertFalse(result.snapshot.mcp_registry.server("knowledge-tools").allows_tool("write"))
        prompt = result.snapshot.prompt_bindings["runtime-system"]
        self.assertEqual(prompt.render("Mode={{request.mode}}", {"request.mode": "safe"}), "Mode=safe")
        self.assertFalse(result.snapshot.observability.trace_enabled)
        self.assertFalse(result.snapshot.observability.capture_content)
        with self.assertRaises(TypeError):
            result.snapshot.model_bindings["other"] = result.snapshot.model_bindings["primary"]  # type: ignore[index]

    def test_rejects_system_ai_model_and_bad_route(self) -> None:
        """Reject system-managed and non-route model references."""

        loader = AgentServingSnapshotLoader()
        profile = valid_profile()
        profile["model_bindings"][0]["ref"] = "system.ai.gpt"

        result = loader.load_mapping(profile)

        self.assertFalse(result.accepted)
        self.assertIsNone(result.snapshot)
        self.assertTrue(any("system.ai" in issue.message for issue in result.issues))

        profile["model_bindings"][0]["ref"] = "model-services/simo_prod.ai_models.model_primary"
        result = loader.load_mapping(profile)
        self.assertFalse(result.accepted)
        self.assertTrue(any("pattern" in issue.message or "valid under" in issue.message for issue in result.issues))

    def test_accepts_gateway_model_services(self) -> None:
        """Accept approved Model Services and direct Serving Endpoint references."""

        loader = AgentServingSnapshotLoader()
        for reference in (
            "model-services/simo_prod.ai_gateway.for_research_bedrock",
            "model-services/simo_prod.ai_gateway.route_glm",
            "serving-endpoint/haystack-router",
            "serving-endpoints/haystack-router-v2",
        ):
            profile = valid_profile()
            profile["model_bindings"][0]["ref"] = reference
            result = loader.load_mapping(profile)
            self.assertTrue(result.accepted, result.issues)

        profile = valid_profile()
        profile["model_bindings"][0]["ref"] = "primary-alias"
        result = loader.load_mapping(profile)
        self.assertTrue(result.accepted, result.issues)

    def test_rejects_unsafe_model_references(self) -> None:
        """Reject URLs, traversal, reserved namespaces, and unapproved resources."""

        loader = AgentServingSnapshotLoader()
        references = (
            "system.ai.gpt",
            "system.foundation-model",
            "system-alias",
            "agent_alias",
            "model-services/simo_prod.ai_models.route_",
            "model-services/simo_prod.ai_other.route_test",
            "model-services/other_prod.ai_models.route_test",
            "https://workspace.example/model-services/simo_prod.ai_models.route_test",
            "alias/name",
            r"alias\name",
            "alias/../name",
            "python:os.system",
            "import os",
            "/absolute/model-services/simo_prod.ai_models.route_test",
            "model-services/simo_prod.ai_gateway.evil/../../route_test",
            "model-services/simo_prod.ai_models.agent_test",
        )
        for reference in references:
            profile = valid_profile()
            profile["model_bindings"][0]["ref"] = reference
            result = loader.load_mapping(profile)
            self.assertFalse(result.accepted, reference)

    def test_rejects_python_import_and_supervisor_agent(self) -> None:
        """Reject Supervisor Agent and arbitrary import configuration."""

        loader = AgentServingSnapshotLoader()
        profile = valid_profile()
        profile["profile"]["description"] = "Supervisor Agent import os"

        result = loader.load_mapping(profile)

        self.assertFalse(result.accepted)
        messages = " ".join(issue.message for issue in result.issues)
        self.assertIn("Supervisor Agent", messages)
        self.assertIn("import", messages)

    def test_last_known_good_is_returned_for_invalid_reload(self) -> None:
        """Keep serving the previous snapshot after an invalid reload."""

        loader = AgentServingSnapshotLoader()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(valid_profile()), encoding="utf-8")
            first = loader.load_file(path)
            self.assertTrue(first.accepted)
            good_snapshot = first.snapshot

            invalid = valid_profile()
            invalid["agent_graph"]["entry_node"] = "missing"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            second = loader.load_file(path)

            self.assertFalse(second.accepted)
            self.assertTrue(second.used_last_known_good)
            self.assertIs(second.snapshot, good_snapshot)

    def test_initial_invalid_load_fails_closed(self) -> None:
        """Fail closed when the first profile cannot be loaded."""

        loader = AgentServingSnapshotLoader()
        with self.assertRaises(SnapshotValidationError):
            loader.require_file("/tmp/does-not-exist/haystack-profile.json")

    def test_rejects_unknown_graph_binding(self) -> None:
        """Reject graph references that do not resolve to bindings."""

        loader = AgentServingSnapshotLoader()
        profile = valid_profile()
        profile["agent_graph"]["nodes"][1]["model_binding"] = "missing"

        result = loader.load_mapping(profile)

        self.assertFalse(result.accepted)
        self.assertTrue(any("unknown reference" in issue.message for issue in result.issues))

    def test_round_trips_agent_tools_and_state_schema(self) -> None:
        """Serialize and reload delegated tools and their state contract."""

        profile = valid_profile()
        profile["agent_graph"]["nodes"].insert(
            2, {"id": "review", "kind": "model", "model_binding": "primary"}
        )
        profile["agent_graph"]["edges"] = [
            {"source": "request", "target": "planner"},
            {"source": "planner", "target": "response"},
        ]
        profile["state_schema"] = {"review_findings": {"type": "string", "persist": True}}
        profile["agent_tools"] = [
            {
                "id": "review-tool",
                "node": "review",
                "name": "review_code",
                "description": "Review code.",
                "outputs_to_state": {
                    "review_findings": {"source": "last_message", "handler": "message_text"}
                },
            }
        ]
        profile["agent_graph"]["nodes"][1]["agent_tools"] = ["review-tool"]

        loader = AgentServingSnapshotLoader()
        result = loader.load_mapping(profile)

        self.assertTrue(result.accepted, result.issues)
        assert result.snapshot is not None
        serialized = result.snapshot.to_dict()
        self.assertEqual(serialized["state_schema"]["review_findings"]["type"], "string")
        self.assertEqual(
            serialized["agent_tools"][0]["outputs_to_state"]["review_findings"]["handler"], "message_text"
        )
        reloaded = loader.load_mapping(serialized)
        self.assertTrue(reloaded.accepted, reloaded.issues)

    def test_rejects_agent_tool_state_unknown_reference(self) -> None:
        """Reject AgentTool mappings that read or write undeclared state."""

        profile = valid_profile()
        profile["agent_tools"] = [
            {
                "id": "review-tool",
                "node": "planner",
                "name": "review_code",
                "description": "Review code.",
                "inputs_from_state": {"missing_state": "review_context"},
            }
        ]
        profile["agent_graph"]["nodes"][1]["agent_tools"] = ["review-tool"]

        result = AgentServingSnapshotLoader().load_mapping(profile)

        self.assertFalse(result.accepted)
        self.assertTrue(any("missing_state" in issue.message for issue in result.issues))

    def test_rejects_agent_tool_delegation_cycle(self) -> None:
        """Reject recursive specialist delegation before runtime construction."""

        profile = valid_profile()
        profile["agent_graph"]["nodes"].insert(2, {"id": "specialist", "kind": "model", "model_binding": "primary"})
        profile["agent_graph"]["edges"] = [
            {"source": "request", "target": "planner"},
            {"source": "planner", "target": "response"},
        ]
        profile["agent_tools"] = [
            {"id": "planner-to-specialist", "node": "specialist", "name": "delegate_specialist", "description": "Delegate."},
            {"id": "specialist-to-planner", "node": "planner", "name": "delegate_planner", "description": "Delegate."},
        ]
        profile["agent_graph"]["nodes"][1]["agent_tools"] = ["planner-to-specialist"]
        profile["agent_graph"]["nodes"][2]["agent_tools"] = ["specialist-to-planner"]

        result = AgentServingSnapshotLoader().load_mapping(profile)

        self.assertFalse(result.accepted)
        self.assertTrue(any("cycle" in issue.message.lower() for issue in result.issues))

    def test_rejects_structurally_invalid_document_without_raising(self) -> None:
        """Return validation errors instead of raising on incomplete JSON."""

        loader = AgentServingSnapshotLoader()
        result = loader.load_mapping({"schema_version": "1.0"})

        self.assertFalse(result.accepted)
        self.assertIsNone(result.snapshot)
        self.assertTrue(result.issues)


if __name__ == "__main__":
    unittest.main()
