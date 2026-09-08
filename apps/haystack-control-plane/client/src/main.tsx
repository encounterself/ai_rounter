import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Endpoint = { name?: string; endpoint_name?: string };
type DraftConfig = {
  agentGraph: string;
  mcpBindings: string;
  skills: string;
  prompts: string;
};
type ParsedDraft = {
  agentGraph: unknown;
  mcpBindings: unknown;
  skills: unknown;
  prompts: unknown;
};
const protocols = [
  ["agent/v1/responses", "Haystack Agent Responses"],
  ["openai/v1/responses", "OpenAI Responses extension"],
  ["openai/v1/chat/completions", "OpenAI Chat extension"],
  ["anthropic/v1/messages", "Anthropic Messages extension"],
];

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...options, headers: { "content-type": "application/json", ...(options?.headers || {}) } });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error?.message || `Request failed (${response.status})`);
  return body;
}

function endpointName(endpoint: Endpoint) { return endpoint.name || endpoint.endpoint_name || "unknown"; }

function extractText(value: unknown): string {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object") return JSON.stringify(value, null, 2);
  const record = value as Record<string, unknown>;
  if (typeof record.output_text === "string") return record.output_text;
  if (typeof record.response === "string") return record.response;
  if (Array.isArray(record.output)) return record.output.map(extractText).filter(Boolean).join("\n") || JSON.stringify(value, null, 2);
  return JSON.stringify(value, null, 2);
}

function formatConfig(value: unknown): string {
  return JSON.stringify(value ?? [], null, 2);
}

function parseConfig(value: string, label: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    throw new Error(`${label} must be valid JSON`);
  }
}

function App() {
  const [boundEndpoint, setBoundEndpoint] = useState("haystack-agent-router");
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [endpoint, setEndpoint] = useState("");
  const [protocol, setProtocol] = useState("agent/v1/responses");
  const [route, setRoute] = useState("model-services/simo_prod.ai_models.route_claude-sonnet-5");
  const [prompt, setPrompt] = useState("Hello from the Haystack control plane");
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState("loading");
  const [draft, setDraft] = useState<DraftConfig>({
    agentGraph: formatConfig({ nodes: [], edges: [] }),
    mcpBindings: formatConfig([]),
    skills: formatConfig([]),
    prompts: formatConfig([]),
  });
  const [draftStatus, setDraftStatus] = useState("");

  useEffect(() => {
    Promise.all([
      api<{ boundEndpoint: string; endpoints: Endpoint[] }>("/api/serving-endpoints"),
      api<{ status: string }>("/api/health"),
      api<{ draft: DraftConfig }>("/api/config"),
    ]).then(([serving, status, config]) => {
      setBoundEndpoint(serving.boundEndpoint);
      setEndpoints(serving.endpoints || []);
      setEndpoint(serving.boundEndpoint);
      setHealth(status.status);
      setDraft({
        agentGraph: formatConfig(config.draft.agentGraph),
        mcpBindings: formatConfig(config.draft.mcpBindings),
        skills: formatConfig(config.draft.skills),
        prompts: formatConfig(config.draft.prompts),
      });
    }).catch((requestError: Error) => { setHealth("degraded"); setError(requestError.message); });
  }, []);

  const endpointOptions = useMemo(() => Array.from(new Set([boundEndpoint, ...endpoints.map(endpointName).filter((name) => name !== "unknown")])), [boundEndpoint, endpoints]);

  function updateDraft(key: keyof DraftConfig, value: string) {
    setDraft((current) => ({ ...current, [key]: value }));
    setDraftStatus("");
  }

  function parsedDraft(): ParsedDraft {
    return {
      agentGraph: parseConfig(String(draft.agentGraph), "Agent graph"),
      mcpBindings: parseConfig(String(draft.mcpBindings), "MCP bindings"),
      skills: parseConfig(String(draft.skills), "Skills"),
      prompts: parseConfig(String(draft.prompts), "Prompts"),
    };
  }

  async function saveDraft() {
    setBusy(true); setError(""); setDraftStatus("");
    try {
      const parsed = parsedDraft();
      await api("/api/config", { method: "PUT", body: JSON.stringify({ ...parsed, status: "draft" }) });
      setDraft({
        agentGraph: formatConfig(parsed.agentGraph),
        mcpBindings: formatConfig(parsed.mcpBindings),
        skills: formatConfig(parsed.skills),
        prompts: formatConfig(parsed.prompts),
      });
      setDraftStatus("Draft saved");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to save draft");
    } finally { setBusy(false); }
  }

  async function invoke() {
    setBusy(true); setError(""); setResult("");
    try {
      const metadata = parsedDraft();
      const response = await api<{ response: unknown }>("/api/gateway/request", {
        method: "POST",
        body: JSON.stringify({ endpoint: endpoint || boundEndpoint, protocol, route, input: [{ role: "user", content: prompt }], stream: false, metadata: { controlPlane: metadata } }),
      });
      setResult(extractText(response.response));
    } catch (requestError) { setError(requestError instanceof Error ? requestError.message : "Invocation failed"); }
    finally { setBusy(false); }
  }

  return <main className="shell">
    <header className="hero"><div><p className="eyebrow">HAYSTACK × DATABRICKS</p><h1>Control Plane</h1><p className="subtitle">Configure routes, test the bound serving endpoint, and keep provider credentials server-side.</p></div><span className={`status ${health}`}>{health}</span></header>
    <section className="grid">
      <div className="card"><h2>Serving route</h2>
        <label>Serving endpoint<select value={endpoint} onChange={(event) => setEndpoint(event.target.value)}>{endpointOptions.map((name) => <option key={name} value={name}>{name}{name === boundEndpoint ? " (bound)" : ""}</option>)}</select></label>
        <label>Protocol<select value={protocol} onChange={(event) => setProtocol(event.target.value)}>{protocols.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label>AI Gateway / model route<input value={route} onChange={(event) => setRoute(event.target.value)} placeholder="model-services/simo_prod.ai_models.route_claude-sonnet-5" /></label>
        <p className="hint">The default request is <code>agent/v1/responses</code>. Other protocol values remain extension points handled by the same server-side gateway.</p>
      </div>
      <div className="card"><h2>Test invocation</h2><label>Input<textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={7} /></label><button onClick={invoke} disabled={busy || !prompt.trim()}>{busy ? "Calling endpoint…" : "Call serving endpoint"}</button>{error && <pre className="error">{error}</pre>}{result && <pre className="result">{result}</pre>}</div>
    </section>
    <section className="card draft-config"><div className="section-heading"><div><h2>Orchestration draft</h2><p className="hint">Edit the JSON sent as request metadata. Saving persists the draft through the App API; invocation uses the current editor state.</p></div><button onClick={saveDraft} disabled={busy}>{busy ? "Saving…" : "Save draft"}</button></div>
      <div className="config-grid">
        <label>Agent graph<textarea value={draft.agentGraph} onChange={(event) => updateDraft("agentGraph", event.target.value)} rows={9} spellCheck={false} /></label>
        <label>MCP bindings<textarea value={draft.mcpBindings} onChange={(event) => updateDraft("mcpBindings", event.target.value)} rows={9} spellCheck={false} /></label>
        <label>Skills<textarea value={draft.skills} onChange={(event) => updateDraft("skills", event.target.value)} rows={7} spellCheck={false} /></label>
        <label>Prompts<textarea value={draft.prompts} onChange={(event) => updateDraft("prompts", event.target.value)} rows={7} spellCheck={false} /></label>
      </div>
      {draftStatus && <p className="saved-status">{draftStatus}</p>}
    </section>
    <section className="card architecture"><h2>Runtime boundary</h2><div className="pill-row"><span>App resource binding</span><span>agent/v1/responses</span><span>AI Gateway route</span><span>MCP / Skills / Prompts ready</span></div><p>Browser requests stay same-origin. The Node server resolves the bound endpoint and authenticates to Databricks; no PAT or provider token is sent to the browser.</p></section>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App />);
