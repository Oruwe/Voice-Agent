import { useState } from "react";
import {
  API_URL,
  ApiRequestError,
  createAgentKey,
  queryKnowledge,
  type KnowledgeQueryResponse,
} from "../lib/api";

interface AgentApiPanelProps {
  accessToken: string | null;
}

function errorText(err: unknown, fallback: string): string {
  return err instanceof ApiRequestError ? err.detail : fallback;
}

function CopyBlock({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard can be blocked (permissions, non-secure context); the text stays selectable.
    }
  }

  return (
    <div className="agent-api__block">
      <div className="agent-api__block-head">
        <span className="agent-api__label">{label}</span>
        <button type="button" className="agent-api__copy" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="agent-api__code">{value}</pre>
    </div>
  );
}

export function AgentApiPanel({ accessToken }: AgentApiPanelProps) {
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<KnowledgeQueryResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [agentKey, setAgentKey] = useState<string | null>(null);
  const [keyDays, setKeyDays] = useState<number | null>(null);
  const [keyError, setKeyError] = useState<string | null>(null);
  const [creatingKey, setCreatingKey] = useState(false);

  const mcpUrl = `${API_URL}/mcp`;
  const keyText = agentKey ?? "<AGENT_KEY>";

  async function search(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!accessToken || !query.trim() || searching) return;
    setSearching(true);
    setSearchError(null);
    try {
      setResult(await queryKnowledge(accessToken, query.trim()));
    } catch (err) {
      setResult(null);
      setSearchError(errorText(err, "Search failed."));
    } finally {
      setSearching(false);
    }
  }

  async function makeKey() {
    if (!accessToken || creatingKey) return;
    setCreatingKey(true);
    setKeyError(null);
    try {
      const res = await createAgentKey(accessToken);
      setAgentKey(res.agent_key);
      setKeyDays(Math.round(res.expires_in / 86400));
    } catch (err) {
      setKeyError(errorText(err, "Could not create a key."));
    } finally {
      setCreatingKey(false);
    }
  }

  if (!accessToken) {
    return <p className="panel-empty">Sign in to query the knowledge base or connect an agent.</p>;
  }

  return (
    <div className="panel-scroll agent-api">
      <section className="agent-api__section">
        <h3 className="agent-api__title">Try the knowledge tool</h3>
        <p className="agent-api__hint">The same search other agents call, served by Moss.</p>
        <form className="chat-input agent-api__search" onSubmit={search}>
          <input
            className="field__input chat-input__field"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. how do I reset the inverter?"
            aria-label="Search the knowledge base"
            maxLength={500}
          />
          <button className="btn btn--primary" type="submit" disabled={searching || !query.trim()}>
            {searching ? "…" : "Search"}
          </button>
        </form>

        {searchError && <p className="doc-upload__feedback doc-upload__feedback--err">{searchError}</p>}

        {result && !result.index_ready && (
          <p className="agent-api__hint">No documents yet — upload one in the Docs tab.</p>
        )}
        {result && result.index_ready && (
          <div className="agent-api__results">
            <p className="agent-api__meta">
              <span className="agent-api__badge">{result.took_ms.toFixed(1)} ms</span>
              {result.hits.length} passage{result.hits.length === 1 ? "" : "s"}
            </p>
            {result.hits.length === 0 && <p className="agent-api__hint">No matching passages.</p>}
            <ol className="agent-api__hits">
              {result.hits.map((hit, i) => (
                <li key={`${hit.source}-${hit.chunk}-${i}`} className="agent-api__hit">
                  <div className="agent-api__hit-head">
                    <span>{hit.source ?? "document"}</span>
                    {hit.score !== null && <span>{hit.score.toFixed(2)}</span>}
                  </div>
                  <p className="agent-api__hit-text">{hit.text}</p>
                </li>
              ))}
            </ol>
          </div>
        )}
      </section>

      <section className="agent-api__section">
        <h3 className="agent-api__title">Connect your agent</h3>
        <p className="agent-api__hint">
          Any MCP client (Claude, Cursor, custom agents) or plain HTTP can use this knowledge base as a tool.
        </p>
        <button type="button" className="btn btn--secondary" onClick={makeKey} disabled={creatingKey}>
          {agentKey ? "Create a new key" : creatingKey ? "Creating…" : "Create agent key"}
        </button>
        {keyError && <p className="doc-upload__feedback doc-upload__feedback--err">{keyError}</p>}
        {agentKey && (
          <>
            <p className="agent-api__hint">Valid for {keyDays} days. Treat it like a password.</p>
            <CopyBlock label="Agent key" value={agentKey} />
          </>
        )}

        <CopyBlock label="MCP server URL" value={mcpUrl} />
        <CopyBlock
          label="Claude Code"
          value={`claude mcp add --transport http fieldops ${mcpUrl} \\\n  --header "Authorization: Bearer ${keyText}"`}
        />
        <CopyBlock
          label="REST"
          value={`curl -X POST ${API_URL}/v1/knowledge/query \\\n  -H "Authorization: Bearer ${keyText}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"query": "how do I reset the inverter?", "top_k": 3}'`}
        />
      </section>
    </div>
  );
}
