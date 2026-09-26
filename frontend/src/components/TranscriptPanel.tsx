import { useEffect, useRef } from "react";
import type { TranscriptEntry, TurnLatency } from "../types/transcript";

interface TranscriptPanelProps {
  entries: TranscriptEntry[];
  isConnected: boolean;
  latencies: TurnLatency[];
}

function speakerLabel(speaker: TranscriptEntry["speaker"]): string {
  if (speaker === "user") return "You";
  if (speaker === "agent") return "Agent";
  return "Unknown";
}

function latencyColor(ms: number): string {
  if (ms < 400) return "var(--latency-fast)";
  if (ms < 700) return "var(--latency-mid)";
  return "var(--latency-slow)";
}

function latencyBg(ms: number): string {
  if (ms < 400) return "var(--latency-fast-soft)";
  if (ms < 700) return "var(--latency-mid-soft)";
  return "var(--latency-slow-soft)";
}

interface LatencyBadgeProps {
  latency: TurnLatency;
}

function LatencyBadge({ latency }: LatencyBadgeProps) {
  const ms = latency.totalMs;
  if (ms === undefined) return null;

  const color = latencyColor(ms);
  const bg = latencyBg(ms);

  return (
    <div className="transcript-item__badges">
      <span
        className="latency-badge"
        style={{ color, background: bg, borderColor: color }}
        title="Turn latency"
      >
        <svg width="9" height="9" viewBox="0 0 9 9" aria-hidden="true" style={{ flexShrink: 0 }}>
          <circle cx="4.5" cy="4.5" r="3.5" fill="none" stroke="currentColor" strokeWidth="1.2" />
          <path d="M4.5 2.5v2l1.2 1.2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
        {ms}ms
      </span>
      {latency.mossMs !== undefined && (
        <span className="latency-badge latency-badge--moss" title="Moss search">
          <svg width="9" height="9" viewBox="0 0 9 9" aria-hidden="true" style={{ flexShrink: 0 }}>
            <circle cx="4.5" cy="4.5" r="3.5" fill="none" stroke="currentColor" strokeWidth="1.2" />
            <circle cx="4.5" cy="4.5" r="1.5" fill="currentColor" />
          </svg>
          {latency.mossHits !== undefined ? `${latency.mossHits} hits` : ""}
          {latency.mossHits !== undefined ? " · " : ""}
          {Math.round(latency.mossMs)}ms
        </span>
      )}
    </div>
  );
}

export function TranscriptPanel({ entries, isConnected, latencies }: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries]);

  // Build a list of agent-turn index -> latency by pairing agent turns with
  // latencies in order. The Nth agent final turn gets the Nth latency entry.
  const agentFinalEntries = entries.filter((e) => e.speaker === "agent" && e.final);
  const latencyByEntryId = new Map<string, TurnLatency>();
  agentFinalEntries.forEach((entry, i) => {
    if (latencies[i]) {
      latencyByEntryId.set(entry.id, latencies[i]);
    }
  });

  return (
    <div className="panel-scroll" ref={scrollRef} tabIndex={0} aria-label="Live transcript">
      {entries.length === 0 ? (
        <p className="panel-empty">
          {isConnected
            ? "Listening for speech — transcript will appear here as the conversation happens."
            : "Connect to a session to see the live transcript."}
        </p>
      ) : (
        <ul className="transcript-list">
          {entries.map((entry) => {
            const lat = entry.speaker === "agent" ? latencyByEntryId.get(entry.id) : undefined;
            return (
              <li key={entry.id} className={`transcript-item transcript-item--${entry.speaker}`}>
                <div className="transcript-item__meta">
                  <span className="transcript-item__speaker">{speakerLabel(entry.speaker)}</span>
                  {!entry.final && <span className="transcript-item__interim">transcribing…</span>}
                </div>
                <p className="transcript-item__text">{entry.text}</p>
                {lat && <LatencyBadge latency={lat} />}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
