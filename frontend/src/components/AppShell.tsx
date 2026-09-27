import { useState, type ReactNode } from "react";
import type { ThemePreference } from "../hooks/useTheme";
import type { AgentState, ToolEvent, TranscriptEntry, TurnLatency } from "../types/transcript";
import type { AppErrorInfo, MicPhase, SessionPhase } from "../types/livekit";
import { ThemeToggle } from "./ThemeToggle";
import { ConnectionStatus } from "./ConnectionStatus";
import { VoiceVisualizer } from "./VoiceVisualizer";
import { VoiceControls } from "./VoiceControls";
import { TranscriptPanel } from "./TranscriptPanel";
import { ChatInput } from "./ChatInput";
import { ToolEventCard } from "./ToolEventCard";
import { ErrorBanner } from "./ErrorBanner";
import { DocumentUpload } from "./DocumentUpload";
import { LatencyMonitor } from "./LatencyMonitor";

interface AppShellProps {
  themePreference: ThemePreference;
  onThemeChange: (preference: ThemePreference) => void;
  connectionPhase: SessionPhase;
  micPhase: MicPhase;
  agentState: AgentState;
  agentIdentity: string | null;
  tenantSlug?: string;
  externalId?: string;
  localAudioLevel: number;
  agentAudioLevel: number;
  transcript: TranscriptEntry[];
  toolEvents: ToolEvent[];
  latencies: TurnLatency[];
  errors: AppErrorInfo[];
  onDismissError: (id: string) => void;
  onConnect: () => void;
  onDisconnect: () => void;
  onStartMic: () => void;
  onStopMic: () => void;
  onToggleMute: () => void;
  onSendChat: (text: string) => Promise<void>;
  signInSlot: ReactNode;
  showSignIn: boolean;
  accessToken?: string | null;
}

type RightTab = "latency" | "documents" | "events";

export function AppShell({
  themePreference,
  onThemeChange,
  connectionPhase,
  micPhase,
  agentState,
  agentIdentity,
  tenantSlug,
  externalId,
  localAudioLevel,
  agentAudioLevel,
  transcript,
  toolEvents,
  latencies,
  errors,
  onDismissError,
  onConnect,
  onDisconnect,
  onStartMic,
  onStopMic,
  onToggleMute,
  onSendChat,
  signInSlot,
  showSignIn,
  accessToken,
}: AppShellProps) {
  const [rightTab, setRightTab] = useState<RightTab>("latency");
  const isConnected = connectionPhase === "connected";

  const lastLatency = latencies[latencies.length - 1];
  const lastMs = lastLatency?.totalMs;

  function latencyDisplayColor(ms: number | undefined): string {
    if (ms === undefined) return "var(--text-muted)";
    if (ms < 400) return "var(--latency-fast)";
    if (ms < 700) return "var(--latency-mid)";
    return "var(--latency-slow)";
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path
                d="M12 15a3.5 3.5 0 0 0 3.5-3.5v-5a3.5 3.5 0 0 0-7 0v5A3.5 3.5 0 0 0 12 15Z"
                fill="currentColor"
              />
              <path
                d="M6.5 11.25a5.5 5.5 0 0 0 11 0M12 17.25V20"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </span>
          <div>
            <p className="app-header__title">Field Voice Console</p>
            <p className="app-header__subtitle">Voice Agent Platform</p>
          </div>
        </div>
        <div className="app-header__end">
          <ConnectionStatus
            phase={connectionPhase}
            tenantSlug={tenantSlug}
            externalId={externalId}
            agentIdentity={agentIdentity}
          />
          <ThemeToggle preference={themePreference} onChange={onThemeChange} />
        </div>
      </header>

      <ErrorBanner errors={errors} onDismiss={onDismissError} />

      {showSignIn ? (
        <main className="app-main app-main--centered">{signInSlot}</main>
      ) : (
        <main className="app-main app-grid">
          {/* Left: voice panel */}
          <section className="panel panel--voice" aria-label="Voice session">
            <VoiceVisualizer
              agentState={agentState}
              connectionPhase={connectionPhase}
              micPhase={micPhase}
              localAudioLevel={localAudioLevel}
              agentAudioLevel={agentAudioLevel}
            />
            <VoiceControls
              connectionPhase={connectionPhase}
              micPhase={micPhase}
              onConnect={onConnect}
              onDisconnect={onDisconnect}
              onStartMic={onStartMic}
              onStopMic={onStopMic}
              onToggleMute={onToggleMute}
            />
            <div className="panel__separator" aria-hidden="true" />
            <div className="panel__session-summary">
              <div className="session-summary__row">
                <span className="session-summary__label">Last latency</span>
                <span
                  className="session-summary__value"
                  style={{ color: latencyDisplayColor(lastMs) }}
                >
                  {lastMs !== undefined ? `${lastMs}ms` : "--"}
                </span>
              </div>
              <div className="session-summary__row">
                <span className="session-summary__label">Turns</span>
                <span className="session-summary__value">{latencies.length}</span>
              </div>
            </div>
          </section>

          {/* Center: conversation */}
          <section className="panel panel--conversation" aria-label="Conversation transcript">
            <div className="panel__header">
              <h2 className="panel__header-title">Conversation</h2>
            </div>
            <TranscriptPanel entries={transcript} isConnected={isConnected} latencies={latencies} />
            <ChatInput isConnected={isConnected} onSend={onSendChat} />
          </section>

          {/* Right: intelligence panel */}
          <section className="panel panel--intelligence" aria-label="Session intelligence">
            <div className="panel-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={rightTab === "latency"}
                className={`panel-tabs__tab${rightTab === "latency" ? " is-active" : ""}`}
                onClick={() => setRightTab("latency")}
              >
                Latency
                {latencies.length > 0 && (
                  <span className="panel-tabs__badge">{latencies.length}</span>
                )}
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={rightTab === "documents"}
                className={`panel-tabs__tab${rightTab === "documents" ? " is-active" : ""}`}
                onClick={() => setRightTab("documents")}
              >
                Docs
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={rightTab === "events"}
                className={`panel-tabs__tab${rightTab === "events" ? " is-active" : ""}`}
                onClick={() => setRightTab("events")}
              >
                Events
                {toolEvents.length > 0 && (
                  <span className="panel-tabs__badge">{toolEvents.length}</span>
                )}
              </button>
            </div>
            <div className="panel-tabs__panels">
              <div role="tabpanel" hidden={rightTab !== "latency"}>
                <LatencyMonitor latencies={latencies} isConnected={isConnected} />
              </div>
              <div role="tabpanel" hidden={rightTab !== "documents"}>
                <DocumentUpload accessToken={accessToken ?? null} />
              </div>
              <div role="tabpanel" hidden={rightTab !== "events"}>
                <ToolEventCard events={toolEvents} isConnected={isConnected} />
              </div>
            </div>
          </section>
        </main>
      )}
    </div>
  );
}
