import type { AgentState } from "../types/transcript";
import type { MicPhase, SessionPhase } from "../types/livekit";
import { WaveformBars } from "./WaveformBars";

interface VoiceVisualizerProps {
  agentState: AgentState;
  connectionPhase: SessionPhase;
  micPhase: MicPhase;
  localAudioLevel: number;
  agentAudioLevel: number;
}

const STATE_LABEL: Record<AgentState, string> = {
  idle: "Idle",
  initializing: "Init",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

const ORB_COLOR: Record<AgentState, string> = {
  idle: "var(--text-faint)",
  initializing: "var(--text-muted)",
  listening: "var(--accent-listen)",
  thinking: "var(--accent-think)",
  speaking: "var(--accent-speak)",
};

const ORB_GLOW: Record<AgentState, string> = {
  idle: "none",
  initializing: "none",
  listening: "0 0 18px 4px var(--accent-listen-soft)",
  thinking: "0 0 18px 4px var(--accent-think-soft)",
  speaking: "0 0 18px 4px var(--accent-speak-soft)",
};

export function VoiceVisualizer({
  agentState,
  connectionPhase,
  micPhase,
  localAudioLevel,
  agentAudioLevel,
}: VoiceVisualizerProps) {
  const isLive = connectionPhase === "connected";
  const displayState = isLive ? agentState : "idle";

  const stateLabel = isLive
    ? STATE_LABEL[agentState]
    : connectionPhase === "connecting"
      ? "Connecting"
      : "Not connected";

  const userActive = isLive && micPhase === "started";
  const agentActive = isLive && (agentState === "speaking");

  return (
    <div className="visualizer">
      <div
        className="visualizer__orb"
        style={{
          background: ORB_COLOR[displayState],
          boxShadow: ORB_GLOW[displayState],
        }}
        aria-hidden="true"
      />
      <p className="visualizer__state" aria-live="polite">
        {stateLabel}
      </p>

      <div className="visualizer__waveforms">
        <div className="visualizer__waveform-group">
          <WaveformBars level={localAudioLevel} isActive={userActive} color="teal" />
          <span className="visualizer__waveform-label">You</span>
        </div>
        <div className="visualizer__waveform-group">
          <WaveformBars level={agentAudioLevel} isActive={agentActive} color="purple" />
          <span className="visualizer__waveform-label">Agent</span>
        </div>
      </div>
    </div>
  );
}
