import type { TurnLatency } from "../types/transcript";

interface LatencyMonitorProps {
  latencies: TurnLatency[];
  isConnected: boolean;
}

function latencyColor(ms: number | undefined): string {
  if (ms === undefined) return "var(--text-muted)";
  if (ms < 400) return "var(--latency-fast)";
  if (ms < 700) return "var(--latency-mid)";
  return "var(--latency-slow)";
}

function latencyBg(ms: number | undefined): string {
  if (ms === undefined) return "transparent";
  if (ms < 400) return "var(--latency-fast-soft)";
  if (ms < 700) return "var(--latency-mid-soft)";
  return "var(--latency-slow-soft)";
}

function Sparkline({ latencies }: { latencies: TurnLatency[] }) {
  const values = latencies
    .slice(-10)
    .map((l) => l.totalMs)
    .filter((v): v is number => v !== undefined);

  if (values.length < 2) {
    return (
      <div className="latency-monitor__sparkline">
        <svg width="200" height="40" aria-hidden="true">
          <line x1="0" y1="20" x2="200" y2="20" stroke="var(--border-strong)" strokeWidth="1" strokeDasharray="4 4" />
        </svg>
      </div>
    );
  }

  const maxVal = Math.max(...values);
  const minVal = Math.min(...values);
  const range = Math.max(maxVal - minVal, 50);
  const W = 200;
  const H = 40;
  const pad = 3;

  const points = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (W - pad * 2);
    const y = H - pad - ((v - minVal) / range) * (H - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const lastVal = values[values.length - 1];
  const strokeColor = latencyColor(lastVal);

  return (
    <div className="latency-monitor__sparkline">
      <svg width={W} height={H} aria-label={`Latency trend over last ${values.length} turns`}>
        <polyline
          points={points.join(" ")}
          fill="none"
          stroke={strokeColor}
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {values.map((v, i) => {
          const x = pad + (i / (values.length - 1)) * (W - pad * 2);
          const y = H - pad - ((v - minVal) / range) * (H - pad * 2);
          return (
            <circle
              key={i}
              cx={x.toFixed(1)}
              cy={y.toFixed(1)}
              r={i === values.length - 1 ? 3 : 2}
              fill={i === values.length - 1 ? strokeColor : "var(--surface-raised)"}
              stroke={strokeColor}
              strokeWidth="1"
            />
          );
        })}
      </svg>
    </div>
  );
}

function BreakdownBar({ eouMs, thinkMs }: { eouMs: number; thinkMs: number }) {
  const total = eouMs + thinkMs;
  const eouPct = (eouMs / total) * 100;
  const thinkPct = (thinkMs / total) * 100;

  return (
    <div className="latency-monitor__breakdown">
      <div className="latency-monitor__breakdown-bar">
        <div
          className="latency-monitor__breakdown-seg latency-monitor__breakdown-seg--eou"
          style={{ width: `${eouPct.toFixed(1)}%` }}
          title={`EOU: ${eouMs}ms`}
        />
        <div
          className="latency-monitor__breakdown-seg latency-monitor__breakdown-seg--think"
          style={{ width: `${thinkPct.toFixed(1)}%` }}
          title={`Think: ${thinkMs}ms`}
        />
      </div>
      <div className="latency-monitor__breakdown-labels">
        <span className="latency-monitor__breakdown-label latency-monitor__breakdown-label--eou">
          EOU {eouMs}ms
        </span>
        <span className="latency-monitor__breakdown-label latency-monitor__breakdown-label--think">
          Think {thinkMs}ms
        </span>
      </div>
    </div>
  );
}

export function LatencyMonitor({ latencies, isConnected }: LatencyMonitorProps) {
  const last = latencies[latencies.length - 1];
  const lastMs = last?.totalMs;

  const validTotals = latencies.map((l) => l.totalMs).filter((v): v is number => v !== undefined);
  const avgMs = validTotals.length > 0 ? Math.round(validTotals.reduce((a, b) => a + b, 0) / validTotals.length) : undefined;
  const bestMs = validTotals.length > 0 ? Math.min(...validTotals) : undefined;

  const color = latencyColor(lastMs);
  const bg = latencyBg(lastMs);

  return (
    <div className="latency-monitor">
      <div className="latency-monitor__hero" style={{ color, background: bg }}>
        <span className="latency-monitor__value">
          {lastMs !== undefined ? `${lastMs}ms` : "--"}
        </span>
        <span className="latency-monitor__hero-label">Last turn latency</span>
      </div>

      {last?.eouMs !== undefined && last?.thinkMs !== undefined && (
        <BreakdownBar eouMs={last.eouMs} thinkMs={last.thinkMs} />
      )}

      <Sparkline latencies={latencies} />

      <div className="latency-monitor__stats">
        <span className="latency-monitor__stat">
          Avg: {avgMs !== undefined ? `${avgMs}ms` : "--"}
        </span>
        <span className="latency-monitor__stat-sep" aria-hidden="true">·</span>
        <span className="latency-monitor__stat">
          Best: {bestMs !== undefined ? `${bestMs}ms` : "--"}
        </span>
        <span className="latency-monitor__stat-sep" aria-hidden="true">·</span>
        <span className="latency-monitor__stat">
          Turns: {validTotals.length}
        </span>
      </div>

      {last?.mossMs !== undefined && (
        <div className="latency-monitor__moss">
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <circle cx="6" cy="6" r="5" fill="none" stroke="var(--accent-listen)" strokeWidth="1.5" />
            <circle cx="6" cy="6" r="2" fill="var(--accent-listen)" />
          </svg>
          <span>
            Moss: {last.mossHits !== undefined ? `${last.mossHits} hits` : ""}{last.mossHits !== undefined ? " · " : ""}{Math.round(last.mossMs)}ms
          </span>
        </div>
      )}

      {!isConnected && latencies.length === 0 && (
        <p className="latency-monitor__empty">
          Connect to a session to see per-turn latency.
        </p>
      )}
      {isConnected && latencies.length === 0 && (
        <p className="latency-monitor__empty">
          Waiting for first turn…
        </p>
      )}
    </div>
  );
}
