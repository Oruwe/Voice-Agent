interface WaveformBarsProps {
  level: number;
  isActive: boolean;
  bars?: number;
  color: "teal" | "purple";
}

export function WaveformBars({ level, isActive, bars = 20, color }: WaveformBarsProps) {
  const clampedLevel = Math.min(1, Math.max(0, level));

  return (
    <div className="waveform" aria-hidden="true">
      {Array.from({ length: bars }, (_, index) => {
        let height: number;
        if (isActive && clampedLevel > 0.01) {
          height = 3 + clampedLevel * 28 + Math.sin(index * 0.7) * clampedLevel * 8;
        } else {
          height = 3;
        }
        return (
          <span
            key={index}
            className={`waveform__bar waveform__bar--${color}`}
            style={{ height: `${Math.max(3, height).toFixed(1)}px` }}
          />
        );
      })}
    </div>
  );
}
