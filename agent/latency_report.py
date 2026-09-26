#!/usr/bin/env python3
"""
Aggregate the per-turn latency JSONL written by app/latency_log.py into
p50/p95/p99 tables, and compare two runs (e.g. before vs after a tuning
change) side by side.

WHY THIS EXISTS: this project's latency work could only ever be described
as "it feels faster". LiveKit measures every turn end to end; app/
latency_log.py persists those measurements; this turns them into numbers
you can actually defend.

HONESTY ABOUT SMALL SAMPLES: a p95 over 5 turns is not a p95, it is the
slowest turn wearing a hat. Every row reports n, and the report says so
plainly when n is too small for the tail percentiles to mean anything.
Quoting a number this tool flagged as thin is how you get taken apart in
the Q&A.

Usage:
    python latency_report.py run.jsonl                 # one run
    python latency_report.py before.jsonl after.jsonl  # A/B with deltas
    python latency_report.py a.jsonl b.jsonl --markdown

To produce a run: set LATENCY_LOG_PATH, start the agent, have a real
conversation, then point this at the file.
"""
from __future__ import annotations

import argparse
import json
import sys

# Display order and labels. Keys match MetricsReport / latency_log.py.
METRICS: tuple[tuple[str, str], ...] = (
    ("e2e_latency", "End-to-end (stopped speaking -> agent speaks)"),
    ("end_of_turn_delay", "Turn-end decision (endpointing)"),
    ("transcription_delay", "STT transcript after speech"),
    ("llm_node_ttft", "LLM first token (incl. Moss recall)"),
    ("llm_node_ttfs", "LLM first sentence -> TTS"),
    ("tts_node_ttfb", "TTS first audio byte"),
    ("playback_latency", "Playback start"),
)

# Rate, not a latency: lower is not better, so it is reported separately and
# never gets a "faster/slower" verdict.
RATE_METRICS: tuple[tuple[str, str], ...] = (
    ("llm_node_tps", "LLM output tokens/sec"),
)

# Below this, the tail percentiles are noise rather than signal.
THIN_SAMPLE = 20


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Deliberately not statistics.quantiles: interpolation invents values
    between observed turns, which reads as false precision on the small
    samples a hand-run voice demo actually produces.
    """
    if not values:
        raise ValueError("percentile() of empty sequence")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-pct * len(ordered) // 100))))
    return ordered[rank - 1]


def load(path: str) -> tuple[dict[str, list[float]], int, int]:
    """Read a JSONL run. Returns (values by metric, rows kept, lines skipped)."""
    series: dict[str, list[float]] = {}
    kept = skipped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A partial final line is normal if the worker was killed
                # mid-write; that must not sink the whole report.
                skipped += 1
                continue
            if not isinstance(row, dict):
                skipped += 1
                continue
            kept += 1
            for key, _ in (*METRICS, *RATE_METRICS):
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    series.setdefault(key, []).append(float(value))
    return series, kept, skipped


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms"


def _render(rows: list[list[str]], headers: list[str], markdown: bool) -> str:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    if markdown:
        out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
        out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
        out += ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |" for r in rows]
        return "\n".join(out)
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    out += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip() for r in rows]
    return "\n".join(out)


def single_report(path: str, markdown: bool) -> str:
    series, kept, skipped = load(path)
    lines = [f"# Latency report: {path}" if markdown else f"Latency report: {path}", ""]
    if not series:
        lines.append(f"No latency rows found ({kept} rows read, {skipped} skipped).")
        lines.append("Was LATENCY_LOG_PATH set while the agent ran?")
        return "\n".join(lines)

    rows = []
    for key, label in METRICS:
        if key not in series:
            continue
        s = stats(series[key])
        rows.append([label, str(s["n"]), _ms(s["p50"]), _ms(s["p95"]), _ms(s["p99"]), _ms(s["max"])])
    if rows:
        lines.append(_render(rows, ["Stage", "n", "p50", "p95", "p99", "max"], markdown))

    rate_rows = []
    for key, label in RATE_METRICS:
        if key not in series:
            continue
        s = stats(series[key])
        rate_rows.append([label, str(s["n"]), f"{s['p50']:.0f}", f"{s['min']:.0f}", f"{s['max']:.0f}"])
    if rate_rows:
        lines += ["", _render(rate_rows, ["Rate", "n", "median", "min", "max"], markdown)]

    lines += ["", *_caveats(max((len(v) for v in series.values()), default=0), skipped)]
    return "\n".join(lines)


def compare_report(before_path: str, after_path: str, markdown: bool) -> str:
    before, _, before_skipped = load(before_path)
    after, _, after_skipped = load(after_path)

    title = f"Latency: {before_path} -> {after_path}"
    lines = [f"# {title}" if markdown else title, ""]

    rows = []
    for key, label in METRICS:
        if key not in before or key not in after:
            continue
        b, a = percentile(before[key], 50), percentile(after[key], 50)
        b95, a95 = percentile(before[key], 95), percentile(after[key], 95)
        delta = a - b
        pct = (delta / b * 100) if b else 0.0
        rows.append([
            label,
            f"{len(before[key])}/{len(after[key])}",
            _ms(b), _ms(a),
            f"{delta * 1000:+.0f} ms ({pct:+.0f}%)",
            f"{_ms(b95)} -> {_ms(a95)}",
        ])
    if not rows:
        lines.append("No metric appears in both runs, so there is nothing to compare.")
        return "\n".join(lines)

    lines.append(_render(
        rows,
        ["Stage", "n before/after", "p50 before", "p50 after", "p50 change", "p95"],
        markdown,
    ))

    smallest = min(
        min(len(before[k]), len(after[k]))
        for k, _ in METRICS if k in before and k in after
    )
    lines += ["", *_caveats(smallest, before_skipped + after_skipped, comparing=True)]
    return "\n".join(lines)


def _caveats(n: int, skipped: int, *, comparing: bool = False) -> list[str]:
    out = []
    if skipped:
        out.append(f"Note: skipped {skipped} unparseable line(s).")
    if n < THIN_SAMPLE:
        out.append(
            f"WARNING: only {n} turn(s) in the smallest series. p95/p99 are not "
            f"meaningful below ~{THIN_SAMPLE} turns -- treat them as the slowest "
            "turn, not a tail. Quote p50, or record a longer run."
        )
        if comparing:
            out.append(
                "A difference this small a sample can also just be variance between "
                "two calls. Run each side a few times before claiming a win."
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate and compare voice-agent per-turn latency runs."
    )
    parser.add_argument("runs", nargs="+", metavar="RUN.jsonl",
                        help="one run to summarize, or two to compare (before after)")
    parser.add_argument("--markdown", action="store_true",
                        help="emit markdown tables (paste-ready)")
    args = parser.parse_args(argv)

    if len(args.runs) > 2:
        parser.error("pass one run to summarize, or two to compare")

    try:
        if len(args.runs) == 1:
            print(single_report(args.runs[0], args.markdown))
        else:
            print(compare_report(args.runs[0], args.runs[1], args.markdown))
    except FileNotFoundError as e:
        print(f"No such run file: {e.filename}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
