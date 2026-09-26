"""
Unit tests for the latency benchmark pair: `app.latency_log` (the per-turn
recorder wired into the agent) and `latency_report` (the standalone
aggregator). Both are pure local computation/IO -- no LiveKit room, no
network -- so these run fully offline.

What is NOT covered here: the recorder against a real session. That needs a
live room, a mic, and network to Sarvam/Groq, none of which exist in CI or
the dev sandbox (see preflight_check.py's docstring).
"""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.latency_log import record_turn_metrics

# latency_report.py is a top-level operator script (sibling of
# preflight_check.py), not part of the `app` package, so it is loaded by path.
_REPORT_PATH = Path(__file__).resolve().parent.parent / "latency_report.py"
_spec = importlib.util.spec_from_file_location("latency_report", _REPORT_PATH)
latency_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(latency_report)


def _turn(**metrics):
    return SimpleNamespace(metrics=metrics, role="assistant", id="item-1")


# --------------------------------------------------------------------------
# percentile
# --------------------------------------------------------------------------

def test_percentile_uses_nearest_rank_not_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    # An interpolating implementation would invent 2.5 here.
    assert latency_report.percentile(values, 50) == 2.0
    assert latency_report.percentile(values, 95) == 4.0


def test_percentile_single_value():
    assert latency_report.percentile([0.42], 50) == 0.42
    assert latency_report.percentile([0.42], 99) == 0.42


def test_percentile_two_values():
    assert latency_report.percentile([1.0, 9.0], 50) == 1.0
    assert latency_report.percentile([1.0, 9.0], 95) == 9.0


def test_percentile_is_order_independent():
    assert latency_report.percentile([5.0, 1.0, 3.0], 50) == 3.0


def test_percentile_of_empty_raises():
    with pytest.raises(ValueError):
        latency_report.percentile([], 50)


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------

def test_load_collects_series_and_counts(tmp_path):
    run = tmp_path / "run.jsonl"
    run.write_text(
        json.dumps({"e2e_latency": 0.5, "tts_node_ttfb": 0.2}) + "\n"
        + json.dumps({"e2e_latency": 0.7}) + "\n",
        encoding="utf-8",
    )
    series, kept, skipped = latency_report.load(str(run))
    assert series["e2e_latency"] == [0.5, 0.7]
    assert series["tts_node_ttfb"] == [0.2]
    assert (kept, skipped) == (2, 0)


def test_load_skips_malformed_lines_instead_of_crashing(tmp_path):
    run = tmp_path / "run.jsonl"
    run.write_text(
        json.dumps({"e2e_latency": 0.5}) + "\n"
        + "{not valid json\n"
        + "\n"  # blank lines are not counted as skipped
        + json.dumps([1, 2, 3]) + "\n"  # valid JSON, wrong shape
        + json.dumps({"e2e_latency": 0.9}) + "\n",
        encoding="utf-8",
    )
    series, kept, skipped = latency_report.load(str(run))
    assert series["e2e_latency"] == [0.5, 0.9]
    assert kept == 2
    assert skipped == 2


def test_load_ignores_non_numeric_and_bool_values(tmp_path):
    run = tmp_path / "run.jsonl"
    run.write_text(
        json.dumps({"e2e_latency": "slow", "tts_node_ttfb": True}) + "\n"
        + json.dumps({"e2e_latency": 0.3}) + "\n",
        encoding="utf-8",
    )
    series, _, _ = latency_report.load(str(run))
    assert series["e2e_latency"] == [0.3]
    assert "tts_node_ttfb" not in series


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def _write(path, values, key="e2e_latency"):
    path.write_text(
        "\n".join(json.dumps({key: v}) for v in values) + "\n", encoding="utf-8"
    )


def test_single_report_renders_values_in_ms(tmp_path):
    run = tmp_path / "run.jsonl"
    _write(run, [0.5, 0.6, 0.7])
    out = latency_report.single_report(str(run), markdown=False)
    assert "End-to-end" in out
    assert "600 ms" in out  # p50 of 3 values, nearest-rank


def test_single_report_on_empty_file_explains_itself(tmp_path):
    run = tmp_path / "empty.jsonl"
    run.write_text("", encoding="utf-8")
    out = latency_report.single_report(str(run), markdown=False)
    assert "No latency rows found" in out
    assert "LATENCY_LOG_PATH" in out


def test_markdown_mode_emits_pipe_table(tmp_path):
    run = tmp_path / "run.jsonl"
    _write(run, [0.5])
    out = latency_report.single_report(str(run), markdown=True)
    assert out.startswith("# Latency report")
    assert "| Stage" in out


def test_thin_sample_is_warned_about(tmp_path):
    run = tmp_path / "run.jsonl"
    _write(run, [0.5, 0.6])
    out = latency_report.single_report(str(run), markdown=False)
    assert "WARNING" in out
    assert "p95/p99 are not meaningful" in out


def test_large_sample_has_no_thin_warning(tmp_path):
    run = tmp_path / "run.jsonl"
    _write(run, [0.5] * (latency_report.THIN_SAMPLE + 1))
    out = latency_report.single_report(str(run), markdown=False)
    assert "WARNING" not in out


def test_compare_report_computes_improvement(tmp_path):
    before, after = tmp_path / "b.jsonl", tmp_path / "a.jsonl"
    _write(before, [1.0] * 25)
    _write(after, [0.5] * 25)
    out = latency_report.compare_report(str(before), str(after), markdown=False)
    assert "1000 ms" in out and "500 ms" in out
    assert "-500 ms (-50%)" in out


def test_compare_report_shows_regression_with_plus_sign(tmp_path):
    before, after = tmp_path / "b.jsonl", tmp_path / "a.jsonl"
    _write(before, [0.5] * 25)
    _write(after, [0.75] * 25)
    out = latency_report.compare_report(str(before), str(after), markdown=False)
    assert "+250 ms (+50%)" in out


def test_compare_report_with_no_shared_metric_says_so(tmp_path):
    before, after = tmp_path / "b.jsonl", tmp_path / "a.jsonl"
    _write(before, [0.5], key="e2e_latency")
    _write(after, [0.5], key="tts_node_ttfb")
    out = latency_report.compare_report(str(before), str(after), markdown=False)
    assert "nothing to compare" in out


def test_compare_warns_about_variance_on_thin_samples(tmp_path):
    before, after = tmp_path / "b.jsonl", tmp_path / "a.jsonl"
    _write(before, [1.0, 1.0])
    _write(after, [0.5, 0.5])
    out = latency_report.compare_report(str(before), str(after), markdown=False)
    assert "variance" in out


def test_main_rejects_more_than_two_runs(tmp_path):
    with pytest.raises(SystemExit):
        latency_report.main(["a.jsonl", "b.jsonl", "c.jsonl"])


def test_main_reports_missing_file_without_traceback(capsys):
    assert latency_report.main(["does-not-exist.jsonl"]) == 1
    assert "No such run file" in capsys.readouterr().err


# --------------------------------------------------------------------------
# recorder (app.latency_log)
# --------------------------------------------------------------------------

def test_recorder_is_inert_when_env_unset(tmp_path):
    target = tmp_path / "should-not-exist.jsonl"
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LATENCY_LOG_PATH", None)
        record_turn_metrics(_turn(e2e_latency=0.5), session_id="s1")
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_recorder_writes_expected_shape(tmp_path):
    path = tmp_path / "run.jsonl"
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        record_turn_metrics(_turn(e2e_latency=0.5, tts_node_ttfb=0.2), session_id="room-7")

    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["e2e_latency"] == 0.5
    assert row["tts_node_ttfb"] == 0.2
    assert row["session_id"] == "room-7"
    assert row["role"] == "assistant"
    assert isinstance(row["ts"], float)


def test_recorder_appends_rather_than_truncating(tmp_path):
    path = tmp_path / "run.jsonl"
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        record_turn_metrics(_turn(e2e_latency=0.5), session_id="s")
        record_turn_metrics(_turn(e2e_latency=0.6), session_id="s")
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_recorder_never_writes_transcript_text(tmp_path):
    """The file is a benchmark artifact, not a conversation store."""
    path = tmp_path / "run.jsonl"
    item = SimpleNamespace(
        metrics={"e2e_latency": 0.5, "transcript": "my bank PIN is 1234"},
        role="user",
        id="i1",
        text_content="my bank PIN is 1234",
        content=["my bank PIN is 1234"],
    )
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        record_turn_metrics(item, session_id="s")

    written = path.read_text(encoding="utf-8")
    assert "1234" not in written
    assert "PIN" not in written
    assert json.loads(written)["e2e_latency"] == 0.5


def test_recorder_skips_turns_with_no_timings(tmp_path):
    """A say() greeting has no metrics; an empty row would skew percentiles."""
    path = tmp_path / "run.jsonl"
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        record_turn_metrics(_turn(), session_id="s")
    assert not path.exists()


def test_recorder_swallows_write_failure(tmp_path):
    """Telemetry must never break a live call."""
    unwritable = tmp_path / "no-such-dir" / "run.jsonl"
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(unwritable)}):
        record_turn_metrics(_turn(e2e_latency=0.5), session_id="s")  # must not raise


def test_recorder_tolerates_item_without_metrics(tmp_path):
    path = tmp_path / "run.jsonl"
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        record_turn_metrics(SimpleNamespace(role="user", id="x"), session_id="s")
    assert not path.exists()


# --------------------------------------------------------------------------
# end to end: recorder output feeds the aggregator
# --------------------------------------------------------------------------

def test_recorded_file_is_readable_by_the_report(tmp_path):
    path = tmp_path / "run.jsonl"
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        for value in (0.4, 0.5, 0.6):
            record_turn_metrics(_turn(e2e_latency=value), session_id="s")

    series, kept, skipped = latency_report.load(str(path))
    assert series["e2e_latency"] == [0.4, 0.5, 0.6]
    assert (kept, skipped) == (3, 0)
    assert "500 ms" in latency_report.single_report(str(path), markdown=False)
