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

from app.latency_log import BargeInTracker, record_turn_metrics

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
# BargeInTracker
#
# Pure state machine driven by three callbacks, so it tests fully offline.
# Real barge-in still needs a live mic -- see the module docstring.
# --------------------------------------------------------------------------

def _user_state(new_state, at):
    return SimpleNamespace(new_state=new_state, created_at=at)


def _agent_state(old_state, new_state, at):
    return SimpleNamespace(old_state=old_state, new_state=new_state, created_at=at)


def _interrupted_reply(interrupted=True):
    return SimpleNamespace(role="assistant", interrupted=interrupted, id="i1")


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_barge_in_happy_path_records_the_delta(tmp_path):
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="room-1")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        tracker.on_user_state(_user_state("speaking", 100.0), agent_state="speaking")
        tracker.on_item(_interrupted_reply())
        tracker.on_agent_state(_agent_state("speaking", "listening", 100.3))

    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["barge_in_latency"] == pytest.approx(0.3)
    assert rows[0]["session_id"] == "room-1"


def test_no_row_when_agent_finished_naturally(tmp_path):
    """User talked over it, but the reply was never actually interrupted."""
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        tracker.on_user_state(_user_state("speaking", 100.0), agent_state="speaking")
        tracker.on_item(_interrupted_reply(interrupted=False))
        tracker.on_agent_state(_agent_state("speaking", "listening", 100.3))
    assert _rows(path) == []


def test_no_row_when_agent_stops_with_no_onset(tmp_path):
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        tracker.on_item(_interrupted_reply())
        tracker.on_agent_state(_agent_state("speaking", "listening", 100.3))
    assert _rows(path) == []


def test_user_speaking_while_agent_idle_is_not_an_onset(tmp_path):
    """Normal turn-taking: the user speaking is not a barge-in."""
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        tracker.on_user_state(_user_state("speaking", 100.0), agent_state="listening")
        tracker.on_item(_interrupted_reply())
        tracker.on_agent_state(_agent_state("speaking", "listening", 100.3))
    assert _rows(path) == []


def test_onset_does_not_leak_into_a_later_turn(tmp_path):
    """An overlap that didn't stop the agent must not arm the next measurement."""
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        # turn 1: user overlaps, agent finishes anyway
        tracker.on_user_state(_user_state("speaking", 100.0), agent_state="speaking")
        tracker.on_item(_interrupted_reply(interrupted=False))
        tracker.on_agent_state(_agent_state("speaking", "listening", 100.5))
        # turn 2: agent interrupted, but no fresh onset was recorded
        tracker.on_item(_interrupted_reply())
        tracker.on_agent_state(_agent_state("speaking", "listening", 130.0))
    assert _rows(path) == []


def test_stale_onset_is_discarded(tmp_path):
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        tracker.on_user_state(_user_state("speaking", 100.0), agent_state="speaking")
        tracker.on_item(_interrupted_reply())
        tracker.on_agent_state(_agent_state("speaking", "listening", 100.0 + 60))
    assert _rows(path) == []


def test_agent_state_change_not_leaving_speaking_is_ignored(tmp_path):
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        tracker.on_user_state(_user_state("speaking", 100.0), agent_state="speaking")
        tracker.on_item(_interrupted_reply())
        tracker.on_agent_state(_agent_state("listening", "thinking", 100.2))
    assert _rows(path) == []


def test_two_successive_barge_ins_each_record(tmp_path):
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        for start, stop in ((100.0, 100.3), (200.0, 200.25)):
            tracker.on_user_state(_user_state("speaking", start), agent_state="speaking")
            tracker.on_item(_interrupted_reply())
            tracker.on_agent_state(_agent_state("speaking", "listening", stop))

    values = [r["barge_in_latency"] for r in _rows(path)]
    assert values == pytest.approx([0.3, 0.25])


def test_barge_in_tracker_is_inert_when_env_unset(tmp_path):
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LATENCY_LOG_PATH", None)
        tracker.on_user_state(_user_state("speaking", 100.0), agent_state="speaking")
        tracker.on_item(_interrupted_reply())
        tracker.on_agent_state(_agent_state("speaking", "listening", 100.3))
    assert list(tmp_path.iterdir()) == []


def test_barge_in_rows_flow_through_the_report(tmp_path):
    path = tmp_path / "run.jsonl"
    tracker = BargeInTracker(session_id="s")
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        for i in range(3):
            tracker.on_user_state(_user_state("speaking", 100.0 * i), agent_state="speaking")
            tracker.on_item(_interrupted_reply())
            tracker.on_agent_state(_agent_state("speaking", "listening", 100.0 * i + 0.3))

    series, _, _ = latency_report.load(str(path))
    assert series["barge_in_latency"] == pytest.approx([0.3, 0.3, 0.3])
    assert "Barge-in stop" in latency_report.single_report(str(path), markdown=False)


def test_turn_rows_carry_the_interrupted_flag(tmp_path):
    path = tmp_path / "run.jsonl"
    item = SimpleNamespace(
        metrics={"e2e_latency": 0.5}, role="assistant", id="i1", interrupted=True
    )
    with patch.dict(os.environ, {"LATENCY_LOG_PATH": str(path)}):
        record_turn_metrics(item, session_id="s")
    assert json.loads(path.read_text(encoding="utf-8"))["interrupted"] is True


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
