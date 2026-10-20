"""Per-tick metrics: JSONL writing and the decider-vs-LLM comparison summary."""

from __future__ import annotations

import json

import pytest

from examples.ctf_agent.metrics import (
    MetricsWriter,
    TickRecord,
    compare,
    format_comparison,
    load_records,
    percentile,
    summarize,
)


def rec(tick, path, decider_ms=None, llm_ms=None, mode="decider", **kw):
    return TickRecord(tick=tick, mode=mode, path=path, decider_ms=decider_ms, llm_ms=llm_ms, **kw)


class TestTickRecord:
    def test_decision_ms_sums_what_the_tick_spent_deciding(self):
        assert rec(1, "llm", decider_ms=30, llm_ms=900).decision_ms == 930
        assert rec(1, "decider", decider_ms=30).decision_ms == 30
        assert rec(1, "wait").decision_ms == 0.0


class TestMetricsWriter:
    def test_writes_one_json_object_per_line(self, tmp_path):
        target = tmp_path / "run.jsonl"
        with MetricsWriter(target) as writer:
            writer.write(rec(1, "decider", decider_ms=12.5, label="move_left", confidence=0.9))
            writer.write(rec(2, "llm", llm_ms=800, llm_input_tokens=100, llm_output_tokens=20))
        lines = target.read_text().splitlines()
        assert [json.loads(line)["tick"] for line in lines] == [1, 2]
        first = json.loads(lines[0])
        assert first["path"] == "decider" and first["decider_ms"] == 12.5 and first["decision_ms"] == 12.5
        assert json.loads(lines[1])["llm_output_tokens"] == 20

    def test_a_none_path_is_a_no_op_writer(self):
        with MetricsWriter(None) as writer:
            writer.write(rec(1, "wait"))  # must not raise

    def test_creates_missing_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "run.jsonl"
        with MetricsWriter(target) as writer:
            writer.write(rec(1, "wait"))
        assert target.exists()

    def test_load_records_round_trips_and_skips_blank_lines(self, tmp_path):
        target = tmp_path / "run.jsonl"
        with MetricsWriter(target) as writer:
            writer.write(rec(1, "decider", decider_ms=10))
        target.write_text(target.read_text() + "\n\n")
        loaded = load_records(target)
        assert len(loaded) == 1 and loaded[0]["path"] == "decider"


class TestStats:
    def test_percentile_nearest_rank(self):
        values = list(range(1, 101))
        assert percentile(values, 95) == 95
        assert percentile([5], 95) == 5
        assert percentile([], 95) is None

    def test_summarize_a_decider_run(self):
        records = [
            rec(1, "decider", decider_ms=20).__dict__ | {"decision_ms": 20},
            rec(2, "decider", decider_ms=40).__dict__ | {"decision_ms": 40},
            rec(3, "llm", decider_ms=30, llm_ms=1000, llm_input_tokens=50, llm_output_tokens=10).__dict__
            | {"decision_ms": 1030},
            rec(4, "budget", decider_ms=30).__dict__ | {"decision_ms": 30},
        ]
        s = summarize(records)
        assert s.ticks == 4 and s.llm_calls == 1
        assert s.decider_share == pytest.approx(0.75)  # decider + budget: the Decider's pick ran, no LLM
        assert s.llm_tokens == 60
        assert s.avg_decision_ms == pytest.approx((20 + 40 + 1030 + 30) / 4)
        assert s.p95_decision_ms == 1030

    def test_summarize_an_empty_run(self):
        s = summarize([])
        assert s.ticks == 0 and s.avg_decision_ms is None and s.decider_share == 0.0

    def test_compare_counts_llm_calls_avoided(self):
        decider_run = summarize(
            [{"path": "decider", "decision_ms": 20}] * 8 + [{"path": "llm", "decision_ms": 900, "llm_ms": 900}] * 2
        )
        llm_run = summarize([{"path": "llm", "decision_ms": 1000, "llm_ms": 1000}] * 10)
        result = compare(decider_run, llm_run)
        assert result.llm_calls_avoided == 8
        assert result.speedup_avg == pytest.approx(1000 / ((8 * 20 + 2 * 900) / 10))

    def test_format_comparison_mentions_the_key_numbers(self):
        decider_run = summarize([{"path": "decider", "decision_ms": 20}] * 4)
        llm_run = summarize([{"path": "llm", "decision_ms": 1000, "llm_ms": 1000}] * 4)
        text = format_comparison(decider_run, llm_run)
        assert "100%" in text and "LLM calls avoided" in text and "p95" in text
