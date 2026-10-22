"""Offline threshold sweep over decider-mode metrics (no network, no AWS)."""

from __future__ import annotations

import json

from scripts import threshold_sweep as ts


def write(path, confidences):
    rows = [{"tick": i, "mode": "decider", "path": "budget", "confidence": c} for i, c in enumerate(confidences)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path

class TestThresholds:
    def test_default_range_is_030_to_095_step_005(self):
        values = ts.thresholds()
        assert values[0] == 0.30 and values[-1] == 0.95 and len(values) == 14

    def test_custom_range(self):
        assert ts.thresholds(0.5, 0.7, 0.1) == [0.5, 0.6, 0.7]

class TestSweep:
    def test_coverage_is_inclusive_at_the_threshold(self):
        rows = ts.sweep([0.2, 0.5, 0.5, 0.9], [0.5, 0.9])
        assert rows[0] == (0.5, 0.75, 0.25)  # three of four ticks have confidence >= 0.5
        assert rows[1] == (0.9, 0.25, 0.75)

    def test_empty_input_gives_zero_coverage(self):
        assert ts.sweep([], [0.5]) == [(0.5, 0.0, 0.0)]

class TestDistribution:
    def test_quantiles(self):
        d = ts.distribution([0.1, 0.2, 0.3, 0.4, 0.5])
        assert (d["min"], d["p25"], d["p50"], d["p75"], d["max"]) == (0.1, 0.2, 0.3, 0.4, 0.5)

    def test_histogram_has_one_bar_per_bin(self):
        lines = ts.histogram([0.05, 0.05, 0.95], bins=10)
        assert len(lines) == 10 and lines[0].count("#") > lines[5].count("#")

class TestLoad:
    def test_only_numeric_confidences_are_used(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text('{"confidence": 0.4}\n{"confidence": null}\n{"path": "wait"}\n{"confidence": 0.8}\n')
        assert ts.load_confidences([path]) == [0.4, 0.8]

    def test_several_files_are_pooled(self, tmp_path):
        a, b = write(tmp_path / "a.jsonl", [0.1]), write(tmp_path / "b.jsonl", [0.9])
        assert sorted(ts.load_confidences([a, b])) == [0.1, 0.9]

class TestMain:
    def test_prints_markdown_and_writes_out_file(self, tmp_path, capsys):
        src = write(tmp_path / "m.jsonl", [0.3, 0.6, 0.9, 0.9])
        out = tmp_path / "sweep.md"
        assert ts.main([str(src), "--out", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "| threshold |" in printed and "| 0.90 |" in printed
        assert "coverage" in printed and "decision quality" in printed
        assert out.read_text() == printed.rstrip("\n") + "\n" or out.read_text().strip() == printed.strip()

    def test_no_confidences_is_an_error(self, tmp_path, capsys):
        src = tmp_path / "empty.jsonl"
        src.write_text("")
        assert ts.main([str(src)]) == 2


def shadow_file(path, rows):
    """rows: (shadow_label, shadow_confidence, executed_label) per llm-only tick."""
    recs = [{"tick": i, "mode": "llm-only", "path": "llm", "confidence": None, "shadow_label": lab,
             "shadow_confidence": conf, "executed_label": ex} for i, (lab, conf, ex) in enumerate(rows)]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return path


ROWS = [("a", 0.95, "a"), ("a", 0.92, "b"), ("b", 0.55, "b"), ("b", 0.52, "a"), ("a", 0.31, None)]


class TestShadow:
    def test_pairs_exclude_null_executed_labels_and_count_them(self):
        recs = [{"shadow_label": lab, "shadow_confidence": c, "executed_label": e} for lab, c, e in ROWS]
        recs += [{"confidence": 0.9}, {"shadow_label": None, "shadow_confidence": None, "shadow_error": "x"}]
        pairs, excluded = ts.shadow_pairs(recs)
        assert pairs == [(0.95, True), (0.92, False), (0.55, True), (0.52, False)] and excluded == 1

    def test_sweep_reports_coverage_agreement_and_counts(self):
        pairs = [(0.95, True), (0.92, False), (0.55, True), (0.52, False)]
        rows = ts.shadow_sweep(pairs, [0.5, 0.9, 0.99])
        assert rows[0] == (0.5, 1.0, 0.5, 4)
        assert rows[1] == (0.9, 0.5, 0.5, 2)
        assert rows[2] == (0.99, 0.0, None, 0)  # nothing covered: agreement undefined

    def test_buckets_are_tenth_wide_and_the_top_bucket_includes_one(self):
        rows = ts.shadow_buckets([(0.95, True), (0.92, False), (0.55, True), (1.0, True)])
        assert rows == [(0.5, 1, 1.0), (0.9, 3, 2 / 3)]

    def test_report_has_shadow_tables_and_the_proxy_disclaimer(self, tmp_path, capsys):
        src = shadow_file(tmp_path / "llm.jsonl", ROWS)
        assert ts.main([str(src)]) == 0
        out = capsys.readouterr().out
        assert "agreement with the LLM is a proxy for quality, not ground truth" in out
        assert "covered" in out and "agreement" in out and "| 0.90 |" in out and "0.9-1.0" in out
        assert "Excluded (no executed label): 1" in out

    def test_decider_only_files_keep_the_old_report(self, tmp_path, capsys):
        assert ts.main([str(write(tmp_path / "m.jsonl", [0.3, 0.9]))]) == 0
        assert "Shadow" not in capsys.readouterr().out
