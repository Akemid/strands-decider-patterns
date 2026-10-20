"""The CTF loop writes one metrics line per tick, whichever path decided."""

from __future__ import annotations

import json
import sys

from examples.ctf_agent import agent as dl
from tests.test_ctf_agent import SequenceClient, route, turn_with


class ScriptedDecider:
    """Answers each ask from a list of (pick, confidence); tracks tokens like the real client."""

    def __init__(self, picks):
        self.picks, self.input_tokens, self.last_latency_ms, self.calls = list(picks), 0, 7.0, 0

    def ask(self, state, questions):
        pick, confidence = self.picks.pop(0)
        self.calls += 1
        self.input_tokens += 100
        return {
            "action": {"type": "choice", "choice": pick, "confidence": confidence, "probabilities": {pick: confidence}},
            "threat": {"type": "noul", "noul": 0.0},
            "route_bad": {"type": "noul", "noul": 0.0},
        }


def parse(*argv):
    old = sys.argv
    sys.argv = ["agent.py", *argv]
    try:
        return dl.parse_args()
    finally:
        sys.argv = old


def run(args, decider, ticks=2):
    turns = [
        {**turn_with([route("enemy_flag_direct", "enemy_flag")], tick=t), "deadline_remaining_ms": 5000}
        for t in range(1, ticks + 1)
    ] + [{"status": "match_over"}]
    client = SequenceClient(turns)

    def fake_llm_tick(agent_factory, state, tick, handler=None, usage=None):
        if usage is not None:
            usage.update(input_tokens=120, output_tokens=15)
        return 1, ("move", {"direction": "left", "tick": tick})

    saved = dl.run_llm_tick, dl.Agent
    dl.run_llm_tick, dl.Agent = fake_llm_tick, lambda **kw: object()
    try:
        dl.play(client, decider, args)
    finally:
        dl.run_llm_tick, dl.Agent = saved
    return client


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_decider_and_llm_paths_are_recorded_with_latencies_and_tokens(tmp_path):
    target = tmp_path / "run.jsonl"
    decider = ScriptedDecider([("enemy_flag_direct", 0.97), ("enemy_flag_direct", 0.2)])
    run(parse("--no-warmup", "--metrics", str(target)), decider)
    first, second = read(target)
    assert (first["path"], first["mode"], first["label"]) == ("decider", "decider", "enemy_flag_direct")
    assert first["decider_ms"] is not None and first["llm_ms"] is None
    assert first["decider_input_tokens"] == 100
    assert second["path"] == "llm" and second["llm_input_tokens"] == 120 and second["llm_output_tokens"] == 15
    assert second["decision_ms"] >= second["llm_ms"]


def test_llm_only_mode_records_llm_ticks_without_decider_time(tmp_path):
    target = tmp_path / "llm.jsonl"
    run(parse("--llm-only", "--metrics", str(target)), None)
    records = read(target)
    assert [r["path"] for r in records] == ["llm", "llm"]
    assert all(r["mode"] == "llm-only" and r["decider_ms"] is None and r["reason"] is None for r in records)


def test_no_metrics_flag_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(parse("--llm-only"), None)
    assert list(tmp_path.iterdir()) == []
