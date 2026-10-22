"""The CTF loop writes one metrics line per tick, whichever path decided."""

from __future__ import annotations

import json
import sys

import pytest

from decider_patterns import ServerUnavailable
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
        }


def parse(*argv):
    old = sys.argv
    sys.argv = ["agent.py", *argv]
    try:
        return dl.parse_args()
    finally:
        sys.argv = old


def run(args, decider, ticks=2, events=None, llm_call=("move", {"direction": "left"})):
    turns = [
        {**turn_with([route("enemy_flag_direct", "enemy_flag")], tick=t), "deadline_remaining_ms": 5000}
        for t in range(1, ticks + 1)
    ] + [{"status": "match_over"}]
    client = SequenceClient(turns)

    def fake_llm_tick(agent_factory, state, tick, handler=None, usage=None):
        if events is not None:
            events.append("llm")
        if usage is not None:
            usage.update(input_tokens=120, output_tokens=15)
        return 1, (llm_call[0], {**llm_call[1], "tick": tick})

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
    assert (first["filter_reason"], first["options_offered"]) == ("unfiltered", 2)  # the route + wait
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


class LoggingDecider(ScriptedDecider):
    def __init__(self, picks, events):
        super().__init__(picks)
        self.events = events

    def ask(self, state, questions):
        self.events.append("decider")
        return super().ask(state, questions)


class BrokenDecider(ScriptedDecider):
    def ask(self, state, questions):
        raise RuntimeError("decider exploded")


def test_shadow_requires_llm_only():
    with pytest.raises(SystemExit):
        parse("--shadow-decider")


def test_shadow_records_decider_choice_next_to_the_executed_label_after_the_llm_acted(tmp_path):
    target, events = tmp_path / "s.jsonl", []
    decider = LoggingDecider([("enemy_flag_direct", 0.9), ("wait", 0.4)], events)
    run(parse("--llm-only", "--shadow-decider", "--no-warmup", "--metrics", str(target)), decider, events=events)
    first, second = read(target)
    assert events == ["llm", "decider", "llm", "decider"]  # the shadow request never precedes the LLM's action
    assert (first["mode"], first["path"], first["decider_ms"]) == ("llm-only", "llm", None)
    assert (first["shadow_label"], first["shadow_confidence"]) == ("enemy_flag_direct", 0.9)
    assert first["shadow_ms"] is not None and first["shadow_error"] is None
    assert (first["executed_label"], first["executed_kind"]) == ("enemy_flag_direct", "route")
    assert (second["shadow_label"], second["shadow_confidence"]) == ("wait", 0.4)


def test_shadow_off_leaves_the_fields_empty_and_never_asks_the_decider(tmp_path):
    target = tmp_path / "n.jsonl"
    decider = ScriptedDecider([])
    run(parse("--llm-only", "--metrics", str(target)), decider)
    assert decider.calls == 0
    assert all(r["shadow_label"] is None and r["executed_label"] is None for r in read(target))


def test_shadow_decider_errors_are_recorded_and_never_stop_the_game(tmp_path):
    target = tmp_path / "e.jsonl"
    client = run(parse("--llm-only", "--shadow-decider", "--no-warmup", "--metrics", str(target)),
                 BrokenDecider([]))
    first, second = read(target)
    assert "decider exploded" in first["shadow_error"] and first["shadow_label"] is None
    assert first["path"] == second["path"] == "llm" and second["shadow_error"]
    assert [n for n, _ in client.calls].count("wait_for_turn") == 3


def test_shadow_warmup_runs_but_is_not_a_tick_measurement(tmp_path):
    target = tmp_path / "w.jsonl"
    decider = ScriptedDecider([("wait", 0.5)] * 3)
    run(parse("--llm-only", "--shadow-decider", "--metrics", str(target)), decider)
    assert decider.calls == 3  # one warm-up plus one request per tick
    assert len(read(target)) == 2


def test_failed_shadow_warmup_disables_the_shadow_without_touching_the_game(tmp_path):
    target = tmp_path / "d.jsonl"
    run(parse("--llm-only", "--shadow-decider", "--metrics", str(target)), ConnectionRefusedDecider([]))
    records = read(target)
    assert [r["path"] for r in records] == ["llm", "llm"]
    assert all(r["shadow_label"] is None and "unavailable" in r["shadow_error"] for r in records)


class ConnectionRefusedDecider(ScriptedDecider):
    def ask(self, state, questions):
        raise ServerUnavailable("down")


@pytest.mark.parametrize("call, expected", [
    (("wait", {}), ("wait", "wait")),
    (("move", {"direction": "down"}), (None, "unmatched")),
    (("move", {"direction": "left", "rationale": "route enemy_flag_direct"}), ("enemy_flag_direct", "route")),
    (("tag", {"enemy_id": "red-9"}), (None, "unmatched")),
    (None, (None, "none")),
])
def test_executed_choice_maps_the_llm_call_to_an_option_label(call, expected):
    turn = turn_with([route("enemy_flag_direct", "enemy_flag")])
    options = dl.build_options(turn)
    assert dl.executed_choice(call, turn, dl.Plan(), options) == expected


def test_the_decider_is_asked_one_question_over_the_filtered_options(tmp_path):
    asked = []

    class Spy(ScriptedDecider):
        def ask(self, state, questions):
            asked.append(questions)
            return super().ask(state, questions)

    routes = [route("enemy_flag_safe", "enemy_flag"), route("enemy_flag_risky", "enemy_flag", risks=[(2, "red-1")]),
              route("own_base_blocked", "own_base", blocked=True)]
    turns = [{**turn_with(routes, tick=1), "deadline_remaining_ms": 5000}, {"status": "match_over"}]
    target = tmp_path / "filtered.jsonl"
    saved = dl.Agent
    dl.Agent = lambda **kw: object()
    try:
        dl.play(SequenceClient(turns), Spy([("enemy_flag_safe", 0.97)]),
                parse("--no-warmup", "--metrics", str(target)))
    finally:
        dl.Agent = saved
    assert len(asked) == 1 and set(asked[0]) == {"action"}
    assert set(asked[0]["action"]["criteria"]) == {"enemy_flag_safe", "wait"}
    (record,) = read(target)
    assert (record["filter_reason"], record["options_offered"], record["reason"]) == ("filtered", 2, "filtered")
