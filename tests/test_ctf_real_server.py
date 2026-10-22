"""Real-server readiness: the agent must not depend on the simulator; turn recording; team selection."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

from examples.ctf_agent import agent as dl
from examples.ctf_agent import options as ctf_options
from tests.test_ctf_agent import MAP_RAW, ExplodingDecider, SequenceClient, StubClient, route, turn_with

ROOT = Path(__file__).resolve().parent.parent


def parse(*argv):
    old = sys.argv
    sys.argv = ["agent.py", *argv]
    try:
        return dl.parse_args()
    finally:
        sys.argv = old


class NoSimulatorImportTests(unittest.TestCase):
    def test_agent_and_patterns_never_import_the_simulator(self):
        offenders = []
        for package in ("examples", "decider_patterns"):
            for path in sorted((ROOT / package).rglob("*.py")):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                             else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                    offenders += [f"{path.relative_to(ROOT)}: {n}" for n in names
                                  if n.split(".")[0] == "simulator"]
        self.assertEqual(offenders, [])


class DefendRadiusTests(unittest.TestCase):
    def test_is_four_on_a_fifteen_wide_map_and_scales_with_width(self):
        self.assertEqual(ctf_options.defend_radius(dl.parse_map(MAP_RAW)), 4)
        wide = dl.parse_map({**MAP_RAW, "fields": {"red": {"min_x": 0, "max_x": 13},
                                                   "blue": {"min_x": 16, "max_x": 29}, "neutral_x": 14}})
        self.assertEqual(ctf_options.defend_radius(wide), 8)

    def test_without_a_map_the_radius_check_is_disabled(self):
        self.assertIsNone(ctf_options.defend_radius(None))


class JoinTeamTests(unittest.TestCase):
    def test_team_flag_defaults_to_none_and_accepts_red_blue(self):
        self.assertIsNone(parse().team)
        self.assertEqual(parse("--team", "blue").team, "blue")

    def test_a_requested_team_is_the_only_one_tried_and_agent_index_is_forwarded(self):
        calls = []

        class Client(StubClient):
            def call_tool_sync(self, tool_use_id, name, arguments):
                calls.append(arguments)
                return super().call_tool_sync(tool_use_id, name, arguments)

        client = Client({"join_match": ({"error": "slot_taken"}, True)})
        with self.assertRaises(RuntimeError):
            dl.join(client, "m1", "p1", team="blue", agent_index=1)
        self.assertEqual(calls, [{"match_id": "m1", "team": "blue", "name": "p1", "agent_index": 1}])


class RecordTurnsTests(unittest.TestCase):
    def play(self, path):
        turns = [{**turn_with([route("enemy_flag_direct", "enemy_flag")], tick=t),
                  "deadline_remaining_ms": 4000} for t in (1, 2)] + [{"status": "match_over"}]
        client = SequenceClient(turns)
        args = parse("--llm-only", "--max-llm-calls", "0", "--record-turns", str(path))
        saved = dl.Agent
        dl.Agent = lambda **kw: object()
        try:
            dl.play(client, ExplodingDecider(), args)
        finally:
            dl.Agent = saved
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    def test_flag_defaults_off(self):
        self.assertIsNone(parse().record_turns)

    def test_records_join_once_then_every_turn_payload_without_internal_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.play(Path(tmp) / "nested" / "turns.jsonl")
        self.assertEqual([r["kind"] for r in rows], ["join", "map", "turn", "turn", "turn"])  # no map in join: get_map
        self.assertEqual(rows[0]["payload"]["team"], "blue")
        self.assertEqual([r["payload"].get("tick") for r in rows[2:4]], [1, 2])
        self.assertEqual(rows[4]["payload"], {"status": "match_over"})
        self.assertTrue(all("_is_error" not in r["payload"] for r in rows))

    def test_the_recording_is_the_only_file_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.play(Path(tmp) / "t.jsonl")
            self.assertEqual(list(Path(tmp).iterdir()), [Path(tmp) / "t.jsonl"])


if __name__ == "__main__":
    unittest.main()
