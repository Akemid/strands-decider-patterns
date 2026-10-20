"""Ported tests for the CTF example agent (pure logic only: no network, no live match)."""

from __future__ import annotations

import json
import sys
import unittest

from examples.ctf_agent import agent as dl
from examples.ctf_agent.options import describe_route
from examples.ctf_agent.render import render_state


def route(rid, goal, step="left", length=5, path=None, blocked=False, risks=(), threat=None):
    """A route in the v2 shape; `threat` is (enemy_id, step, distance) for `closest_threat`, or None."""
    return {"id": rid, "goal": goal, "first_step": step, "length": length, "path": path or [],
            "blocked_now": blocked, "blocked_reason": "contention" if blocked else None,
            "risks": [{"step": s, "enemy_id": e} for s, e in risks],
            "closest_threat": None if threat is None
            else {"enemy_id": threat[0], "step": threat[1], "distance": threat[2]}}


def turn_with(routes, me=(9, 7), tick=5):
    return {"status": "turn", "tick": tick,
            "state": {"tick": tick, "self": "blue-1", "self_position": {"x": me[0], "y": me[1]},
                      "agents": [{"id": "blue-1", "team": "blue", "position": {"x": me[0], "y": me[1]},
                                  "carrying": None, "disconnected": False}],
                      "routes": routes}}


def options_for(turn, plan=None):
    return dl.build_options(turn, plan)


def answers(threat=0.0, route_bad=0.0, pick="enemy_flag_direct", conf=0.9, probs=None):
    return {"threat": {"type": "noul", "noul": threat},
            "route_bad": {"type": "noul", "noul": route_bad},
            "action": {"type": "choice", "choice": pick, "confidence": conf,
                       "probabilities": probs if probs is not None else {pick: conf}}}


class CombineAnswersTests(unittest.TestCase):
    def setUp(self):
        self.routes = [
            route("enemy_flag_direct", "enemy_flag", length=6, risks=[(3, "red-1")]),
            route("enemy_flag_safe", "enemy_flag", step="up", length=9),
            route("enemy_flag_blocked", "enemy_flag", length=4, blocked=True),
            route("tag:red-1", "tag", step="left", length=3),
            route("tag:red-2", "tag", step=None, length=0),
            route("defend_flag", "defend_flag", step="right", length=2),
            route("own_base_direct", "own_base", length=8),
        ]
        self.turn = turn_with(self.routes)
        self.options = options_for(self.turn)

    def combine(self, ans, **kw):
        return dl.combine_answers(ans, self.options, self.turn, dl.Plan(), **kw)

    def test_questions_cover_threat_route_bad_and_action_in_one_request(self):
        questions = dl.build_questions(self.options)
        self.assertEqual(set(questions), {"threat", "route_bad", "action"})
        self.assertEqual(questions["threat"]["type"], "noul")
        self.assertEqual(questions["route_bad"]["type"], "noul")
        self.assertEqual(questions["action"]["type"], "choice")
        self.assertEqual(set(questions["action"]["criteria"]), set(self.options))
        self.assertEqual(set(questions["threat"]["criteria"]), {"true", "false"})
        self.assertEqual(set(questions["route_bad"]["criteria"]), {"true", "false"})

    def test_no_recommendation_wording_in_options(self):
        for text in self.options.values():
            self.assertNotIn("recommend", text.lower())

    def test_fallback_to_choice(self):
        label, conf, reason = self.combine(answers(threat=0.1, route_bad=0.2, pick="enemy_flag_safe", conf=0.77))
        self.assertEqual((label, reason), ("enemy_flag_safe", "choice"))
        self.assertAlmostEqual(conf, 0.77)

    def test_threat_prefers_adjacent_tag_without_probabilities(self):
        ans = answers(threat=0.8, pick="enemy_flag_direct")
        ans["action"]["probabilities"] = {}
        label, conf, reason = self.combine(ans)
        self.assertEqual((label, reason), ("tag:red-2", "threat"))
        self.assertAlmostEqual(conf, 0.8)

    def test_threat_uses_choice_probabilities_when_available(self):
        ans = answers(threat=0.6, pick="enemy_flag_direct",
                      probs={"enemy_flag_direct": 0.5, "tag:red-1": 0.2, "defend_flag": 0.25, "tag:red-2": 0.05})
        label, conf, reason = self.combine(ans)
        self.assertEqual((label, reason), ("defend_flag", "threat"))
        self.assertAlmostEqual(conf, 0.6)

    def test_threat_without_adjacent_picks_shortest_tag_then_defend(self):
        routes = [r for r in self.routes if r["id"] != "tag:red-2"]
        turn = turn_with(routes)
        opts = options_for(turn)
        ans = answers(threat=0.9)
        ans["action"]["probabilities"] = {}
        self.assertEqual(dl.combine_answers(ans, opts, turn, dl.Plan())[0], "tag:red-1")
        routes = [r for r in routes if r["id"] != "tag:red-1"]
        turn = turn_with(routes)
        self.assertEqual(dl.combine_answers(ans, options_for(turn), turn, dl.Plan())[0], "defend_flag")

    def test_threat_without_tag_or_defend_option_falls_through(self):
        routes = [r for r in self.routes if r["goal"] not in ("tag", "defend_flag")]
        turn = turn_with(routes)
        label, _, reason = dl.combine_answers(answers(threat=0.9, pick="enemy_flag_safe"), options_for(turn), turn, dl.Plan())
        self.assertEqual((label, reason), ("enemy_flag_safe", "choice"))

    def test_threat_threshold_is_configurable(self):
        ans = answers(threat=0.6, pick="enemy_flag_safe")
        self.assertEqual(self.combine(ans, threat_threshold=0.7)[2], "choice")
        self.assertEqual(self.combine(ans, threat_threshold=0.5)[2], "threat")

    def test_blocked_tag_is_not_a_threat_target(self):
        routes = [dict(r, blocked_now=True) if r["goal"] in ("tag", "defend_flag") else r for r in self.routes]
        turn = turn_with(routes)
        label, _, reason = dl.combine_answers(answers(threat=0.9, pick="enemy_flag_safe"), options_for(turn), turn, dl.Plan())
        self.assertEqual((label, reason), ("enemy_flag_safe", "choice"))

    def test_blocked_pick_switches_to_safe_same_goal_route(self):
        label, conf, reason = self.combine(answers(route_bad=0.81, pick="enemy_flag_blocked", conf=0.31))
        self.assertEqual((label, reason), ("enemy_flag_safe", "avoid blocked/risky"))
        self.assertAlmostEqual(conf, 0.81)

    def test_risky_pick_switches_to_safe_same_goal_route(self):
        label, _, reason = self.combine(answers(route_bad=0.6, pick="enemy_flag_direct"))
        self.assertEqual((label, reason), ("enemy_flag_safe", "avoid blocked/risky"))

    def test_no_switch_when_route_bad_low(self):
        self.assertEqual(self.combine(answers(route_bad=0.49, pick="enemy_flag_blocked"))[2], "choice")

    def test_no_switch_without_safe_same_goal_alternative(self):
        # own_base_direct has no sibling; make it risky
        routes = [dict(r, risks=[{"step": 2, "enemy_id": "red-1"}]) if r["id"] == "own_base_direct" else r
                  for r in self.routes]
        turn = turn_with(routes)
        label, _, reason = dl.combine_answers(answers(route_bad=0.9, pick="own_base_direct"), options_for(turn), turn, dl.Plan())
        self.assertEqual((label, reason), ("own_base_direct", "choice"))

    def test_safe_pick_is_kept_even_when_route_bad_high(self):
        self.assertEqual(self.combine(answers(route_bad=0.9, pick="enemy_flag_safe"))[2], "choice")

    def test_threat_beats_route_bad(self):
        self.assertEqual(self.combine(answers(threat=0.9, route_bad=0.9, pick="enemy_flag_blocked"))[2], "threat")

    def test_unusable_answers_do_not_crash(self):
        label, conf, reason = self.combine({"action": {"choice": None}})
        self.assertEqual((label, conf, reason), (None, 0.0, "choice"))
        label, _, reason = self.combine({"threat": "garbage", "action": {"choice": "wait", "confidence": 0.9}})
        self.assertEqual((label, reason), ("wait", "choice"))

    def test_format_decision_log_line(self):
        line = dl.format_decision(7, "own_base_safe", 0.31, "avoid blocked/risky", 0.12, 0.81)
        self.assertEqual(line, "[tick 7] decider -> own_base_safe (0.31) reason=avoid blocked/risky "
                               "threat=0.12 route_bad=0.81")


class BlockedAnnotationTests(unittest.TestCase):
    def plan_and_turn(self, blocked):
        path = [{"x": 8, "y": 7}, {"x": 7, "y": 7}, {"x": 6, "y": 7}]
        turn = turn_with([route("enemy_flag_direct", "enemy_flag", path=path, blocked=blocked)], me=(9, 7))
        plan = dl.Plan("enemy_flag_direct", "enemy_flag", 3, "move left",
                       path=((9, 7), (8, 7), (7, 7), (6, 7)), position=(10, 7))
        return turn, plan

    def test_unblocked_route_keeps_continues_annotation(self):
        turn, plan = self.plan_and_turn(False)
        self.assertIn("continues your current path", dl.build_options(turn, plan)["enemy_flag_direct"])

    def test_blocked_route_is_not_annotated_as_continuing(self):
        turn, plan = self.plan_and_turn(True)
        text = dl.build_options(turn, plan)["enemy_flag_direct"]
        self.assertNotIn("continues your current path", text)
        self.assertIn("| would continue your path but is BLOCKED now", text)


class CarrierRouteTests(unittest.TestCase):
    """A carrier is offered a length-0 `enemy_flag_*` route (the flag sits on it): acting on it would be a no-op."""

    def turn(self, carrying, flag_carried_by=None):
        turn = turn_with([route("enemy_flag_direct", "enemy_flag", step=None, length=0),
                          route("own_base_direct", "own_base", step="left", length=6)], me=(9, 7))
        state = turn["state"]
        if carrying:
            state["agents"][0]["carrying"] = "red"
        state["flags"] = [{"team": "red", "position": {"x": 1, "y": 7}, "carried_by": None},
                          {"team": "red", "position": {"x": 9, "y": 7},
                           "carried_by": flag_carried_by}]
        return turn

    def test_a_carrier_is_not_offered_the_length_zero_enemy_flag_route(self):
        options = dl.build_options(self.turn(True, flag_carried_by="blue-1"))
        self.assertNotIn("enemy_flag_direct", options)
        self.assertNotIn("grab_flag", options)
        self.assertIn("own_base_direct", options)

    def test_the_prompt_does_not_list_the_length_zero_route_for_a_carrier(self):
        text = render_state(self.turn(True, flag_carried_by="blue-1"))
        self.assertNotIn("enemy_flag_direct", text)

    def test_a_non_carrier_standing_on_the_enemy_flag_still_maps_to_grab_flag(self):
        turn = self.turn(False)
        options = dl.build_options(turn)
        self.assertIn("enemy_flag_direct", options)
        self.assertEqual(dl.label_to_call("enemy_flag_direct", 5, turn)[0], "grab_flag")


class DeciderLoopCliTests(unittest.TestCase):
    def test_defaults(self):
        old = sys.argv
        sys.argv = ["agent.py"]
        try:
            args = dl.parse_args()
        finally:
            sys.argv = old
        self.assertEqual(args.name, "decider-agent")
        self.assertEqual(args.threat_threshold, 0.5)

    def test_warmup_asks_all_three_questions(self):
        _, questions = dl.warmup_question()
        self.assertEqual(set(questions), {"threat", "route_bad", "action"})


class ExplodingDecider:
    """A decider whose every use fails the test: --llm-only must never touch it."""

    last_latency_ms = None

    def __getattr__(self, name):
        raise AssertionError(f"decider.{name} used in llm-only mode")


class SequenceClient:
    """MCPClient stand-in: scripted wait_for_turn payloads, every other tool answers {}."""

    def __init__(self, turns):
        self.turns, self.calls = list(turns), []

    def list_tools_sync(self):
        return []

    def call_tool_sync(self, tool_use_id, name, arguments):
        self.calls.append((name, arguments))
        if name == "wait_for_turn":
            payload = self.turns.pop(0)
        elif name == "join_match":
            payload = {"agent_id": "blue-1", "team": "blue"}
        else:
            payload = {}
        return {"status": "success", "toolUseId": tool_use_id,
                "content": [{"text": json.dumps(payload)}], "isError": False}


class LlmOnlyTests(unittest.TestCase):
    def parse(self, *argv):
        old = sys.argv
        sys.argv = ["agent.py", *argv]
        try:
            return dl.parse_args()
        finally:
            sys.argv = old

    def play(self, args, ticks=3, llm_result=None, timed_out=False):
        turns = [{**turn_with([route("enemy_flag_direct", "enemy_flag")], tick=t),
                  "deadline_remaining_ms": 4000} for t in range(1, ticks + 1)]
        turns.append({"status": "match_over"})
        client = SequenceClient(turns)
        llm_calls = []

        def fake_llm_tick(agent_factory, state, tick, handler=None, usage=None):
            llm_calls.append((tick, state))
            return llm_result if llm_result is not None else (1, ("move", {"direction": "left", "tick": tick}))

        def fake_timeout(fn, timeout_s):
            return (True, None) if timed_out else (False, fn())

        saved = dl.run_llm_tick, dl.run_with_timeout, dl.Agent
        dl.run_llm_tick, dl.run_with_timeout, dl.Agent = fake_llm_tick, fake_timeout, lambda **kw: object()
        try:
            stats = dl.play(client, ExplodingDecider(), args)
        finally:
            dl.run_llm_tick, dl.run_with_timeout, dl.Agent = saved
        return stats, client, llm_calls

    def test_flag_defaults_off_and_parses(self):
        self.assertFalse(self.parse().llm_only)
        self.assertTrue(self.parse("--llm-only").llm_only)

    def test_decider_is_never_used_and_llm_decides_every_tick(self):
        stats, _, llm_calls = self.play(self.parse("--llm-only"), ticks=3)
        self.assertEqual([t for t, _ in llm_calls], [1, 2, 3])
        self.assertEqual((stats.llm_ticks, stats.decider_ticks, stats.budget_ticks), (3, 0, 0))

    def test_llm_gets_the_same_state_text_as_the_fallback_path(self):
        _, _, llm_calls = self.play(self.parse("--llm-only"), ticks=1)
        turn = turn_with([route("enemy_flag_direct", "enemy_flag")], tick=1)
        self.assertEqual(llm_calls[0][1], dl.render_state(turn, dl.Plan(), None, include_last_action=True))

    def test_llm_runs_even_with_less_time_than_the_llm_budget(self):
        args = self.parse("--llm-only", "--llm-budget-ms", "999999")
        stats, _, llm_calls = self.play(args, ticks=2)
        self.assertEqual(len(llm_calls), 2)
        self.assertEqual(stats.budget_ticks, 0)

    def test_llm_timeout_falls_back_to_wait(self):
        stats, client, _ = self.play(self.parse("--llm-only"), ticks=2, timed_out=True)
        self.assertEqual([c for c in client.calls if c[0] == "wait"], [("wait", {"tick": 1}), ("wait", {"tick": 2})])
        self.assertEqual((stats.llm_ticks, stats.llm_timeouts), (0, 2))

    def test_no_warmup_request_in_llm_only_mode(self):
        # ExplodingDecider raises on any attribute access, so reaching the end proves no warm-up.
        self.play(self.parse("--llm-only"), ticks=1)


MAP_RAW = {
    "grid": {"width": 15, "height": 15},
    "walls": [{"x": 1, "y": 1}],
    "fields": {"red": {"min_x": 0, "max_x": 6}, "blue": {"min_x": 8, "max_x": 14}, "neutral_x": 7},
    "flag_bases": {"red": {"x": 1, "y": 7}, "blue": {"x": 13, "y": 7}},
    "ascii": ["..."], "legend": {"#": "wall"},
}


class StaticMapTests(unittest.TestCase):
    def test_parse_map_keeps_what_the_logic_needs(self):
        info = dl.parse_map(MAP_RAW)
        self.assertEqual(info.fields, {"red": (0, 6), "blue": (8, 14)})
        self.assertEqual(info.neutral_x, 7)
        self.assertEqual(info.walls, frozenset({(1, 1)}))
        self.assertEqual(info.flag_bases, {"red": (1, 7), "blue": (13, 7)})

    def test_an_unusable_map_is_none(self):
        for raw in (None, {}, {"fields": {}}, "map", {"fields": {"neutral_x": 7}}):
            self.assertIsNone(dl.parse_map(raw), raw)

    def test_position_needs_the_map(self):
        turn = turn_with([], me=(9, 7))
        info = dl.parse_map(MAP_RAW)
        self.assertIn("at (9, 7), in your own field", dl.render_state(turn, None, info))
        self.assertIn("at (9, 7), in the enemy field",
                      dl.render_state(turn, None, dl.parse_map({**MAP_RAW, "fields": {
                          **MAP_RAW["fields"], "blue": {"min_x": 12, "max_x": 14}}})))
        self.assertIn("on the neutral line", dl.render_state(turn_with([], me=(7, 3)), None, info))
        plain = dl.render_state(turn, None, None)
        self.assertIn("at (9, 7)", plain)
        self.assertNotIn("field", plain)

    def test_the_decider_text_never_carries_the_servers_last_action(self):
        turn = {**turn_with([]), "last_action": {"action": "move:left", "tick": 4, "result": "blocked",
                                                 "reason": "contention"}}
        text = dl.render_state(turn)
        self.assertNotIn("previous action", text)
        self.assertNotIn("last action result", text)
        state, _ = dl.warmup_question()
        self.assertNotIn("previous action", state)

    def test_the_fallback_text_renders_the_servers_last_action(self):
        turn = {**turn_with([]), "last_action": {"action": "move:left", "tick": 4, "result": "blocked",
                                                 "reason": "contention"}}
        self.assertIn("outcome of your previous action: move:left at tick 4 -> blocked (contention)",
                      dl.render_state(turn, include_last_action=True))
        self.assertNotIn("previous action", dl.render_state({**turn, "last_action": None},
                                                            include_last_action=True))

    def test_closest_threat_replaces_nearest_enemy(self):
        text = describe_route(route("r", "enemy_flag", length=4, threat=("red-1", 3, 2)))
        self.assertIn("closest threat red-1 at step 3, distance 2", text)
        # a null threat is missing data, like an absent key: no clause at all
        self.assertNotIn("threat", describe_route(route("r", "enemy_flag")))
        legacy = {"id": "r", "goal": "enemy_flag", "first_step": "up", "length": 2}
        self.assertEqual(describe_route(legacy), "goal enemy_flag, 2 steps, first step up")

    def test_the_tag_fallback_needs_the_map(self):
        turn = turn_with([], me=(5, 5))
        turn["state"]["self"] = "red-1"
        turn["state"]["agents"] = [
            {"id": "red-1", "team": "red", "position": {"x": 5, "y": 5}, "carrying": None},
            {"id": "blue-1", "team": "blue", "position": {"x": 5, "y": 6}, "carrying": None}]
        self.assertIn("tag_blue-1", dl.build_options(turn, None, dl.parse_map(MAP_RAW)))
        self.assertNotIn("tag_blue-1", dl.build_options(turn, None, None))

    def test_match_over_and_rejections(self):
        self.assertEqual(dl.describe_match_over({"status": "match_over", "score": {"red": 1, "blue": 3},
                                                 "winner": "blue", "reason": "completed"}),
                         "match over (completed): winner blue, score red 1, blue 3")
        self.assertEqual(dl.describe_match_over({"status": "match_over", "score": {"red": 0, "blue": 0},
                                                 "winner": None, "reason": "stopped"}),
                         "match over (stopped): no winner, score red 0, blue 0")
        self.assertIsNone(dl.action_rejection({"status": "accepted", "tick": 1}))
        self.assertEqual(dl.action_rejection({"status": "rejected", "tick": 1, "reason": "match_over"}), "match_over")
        self.assertEqual(dl.action_rejection({"status": "rejected", "tick": 1}), "unknown")

    def test_the_warm_up_state_has_the_v2_shape(self):
        state, _ = dl.warmup_question()
        self.assertIn("closest threat red-1 at step 6, distance 2", state)
        self.assertIn("in your own field", state)
        self.assertNotIn("nearest enemy", state)


class StubClient:
    """call_tool_sync stand-in: answers by tool name from a dict of payloads (or isError results)."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def call_tool_sync(self, tool_use_id, name, arguments):
        self.calls.append(name)
        payload, is_error = self.answers[name]
        return {"status": "error" if is_error else "success", "toolUseId": tool_use_id,
                "content": [{"text": json.dumps(payload)}], "isError": is_error}


class JoinMapTests(unittest.TestCase):
    JOINED = {"agent_id": "red-1", "team": "red", "agent_index": 0, "ticks": 10, "deadline_ms": 5000,
              "grid": {"width": 15, "height": 15}, "min_tick_ms": 0, "map": MAP_RAW}

    def test_join_returns_the_team_and_the_map_from_the_result(self):
        client = StubClient({"join_match": (self.JOINED, False)})
        team, info = dl.join(client, "m1", "p1")
        self.assertEqual(team, "red")
        self.assertEqual(info.neutral_x, 7)
        self.assertEqual(client.calls, ["join_match"])  # get_map is not needed

    def test_a_join_result_without_map_yields_none_and_get_map_is_the_fallback(self):
        bare = {k: v for k, v in self.JOINED.items() if k != "map"}
        client = StubClient({"join_match": (bare, False), "get_map": (MAP_RAW, False)})
        _, info = dl.join(client, "m1", "p1")
        self.assertIsNone(info)
        self.assertEqual(dl.fetch_map(client).fields["blue"], (8, 14))
        self.assertEqual(client.calls, ["join_match", "get_map"])

    def test_get_map_failing_is_not_fatal(self):
        client = StubClient({"get_map": ({"reason": "not_joined"}, True)})
        self.assertIsNone(dl.fetch_map(client))
