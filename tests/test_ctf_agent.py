"""Ported tests for the CTF example agent (pure logic only: no network, no live match)."""

from __future__ import annotations

import json
import sys
import unittest

from examples.ctf_agent import agent as dl
from examples.ctf_agent import options as ctf_options
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


def answers(pick="enemy_flag_direct", conf=0.9, probs=None):
    return {"action": {"type": "choice", "choice": pick, "confidence": conf,
                       "probabilities": probs if probs is not None else {pick: conf}}}


class SingleQuestionTests(unittest.TestCase):
    def setUp(self):
        self.routes = [
            route("enemy_flag_direct", "enemy_flag", length=6, risks=[(3, "red-1")]),
            route("enemy_flag_safe", "enemy_flag", step="up", length=9),
            route("tag:red-1", "tag", step="left", length=3),
        ]
        self.turn = turn_with(self.routes)
        self.options = options_for(self.turn)

    def test_the_request_holds_only_the_action_choice(self):
        questions = dl.build_questions(self.options)
        self.assertEqual(set(questions), {"action"})
        self.assertEqual(questions["action"]["type"], "choice")
        self.assertEqual(set(questions["action"]["criteria"]), set(self.options))

    def test_no_recommendation_wording_in_options(self):
        for text in self.options.values():
            self.assertNotIn("recommend", text.lower())

    def test_the_decision_is_the_choice_with_its_confidence_and_the_filter_reason(self):
        label, conf, reason = dl.combine_answers(answers(pick="enemy_flag_safe", conf=0.77), "filtered")
        self.assertEqual((label, reason), ("enemy_flag_safe", "filtered"))
        self.assertAlmostEqual(conf, 0.77)

    def test_the_filter_reason_defaults_to_unfiltered(self):
        self.assertEqual(dl.combine_answers(answers(), )[2], "unfiltered")

    def test_unusable_answers_do_not_crash(self):
        self.assertEqual(dl.combine_answers({"action": {"choice": None}}, "filtered"), (None, 0.0, "filtered"))
        self.assertEqual(dl.combine_answers("garbage", "lesser_evil"), (None, 0.0, "lesser_evil"))

    def test_extra_answers_from_an_older_server_are_ignored(self):
        label, _, _ = dl.combine_answers({**answers(pick="wait"), "threat": {"type": "noul", "noul": 0.9}})
        self.assertEqual(label, "wait")

    def test_format_decision_log_line(self):
        line = dl.format_decision(7, "own_base_safe", 0.31, "filtered")
        self.assertEqual(line, "[tick 7] decider -> own_base_safe (0.31) reason=filtered")


def agent_entry(aid, team, x, y):
    return {"id": aid, "team": team, "position": {"x": x, "y": y}, "carrying": None, "disconnected": False}


def threat_turn(routes=(), enemies=(), my_flag_carried_by=None, me=(10, 3)):
    """Blue is me (base (13, 7), field x 8..14); `enemies` are (id, x, y) red agents."""
    turn = turn_with(list(routes), me=me)
    state = turn["state"]
    state["agents"] += [agent_entry("blue-2", "blue", 13, 12)]
    state["agents"] += [agent_entry(eid, "red", x, y) for eid, x, y in enemies]
    state["flags"] = [{"team": "red", "position": {"x": 1, "y": 7}, "carried_by": None},
                      {"team": "blue", "position": {"x": 13, "y": 7}, "carried_by": my_flag_carried_by}]
    return turn


class FlagThreatenedTests(unittest.TestCase):
    def setUp(self):
        self.map = dl.parse_map(MAP_RAW)

    def threatened(self, turn, map_info="default"):
        return ctf_options.flag_threatened(dl.turn_state(turn), self.map if map_info == "default" else map_info)

    def test_quiet_board_is_not_a_threat(self):
        self.assertFalse(self.threatened(threat_turn(enemies=[("red-1", 2, 7)])))

    def test_my_flag_carried_by_an_enemy_is_a_threat(self):
        self.assertTrue(self.threatened(threat_turn(my_flag_carried_by="red-1", enemies=[("red-1", 2, 7)])))

    def test_my_flag_carried_by_a_teammate_is_not_a_threat(self):
        self.assertFalse(self.threatened(threat_turn(my_flag_carried_by="blue-2")))

    def test_an_enemy_in_my_field_within_the_defend_radius_is_a_threat(self):
        self.assertTrue(self.threatened(threat_turn(enemies=[("red-1", 10, 7)])))  # 3 steps from (13, 7)
        self.assertTrue(self.threatened(threat_turn(enemies=[("red-1", 9, 7)])))  # exactly 4 steps

    def test_an_enemy_just_outside_the_radius_is_not(self):
        self.assertFalse(self.threatened(threat_turn(enemies=[("red-1", 8, 7)])))  # 5 steps

    def test_steps_follow_the_walls_not_the_straight_line(self):
        walls = [{"x": 11, "y": y} for y in range(3, 12)]
        info = dl.parse_map({**MAP_RAW, "walls": walls})
        self.assertFalse(self.threatened(threat_turn(enemies=[("red-1", 10, 7)]), info))

    def test_an_enemy_close_to_the_base_but_outside_my_field_is_not(self):
        info = dl.parse_map({**MAP_RAW, "fields": {**MAP_RAW["fields"], "blue": {"min_x": 10, "max_x": 14}}})
        self.assertFalse(self.threatened(threat_turn(enemies=[("red-1", 9, 7)]), info))

    def test_without_the_map_only_the_carried_flag_counts(self):
        self.assertFalse(self.threatened(threat_turn(enemies=[("red-1", 12, 7)]), None))
        self.assertTrue(self.threatened(threat_turn(my_flag_carried_by="red-1"), None))


class FilterOptionsTests(unittest.TestCase):
    def setUp(self):
        self.map = dl.parse_map(MAP_RAW)
        self.routes = [
            route("enemy_flag_direct", "enemy_flag", length=6, risks=[(3, "red-1")]),
            route("enemy_flag_safe", "enemy_flag", step="up", length=9),
            route("enemy_flag_blocked", "enemy_flag", length=4, blocked=True),
            route("tag:red-1", "tag", step="left", length=3),
            route("tag:red-2", "tag", step=None, length=0, blocked=True),
            route("defend_flag", "defend_flag", step="right", length=2),
            route("own_base_direct", "own_base", length=8),
        ]

    def run_filter(self, routes=None, **kw):
        turn = threat_turn(self.routes if routes is None else routes, **kw)
        return dl.filter_options(options_for(turn), turn, self.map)

    def test_without_a_threat_bad_routes_are_dropped_and_wait_stays(self):
        result = self.run_filter(enemies=[("red-1", 2, 7)])
        self.assertEqual(set(result.options), {"enemy_flag_safe", "tag:red-1", "defend_flag", "own_base_direct", "wait"})
        self.assertEqual(result.reason, "filtered")

    def test_a_threat_offers_only_unblocked_tag_or_defend_options_and_wait(self):
        result = self.run_filter(enemies=[("red-1", 10, 7)])
        self.assertEqual(set(result.options), {"tag:red-1", "defend_flag", "wait"})
        self.assertEqual(result.reason, "filtered")

    def test_a_threat_with_risky_tag_route_still_offers_it(self):
        # a tag route is unblocked even with a risk: only `blocked_now` disqualifies it in the threat branch
        routes = [dict(r, risks=[{"step": 1, "enemy_id": "red-1"}]) if r["id"] == "tag:red-1" else r
                  for r in self.routes]
        result = self.run_filter(routes, enemies=[("red-1", 10, 7)])
        self.assertIn("tag:red-1", result.options)

    def test_a_carried_flag_is_a_threat_too(self):
        result = self.run_filter(my_flag_carried_by="red-1", enemies=[("red-1", 12, 7)])
        self.assertEqual(set(result.options), {"tag:red-1", "defend_flag", "wait"})

    def test_a_threat_without_an_unblocked_defender_falls_back_to_dropping_bad_routes(self):
        routes = [dict(r, blocked_now=True) if r["goal"] in ("tag", "defend_flag") else r for r in self.routes]
        result = self.run_filter(routes, enemies=[("red-1", 10, 7)])
        self.assertEqual(set(result.options), {"enemy_flag_safe", "own_base_direct", "wait"})

    def test_when_only_wait_would_remain_the_full_set_is_offered(self):
        routes = [route("enemy_flag_direct", "enemy_flag", risks=[(2, "red-1")]),
                  route("own_base_direct", "own_base", blocked=True)]
        turn = threat_turn(routes, enemies=[("red-1", 2, 7)])
        options = options_for(turn)
        result = dl.filter_options(options, turn, self.map)
        self.assertEqual((result.options, result.reason), (options, "lesser_evil"))

    def test_nothing_to_drop_is_unfiltered(self):
        routes = [route("enemy_flag_safe", "enemy_flag", length=9)]
        turn = threat_turn(routes, enemies=[("red-1", 2, 7)])
        self.assertEqual(dl.filter_options(options_for(turn), turn, self.map).reason, "unfiltered")

    def test_non_route_options_survive_the_bad_route_filter(self):
        turn = turn_with([], me=(5, 5))
        turn["state"]["moves"] = {"up": "open", "down": "wall", "left": "open", "right": "open"}
        options = options_for(turn)
        self.assertEqual(dl.filter_options(options, turn, self.map).options, options)

    def test_the_carrier_exclusion_of_the_length_zero_flag_route_is_kept(self):
        turn = threat_turn([route("enemy_flag_direct", "enemy_flag", step=None, length=0),
                            route("own_base_direct", "own_base", length=6)], enemies=[("red-1", 2, 7)])
        turn["state"]["agents"][0]["carrying"] = "red"
        result = dl.filter_options(options_for(turn), turn, self.map)
        self.assertNotIn("enemy_flag_direct", result.options)
        self.assertIn("own_base_direct", result.options)

    def test_the_gate_validates_the_pick_against_the_filtered_set(self):
        result = self.run_filter(enemies=[("red-1", 2, 7)])
        gate = dl.build_gate()
        self.assertEqual(gate.route("enemy_flag_blocked", 0.99, result.options, 5000), "llm")
        self.assertEqual(gate.route("enemy_flag_blocked", 0.99, result.options, 100), "wait")
        self.assertEqual(gate.route("enemy_flag_safe", 0.99, result.options, 5000), "decider")


class RequestSizeTests(unittest.TestCase):
    def test_the_rendered_request_stays_under_the_character_budget(self):
        """Request size for the warm-up state (what the Decider is billed for, ~1 ms per input token).

        Baseline measured before Option A (3 questions, long instructions, verbose options, all routes listed
        twice): 2856 characters of JSON (state 789 + questions 2010).
        After Option A (one question, one-sentence instructions, compact options, filtered options and routes):
        1041 characters (state 499 + questions 492). The budget is that value plus ~10% headroom.
        """
        state, questions = dl.warmup_question()
        size = len(json.dumps({"state": state, "questions": questions}))
        self.assertLessEqual(size, REQUEST_CHAR_BUDGET, size)

    def test_the_instructions_are_one_sentence(self):
        self.assertEqual(ctf_options.DECIDER_INSTRUCTIONS.count(". "), 0)
        self.assertLess(len(ctf_options.DECIDER_INSTRUCTIONS), 250)


REQUEST_CHAR_BUDGET = 1150


class BlockedAnnotationTests(unittest.TestCase):
    def plan_and_turn(self, blocked):
        path = [{"x": 8, "y": 7}, {"x": 7, "y": 7}, {"x": 6, "y": 7}]
        turn = turn_with([route("enemy_flag_direct", "enemy_flag", path=path, blocked=blocked)], me=(9, 7))
        plan = dl.Plan("enemy_flag_direct", "enemy_flag", 3, "move left",
                       path=((9, 7), (8, 7), (7, 7), (6, 7)), position=(10, 7))
        return turn, plan

    def test_unblocked_route_keeps_continues_annotation(self):
        turn, plan = self.plan_and_turn(False)
        self.assertIn("your plan's path (", dl.build_options(turn, plan)["enemy_flag_direct"])

    def test_blocked_route_is_not_annotated_as_continuing(self):
        turn, plan = self.plan_and_turn(True)
        text = dl.build_options(turn, plan)["enemy_flag_direct"]
        self.assertNotIn("your plan's path (", text)
        self.assertIn("| your plan's path, now blocked", text)


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
        self.assertFalse(hasattr(args, "threat_threshold"))

    def test_warmup_asks_the_single_action_question(self):
        _, questions = dl.warmup_question()
        self.assertEqual(set(questions), {"action"})


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
        # The abandoned thread is still alive on tick 2, so no second LLM call starts (llm_busy -> wait).
        stats, client, _ = self.play(self.parse("--llm-only"), ticks=2, timed_out=True)
        self.assertEqual([c for c in client.calls if c[0] == "wait"], [("wait", {"tick": 1}), ("wait", {"tick": 2})])
        self.assertEqual((stats.llm_ticks, stats.llm_timeouts), (0, 1))

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
        self.assertIn("threat red-1 dist 2", text)
        # a null threat is missing data, like an absent key: no clause at all
        self.assertNotIn("threat", describe_route(route("r", "enemy_flag")))
        legacy = {"id": "r", "goal": "enemy_flag", "first_step": "up", "length": 2}
        self.assertEqual(describe_route(legacy), "goal enemy_flag, 2 steps, first up")

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
        self.assertIn("threat red-2 dist 4", state)  # a routes line survives the filter
        self.assertNotIn("enemy_flag_direct", state)  # the risky route was filtered out of the state too
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
