"""The minimal dynamic game behind the simulator."""

from __future__ import annotations

import pytest

from simulator.engine import SimMatch


def match(**kw):
    m = SimMatch(**kw)
    m.join("red", 0)
    return m


def route_ids(m):
    return {r["id"] for r in m.state(1)["routes"]}


class TestState:
    def test_state_has_the_v2_keys_and_no_static_map_keys(self):
        state = match().state(1)
        assert {"self", "self_position", "agents", "flags", "score", "routes", "teammates", "moves"} <= set(state)
        assert not {"grid", "walls", "fields"} & set(state)
        assert all("base" not in f for f in state["flags"])
        assert state["self"] == "red-1" and state["self_position"] == {"x": 1, "y": 7}

    def test_routes_offer_the_enemy_flag_with_a_first_step_and_length(self):
        route = next(r for r in match().state(1)["routes"] if r["goal"] == "enemy_flag")
        assert route["first_step"] in {"up", "down", "left", "right"}
        assert route["length"] >= 12 and route["path"][-1] == {"x": 13, "y": 7}
        assert "min_enemy_distance" not in route

    def test_no_own_base_route_until_carrying(self):
        assert "own_base" not in {r["goal"] for r in match().state(1)["routes"]}


class TestMoves:
    def test_move_changes_position_and_blocked_moves_do_not(self):
        m = match()
        m.apply("move", {"direction": "right"})
        assert m.position == (2, 7)
        m.position = (0, 0)
        m.apply("move", {"direction": "up"})
        assert m.position == (0, 0)

    def test_walls_block(self):
        m = match()
        m.position = (1, 2)  # a wall sits at (1, 1)
        m.apply("move", {"direction": "up"})
        assert m.position == (1, 2)


class TestFlag:
    def test_grab_needs_standing_on_the_enemy_flag(self):
        m = match()
        m.apply("grab_flag", {})
        assert not m.carrying
        m.position = (13, 7)
        m.apply("grab_flag", {})
        assert m.carrying and "own_base" in {r["goal"] for r in m.state(1)["routes"]}

    def test_carrying_the_flag_home_scores_and_ends_the_match(self):
        m = match()
        m.position, m.carrying = (2, 7), True
        m.apply("move", {"direction": "left"})
        m.end_tick()
        assert m.over and m.winner == "red" and m.score == {"red": 1, "blue": 0}

    def test_the_match_ends_in_a_tie_after_max_ticks(self):
        m = match(max_ticks=2)
        for _ in range(2):
            m.end_tick()
        assert m.over and m.winner == "tie"


class TestEnemies:
    def test_an_enemy_next_to_you_in_its_field_tags_you_back_to_base(self):
        m = match(seed=1)
        m.position, m.carrying = (9, 7), True
        m.enemies = {"blue-1": (10, 7), "blue-2": (14, 14)}
        m.end_tick()
        assert m.position == (1, 7) and not m.carrying

    def test_tag_sends_an_adjacent_enemy_home_only_in_your_own_field(self):
        m = match()
        m.enemies = {"blue-1": (3, 7), "blue-2": (14, 14)}
        m.position = (2, 7)
        m.apply("tag", {"enemy_id": "blue-1"})
        assert m.enemies["blue-1"] == (13, 7)
        m.enemies["blue-1"] = (10, 7)
        m.position = (9, 7)
        m.apply("tag", {"enemy_id": "blue-1"})
        assert m.enemies["blue-1"] == (10, 7)

    def test_enemies_chase_you_when_close_but_only_every_other_tick(self):
        m = match()
        m.position = (6, 7)
        m.enemies = {"blue-1": (4, 7), "blue-2": (14, 14)}
        m.end_tick()
        assert m.enemies["blue-1"] == (4, 7)
        m.end_tick()
        assert m.enemies["blue-1"] == (5, 7)

    def test_a_route_passing_next_to_an_enemy_reports_the_risk_and_the_threat(self):
        m = match()
        m.position = (5, 7)
        m.enemies = {"blue-1": (9, 7), "blue-2": (14, 14)}
        route = next(r for r in m.state(1)["routes"] if r["goal"] == "enemy_flag")
        assert any(r["enemy_id"] == "blue-1" for r in route["risks"])
        assert route["closest_threat"]["enemy_id"] == "blue-1"

    def test_the_game_is_deterministic_for_a_seed(self):
        def play(seed):
            m = match(seed=seed)
            for _ in range(10):
                m.apply("move", {"direction": "right"})
                m.end_tick()
            return m.position, dict(m.enemies)

        assert play(3) == play(3)
        assert isinstance(play(3)[0], tuple)


def test_unknown_actions_are_rejected_loudly():
    with pytest.raises(KeyError):
        match().apply("fly", {})
