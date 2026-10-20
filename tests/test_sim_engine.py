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
        route = next(r for r in match().state(1)["routes"] if r["id"] == "enemy_flag_direct")
        assert route["first_step"] in {"up", "down", "left", "right"}
        assert route["length"] == 12 and len(route["path"]) == 8  # path is capped, length is the full count
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


def routes_by_id(m):
    return {r["id"]: r for r in m.state(1)["routes"]}


def bfs_distance(a, b, walls=frozenset({(1, 1), (2, 1)})):
    """Reference grid distance on the 15x15 board, independent of the engine."""
    from collections import deque

    seen, queue = {a: 0}, deque([a])
    while queue:
        x, y = queue.popleft()
        if (x, y) == b:
            return seen[b]
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= n[0] < 15 and 0 <= n[1] < 15 and n not in walls and n not in seen:
                seen[n] = seen[(x, y)] + 1
                queue.append(n)
    return None


class TestRouteContract:
    def test_closest_threat_is_the_nearest_point_along_the_route_not_manhattan_from_me(self):
        m = match()
        m.position = (8, 7)
        m.enemies = {"blue-1": (11, 9), "blue-2": (14, 14)}
        route = routes_by_id(m)["enemy_flag_direct"]
        cells = [(8, 7)] + [(c["x"], c["y"]) for c in route["path"]]
        assert route["length"] == len(cells) - 1  # short enough that path is the full route
        best = min(
            ((bfs_distance(c, e), step, eid) for step, c in enumerate(cells) for eid, e in m.enemies.items()),
        )
        threat = route["closest_threat"]
        assert (threat["distance"], threat["step"], threat["enemy_id"]) == best
        assert threat["step"] > 0 and threat["step"] != threat["distance"]

    def test_closest_threat_ignores_enemies_outside_their_own_field(self):
        m = match()
        m.enemies = {"blue-1": (3, 7), "blue-2": (4, 7)}  # both stand in MY field
        assert routes_by_id(m)["enemy_flag_direct"]["closest_threat"] is None

    def test_path_holds_the_first_eight_cells_while_length_stays_full(self):
        route = routes_by_id(match())["enemy_flag_direct"]
        assert len(route["path"]) == 8 and route["length"] >= 12

    def test_risks_report_the_real_cell_next_to_the_enemy(self):
        m = match()
        m.position = (5, 7)
        m.enemies = {"blue-1": (10, 7), "blue-2": (14, 14)}
        risks = routes_by_id(m)["enemy_flag_direct"]["risks"]
        assert risks
        for risk in risks:
            cell = (risk["cell"]["x"], risk["cell"]["y"])
            assert abs(cell[0] - 10) + abs(cell[1] - 7) == 1 and risk["enemy_id"] == "blue-1"

    def test_a_safe_variant_avoids_every_risk_and_a_detour_starts_elsewhere(self):
        m = match()
        m.position = (5, 7)
        m.enemies = {"blue-1": (10, 7), "blue-2": (14, 14)}
        routes = routes_by_id(m)
        assert routes["enemy_flag_direct"]["risks"]
        assert routes["enemy_flag_safe"]["risks"] == []
        assert routes["enemy_flag_safe"]["path"] != routes["enemy_flag_direct"]["path"]
        assert "enemy_flag_detour" in routes  # the open board always leaves a third first step
        taken = {routes[i]["first_step"] for i in ("enemy_flag_direct", "enemy_flag_safe")}
        assert routes["enemy_flag_detour"]["first_step"] not in taken

    def test_an_unreachable_target_leaves_the_family_empty(self):
        m = match()
        m.position = (1, 0)  # walled in: (1, 1) is already a wall
        m.walls |= {(0, 0), (2, 0)}
        assert not [r for r in m.state(1)["routes"] if r["goal"] == "enemy_flag"]

    def test_a_destination_on_a_risk_cell_has_no_safe_variant(self):
        m = match()
        m.position = (9, 7)
        m.enemies = {"blue-1": (12, 7), "blue-2": (14, 14)}  # the enemy flag base (13, 7) is next to blue-1
        routes = routes_by_id(m)
        assert routes["enemy_flag_direct"]["risks"]
        assert "enemy_flag_safe" not in routes

    def test_a_carrier_gets_a_length_zero_enemy_flag_route_and_nothing_else_in_that_family(self):
        m = match()
        m.position, m.carrying = (5, 7), True
        family = [r for r in m.state(1)["routes"] if r["goal"] == "enemy_flag"]
        assert [(r["id"], r["length"], r["first_step"], r["path"]) for r in family] == [
            ("enemy_flag_direct", 0, None, [])
        ]

    def test_identical_paths_in_a_family_are_deduped_before_the_cap(self):
        m = match()
        m.position = (5, 7)
        m.enemies = {"blue-1": (14, 14), "blue-2": (14, 0)}  # no risks: safe would repeat direct
        routes = routes_by_id(m)
        assert "enemy_flag_safe" not in routes
        paths = [tuple((c["x"], c["y"]) for c in r["path"]) for r in routes.values() if r["goal"] == "enemy_flag"]
        assert len(paths) == len(set(paths))

    def test_a_crowded_state_keeps_the_first_seven_ids_so_tag_routes_drop_first(self):
        m = match()
        m.position, m.carrying = (12, 7), True
        m.enemies = {"blue-1": (10, 7), "blue-2": (3, 6), "blue-3": (3, 8), "blue-4": (4, 9)}
        ids = [r["id"] for r in m.state(1)["routes"]]
        assert len(ids) <= 7 and ids == sorted(ids)
        assert ids == ["defend_flag", "enemy_flag_direct", "own_base_detour", "own_base_direct",
                       "own_base_safe", "tag:blue-2", "tag:blue-3"]  # tag:blue-4 is the one cut

    def test_ids_are_sorted_and_capped_at_seven(self):
        ids = [r["id"] for r in match().state(1)["routes"]]
        assert ids == sorted(ids) and len(ids) <= 7

    def test_other_agents_do_not_block_the_search_but_mark_the_route_occupied(self):
        m = match()
        free = routes_by_id(m)["enemy_flag_direct"]
        assert free["blocked_now"] is False and free["blocked_reason"] is None
        dx, dy = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}[free["first_step"]]
        m.teammates["red-2"] = (m.position[0] + dx, m.position[1] + dy)
        blocked = routes_by_id(m)["enemy_flag_direct"]
        assert blocked["path"] == free["path"]
        assert blocked["blocked_now"] is True and blocked["blocked_reason"] == "occupied"

    def test_tag_routes_exist_for_enemies_in_my_field_and_end_next_to_them(self):
        m = match()
        m.enemies = {"blue-1": (4, 7), "blue-2": (14, 14)}
        routes = routes_by_id(m)
        assert set(routes) >= {"tag:blue-1"} and "tag:blue-2" not in routes
        tag = routes["tag:blue-1"]
        assert tag["goal"] == "tag" and tag["length"] == 2
        end = tag["path"][-1]
        assert abs(end["x"] - 4) + abs(end["y"] - 7) == 1

    def test_an_adjacent_intruder_gives_an_empty_tag_route(self):
        m = match()
        m.position, m.enemies = (3, 7), {"blue-1": (4, 7), "blue-2": (14, 14)}
        tag = routes_by_id(m)["tag:blue-1"]
        assert tag["length"] == 0 and tag["path"] == [] and tag["first_step"] is None

    def test_defend_flag_needs_an_enemy_within_the_radius_of_my_flag(self):
        m = match()
        assert "defend_flag" not in routes_by_id(m)
        m.enemies = {"blue-1": (5, 7), "blue-2": (14, 14)}  # 4 steps from my base at (1, 7)
        assert "defend_flag" in routes_by_id(m)
        m.enemies = {"blue-1": (6, 7), "blue-2": (14, 14)}  # 5 steps: outside
        assert "defend_flag" not in routes_by_id(m)

    def test_own_base_routes_appear_only_for_a_carrier_on_my_side(self):
        m = match()
        m.position, m.carrying = (13, 7), True
        ids = set(routes_by_id(m))
        assert "own_base_direct" in ids and not any(i.startswith("own_base") for i in routes_by_id(match()))

    def test_agents_report_the_team_name_of_the_carried_flag(self):
        m = match()
        m.position, m.carrying = (13, 7), True
        agents = {a["id"]: a for a in m.state(1)["agents"]}
        assert agents["red-1"]["carrying"] == "blue"
        assert all(a["carrying"] is None for aid, a in agents.items() if aid != "red-1")

    def test_teammates_are_a_list_of_id_strings(self):
        assert match().state(1)["teammates"] == ["red-2", "red-3"]

    def test_moves_are_static_and_never_report_agents(self):
        m = match()
        m.teammates["red-2"] = (2, 7)
        assert m.state(1)["moves"]["right"] == "open"
        m.position = (0, 0)
        assert m.state(1)["moves"] == {"up": "out_of_bounds", "down": "open", "left": "out_of_bounds",
                                       "right": "open"}
        m.position = (1, 2)
        assert m.state(1)["moves"]["up"] == "wall"

    def test_an_unknown_direction_is_a_value_error_not_a_key_error(self):
        with pytest.raises(ValueError, match="direction"):
            match().apply("move", {"direction": "north"})
