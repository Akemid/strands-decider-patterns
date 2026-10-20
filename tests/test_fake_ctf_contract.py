"""The FakeGame must mirror the MCP turn contract (v2).

The contract is written out here, independently of the fake. If the fake drifts to an older shape these
tests fail, which is the point: a fake that encodes an outdated contract makes every other test pass while
agents degrade silently against a real server.
"""

import json
from contextlib import AsyncExitStack

import pytest
from mcp.client.session import ClientSession

from simulator.fake_ctf_mcp import FakeGame, in_memory_transport, make_route

JOIN_KEYS = {"agent_id", "team", "agent_index", "ticks", "deadline_ms", "grid", "min_tick_ms", "map"}
MAP_KEYS = {"grid", "walls", "fields", "flag_bases", "ascii", "legend"}
TURN_KEYS = {"status", "tick", "deadline_ms", "deadline_remaining_ms", "state", "events", "last_action"}
STATIC_STATE_KEYS = {"grid", "walls", "fields"}


async def call(session, name, **arguments):
    """(is_error, payload) of one tool call."""
    result = await session.call_tool(name, arguments)
    return result.isError, json.loads(result.content[0].text)


class Client:
    def __init__(self, game):
        self.game = game
        self._stack = AsyncExitStack()
        self.session: ClientSession

    async def __aenter__(self):
        try:
            read, write, _ = await self._stack.enter_async_context(in_memory_transport(self.game.server))
            self.session = await self._stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc):
        return await self._stack.__aexit__(*exc)

    async def join(self, team="red", **extra):
        return await call(self.session, "join_match", match_id="m1", team=team, **extra)

    async def tool(self, name, **arguments):
        return await call(self.session, name, **arguments)


class TestJoinResult:
    async def test_join_returns_the_documented_keys_and_the_static_map(self):
        async with Client(FakeGame(turns=1)) as c:
            is_error, joined = await c.join()
        assert not is_error
        assert set(joined) == JOIN_KEYS
        assert set(joined["map"]) == MAP_KEYS
        assert joined["team"] == "red" and joined["agent_id"].startswith("red-")

    async def test_the_map_is_self_consistent(self):
        async with Client(FakeGame(turns=1)) as c:
            _, joined = await c.join()
        board = joined["map"]
        assert joined["grid"] == board["grid"]["width"]  # the board is square: join reports the width only
        assert len(board["ascii"]) == board["grid"]["height"]
        assert {len(row) for row in board["ascii"]} == {board["grid"]["width"]}
        fields = board["fields"]
        assert set(fields) == {"red", "blue", "neutral_x"}
        assert set(fields["red"]) == {"min_x", "max_x"}
        assert set(board["flag_bases"]) == {"red", "blue"}

    async def test_a_full_team_is_a_tool_error_with_a_reason(self):
        async with Client(FakeGame(turns=1, reject_teams=("red",))) as c:
            is_error, payload = await c.join("red")
        assert is_error and payload["reason"] == "match_full"

    async def test_joining_a_finished_match_is_a_match_finished_error(self):
        async with Client(FakeGame(turns=1, finished=True)) as c:
            is_error, payload = await c.join()
        assert is_error and payload["reason"] == "match_finished"


SLOT_ERRORS = {"slot_taken", "match_full", "match_already_started", "match_finished", "unauthorized"}


class TestJoinSlots:
    """`agent_index` seats, and the server's tool-error shape `{reason, detail?}` for every refusal."""

    async def test_agent_index_is_echoed_in_the_join_result(self):
        async with Client(FakeGame(turns=1)) as c:
            is_error, joined = await c.join("blue", agent_index=2)
        assert not is_error
        assert (joined["team"], joined["agent_index"], joined["agent_id"]) == ("blue", 2, "blue-3")

    async def test_omitted_agent_index_takes_the_lowest_free_slot(self):
        game = FakeGame(turns=1, taken_slots=(("red", 0), ("red", 1)))
        async with Client(game) as c:
            _, joined = await c.join("red")
        assert joined["agent_index"] == 2

    async def test_a_taken_slot_is_slot_taken_and_never_falls_back(self):
        game = FakeGame(turns=1, taken_slots=(("red", 1),))
        async with Client(game) as c:
            is_error, payload = await c.join("red", agent_index=1)
        assert is_error and payload["reason"] == "slot_taken"
        assert game.joined is False  # the refused session is not seated anywhere

    async def test_two_sessions_cannot_share_a_slot(self):
        game = FakeGame(turns=1)
        async with Client(game) as first, Client(game) as second:
            assert not (await first.join("red", agent_index=0))[0]
            is_error, payload = await second.join("red", agent_index=0)
        assert is_error and payload["reason"] == "slot_taken"

    async def test_a_second_join_on_the_same_session_is_already_joined(self):
        async with Client(FakeGame(turns=1)) as c:
            assert not (await c.join("red", agent_index=0))[0]
            is_error, payload = await c.join("red", agent_index=1)
        assert is_error and payload["reason"] == "already_joined"

    @pytest.mark.parametrize(
        ("flag", "reason"),
        [("full", "match_full"), ("started", "match_already_started")],
    )
    async def test_lobby_refusals_are_tool_errors_with_a_reason(self, flag, reason):
        async with Client(FakeGame(turns=1, **{flag: True})) as c:
            is_error, payload = await c.join("red", agent_index=0)
        assert is_error and payload["reason"] == reason
        assert set(payload) <= {"reason", "detail"}
        assert reason in SLOT_ERRORS

    @pytest.mark.parametrize("index", [-1, 3])
    async def test_an_out_of_range_agent_index_is_invalid_arguments(self, index):
        async with Client(FakeGame(turns=1)) as c:
            is_error, payload = await c.join("red", agent_index=index)
        assert is_error and payload["reason"] == "invalid_arguments"


class TestJoinRefusalOrder:
    """The server checks: invalid_arguments, then already_joined, then the lobby reasons in this order:
    unknown_match, match_full, slot_taken, unauthorized, match_already_started, match_finished."""

    async def test_invalid_arguments_beats_already_joined(self):
        async with Client(FakeGame(turns=1)) as c:
            assert not (await c.join("red", agent_index=0))[0]
            is_error, payload = await c.join("red", agent_index=7)
        assert is_error and payload["reason"] == "invalid_arguments"

    @pytest.mark.parametrize("flag", ["full", "started", "finished"])
    async def test_already_joined_beats_every_lobby_reason(self, flag):
        game = FakeGame(turns=1)
        async with Client(game) as c:
            assert not (await c.join("red", agent_index=0))[0]
            setattr(game, flag, True)  # the lobby state changes after the seat was taken
            is_error, payload = await c.join("red", agent_index=1)
        assert is_error and payload["reason"] == "already_joined"

    async def test_an_unknown_match_is_unknown_match_and_beats_the_other_lobby_reasons(self):
        async with Client(FakeGame(turns=1, known_match="other", full=True, started=True)) as c:
            is_error, payload = await c.join("red", agent_index=0)
        assert is_error and payload["reason"] == "unknown_match"

    async def test_match_full_beats_slot_taken(self):
        async with Client(FakeGame(turns=1, full=True, taken_slots=(("red", 1),))) as c:
            _, payload = await c.join("red", agent_index=1)
        assert payload["reason"] == "match_full"

    async def test_slot_taken_beats_match_already_started_and_match_finished(self):
        async with Client(FakeGame(turns=1, started=True, finished=True, taken_slots=(("red", 1),))) as c:
            _, payload = await c.join("red", agent_index=1)
        assert payload["reason"] == "slot_taken"

    async def test_unauthorized_beats_match_already_started_and_match_finished(self):
        game = FakeGame(turns=1, started=True, finished=True, disconnected_slots=(("red", 1),))
        async with Client(game) as c:
            _, payload = await c.join("red", agent_index=1)
        assert payload["reason"] == "unauthorized"

    async def test_match_already_started_beats_match_finished(self):
        async with Client(FakeGame(turns=1, started=True, finished=True)) as c:
            _, payload = await c.join("red", agent_index=0)
        assert payload["reason"] == "match_already_started"

    async def test_unauthorized_is_only_for_a_disconnected_slot_owned_by_another_subject(self):
        game = FakeGame(turns=1, disconnected_slots=(("red", 1),))
        async with Client(game) as c:
            is_error, payload = await c.join("red", agent_index=1)
            assert is_error and payload["reason"] == "unauthorized"
            assert not (await c.join("red", agent_index=0))[0]  # any other seat is fine

    async def test_an_unrelated_player_is_never_unauthorized(self):
        async with Client(FakeGame(turns=1)) as c:
            is_error, joined = await c.join("red", agent_index=1)
        assert not is_error and joined["agent_index"] == 1


class TestGetMap:
    async def test_before_joining_it_is_a_not_joined_tool_error(self):
        async with Client(FakeGame(turns=1)) as c:
            is_error, payload = await c.tool("get_map")
        assert is_error and payload == {"reason": "not_joined"}

    async def test_after_joining_it_returns_the_same_object_as_join_match_map(self):
        async with Client(FakeGame(turns=1)) as c:
            _, joined = await c.join()
            is_error, board = await c.tool("get_map")
            _, again = await c.tool("get_map")
        assert not is_error and board == joined["map"] == again


class TestTurn:
    async def test_a_turn_has_the_documented_keys_and_no_static_board_data(self):
        async with Client(
            FakeGame(turns=2, state={"flags": [{"team": "red", "position": {"x": 1, "y": 7}}]})
        ) as c:
            await c.join()
            _, turn = await c.tool("wait_for_turn")
        assert set(turn) == TURN_KEYS and turn["status"] == "turn"
        assert STATIC_STATE_KEYS.isdisjoint(turn["state"])
        assert all("base" not in flag for flag in turn["state"]["flags"])
        assert turn["last_action"] is None  # first turn

    async def test_wait_for_turn_before_joining_is_a_not_joined_tool_error(self):
        async with Client(FakeGame(turns=1)) as c:
            is_error, payload = await c.tool("wait_for_turn")
        assert is_error and payload == {"reason": "not_joined"}

    @pytest.mark.parametrize(
        "tool,args,expected",
        [
            ("move", {"direction": "up"}, {"action": "move:up", "result": "moved"}),
            ("tag", {"enemy_id": "blue-1"}, {"action": "tag:blue-1", "result": "tagged"}),
            ("grab_flag", {}, {"action": "grab_flag", "result": "grabbed"}),
            ("wait", {}, {"action": "wait", "result": "waited"}),
        ],
    )
    async def test_last_action_reports_the_previous_action_of_this_session(self, tool, args, expected):
        async with Client(FakeGame(turns=3)) as c:
            await c.join()
            _, first = await c.tool("wait_for_turn")
            await c.tool(tool, tick=first["tick"], **args)
            _, second = await c.tool("wait_for_turn")
        assert second["last_action"] == {"tick": first["tick"], **expected}

    async def test_routes_carry_closest_threat_and_never_min_enemy_distance(self):
        route = make_route("enemy_flag_direct", "enemy_flag", "left", 8, threat=("blue-1", 6, 2))
        assert "min_enemy_distance" not in route
        assert route["closest_threat"] == {"enemy_id": "blue-1", "step": 6, "distance": 2}
        assert make_route("x", "enemy_flag", None, 0)["closest_threat"] is None
        async with Client(FakeGame(turns=1, state={"routes": [route]})) as c:
            await c.join()
            _, turn = await c.tool("wait_for_turn")
        assert turn["state"]["routes"] == [route]

    @pytest.mark.parametrize(
        "extra",
        [
            {"walls": []},
            {"grid": {"width": 15, "height": 15}},
            {"fields": {}},
            {"flags": [{"team": "red", "base": {"x": 1, "y": 7}}]},
            {"routes": [{"id": "r", "min_enemy_distance": 3}]},
        ],
    )
    def test_the_fake_refuses_state_that_belongs_to_the_old_contract(self, extra):
        with pytest.raises(ValueError, match="contract"):
            FakeGame(turns=1, state=extra)


class TestMatchOverAndAcks:
    async def test_match_over_has_score_winner_and_reason(self):
        async with Client(FakeGame(turns=0)) as c:
            await c.join()
            _, over = await c.tool("wait_for_turn")
        assert set(over) == {"status", "score", "winner", "reason"}
        assert over["status"] == "match_over" and over["reason"] == "completed"
        assert over["winner"] in {"red", "blue", "tie"}  # the server never sends "draw"
        assert set(over["score"]) == {"red", "blue"}

    async def test_a_stopped_match_has_no_winner(self):
        async with Client(FakeGame(turns=0, stopped=True)) as c:
            await c.join()
            _, over = await c.tool("wait_for_turn")
        assert over["reason"] == "stopped" and over["winner"] is None

    async def test_a_skipped_tick_is_unknown_unless_the_server_recorded_a_deadline_rejection(self):
        # Real server last-action rule: without an accepted submission for the previous tick, `rejected/deadline` needs the
        # agent's own `action_rejected` event; otherwise the outcome is `unknown`.
        async with Client(FakeGame(turns=3)) as c:
            await c.join()
            await c.tool("wait_for_turn")  # tick 1, no action submitted
            _, turn = await c.tool("wait_for_turn")
        assert turn["last_action"] == {"action": "none", "tick": 1, "result": "unknown"}
        assert turn["events"] == []

    async def test_a_deadline_rejection_is_reported_when_the_server_would_emit_one(self):
        async with Client(FakeGame(turns=3, deadline_misses=True)) as c:
            await c.join()
            await c.tool("wait_for_turn")
            _, turn = await c.tool("wait_for_turn")
        rejected = {"result": "rejected", "reason": "deadline"}
        assert turn["last_action"] == {"action": "none", "tick": 1, **rejected}
        assert {"type": "action_rejected", "agent_id": "red-1", "tick": 1, "reason": "deadline"} in turn[
            "events"
        ]

    async def test_actions_are_acknowledged_with_status_and_tick(self):
        async with Client(FakeGame(turns=2)) as c:
            await c.join()
            _, turn = await c.tool("wait_for_turn")
            is_error, ack = await c.tool("move", tick=turn["tick"], direction="up")
        assert not is_error and ack == {"status": "accepted", "tick": turn["tick"]}

    async def test_an_action_before_joining_is_rejected_not_joined(self):
        async with Client(FakeGame(turns=1)) as c:
            is_error, ack = await c.tool("wait", tick=1)
        assert not is_error and ack == {"status": "rejected", "tick": 1, "reason": "not_joined"}

    async def test_an_action_after_match_over_is_rejected_match_over(self):
        async with Client(FakeGame(turns=0)) as c:
            await c.join()
            await c.tool("wait_for_turn")  # match_over
            is_error, ack = await c.tool("wait", tick=1)
        assert not is_error and ack == {"status": "rejected", "tick": 1, "reason": "match_over"}

    async def test_an_action_can_be_rejected_with_a_gateway_reason(self):
        async with Client(FakeGame(turns=1, reject_actions="match_not_started")) as c:
            await c.join()
            _, turn = await c.tool("wait_for_turn")
            _, ack = await c.tool("wait", tick=turn["tick"])
        assert ack == {"status": "rejected", "tick": turn["tick"], "reason": "match_not_started"}


class TestJoinAndActionSchemas:
    async def test_join_reports_grid_as_the_board_width_and_the_default_min_tick(self):
        async with Client(FakeGame(turns=1)) as c:
            _, joined = await c.join()
        assert joined["grid"] == 15 and joined["min_tick_ms"] == 1000

    async def test_action_tools_accept_a_bounded_rationale_and_a_direction_enum(self):
        async with Client(FakeGame(turns=1)) as c:
            tools = {t.name: t.inputSchema["properties"] for t in (await c.session.list_tools()).tools}
        for name in ("move", "tag", "grab_flag", "wait"):
            rationale = tools[name]["rationale"]
            assert rationale["anyOf"][0]["type"] == "string" and rationale["anyOf"][0]["maxLength"] == 280
        assert tools["move"]["direction"]["enum"] == ["up", "down", "left", "right"]

    async def test_a_rationale_is_accepted_on_every_action(self):
        async with Client(FakeGame(turns=2)) as c:
            await c.join()
            _, turn = await c.tool("wait_for_turn")
            is_error, ack = await c.tool("wait", tick=turn["tick"], rationale="holding")
        assert not is_error and ack["status"] == "accepted"

    async def test_an_invalid_direction_is_a_rejected_ack_not_a_crash(self):
        from simulator.engine import SimMatch

        async with Client(FakeGame(turns=2, engine=SimMatch())) as c:
            await c.join()
            _, turn = await c.tool("wait_for_turn")
            is_error, ack = await c.tool("move", tick=turn["tick"], direction="north")
        assert not is_error and ack == {"status": "rejected", "tick": turn["tick"], "reason": "invalid_message"}

    def test_make_route_carries_real_risk_cells_and_blocked_reason(self):
        route = make_route("r", "tag", None, 0, risks=((2, "blue-1", (9, 6)),), blocked_reason="occupied")
        assert route["risks"] == [{"cell": {"x": 9, "y": 6}, "step": 2, "enemy_id": "blue-1"}]
        assert route["blocked_now"] is True and route["blocked_reason"] == "occupied"
        assert make_route("r", "tag", None, 0)["blocked_now"] is False
