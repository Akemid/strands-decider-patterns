"""FakeGame driven by the dynamic SimMatch engine, over the real MCP protocol (in memory)."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession

from simulator.engine import SimMatch
from simulator.fake_ctf_mcp import FakeGame, in_memory_transport


async def call(session, name, **arguments):
    result = await session.call_tool(name, arguments)
    return json.loads(result.content[0].text)


async def play(game, policy, max_turns=200):
    """Join, then act with `policy(state) -> (tool, args)` until match_over. Returns the last payload."""
    async with AsyncExitStack() as stack:
        read, write, _ = await stack.enter_async_context(in_memory_transport(game.server))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        await call(session, "join_match", match_id="m1", team="red")
        for _ in range(max_turns):
            turn = await call(session, "wait_for_turn")
            if turn["status"] == "match_over":
                return turn
            tool, args = policy(turn["state"])
            await call(session, tool, tick=turn["tick"], **args)
    raise AssertionError("match never ended")


def greedy(state):
    me = next(a for a in state["agents"] if a["id"] == state["self"])
    enemy_flag = next(f for f in state["flags"] if f["team"] == "blue")
    if me["carrying"] is None and state["self_position"] == enemy_flag["position"]:
        return "grab_flag", {}
    routes = [r for r in state["routes"] if r["goal"] in ("enemy_flag", "own_base") and not r["blocked_now"]]
    safe = [r for r in routes if not r["risks"]] or routes
    if safe and safe[-1]["first_step"]:
        return "move", {"direction": safe[-1]["first_step"]}
    return "wait", {}


def waiting(state):
    return "wait", {}


async def test_a_greedy_agent_can_capture_the_flag_and_win():
    turn = await play(FakeGame(turns=120, engine=SimMatch(max_ticks=120, seed=0)), greedy)
    assert turn["winner"] == "red" and turn["score"] == {"red": 1, "blue": 0}


async def test_an_idle_agent_ends_in_a_tie_at_the_tick_limit():
    game = FakeGame(turns=5, engine=SimMatch(max_ticks=5))
    turn = await play(game, waiting)
    assert turn["status"] == "match_over" and turn["winner"] == "tie"
    assert game.tick == 5


async def test_turn_state_reflects_the_agents_own_moves():
    positions = []

    def mover(state):
        positions.append(state["self_position"]["x"])
        return "move", {"direction": "right"}

    await play(FakeGame(turns=4, engine=SimMatch(max_ticks=4)), mover)
    assert positions == [1, 2, 3, 4]
