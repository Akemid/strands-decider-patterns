"""CTF turn state: parsing `wait_for_turn` payloads, the static map, and the loop-owned path plan.

Everything here is pure (no I/O). MCP turn contract v2: the turn carries `tick`, `events`, `last_action` and
a `state` (agents, flags, score, self, self_position, moves, teammates, routes); the static map comes once
from `join_match.map`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

# ---------------------------------------------------------------------------
# Tools and directions
# ---------------------------------------------------------------------------

ACTION_TOOLS = ("move", "tag", "grab_flag", "wait")


DIRECTIONS = ("up", "down", "left", "right")


def parse_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Decode an MCPToolResult into a dict. JSON text blocks are parsed; other text is wrapped."""
    for block in result.get("content") or []:
        text = block.get("text") if isinstance(block, dict) else None
        if text is None:
            continue
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
        return decoded if isinstance(decoded, dict) else {"value": decoded}
    return {}


def turn_state(turn: dict[str, Any]) -> dict[str, Any]:
    """The game state: `turn["state"]`, falling back to `turn["observation"]` if absent."""
    for key in ("state", "observation"):
        value = turn.get(key)
        if isinstance(value, dict):
            return value
    return {}


def extract_tick(turn: dict[str, Any]) -> int:
    """The tick number every action must echo back. Top-level first, then inside state."""
    for source in (turn, turn_state(turn)):
        tick = source.get("tick")
        if isinstance(tick, int) and not isinstance(tick, bool):
            return tick
    raise ValueError("no tick found in wait_for_turn result")


def xy(value: Any) -> tuple[int, int] | None:
    """Coerce an {x, y} dict into a tuple."""
    if isinstance(value, dict) and isinstance(value.get("x"), int) and isinstance(value.get("y"), int):
        return value["x"], value["y"]
    return None


def my_entry(state: dict[str, Any]) -> dict[str, Any] | None:
    """The agent entry whose id equals `state.self`."""
    for agent in state.get("agents") or []:
        if isinstance(agent, dict) and agent.get("id") == state.get("self"):
            return agent
    return None


def my_position(state: dict[str, Any]) -> tuple[int, int] | None:
    return xy(state.get("self_position")) or xy((my_entry(state) or {}).get("position"))


def my_team(state: dict[str, Any]) -> str | None:
    team = (my_entry(state) or {}).get("team")
    return team if isinstance(team, str) else None


def flag_of(state: dict[str, Any], team: str | None) -> dict[str, Any] | None:
    for flag in state.get("flags") or []:
        if isinstance(flag, dict) and flag.get("team") == team:
            return flag
    return None


def enemy_team(team: str | None) -> str | None:
    return {"red": "blue", "blue": "red"}.get(team or "")


def enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Connected agents on the other team."""
    team = my_team(state)
    return [
        a for a in state.get("agents") or []
        if isinstance(a, dict) and a.get("team") != team and a.get("team") is not None
        and not a.get("disconnected") and a.get("id") != state.get("self")
    ]


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class MapInfo:
    """The static board, captured once from `join_match.map` (or `get_map`). It never changes in a match.

    `fields` maps a team to its `(min_x, max_x)` columns, `neutral_x` is the neutral column, `flag_bases`
    maps a team to its base cell. `walls` is kept for completeness: movement and routes are server-computed.
    """

    fields: dict[str, tuple[int, int]]
    neutral_x: int | None
    walls: frozenset[tuple[int, int]]
    flag_bases: dict[str, tuple[int, int]]


def parse_map(raw: Any) -> MapInfo | None:
    """A `MapInfo` from a `join_match.map` / `get_map` payload; None when it has no usable `fields` (pure)."""
    raw_fields = raw.get("fields") if isinstance(raw, dict) else None
    if not isinstance(raw_fields, dict):
        return None
    fields: dict[str, tuple[int, int]] = {}
    for team in ("red", "blue"):
        bounds = raw_fields.get(team)
        if isinstance(bounds, dict) and is_int(bounds.get("min_x")) and is_int(bounds.get("max_x")):
            fields[team] = (bounds["min_x"], bounds["max_x"])
    if not fields:
        return None
    walls = raw.get("walls")
    bases = raw.get("flag_bases")
    return MapInfo(
        fields=fields,
        neutral_x=raw_fields["neutral_x"] if is_int(raw_fields.get("neutral_x")) else None,
        walls=frozenset(c for c in map(xy, walls if isinstance(walls, list) else []) if c is not None),
        flag_bases={
            team: cell
            for team, cell in (
                (t, xy(bases.get(t) if isinstance(bases, dict) else None)) for t in ("red", "blue")
            )
            if cell is not None
        },
    )


def in_my_field(state: dict[str, Any], map_info: MapInfo | None = None) -> bool:
    """Whether we stand in our own field. False without the map: the turn state no longer says."""
    me = my_position(state)
    bounds = map_info.fields.get(my_team(state) or "") if map_info is not None else None
    if me is None or bounds is None:
        return False
    return bounds[0] <= me[0] <= bounds[1]


def is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def is_carrying(state: dict[str, Any]) -> bool:
    return (my_entry(state) or {}).get("carrying") is not None


def open_directions(state: dict[str, Any]) -> list[str]:
    moves = state.get("moves")
    moves = moves if isinstance(moves, dict) else {}
    return [d for d in DIRECTIONS if moves.get(d) == "open"]


def offered_routes(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Well-formed server routes (dicts with a string id), in server order."""
    routes = state.get("routes")
    return [r for r in routes if isinstance(r, dict) and isinstance(r.get("id"), str)] if isinstance(routes, list) else []


Cell = tuple[int, int]


@dataclass
class Plan:
    """Loop-owned memory of the path the agent last acted on. Informational only: the model
    still decides every tick and may keep or change it.

    Route ids are recomputed by the server every tick, so continuity is judged by `path` (the
    executed route's path when it was chosen) and `position` (where the agent stood then).
    `route_id` is kept for display and logging only.
    """

    route_id: str | None = None
    goal: str | None = None
    since_tick: int | None = None
    last_action: str | None = None
    path: tuple[Cell, ...] | None = None
    position: Cell | None = None


def as_cells(path: Any) -> tuple[Cell, ...] | None:
    """Normalise a path (cells as tuples or {x, y} dicts) to a tuple of tuples; None if malformed/empty."""
    if not isinstance(path, (list, tuple)) or not path:
        return None
    cells: list[Cell] = []
    for cell in path:
        if isinstance(cell, dict):
            cell = xy(cell)
        if (not isinstance(cell, (list, tuple)) or len(cell) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in cell)):
            return None
        cells.append((cell[0], cell[1]))
    return tuple(cells)


def _same_prefix(expected: tuple[Cell, ...], route: tuple[Cell, ...]) -> bool:
    """True when both agree on their overlapping prefix (both may be truncated); needs >= 1 cell."""
    overlap = min(len(expected), len(route))
    return overlap >= 1 and expected[:overlap] == route[:overlap]


def continues_path(prev_path: Any, prev_position: Any, my_position: Any, route_path: Any) -> bool:
    """Whether `route_path` (from `my_position`) continues the previously executed path (pure).

    Moved one step along the previous path: the route must start with the rest of that path.
    Did not move (e.g. move_blocked): the route must start with the previous path again.
    Anything else (respawned, tagged, moved elsewhere) or malformed input: False.
    """
    prev, route = as_cells(prev_path), as_cells(route_path)
    here, before = as_cells([my_position]), as_cells([prev_position])
    if prev is None or route is None or here is None:
        return False
    here_cell = here[0]
    if here_cell == prev[0] and (before is None or before[0] != here_cell):
        return _same_prefix(prev[1:], route)
    if before is not None and here_cell == before[0]:
        return _same_prefix(prev, route)
    return False


def route_continues_plan(plan: Plan | None, my_position: Cell | None, route: dict[str, Any]) -> bool:
    """Whether an offered route's path continues the plan's stored path."""
    return plan is not None and continues_path(plan.path, plan.position, my_position, route.get("path"))


def route_path(turn: dict[str, Any], route_id: str | None) -> tuple[Cell, ...] | None:
    """Path of the offered route with this id, if any."""
    route = next((r for r in offered_routes(turn_state(turn)) if r["id"] == route_id), None)
    return as_cells(route.get("path")) if route else None


def update_plan(plan: Plan, executed_route_id: str | None, executed_goal: str | None,
                action_text: str, tick: int, executed_path: Any = None,
                position: Cell | None = None) -> Plan:
    """New plan after an executed action (pure).

    A route action keeps `since_tick` only while the goal is unchanged AND the executed route's
    path continues the plan's path (the route id is irrelevant: ids are recomputed every tick).
    Otherwise it starts a new plan at this tick. `position` is where the agent stood when
    acting. `grab_flag` clears the plan; any other non-route action (e.g. `wait`) only records
    `last_action`.
    """
    if executed_route_id is None:
        if action_text.split(" ", 1)[0] == "grab_flag":
            return Plan(last_action=action_text)
        return replace(plan, last_action=action_text)
    keeps = (plan.since_tick is not None and plan.goal is not None and plan.goal == executed_goal
             and continues_path(plan.path, plan.position, position, executed_path))
    since = plan.since_tick if keeps else tick
    return Plan(executed_route_id, executed_goal, since, action_text, as_cells(executed_path), position)


def plan_suffix(plan: Plan) -> str:
    """Short per-tick log suffix, empty when there is no plan."""
    return f" [plan {plan.route_id} since {plan.since_tick}]" if plan.route_id else ""


def describe_action(tool: str, arguments: dict[str, Any]) -> str:
    """Short text for an executed action, e.g. `move down` or `tag blue-2`."""
    if tool == "move" and arguments.get("direction"):
        return f"move {arguments['direction']}"
    if tool == "tag" and arguments.get("enemy_id"):
        return f"tag {arguments['enemy_id']}"
    return tool


def route_goal(turn: dict[str, Any], route_id: str | None) -> str | None:
    """Goal of the offered route with this id, if any."""
    route = next((r for r in offered_routes(turn_state(turn)) if r["id"] == route_id), None)
    return route.get("goal") if route else None


def extract_action_call(messages: Any) -> tuple[str, dict[str, Any]] | None:
    """First (tool name, input) action toolUse block in an agent's messages, if any."""
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else []:
            use = block.get("toolUse") if isinstance(block, dict) else None
            if isinstance(use, dict) and use.get("name") in ACTION_TOOLS:
                tool_input = use.get("input")
                return use["name"], tool_input if isinstance(tool_input, dict) else {}
    return None


def infer_executed_route(call: tuple[str, dict[str, Any]] | None, turn: dict[str, Any],
                         plan: Plan) -> str | None:
    """Which offered route an LLM action call corresponds to, or None for non-route actions.

    Order: `rationale` of the form `route <id>` naming an offered route; `tag` -> `tag:<id>`;
    `move` -> route whose first step matches (current plan's goal preferred, then lowest id).
    """
    if call is None:
        return None
    tool, arguments = call
    routes = offered_routes(turn_state(turn))
    offered = {r["id"]: r for r in routes}

    match = re.search(r"route\s+(\S+)", str(arguments.get("rationale") or ""))
    if match:
        candidate = match.group(1).rstrip(".,;:!)\"'`")
        if candidate in offered:
            return candidate

    if tool == "tag":
        candidate = f"tag:{arguments.get('enemy_id')}"
        return candidate if candidate in offered else None
    if tool == "move":
        matches = sorted((r for r in routes if r.get("first_step") == arguments.get("direction")),
                         key=lambda r: (r.get("goal") != plan.goal, r["id"]))
        return matches[0]["id"] if matches else None
    return None
