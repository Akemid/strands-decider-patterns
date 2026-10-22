"""CTF options and decision policy: state -> options for the Decider, label -> tool call.

Code decides FACTS, the Decider decides JUDGMENT. Whether a route is blocked or risky, and whether my flag is
threatened, are facts the state already contains, so `filter_options` applies them before the request is built
and the Decider only judges what is left with ONE question: the `action` choice built by
`decider_patterns.ToolSelectionGate`.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from decider_patterns import FilterResult, ToolSelectionGate, narrow_options
from decider_patterns.tool_selection import ACTION_KEY, FilterReason, parse_choice
from examples.ctf_agent.state import (
    DIRECTIONS,
    MapInfo,
    Plan,
    enemies,
    enemy_team,
    extract_tick,
    flag_of,
    in_my_field,
    is_adjacent,
    my_position,
    my_team,
    offered_routes,
    open_directions,
    route_continues_plan,
    turn_state,
    xy,
)

REFERENCE_DEFEND_RADIUS = 4  # BFS steps that count as "near my base" on a REFERENCE_MAP_WIDTH-wide map
REFERENCE_MAP_WIDTH = 15

DECIDER_INSTRUCTIONS = (
    "Capture the Flag: pick the single best option for this tick (grab the enemy flag and carry it home, "
    "tag enemies in your field, defend your flag), weighing length, blocked routes and risk, and keep your "
    "plan unless the situation changed."
)


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def route_enemy_id(route: dict[str, Any]) -> str | None:
    """Enemy id of a `tag:<enemy_id>` route."""
    route_id = route["id"]
    return route_id[4:] if route_id.startswith("tag:") and len(route_id) > 4 else None


def describe_route(route: dict[str, Any]) -> str:
    """Compact, fact-only description of one server route."""
    goal = str(route.get("goal", "unknown"))
    enemy_id = route_enemy_id(route)
    parts = [f"goal {goal} {enemy_id}" if goal == "tag" and enemy_id else f"goal {goal}"]
    length, first_step = route.get("length"), route.get("first_step")
    if goal == "tag" and length == 0 and first_step is None:
        parts.append("adjacent, tag now")
    else:
        if isinstance(length, int):
            parts.append(plural(length, "step"))
        if first_step:
            parts.append(f"first {first_step}")
    if route.get("blocked_now"):
        reason = route.get("blocked_reason")
        parts.append(f"BLOCKED ({reason})" if reason else "BLOCKED")
    risks = [r for r in route.get("risks") or [] if isinstance(r, dict)]
    if risks:
        parts.append("risk " + ", ".join(f"{r.get('enemy_id', '?')}@{r.get('step', '?')}" for r in risks))
    threat = route.get("closest_threat")
    if isinstance(threat, dict):
        parts.append(f"threat {threat.get('enemy_id', '?')} dist {threat.get('distance', '?')}")
    return ", ".join(parts)


def build_options(turn: dict[str, Any], plan: Plan | None = None,
                  map_info: MapInfo | None = None) -> dict[str, str]:
    """Options for the decider `choice` question, built from the state. Always includes `wait`.

    One option per server route (label = route id). Without routes, falls back to one
    `move_<dir>` option per open direction (and adjacent tags). No option is ever recommended.
    With a `plan`, every route whose path continues the plan's path gets a neutral note, and
    other routes with the same goal are marked as a different path. The adjacent-`tag_` fallback needs
    `map_info` (to know we stand in our own field); without it no tag option is invented.
    """
    state = turn_state(turn)
    me, team = my_position(state), my_team(state)

    options: dict[str, str] = {}
    routes = offered_routes(state)
    for route in routes:
        text = describe_route(route)
        if plan_active(plan):
            if route_continues_plan(plan, me, route):
                if route.get("blocked_now"):
                    text += " | your plan's path, now blocked"
                else:
                    text += f" | your plan's path ({plural(plan_age(turn, plan), 'tick')})"
            elif plan.goal is not None and route.get("goal") == plan.goal:
                text += " | same goal as your plan, other path"
        options[route["id"]] = text
    if not routes:
        for direction in open_directions(state):
            options[f"move_{direction}"] = f"move one cell {direction}"

    enemy_flag = flag_of(state, enemy_team(team))
    if (me is not None and enemy_flag is not None and enemy_flag.get("carried_by") is None
            and xy(enemy_flag.get("position")) == me):
        options["grab_flag"] = "pick up the enemy flag here"

    if not routes and me is not None and in_my_field(state, map_info):
        for enemy in enemies(state):
            pos = xy(enemy.get("position"))
            if pos is not None and is_adjacent(me, pos):
                options[f"tag_{enemy['id']}"] = f"tag adjacent enemy {enemy['id']}"

    options["wait"] = "do nothing"
    return options


def label_to_call(label: str, tick: int, turn: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Map a chosen option label to (raw MCP tool name, arguments).

    Route labels need `turn` to look up the route's first step.
    """
    if label.startswith("move_") and label[5:] in DIRECTIONS:
        return "move", {"direction": label[5:], "tick": tick}
    if label.startswith("tag_") and len(label) > 4:
        return "tag", {"enemy_id": label[4:], "tick": tick}
    if label == "grab_flag":
        return "grab_flag", {"tick": tick}
    if label == "wait":
        return "wait", {"tick": tick}
    route = next((r for r in offered_routes(turn_state(turn or {})) if r["id"] == label), None)
    if route is not None:
        rationale = f"route {label}"
        first_step = route.get("first_step")
        if first_step in DIRECTIONS:
            return "move", {"direction": first_step, "tick": tick, "rationale": rationale}
        if route.get("length") == 0:  # already on the route's target (first_step is null)
            enemy_id = route_enemy_id(route)
            if enemy_id:
                return "tag", {"enemy_id": enemy_id, "tick": tick, "rationale": rationale}
            if route.get("goal") == "enemy_flag":
                return "grab_flag", {"tick": tick, "rationale": rationale}
            return "wait", {"tick": tick, "rationale": rationale}  # own_base / defend_flag: stay put
    raise ValueError(f"unknown option label: {label!r}")


def build_gate(**kwargs: Any) -> ToolSelectionGate:
    """The CTF tool-selection gate: the generic pattern configured with the CTF instructions."""
    return ToolSelectionGate(DECIDER_INSTRUCTIONS, **kwargs)


def build_questions(options: dict[str, str]) -> dict[str, dict[str, Any]]:
    """The ONE question sent to the Decider: the `action` choice over the (already filtered) options."""
    return build_gate().questions(options)


def is_bad(route: dict[str, Any]) -> bool:
    return bool(route.get("blocked_now")) or bool(route.get("risks"))


def defend_radius(map_info: MapInfo | None) -> int | None:
    """BFS steps from my flag base within which an enemy counts as a threat; None when the map is unknown.

    Approximates the game's defend radius: 4 steps on a 15-wide map, scaled linearly by the map width (the span
    of the team fields), never below 1. The width comes only from the MCP map, so no map means no radius.
    """
    if map_info is None or not map_info.fields:
        return None
    width = max(hi for _, hi in map_info.fields.values()) - min(lo for lo, _ in map_info.fields.values()) + 1
    return max(1, round(REFERENCE_DEFEND_RADIUS * width / REFERENCE_MAP_WIDTH))

def steps_within(start: tuple[int, int], goal: tuple[int, int], limit: int, map_info: MapInfo) -> bool:
    """Whether `goal` is reachable from `start` in at most `limit` grid steps around the walls (BFS)."""
    low = min(lo for lo, _ in map_info.fields.values())
    high = max(hi for _, hi in map_info.fields.values())
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == goal:
            return True
        if dist == limit:
            continue
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cell = (x + dx, y + dy)
            if cell in seen or cell in map_info.walls or not (low <= cell[0] <= high) or cell[1] < 0:
                continue
            seen.add(cell)
            queue.append((cell, dist + 1))
    return False


def flag_threatened(state: dict[str, Any], map_info: MapInfo | None) -> bool:
    """My flag is carried by an enemy, or an enemy inside my field is within the defend radius of my base (pure).

    Without the map only the carried flag can be told (the field and the walls are unknown).
    """
    team = my_team(state)
    mine = flag_of(state, team)
    if mine is None:
        return False
    carrier = mine.get("carried_by")
    mates = {a.get("id") for a in state.get("agents") or [] if isinstance(a, dict) and a.get("team") == team}
    if carrier is not None and carrier not in mates:
        return True
    base = xy(mine.get("position")) or (map_info.flag_bases.get(team or "") if map_info else None)
    bounds = map_info.fields.get(team or "") if map_info is not None else None
    if map_info is None or base is None or bounds is None:
        return False
    radius = defend_radius(map_info)
    if radius is None:
        return False
    for enemy in enemies(state):
        pos = xy(enemy.get("position"))
        if pos is not None and bounds[0] <= pos[0] <= bounds[1] and steps_within(base, pos, radius, map_info):
            return True
    return False


def filter_options(options: dict[str, str], turn: dict[str, Any], map_info: MapInfo | None = None) -> FilterResult:
    """Narrow the options with code-decidable facts before the Decider judges them (pure).

    Flag threatened and an unblocked `tag:*` / `tag_*` / `defend_flag` option exists: offer only those and
    `wait`. Otherwise drop the bad routes (blocked now, or risks). `wait` always stays; when nothing but `wait`
    would remain, the full set is offered ("lesser_evil"). Non-route options (`move_*`, `grab_flag`) are not routes
    and are never dropped by the bad-route rule.
    """
    state = turn_state(turn)
    routes = {r["id"]: r for r in offered_routes(state)}
    if flag_threatened(state, map_info):
        defenders = {label for label in options
                     if label.startswith(("tag:", "tag_")) or label == "defend_flag"
                     if not routes.get(label, {}).get("blocked_now")}
        if defenders:
            return narrow_options(options, defenders.__contains__)
    return narrow_options(options, lambda label: label not in routes or not is_bad(routes[label]))


def combine_answers(answers: Any, reason: FilterReason = "unfiltered") -> tuple[str | None, float, str]:
    """The decision from the Decider's answers (pure): the `action` choice, its confidence, and the filter reason.

    The Decider only judges what the filter left, so there is nothing to merge or override. Unusable answers
    give `(None, 0.0, reason)`. Whether the pick is an offered option is checked by `ToolSelectionGate.route`.
    """
    action = answers.get(ACTION_KEY) if isinstance(answers, dict) else None
    label, confidence = parse_choice(action)
    return label, confidence, reason


def format_decision(tick: int, label: str | None, confidence: float, reason: str) -> str:
    """Per-tick log line for a decider decision."""
    return f"[tick {tick}] decider -> {label} ({confidence:.2f}) reason={reason}"


def plan_age(turn: dict[str, Any], plan: Plan) -> int:
    """Ticks elapsed since the plan started (0 when unknown)."""
    if plan.since_tick is None:
        return 0
    return max(0, extract_tick(turn) - plan.since_tick)


def plan_active(plan: Plan | None) -> bool:
    return plan is not None and bool(plan.route_id or plan.goal)
