"""CTF options and decision policy: state -> options for the Decider, label -> tool call, answer combining.

The Decider's tool-selection question (the `action` choice) is built by `decider_patterns.ToolSelectionGate`.
This module adds two CTF-specific yes/no questions (`threat`, `route_bad`) that share the same encoded state,
and a small pure policy, `combine_answers`, that merges the three answers.
"""

from __future__ import annotations

from typing import Any

from decider_patterns import Decider, ToolSelectionGate
from decider_patterns.tool_selection import ACTION_KEY, parse_choice, parse_noul
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

THREAT_YES = 0.5  # P(true) from the `threat` noul at which the threat rule fires


ROUTE_BAD_YES = 0.5  # P(true) from the `route_bad` noul at which the avoid rule fires


DECIDER_INSTRUCTIONS = (
    "You are a player in a Capture the Flag game. Choose the single best action for this "
    "tick. Reach the enemy flag, grab it, and carry it back to your own base. The options "
    "are routes offered by the server: choose between them by weighing length, whether "
    "they are blocked right now, and the risk of being tagged along the way. Tag enemies "
    "inside your own field when it helps, and defend your flag when it is threatened. "
    "Your current plan, if any, is shown, and routes that continue your current path are "
    "marked; route ids are recomputed every tick and do not mean the same path from one tick "
    "to the next. Keep or change the plan based on the current situation."
)


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def route_enemy_id(route: dict[str, Any]) -> str | None:
    """Enemy id of a `tag:<enemy_id>` route."""
    route_id = route["id"]
    return route_id[4:] if route_id.startswith("tag:") and len(route_id) > 4 else None


def describe_route(route: dict[str, Any]) -> str:
    """Neutral, fact-only description of one server route."""
    goal = str(route.get("goal", "unknown"))
    enemy_id = route_enemy_id(route)
    parts = [f"goal {goal} {enemy_id}" if goal == "tag" and enemy_id else f"goal {goal}"]
    length, first_step = route.get("length"), route.get("first_step")
    if goal == "tag" and length == 0 and first_step is None:
        parts.append("adjacent now: tag immediately")
    else:
        if isinstance(length, int):
            parts.append(plural(length, "step"))
        if first_step:
            parts.append(f"first step {first_step}")
    if route.get("blocked_now"):
        reason = route.get("blocked_reason")
        parts.append(f"BLOCKED now ({reason})" if reason else "BLOCKED now")
    risks = [r for r in route.get("risks") or [] if isinstance(r, dict)]
    if risks:
        detail = "; ".join(f"can be tagged by {r.get('enemy_id', '?')} at step {r.get('step', '?')}" for r in risks)
        parts.append(f"{plural(len(risks), 'risk')}: {detail}")
    if "closest_threat" in route:
        threat = route["closest_threat"]
        if isinstance(threat, dict):
            parts.append(f"closest threat {threat.get('enemy_id', '?')} at step {threat.get('step', '?')}, "
                         f"distance {threat.get('distance', '?')}")
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
                    text += " | would continue your path but is BLOCKED now"
                else:
                    text += f" | continues your current path ({plural(plan_age(turn, plan), 'tick')})"
            elif plan.goal is not None and route.get("goal") == plan.goal:
                text += " | same goal as your current plan, different path"
        options[route["id"]] = text
    if not routes:
        for direction in open_directions(state):
            options[f"move_{direction}"] = f"move one cell {direction}"

    enemy_flag = flag_of(state, enemy_team(team))
    if (me is not None and enemy_flag is not None and enemy_flag.get("carried_by") is None
            and xy(enemy_flag.get("position")) == me):
        options["grab_flag"] = "pick up the enemy flag you are standing on"

    if not routes and me is not None and in_my_field(state, map_info):
        for enemy in enemies(state):
            pos = xy(enemy.get("position"))
            if pos is not None and is_adjacent(me, pos):
                options[f"tag_{enemy['id']}"] = f"tag the adjacent enemy {enemy['id']} (you are in your own field)"

    options["wait"] = "do nothing this tick"
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


THREAT_INSTRUCTIONS = (
    "Is there a threat to your flag right now? Answer true when an enemy is inside your own "
    "field close to your flag, or an enemy is carrying your flag."
)


THREAT_CRITERIA = {
    "true": "an enemy is inside my own field and close to my flag, or is carrying my flag",
    "false": "no enemy is inside my own field close to my flag and none is carrying my flag",
}


ROUTE_BAD_INSTRUCTIONS = (
    "Is the route you would continue now a bad one? Answer true when it is blocked right now "
    "or passes next to an enemy who can tag you."
)


ROUTE_BAD_CRITERIA = {
    "true": "the route I would continue is blocked now, or passes next to an enemy who can tag me",
    "false": "the route I would continue is open and no enemy can tag me along it",
}


EXTRA_QUESTIONS = {
    "threat": Decider.noul(THREAT_INSTRUCTIONS, THREAT_CRITERIA),
    "route_bad": Decider.noul(ROUTE_BAD_INSTRUCTIONS, ROUTE_BAD_CRITERIA),
}


def build_gate(**kwargs: Any) -> ToolSelectionGate:
    """The CTF tool-selection gate: the generic pattern configured with the CTF instructions."""
    return ToolSelectionGate(DECIDER_INSTRUCTIONS, **kwargs)


def build_questions(options: dict[str, str]) -> dict[str, dict[str, Any]]:
    """The three questions sent in ONE Decider request (the state is encoded once and shared)."""
    return build_gate().questions(options, extra=EXTRA_QUESTIONS)


def is_bad(route: dict[str, Any]) -> bool:
    return bool(route.get("blocked_now")) or bool(route.get("risks"))


def _threat_candidates(options: dict[str, str], routes: dict[str, dict[str, Any]]) -> list[str]:
    """Offered, unblocked `tag:<id>` / `tag_<id>` / `defend_flag` labels."""
    found = []
    for label in options:
        if label.startswith(("tag:", "tag_")) or label == "defend_flag":
            if not (label in routes and routes[label].get("blocked_now")):
                found.append(label)
    return found


def _threat_rank(label: str, routes: dict[str, dict[str, Any]]) -> tuple[int, int]:
    """Fallback order: adjacent tag, then shortest tag, then defend_flag."""
    if label == "defend_flag":
        return 2, 0
    length = routes.get(label, {}).get("length")
    length = length if isinstance(length, int) else 0  # tag_<id> fallback options are adjacent
    return (0, 0) if length == 0 else (1, length)


def combine_answers(answers: dict[str, Any], options: dict[str, str], turn: dict[str, Any],
                    plan: Plan | None = None, *,
                    threat_threshold: float = THREAT_YES,
                    route_bad_threshold: float = ROUTE_BAD_YES) -> tuple[str | None, float, str]:
    """Merge the three decider answers into (label, confidence, reason) (pure).

    reason "threat": the threat noul fired and a tag/defend option is offered; confidence is the
    threat probability. reason "avoid blocked/risky": the choice picked a blocked or risky route,
    `route_bad` fired and a same-goal route that is not blocked and has no risks is offered;
    confidence is the `route_bad` probability. reason "choice": the choice's own pick with its
    own confidence. The decider's answers are combined; no strategy is hardcoded beyond this.
    """
    state = turn_state(turn)
    routes = {r["id"]: r for r in offered_routes(state)}
    action = answers.get(ACTION_KEY) if isinstance(answers, dict) else None
    pick, pick_conf = parse_choice(action)
    threat = parse_noul(answers.get("threat") if isinstance(answers, dict) else None)
    route_bad = parse_noul(answers.get("route_bad") if isinstance(answers, dict) else None)

    if threat >= threat_threshold:
        candidates = _threat_candidates(options, routes)
        if candidates:
            probs = action.get("probabilities") if isinstance(action, dict) else None
            probs = probs if isinstance(probs, dict) else {}
            scored = [c for c in candidates if isinstance(probs.get(c), (int, float))]
            if scored:
                best = max(scored, key=lambda c: (probs[c], -_threat_rank(c, routes)[0]))
            else:
                best = min(candidates, key=lambda c: _threat_rank(c, routes))
            return best, threat, "threat"

    picked = routes.get(pick) if pick else None
    if picked is not None and is_bad(picked) and route_bad >= route_bad_threshold:
        me = my_position(state)
        safe = [r for r in routes.values()
                if r["id"] in options and r["id"] != pick and r.get("goal") == picked.get("goal")
                and not is_bad(r)]
        if safe:
            best = min(safe, key=lambda r: (not route_continues_plan(plan, me, r),
                                            r["length"] if isinstance(r.get("length"), int) else 10**6,
                                            r["id"]))
            return best["id"], route_bad, "avoid blocked/risky"

    return pick, pick_conf, "choice"


def format_decision(tick: int, label: str | None, confidence: float, reason: str,
                    threat: float, route_bad: float) -> str:
    """Per-tick log line for a decider decision."""
    return (f"[tick {tick}] decider -> {label} ({confidence:.2f}) reason={reason} "
            f"threat={threat:.2f} route_bad={route_bad:.2f}")


def plan_age(turn: dict[str, Any], plan: Plan) -> int:
    """Ticks elapsed since the plan started (0 when unknown)."""
    if plan.since_tick is None:
        return 0
    return max(0, extract_tick(turn) - plan.since_tick)


def plan_active(plan: Plan | None) -> bool:
    return plan is not None and bool(plan.route_id or plan.goal)
