"""Human-readable rendering of a CTF turn for the Decider and for the LLM fallback."""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

from examples.ctf_agent.options import describe_route, plan_active, plan_age, plural
from examples.ctf_agent.state import (
    DIRECTIONS,
    MapInfo,
    Plan,
    enemies,
    extract_tick,
    in_my_field,
    is_carrying,
    my_position,
    my_team,
    offered_routes,
    route_continues_plan,
    turn_state,
    xy,
)


def describe_event(event: Any) -> str:
    if not isinstance(event, dict):
        return str(event)
    etype = str(event.get("type", "event"))
    if etype == "move_blocked":
        parts = [etype, str(event.get("agent_id", "?")), str(event.get("direction", "?"))]
        target = xy(event.get("target"))
        text = " ".join(parts) + (f" -> {target}" if target else "")
        return f"{text}: {event.get('reason', 'unknown')}"
    parts = [etype]
    if "agent_id" in event:
        parts.append(str(event["agent_id"]))
    src, dst = xy(event.get("from")), xy(event.get("to"))
    if src and dst:
        parts.append(f"{src}->{dst}")
    for key, value in event.items():
        if key in ("type", "agent_id", "from", "to"):
            continue
        if xy(value):
            parts.append(f"{key}={xy(value)}")
        elif isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    return " ".join(parts)


def plan_line(turn: dict[str, Any], plan: Plan) -> str:
    target = f", heading to {plan.path[-1]}" if plan.path else ""
    head = f"plan: {plan.goal}{target}, since tick {plan.since_tick}"
    me = my_position(turn_state(turn))
    if not any(route_continues_plan(plan, me, r) for r in offered_routes(turn_state(turn))):
        return f"{head}; no route continues it"
    line = f"{head}, {plural(plan_age(turn, plan), 'tick')}"
    return f"{line}; last action: {plan.last_action}" if plan.last_action else line


def describe_last_action(last_action: Any) -> str | None:
    """The server's `last_action` ({action, tick, result, reason?}) as one line, None when absent."""
    if not isinstance(last_action, dict) or not isinstance(last_action.get("action"), str):
        return None
    result = str(last_action.get("result", "unknown"))
    reason = last_action.get("reason")
    outcome = f"{result} ({reason})" if reason else result
    return f"outcome of your previous action: {last_action['action']} at tick {last_action.get('tick', '?')} -> {outcome}"


def render_state(
    turn: dict[str, Any],
    plan: Plan | None = None,
    map_info: MapInfo | None = None,
    include_last_action: bool = False,
    only_routes: Collection[str] | None = None,
) -> str:
    """Short, human-readable summary of a wait_for_turn result for the decider and the LLM fallback.

    Where we stand relative to the fields needs `map_info` (the turn no longer carries `fields`); without it
    the position is shown plainly rather than guessed.

    The server's `last_action` is rendered only with `include_last_action` (the LLM fallback): the decider is a
    trained model and its input text must not change until that wording has been evaluated.

    `only_routes` lists just those route ids (the options the Decider is offered after filtering); the
    LLM fallback leaves it unset and sees every route.
    """
    state = turn_state(turn)
    me = my_position(state)
    lines = [f"tick: {extract_tick(turn)}"]

    if me is not None:
        where = ""
        if map_info is not None:
            if in_my_field(state, map_info):
                where = ", in your own field"
            elif me[0] == map_info.neutral_x:
                where = ", on the neutral line"
            else:
                where = ", in the enemy field"
        lines.append(f"you: {state.get('self')} ({my_team(state)}) at {me}{where}")
    lines.append(f"carrying the flag: {'yes' if is_carrying(state) else 'no'}")
    last_action = describe_last_action(turn.get("last_action")) if include_last_action else None
    if last_action:
        lines.append(last_action)
    if plan_active(plan):
        lines.append(plan_line(turn, plan))

    score = state.get("score")
    if isinstance(score, dict):
        lines.append("score: " + ", ".join(f"{k} {v}" for k, v in score.items()))

    for enemy in enemies(state):
        pos = xy(enemy.get("position"))
        lines.append(f"enemy {enemy.get('id')} at {pos}, carrying: {'yes' if enemy.get('carrying') is not None else 'no'}")

    mates = [t for t in state.get("teammates") or [] if t]
    if mates:
        lines.append(f"teammates: {json.dumps(mates)}")

    routes = [r for r in offered_routes(state) if only_routes is None or r["id"] in only_routes]
    if routes:
        lines.append("routes:")
        lines.extend(f"  {route['id']}: {describe_route(route)}" for route in routes)
    else:
        moves = state.get("moves") if isinstance(state.get("moves"), dict) else {}
        if moves:
            lines.append("moves: " + ", ".join(f"{d}: {moves.get(d, 'unknown')}" for d in DIRECTIONS))

    events = [describe_event(e) for e in turn.get("events") or []]
    if events:
        lines.append("recent events: " + "; ".join(events))
    return "\n".join(lines)


def describe_match_over(turn: dict[str, Any]) -> str:
    """One log line for a `match_over` result. `winner` is null when the match was stopped early."""
    reason = turn.get("reason") if isinstance(turn.get("reason"), str) else "unknown"
    winner = turn.get("winner")
    verdict = f"winner {winner}" if isinstance(winner, str) and winner else "no winner"
    score = turn.get("score")
    tail = ", score " + ", ".join(f"{k} {v}" for k, v in score.items()) if isinstance(score, dict) else ""
    return f"match over ({reason}): {verdict}{tail}"


def action_rejection(outcome: dict[str, Any]) -> str | None:
    """The reason of a rejected action ack (`status: rejected`), `unknown` if it has none; None otherwise.

    Rejections are normal results, not MCP errors. New in v2: `match_over`, `match_not_started`, `not_joined`.
    """
    if outcome.get("status") != "rejected":
        return None
    reason = outcome.get("reason")
    return reason if isinstance(reason, str) and reason else "unknown"
