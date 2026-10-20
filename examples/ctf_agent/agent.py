"""CTF example agent: plain Python owns the game loop, a local Strands Decider picks the action.

Pattern: #tool-selection. Each tick the loop renders the state, builds one option per server route, and asks
the Decider (through `decider_patterns.ToolSelectionGate`) which action tool to call. Only when the Decider is
unsure (or unreachable) does a Strands `Agent` on Amazon Bedrock run, once for that tick, restricted to the
action tools.

    uv run strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8099
    uv run python simulator/serve.py                      # in another terminal
    uv run python -m examples.ctf_agent.agent --match demo --metrics runs/decider.jsonl

`STRANDS_DECIDER_URL` overrides the Decider endpoint, `CTF_MCP_URL` the game MCP endpoint.

`--llm-only` is an ablation: same loop, state rendering and options, but the LLM decides every tick and the
Decider is never consulted (no Decider server needed).

One Decider request carries THREE questions about the same encoded state: `action` (choice over the routes),
`threat` and `route_bad` (yes/no). `combine_answers` merges them; see `options.py`.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import threading
import time
import uuid
from typing import Any

from strands import Agent
from strands.tools.mcp import MCPClient

from decider_patterns import (
    DECIDER_ERRORS,
    Decider,
    ServerUnavailable,
    StopAfterToolsHook,
    TickCancelHandler,
)
from decider_patterns.client import ask_serialized
from decider_patterns.timing import llm_timeout_s, remaining_ms, run_with_timeout
from decider_patterns.tool_selection import (
    DEFAULT_LLM_BUDGET_MS,
    DEFAULT_SAFETY_MARGIN_MS,
    DEFAULT_THRESHOLD,
    parse_noul,
    timeout_action,
)
from examples.ctf_agent.metrics import MetricsWriter, TickRecord
from examples.ctf_agent.options import (
    THREAT_YES,
    build_gate,
    build_options,
    build_questions,
    combine_answers,
    format_decision,
    label_to_call,
)
from examples.ctf_agent.process import exit_if_threads_linger, run_guarded, with_client
from examples.ctf_agent.render import action_rejection, describe_match_over, render_state
from examples.ctf_agent.state import (
    ACTION_TOOLS,
    MapInfo,
    Plan,
    describe_action,
    extract_action_call,
    extract_tick,
    infer_executed_route,
    my_position,
    parse_map,
    parse_tool_result,
    plan_suffix,
    route_goal,
    route_path,
    turn_state,
    update_plan,
)

MODELS = {
    "haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "opus": "global.anthropic.claude-opus-4-6-v1",
    "nova-lite": "global.amazon.nova-2-lite-v1:0",
}
SYSTEM_PROMPT = (
    "You are AmazingPlayer, a Capture the Flag game agent competing as a member of a "
    "team. You are direct and tactical. You are given the current game state for ONE tick. "
    "Pick the single best action and call exactly one action tool with the tick number "
    "provided. Do not call any other tool and do not answer with text only."
)
DEFAULT_MCP_URL = "http://127.0.0.1:8765/mcp"  # the bundled simulator; override with CTF_MCP_URL
MCP_SERVER_NAME = "ctf-game"
MCP_SERVERS = {
    MCP_SERVER_NAME: {
        "url": os.environ.get("CTF_MCP_URL", DEFAULT_MCP_URL),
        "transport": "streamable-http",
    },
}


# Raw tool names (VERIFIED via list_tools_sync). No server-name prefix on direct calls.
TOOL_JOIN = "join_match"
TOOL_GET_MAP = "get_map"
TOOL_WAIT_FOR_TURN = "wait_for_turn"
MAX_CONSECUTIVE_TOOL_ERRORS = 5

LLM_PLAN_NOTE = (
    "Your current plan, if any, is shown, and routes that continue your current path are "
    "marked; route ids are recomputed every tick and do not mean the same path from one tick "
    "to the next. Keep or change the plan based on the current situation. When you call move or tag "
    "for a route, set `rationale` to `route <route_id>` (e.g. `route enemy_flag_direct`)."
)


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def turn_remaining_ms(turn: dict[str, Any], elapsed_ms: float) -> float:
    """`remaining_ms` for a turn: `deadline_remaining_ms`, else `deadline_ms`, else 5000."""
    base = number(turn.get("deadline_remaining_ms"))
    if base is None:
        base = number(turn.get("deadline_ms"))
    return remaining_ms(base, elapsed_ms)


def warmup_question() -> tuple[str, dict[str, dict[str, Any]]]:
    """A representative (state, questions) pair for the decider warm-up (pure, no server)."""
    def route(rid, goal, step, length, risks=(), blocked=False, threat=None):
        return {"id": rid, "goal": goal, "first_step": step, "length": length, "path": [],
                "blocked_now": blocked, "blocked_reason": "contention" if blocked else None,
                "risks": [{"step": st, "enemy_id": en} for st, en in risks],
                "closest_threat": (None if threat is None
                                   else {"enemy_id": threat[0], "step": threat[1], "distance": threat[2]})}

    def agent(aid, team, x, y):
        return {"id": aid, "team": team, "position": {"x": x, "y": y}, "carrying": None, "disconnected": False}

    turn = {
        "status": "turn", "tick": 12,
        "state": {
            "tick": 12,
            "agents": [agent("blue-1", "blue", 9, 7), agent("blue-2", "blue", 11, 4),
                       agent("red-1", "red", 5, 6), agent("red-2", "red", 4, 9), agent("red-3", "red", 2, 7)],
            "flags": [{"team": "red", "position": {"x": 1, "y": 7}, "carried_by": None},
                      {"team": "blue", "position": {"x": 13, "y": 7}, "carried_by": None}],
            "score": {"red": 1, "blue": 0},
            "self": "blue-1", "self_position": {"x": 9, "y": 7},
            "teammates": [{"id": "blue-2", "position": {"x": 11, "y": 4}}],
            "routes": [route("enemy_flag_direct", "enemy_flag", "left", 8, [(3, "red-1")], threat=("red-1", 6, 2)),
                       route("enemy_flag_detour", "enemy_flag", "up", 11, threat=("red-2", 4, 4)),
                       route("tag:red-1", "tag", "left", 4),
                       route("home_base", "own_base", "right", 4, blocked=True)],
        },
        "events": [{"type": "moved", "agent_id": "red-1", "from": {"x": 6, "y": 6}, "to": {"x": 5, "y": 6}}],
        "last_action": {"action": "move:left", "tick": 11, "result": "moved"},
    }
    plan = Plan("enemy_flag_direct", "enemy_flag", 9, "move left")
    map_info = parse_map({"fields": {"red": {"min_x": 0, "max_x": 6}, "blue": {"min_x": 8, "max_x": 14},
                                     "neutral_x": 7},
                          "walls": [], "flag_bases": {"red": {"x": 1, "y": 7}, "blue": {"x": 13, "y": 7}}})
    return render_state(turn, plan, map_info), build_questions(build_options(turn, plan, map_info))


def sanitize_name(name: str) -> str:
    """join_match requires [a-z0-9_-]{1,32}."""
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")
    return (cleaned or "refined-decider")[:32]


def report_outcome(outcome: dict[str, Any]) -> None:
    """Print a tool error or a rejected action ack (a normal result with a reason, not an MCP error)."""
    if outcome["_is_error"]:
        print(f"    action error: {outcome}")
    elif action_rejection(outcome):
        print(f"    action rejected: {action_rejection(outcome)}")


class Stats:
    def __init__(self) -> None:
        self.decider_ticks = 0
        self.llm_ticks = 0
        self.budget_ticks = 0
        self.llm_timeouts = 0
        self.latencies_ms: list[float] = []

    def summary(self) -> str:
        avg = f"{sum(self.latencies_ms) / len(self.latencies_ms):.0f} ms" if self.latencies_ms else "n/a"
        return (
            f"ticks decided by decider: {self.decider_ticks}, by LLM fallback: {self.llm_ticks}, "
            f"by budget: {self.budget_ticks}, LLM timeouts: {self.llm_timeouts}, "
            f"decider avg latency: {avg}"
        )


def call_tool(client: MCPClient, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = client.call_tool_sync(f"{name}-{uuid.uuid4().hex[:8]}", name, arguments)
    parsed = parse_tool_result(result)
    # Tool errors set isError; transport errors (e.g. "session not found") only set status="error".
    parsed["_is_error"] = result.get("status") == "error" or bool(result.get("isError"))
    return parsed


def join(client: MCPClient, match: str, name: str) -> tuple[str, MapInfo | None]:
    """Join as red first, then blue. Returns the team joined and the static map (None if absent)."""
    for team in ("red", "blue"):
        outcome = call_tool(client, TOOL_JOIN, {"match_id": match, "team": team, "name": name})
        if not outcome["_is_error"]:
            print(f"joined match {match} as {name} on team {team}")
            return team, parse_map(outcome.get("map"))
        print(f"join as {team} failed: {outcome}")
    raise RuntimeError(f"could not join match {match} on either team")


def fetch_map(client: MCPClient) -> MapInfo | None:
    """The static map from `get_map`: the fallback when `join_match` did not carry it."""
    outcome = call_tool(client, TOOL_GET_MAP, {})
    if outcome["_is_error"]:
        print(f"get_map failed: {outcome}")
        return None
    return parse_map(outcome)


def run_llm_tick(agent_factory, state: str, tick: int, handler: Any | None = None,
                 usage: dict[str, int] | None = None) -> tuple[int, tuple[str, dict[str, Any]] | None]:
    """Run the fallback Agent for one tick (a fresh agent each time, guarded by `handler`).

    Returns (how many action tools it called, the (tool name, input) it executed or None). When `usage` is
    given it is filled with `input_tokens` / `output_tokens` if the model reported them.
    """
    prompt = (
        f"{state}\n\nThis is tick {tick}. Pick ONE of the listed routes or actions and call "
        f"exactly one action tool ({', '.join(ACTION_TOOLS)}) with tick={tick}: move with the "
        "route's first step, tag with the enemy id, grab_flag, or wait.\n" + LLM_PLAN_NOTE
    )
    agent = agent_factory(handler)
    result = agent(prompt)
    tool_metrics = getattr(getattr(result, "metrics", None), "tool_metrics", {}) or {}
    calls = sum(getattr(m, "call_count", 0) for m in tool_metrics.values())
    if usage is not None:
        tokens = getattr(getattr(result, "metrics", None), "accumulated_usage", None) or {}
        usage["input_tokens"] = int(tokens.get("inputTokens", 0))
        usage["output_tokens"] = int(tokens.get("outputTokens", 0))
    return calls, extract_action_call(getattr(agent, "messages", None))


def warm_up(decider: Decider) -> bool:
    """Send one throwaway question through the shared lock. False if the decider is unreachable."""
    state, questions = warmup_question()
    started = time.monotonic()
    try:
        ask_serialized(decider, state, questions)
    except DECIDER_ERRORS as exc:
        print(f"[decider] warm-up failed: {exc}\n    -> LLM fallback for the rest of the match")
        return False
    print(f"decider warm-up: {(time.monotonic() - started) * 1000:.0f} ms (server {decider.last_latency_ms} ms)")
    return True


def build_agent_factory(model: str, action_tools: list[Any], system_prompt: str):
    """Return `handler -> Agent`, the per-tick LLM agent restricted to the action tools."""
    return lambda handler: Agent(
        model=model,
        tools=action_tools,
        system_prompt=system_prompt,
        callback_handler=None,
        interventions=[handler] if handler is not None else [],
        hooks=[StopAfterToolsHook(ACTION_TOOLS)],  # end the turn after the action: no wasted 2nd model call
    )


def play(client: MCPClient, decider: Decider | None, args: argparse.Namespace,
         system_prompt: str = SYSTEM_PROMPT) -> Stats:
    """Run the match loop. With `args.llm_only` the decider is never used (it may be None).

    With `args.metrics` (a path) one JSONL line per tick is written, see `metrics.py`.
    """
    with MetricsWriter(getattr(args, "metrics", None)) as metrics:
        return _play(client, decider, args, system_prompt, metrics)


def _play(client: MCPClient, decider: Decider | None, args: argparse.Namespace, system_prompt: str,
          metrics: MetricsWriter) -> Stats:
    llm_only = bool(getattr(args, "llm_only", False))
    mode = "llm-only" if llm_only else "decider"
    stats = Stats()
    gate = build_gate(threshold=args.threshold, llm_budget_ms=args.llm_budget_ms,
                      safety_margin_ms=args.safety_margin_ms)
    errors = 0
    plan = Plan()

    action_tools = [t for t in client.list_tools_sync() if t.tool_name in ACTION_TOOLS]
    agent_factory = build_agent_factory(MODELS[args.model], action_tools, system_prompt)

    if llm_only:
        print("llm-only mode: decider disabled (no warm-up, no per-tick decider request)")
    elif not args.no_warmup and not warm_up(decider):
        gate.decider_down = True

    _, map_info = join(client, args.match, sanitize_name(args.name))
    if map_info is None:  # an older server: fetch the static map once instead
        map_info = fetch_map(client)
    if map_info is None:
        print("warning: no static map (join_match.map / get_map): field-dependent hints are off")

    while True:
        turn = call_tool(client, TOOL_WAIT_FOR_TURN, {})
        status = turn.get("status")
        if status == "match_over":
            print(describe_match_over(turn))
            return stats
        if status != "turn":  # 'pending' (nothing arrived within the wait cap) or an error
            if turn["_is_error"]:
                errors += 1
                if errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
                    raise RuntimeError(f"wait_for_turn kept failing: {turn}")
            continue
        errors = 0

        tick = extract_tick(turn)
        turn_received = time.monotonic()
        state = render_state(turn, plan, map_info)
        options = build_options(turn, plan, map_info)
        if args.raw_turns:
            print(f"[tick {tick}] raw turn:\n{json.dumps(turn, indent=2, ensure_ascii=False)}")
            print(f"[tick {tick}] state sent to decider:\n{state}")
            print(f"[tick {tick}] options: {sorted(options)}")

        chosen: str | None = None
        confidence = 0.0
        reason, threat_p, route_bad_p = "choice", 0.0, 0.0
        decider_ms: float | None = None
        decider_tokens: int | None = None
        if not gate.decider_down and not llm_only:
            questions = build_questions(options)  # ONE request, three questions
            started = time.monotonic()
            tokens_before = decider.input_tokens
            try:
                answers = ask_serialized(decider, state, questions)
            except DECIDER_ERRORS as exc:
                gate.decider_down = True
                print(f"[decider] {exc}\n    -> LLM fallback for the rest of the match")
            else:
                chosen, confidence, reason = combine_answers(answers, options, turn, plan,
                                                             threat_threshold=args.threat_threshold)
                threat_p, route_bad_p = parse_noul(answers.get("threat")), parse_noul(answers.get("route_bad"))
                decider_ms = (time.monotonic() - started) * 1000
                decider_tokens = decider.input_tokens - tokens_before
                if decider.last_latency_ms is not None:
                    stats.latencies_ms.append(float(decider.last_latency_ms))
                print(f"[tick {tick}] timing: decider wall {(time.monotonic() - started) * 1000:.0f} ms "
                      f"(server {decider.last_latency_ms} ms), deadline_remaining at receipt "
                      f"{turn.get('deadline_remaining_ms')} ms")

        base = {"tick": tick, "mode": mode, "decider_ms": decider_ms, "decider_input_tokens": decider_tokens,
                "label": chosen, "confidence": confidence if chosen is not None else None,
                "reason": None if llm_only else reason}
        decider_down = gate.decider_down
        remaining = turn_remaining_ms(turn, (time.monotonic() - turn_received) * 1000)
        if llm_only:
            path = "llm"  # no decider choice to fall back on: the LLM always runs with the time left
        else:
            path = gate.route(chosen, confidence, options, remaining)

        if path in ("decider", "budget"):
            try:
                tool, arguments = label_to_call(chosen or "", tick, turn)
            except ValueError as exc:
                # Never crash mid-match on an option we cannot map: a wait beats a dead agent.
                print(f"[tick {tick}] {exc} -> wait")
                tool, arguments = "wait", {"tick": tick}
            outcome = call_tool(client, tool, arguments)
            route_id = chosen if route_goal(turn, chosen) is not None else None
            plan = update_plan(plan, route_id, route_goal(turn, route_id), describe_action(tool, arguments), tick,
                               route_path(turn, route_id), my_position(turn_state(turn)))
            if path == "decider":
                stats.decider_ticks += 1
                print(format_decision(tick, chosen, confidence, reason, threat_p, route_bad_p) + plan_suffix(plan))
            else:
                stats.budget_ticks += 1
                print(f"[tick {tick}] budget: {remaining:.0f} ms left < "
                      f"{args.llm_budget_ms + args.safety_margin_ms:.0f} ms needed for LLM -> "
                      f"executing decider choice {chosen} ({confidence:.2f}){plan_suffix(plan)}")
            report_outcome(outcome)
            metrics.write(TickRecord(**base, path=path))
            continue

        if path == "wait":
            outcome = call_tool(client, "wait", {"tick": tick})
            stats.budget_ticks += 1
            plan = update_plan(plan, None, None, "wait", tick)
            why = "decider unavailable" if decider_down else f"no valid decider choice ({chosen}, {confidence:.2f})"
            print(f"[tick {tick}] budget: {remaining:.0f} ms left < "
                  f"{args.llm_budget_ms + args.safety_margin_ms:.0f} ms needed for LLM, {why} -> wait")
            report_outcome(outcome)
            metrics.write(TickRecord(**base, path="wait"))
            continue

        if llm_only:
            reason = "llm-only mode"
        else:
            reason = "decider unavailable" if decider_down else f"low confidence ({chosen}, {confidence:.2f})"
        print(f"[tick {tick}] LLM fallback: {reason}")
        stats.llm_ticks += 1
        llm_started = time.monotonic()
        limit_ms = llm_timeout_s(turn_remaining_ms(turn, (llm_started - turn_received) * 1000),
                                 args.safety_margin_ms) * 1000
        llm_state = render_state(turn, plan, map_info, include_last_action=True)  # fallback-only text
        handler = TickCancelHandler(ACTION_TOOLS, threading.Event())
        llm_usage: dict[str, int] = {}
        timed_out, llm_result = run_with_timeout(
            functools.partial(run_llm_tick, agent_factory, llm_state, tick, handler, llm_usage), limit_ms / 1000)
        llm_ms = (time.monotonic() - llm_started) * 1000
        llm_fields = {"llm_ms": llm_ms, "llm_input_tokens": llm_usage.get("input_tokens"),
                      "llm_output_tokens": llm_usage.get("output_tokens")}
        if timed_out:
            already_acted = handler.cancel_tick()  # BEFORE any fallback action
            stats.llm_ticks -= 1
            stats.llm_timeouts += 1
            waited = f"[tick {tick}] LLM timeout after {(time.monotonic() - llm_started) * 1000:.0f} ms (limit {limit_ms:.0f} ms)"
            # llm-only: no decider choice exists, so treat the decider as down (-> wait on timeout)
            action = timeout_action(chosen, options, gate.decider_down or llm_only, already_acted)
            if action == "none":
                print(f"{waited} -> LLM already executed an action, not sending another")
                metrics.write(TickRecord(**base, path="llm", llm_timeout=True, **llm_fields))
                continue
            if action == "decider":
                try:
                    tool, arguments = label_to_call(chosen or "", tick, turn)
                except ValueError as exc:
                    print(f"[tick {tick}] {exc} -> wait")
                    action, tool, arguments = "wait", "wait", {"tick": tick}
            else:
                tool, arguments = "wait", {"tick": tick}
            outcome = call_tool(client, tool, arguments)
            route_id = chosen if action == "decider" and route_goal(turn, chosen) is not None else None
            plan = update_plan(plan, route_id, route_goal(turn, route_id), describe_action(tool, arguments), tick,
                               route_path(turn, route_id), my_position(turn_state(turn)))
            verdict = f"executing decider choice {chosen} ({confidence:.2f})" if action == "decider" else "wait"
            print(f"{waited} -> {verdict}{plan_suffix(plan)}")
            report_outcome(outcome)
            metrics.write(TickRecord(**base, path="llm", llm_timeout=True, **llm_fields))
            continue
        calls, call = llm_result
        print(f"[tick {tick}] timing: LLM wall {(time.monotonic() - llm_started) * 1000:.0f} ms, "
              f"total since turn {(time.monotonic() - turn_received) * 1000:.0f} ms")
        if calls == 0:
            print("    warning: the LLM made no action call; the server will count this tick as wait")
        if call is not None:
            route_id = infer_executed_route(call, turn, plan)
            plan = update_plan(plan, route_id, route_goal(turn, route_id), describe_action(*call), tick,
                               route_path(turn, route_id), my_position(turn_state(turn)))
            print(f"    LLM executed {describe_action(*call)}{plan_suffix(plan)}")
        metrics.write(TickRecord(**base, path="llm", **llm_fields))


DEFAULT_NAME = "decider-agent"


def parse_args(default_name: str = DEFAULT_NAME) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTF example agent: a code-owned loop where a Strands Decider selects the action tool.")
    parser.add_argument("--model", choices=sorted(MODELS.keys()), default="haiku",
                        help="Model used for the LLM fallback (default: haiku).")
    parser.add_argument("--match", default="llm-1", help="Match id to join (default: llm-1).")
    parser.add_argument("--name", default=default_name,
                        help=f"Agent name used to join the match (default: {default_name}; lowercased to fit [a-z0-9_-]).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Minimum decider confidence to act without the LLM (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--threat-threshold", type=float, default=THREAT_YES,
                        help=f"Minimum `threat` probability to prefer a tag/defend option (default: {THREAT_YES}).")
    parser.add_argument("--raw-turns", action="store_true",
                        help="Print each raw wait_for_turn payload, the state sent to the decider and its options.")
    parser.add_argument("--llm-budget-ms", type=float, default=DEFAULT_LLM_BUDGET_MS,
                        help=f"Expected LLM fallback time; skipped when less time is left (default: {DEFAULT_LLM_BUDGET_MS}).")
    parser.add_argument("--safety-margin-ms", type=float, default=DEFAULT_SAFETY_MARGIN_MS,
                        help=f"Extra margin added to the LLM budget (default: {DEFAULT_SAFETY_MARGIN_MS}).")
    parser.add_argument("--llm-only", action="store_true",
                        help="Ablation: the LLM fallback agent decides EVERY tick and the decider is never "
                             "consulted (no server needed). The LLM always runs with the time left, no budget "
                             "skip; on timeout the tick is a wait. combine_answers is not used.")
    parser.add_argument("--metrics", metavar="PATH", default=None,
                        help="Write one JSON line of per-tick metrics (path, latencies, tokens) to PATH.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the decider warm-up before joining the match.")
    return parser.parse_args()


def main(system_prompt: str = SYSTEM_PROMPT, default_name: str = DEFAULT_NAME) -> int:
    args = parse_args(default_name)

    decider: Decider | None = None
    decider_down_at_start = False
    if not args.llm_only:  # llm-only never touches the decider server
        decider = Decider()
        try:
            print(decider.banner(), "\n")
        except ServerUnavailable as exc:
            print(f"{exc}\n    -> starting in LLM-fallback mode")
            decider_down_at_start = True

    client = MCPClient.load_servers(MCP_SERVERS)[0]
    if decider_down_at_start:
        # Point the decider at nothing useful so the first ask fails open immediately.
        decider.timeout = 1.0
    stats = with_client(client, lambda: play(client, decider, args, system_prompt=system_prompt))

    print(f"\n{stats.summary()}; {decider if decider is not None else 'decider disabled (llm-only)'}")
    return 0


if __name__ == "__main__":
    exit_code = run_guarded(main)
    sys.stdout.flush()
    sys.stderr.flush()
    exit_if_threads_linger(exit_code)
    sys.exit(exit_code)
