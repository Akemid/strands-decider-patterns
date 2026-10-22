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
Decider is never consulted (no Decider server needed). Add `--shadow-decider` to ask the local Decider each tick
AFTER the LLM acted, record its pick and confidence next to what the LLM executed, and never execute it: it
calibrates `--threshold` against the LLM at no Bedrock cost.

Code decides facts first: `filter_options` drops blocked or risky routes (or keeps only tag/defend options when my
flag is threatened). The Decider request then carries ONE question, the `action` choice over what is left;
see `options.py`.
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

from botocore.config import Config as BotocoreConfig
from strands import Agent, ModelRetryStrategy
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

from decider_patterns import (
    DECIDER_ERRORS,
    Decider,
    LlmBudget,
    LlmCallCounter,
    ServerUnavailable,
    StopAfterToolsHook,
    TickCancelHandler,
    is_cap_error,
    is_max_tokens_error,
    transient_llm_error_code,
)
from decider_patterns.client import ask_serialized
from decider_patterns.timing import llm_timeout_s, remaining_ms, run_with_timeout
from decider_patterns.tool_selection import (
    DEFAULT_LLM_BUDGET_MS,
    DEFAULT_SAFETY_MARGIN_MS,
    DEFAULT_THRESHOLD,
    timeout_action,
)
from examples.ctf_agent.metrics import MetricsWriter, TickRecord, TurnRecorder
from examples.ctf_agent.options import (
    build_gate,
    build_options,
    build_questions,
    combine_answers,
    filter_options,
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
# Cost limits for every Bedrock call: a tiny output budget (one tool call), no hidden retries (botocore and
# Strands both retry throttling by default, which multiplies billed calls) and short network timeouts.
MODEL_MAX_TOKENS = 1024  # 256 truncated tool-call turns (MaxTokensReachedException); override with --max-tokens
MODEL_READ_TIMEOUT_S = 10
MODEL_CONNECT_TIMEOUT_S = 5
DEFAULT_MAX_LLM_CALLS = 60
DEFAULT_MAX_CONSECUTIVE_LLM_ERRORS = 3
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
    offered = filter_options(build_options(turn, plan, map_info), turn, map_info).options
    return render_state(turn, plan, map_info, only_routes=offered), build_questions(offered)


def executed_choice(call: tuple[str, dict[str, Any]] | None, turn: dict[str, Any], plan: Plan,
                    options: dict[str, str]) -> tuple[str | None, str]:
    """The option label matching the LLM's action call, and how it matched: (label, kind).

    kind is `route` / `action` (a `move_<dir>`, `tag_<id>` or `grab_flag` option), `wait`, `unmatched`
    (the call maps to no offered option, label None) or `none` (no call at all, label None).
    """
    if call is None:
        return None, "none"
    tool, arguments = call
    if tool == "wait":
        return "wait", "wait"
    route_id = infer_executed_route(call, turn, plan)
    if route_id in options:
        return route_id, "route"
    candidates = {"move": f"move_{arguments.get('direction')}", "tag": f"tag_{arguments.get('enemy_id')}",
                  "grab_flag": "grab_flag"}
    label = candidates.get(tool)
    return (label, "action") if label in options else (None, "unmatched")

def shadow_probe(decider: Decider, state: str, options: dict[str, str], reason: str) -> dict[str, Any]:
    """Ask the Decider what it would pick, with the same filtered single-question request. Never raises.

    `state` and `options` are the ones built for the Decider (after `filter_options`); `reason` is the filter reason.
    """
    started = time.monotonic()
    try:
        answers = ask_serialized(decider, state, build_questions(options))
        label, confidence, _ = combine_answers(answers, reason)
    except Exception as exc:  # shadow mode must never affect the game
        return {"shadow_ms": (time.monotonic() - started) * 1000, "shadow_error": f"{type(exc).__name__}: {exc}"}
    return {"shadow_label": label, "shadow_confidence": confidence if label is not None else None,
            "shadow_ms": (time.monotonic() - started) * 1000}

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


class LlmErrorLimitReached(RuntimeError):
    """Raised after too many transient LLM errors in a row: the agent stops instead of hammering a sick endpoint."""


class Stats:
    def __init__(self) -> None:
        self.decider_ticks = 0
        self.llm_ticks = 0
        self.budget_ticks = 0
        self.llm_timeouts = 0
        self.llm_errors = 0
        self.latencies_ms: list[float] = []

    def summary(self) -> str:
        avg = f"{sum(self.latencies_ms) / len(self.latencies_ms):.0f} ms" if self.latencies_ms else "n/a"
        return (
            f"ticks decided by decider: {self.decider_ticks}, by LLM fallback: {self.llm_ticks}, "
            f"by budget: {self.budget_ticks}, LLM timeouts: {self.llm_timeouts}, LLM errors: {self.llm_errors}, "
            f"decider avg latency: {avg}"
        )


def call_tool(client: MCPClient, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = client.call_tool_sync(f"{name}-{uuid.uuid4().hex[:8]}", name, arguments)
    parsed = parse_tool_result(result)
    # Tool errors set isError; transport errors (e.g. "session not found") only set status="error".
    parsed["_is_error"] = result.get("status") == "error" or bool(result.get("isError"))
    return parsed


def join(client: MCPClient, match: str, name: str, team: str | None = None, agent_index: int | None = None,
         recorder: TurnRecorder | None = None) -> tuple[str, MapInfo | None]:
    """Join the requested `team` or, with none, red first then blue. Returns the team and the static map (or None).

    `agent_index` is forwarded when given. A successful join result is written to `recorder`.
    """
    for candidate in (team,) if team else ("red", "blue"):
        arguments: dict[str, Any] = {"match_id": match, "team": candidate, "name": name}
        if agent_index is not None:
            arguments["agent_index"] = agent_index
        outcome = call_tool(client, TOOL_JOIN, arguments)
        if not outcome["_is_error"]:
            if recorder is not None:
                recorder.write("join", outcome)
            print(f"joined match {match} as {name} on team {candidate}")
            return candidate, parse_map(outcome.get("map"))
        print(f"join as {candidate} failed: {outcome}")
    raise RuntimeError(f"could not join match {match} on {team or 'either team'}")


def fetch_map(client: MCPClient, recorder: TurnRecorder | None = None) -> MapInfo | None:
    """The static map from `get_map`: the fallback when `join_match` did not carry it."""
    outcome = call_tool(client, TOOL_GET_MAP, {})
    if outcome["_is_error"]:
        print(f"get_map failed: {outcome}")
        return None
    if recorder is not None:
        recorder.write("map", outcome)
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


def build_model(model_key: str, max_tokens: int = MODEL_MAX_TOKENS) -> BedrockModel:
    """A `BedrockModel` with bounded output, one attempt and short timeouts (no network until first use)."""
    return BedrockModel(
        model_id=MODELS[model_key],
        max_tokens=max_tokens,
        boto_client_config=BotocoreConfig(
            retries={"max_attempts": 1, "mode": "standard"},
            read_timeout=MODEL_READ_TIMEOUT_S,
            connect_timeout=MODEL_CONNECT_TIMEOUT_S,
        ),
    )


def build_agent_factory(model: Any, action_tools: list[Any], system_prompt: str, budget: LlmBudget | None = None,
                        max_tokens: int = MODEL_MAX_TOKENS):
    """Return `handler -> Agent`, the per-tick LLM agent restricted to the action tools.

    `model` is a model object, or a model key (built lazily once and shared by every tick). Every model
    invocation spends one call of `budget`; the turn also ends once the tick's cancel event is set.
    """
    shared: list[Any] = []

    def factory(handler: Any) -> Agent:
        if isinstance(model, str) and not shared:
            shared.append(build_model(model, max_tokens))
        hooks: list[Any] = [StopAfterToolsHook(ACTION_TOOLS, cancel=getattr(handler, "cancel", None))]
        if budget is not None:
            hooks.append(LlmCallCounter(budget))
        return Agent(
            model=shared[0] if shared else model,
            tools=action_tools,
            system_prompt=system_prompt,
            callback_handler=None,
            interventions=[handler] if handler is not None else [],
            hooks=hooks,
            retry_strategy=ModelRetryStrategy(max_attempts=1),
        )

    return factory


def play(client: MCPClient, decider: Decider | None, args: argparse.Namespace,
         system_prompt: str = SYSTEM_PROMPT) -> Stats:
    """Run the match loop. With `args.llm_only` the decider is never used (it may be None).

    With `args.metrics` (a path) one JSONL line per tick is written, see `metrics.py`.
    """
    with (MetricsWriter(getattr(args, "metrics", None)) as metrics,
          TurnRecorder(getattr(args, "record_turns", None)) as recorder):
        return _play(client, decider, args, system_prompt, metrics, recorder)


def _play(client: MCPClient, decider: Decider | None, args: argparse.Namespace, system_prompt: str,
          metrics: MetricsWriter, recorder: TurnRecorder | None = None) -> Stats:
    llm_only = bool(getattr(args, "llm_only", False))
    mode = "llm-only" if llm_only else "decider"
    stats = Stats()
    gate = build_gate(threshold=args.threshold, llm_budget_ms=args.llm_budget_ms,
                      safety_margin_ms=args.safety_margin_ms)
    errors = 0
    llm_error_streak = 0
    max_llm_errors = getattr(args, "max_consecutive_llm_errors", DEFAULT_MAX_CONSECUTIVE_LLM_ERRORS)
    plan = Plan()

    action_tools = [t for t in client.list_tools_sync() if t.tool_name in ACTION_TOOLS]
    llm_budget = LlmBudget(getattr(args, "max_llm_calls", DEFAULT_MAX_LLM_CALLS))
    agent_factory = build_agent_factory(args.model, action_tools, system_prompt, llm_budget,
                                       getattr(args, "max_tokens", MODEL_MAX_TOKENS))

    shadow = llm_only and bool(getattr(args, "shadow_decider", False)) and decider is not None
    shadow_down = False
    if llm_only and not shadow:
        print("llm-only mode: decider disabled (no warm-up, no per-tick decider request)")
    elif llm_only:
        print("llm-only mode with shadow decider: it is asked after the LLM acted and never executed")
        shadow_down = not args.no_warmup and not warm_up(decider)  # warm-up is not a measurement
    elif not args.no_warmup and not warm_up(decider):
        gate.decider_down = True

    _, map_info = join(client, args.match, sanitize_name(args.name), getattr(args, "team", None),
                       getattr(args, "agent_index", None), recorder)
    if map_info is None:  # an older server: fetch the static map once instead
        map_info = fetch_map(client, recorder)
    if map_info is None:
        print("warning: no static map (join_match.map / get_map): field-dependent hints and the defend-radius "
              "check are off (only a flag carried by an enemy counts as a threat)")

    while True:
        turn = call_tool(client, TOOL_WAIT_FOR_TURN, {})
        if recorder is not None:
            recorder.write("turn", turn)
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
        all_options = build_options(turn, plan, map_info)
        filtered = filter_options(all_options, turn, map_info)
        options = filtered.options  # what the Decider is offered; the LLM fallback still sees every route
        state = render_state(turn, plan, map_info, only_routes=options)
        plan_at_tick = plan
        if args.raw_turns:
            print(f"[tick {tick}] raw turn:\n{json.dumps(turn, indent=2, ensure_ascii=False)}")
            print(f"[tick {tick}] state sent to decider:\n{state}")
            print(f"[tick {tick}] options ({filtered.reason}): {sorted(options)}")

        chosen: str | None = None
        confidence = 0.0
        reason = filtered.reason
        decider_ms: float | None = None
        decider_tokens: int | None = None
        if not gate.decider_down and not llm_only:
            questions = build_questions(options)  # ONE request, ONE question
            started = time.monotonic()
            tokens_before = decider.input_tokens
            try:
                answers = ask_serialized(decider, state, questions)
            except DECIDER_ERRORS as exc:
                gate.decider_down = True
                print(f"[decider] {exc}\n    -> LLM fallback for the rest of the match")
            else:
                chosen, confidence, reason = combine_answers(answers, filtered.reason)
                decider_ms = (time.monotonic() - started) * 1000
                decider_tokens = decider.input_tokens - tokens_before
                if decider.last_latency_ms is not None:
                    stats.latencies_ms.append(float(decider.last_latency_ms))
                print(f"[tick {tick}] timing: decider wall {(time.monotonic() - started) * 1000:.0f} ms "
                      f"(server {decider.last_latency_ms} ms), deadline_remaining at receipt "
                      f"{turn.get('deadline_remaining_ms')} ms")

        base = {"tick": tick, "mode": mode, "decider_ms": decider_ms, "decider_input_tokens": decider_tokens,
                "filter_reason": None if llm_only else filtered.reason,
                "options_offered": None if llm_only else len(options),
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
                print(format_decision(tick, chosen, confidence, reason) + plan_suffix(plan))
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

        skip = llm_budget.skip_reason()
        if skip is not None:  # capped or an earlier (abandoned) LLM thread still alive: never spend more
            action = timeout_action(chosen, options, decider_down or llm_only, already_acted=False)
            tool, arguments = "wait", {"tick": tick}
            if action == "decider":
                try:
                    tool, arguments = label_to_call(chosen or "", tick, turn)
                except ValueError as exc:
                    print(f"[tick {tick}] {exc} -> wait")
                    action = "wait"
            outcome = call_tool(client, tool, arguments)
            route_id = chosen if action == "decider" and route_goal(turn, chosen) is not None else None
            plan = update_plan(plan, route_id, route_goal(turn, route_id), describe_action(tool, arguments), tick,
                               route_path(turn, route_id), my_position(turn_state(turn)))
            stats.budget_ticks += 1
            print(f"[tick {tick}] LLM skipped ({skip}: {llm_budget.calls}/{llm_budget.max_calls} calls) -> "
                  f"{'decider choice ' + str(chosen) if action == 'decider' else 'wait'}{plan_suffix(plan)}")
            report_outcome(outcome)
            metrics.write(TickRecord(**base, path=skip))
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
        llm_call = functools.partial(run_llm_tick, agent_factory, llm_state, tick, handler, llm_usage)

        def guarded_llm_call(call: Any = llm_call) -> Any:
            try:
                return call()
            finally:
                llm_budget.end()  # the thread is really done (an abandoned one ends here, later)

        llm_budget.begin()
        llm_error: str | None = None
        try:
            timed_out, llm_result = run_with_timeout(guarded_llm_call, limit_ms / 1000)
        except Exception as exc:
            if is_cap_error(exc):
                timed_out, llm_result = False, (0, None)
                print(f"[tick {tick}] LLM call cap reached mid-turn ({llm_budget.calls}/{llm_budget.max_calls})")
            elif is_max_tokens_error(exc):
                # The model ran out of output tokens: a failed tick, handled like a timeout (fallback action).
                timed_out, llm_result, llm_error = False, (0, None), "max_tokens"
            elif (code := transient_llm_error_code(exc)) is not None:
                # Throttling / service unavailable: same fallback, counted; the breaker below stops a streak.
                timed_out, llm_result, llm_error = False, (0, None), f"transient:{code}"
                llm_error_streak += 1
                print(f"[tick {tick}] transient LLM error {code} ({llm_error_streak}/{max_llm_errors} in a row)")
            else:
                # Unknown error: crash (run_guarded exits 1). Never fall back and loop, which could multiply calls.
                raise
        if not llm_error and not timed_out and llm_result != (0, None):
            llm_error_streak = 0  # a model call that completed
        breaker_tripped = llm_error is not None and llm_error.startswith("transient:") \
            and max_llm_errors > 0 and llm_error_streak >= max_llm_errors
        llm_ms = (time.monotonic() - llm_started) * 1000
        llm_fields = {"llm_ms": llm_ms, "llm_input_tokens": llm_usage.get("input_tokens"),
                      "llm_output_tokens": llm_usage.get("output_tokens")}
        if timed_out or llm_error:
            already_acted = handler.cancel_tick()  # BEFORE any fallback action
            stats.llm_ticks -= 1
            if llm_error:
                stats.llm_errors += 1
                waited = (f"[tick {tick}] LLM error ({llm_error}) after "
                          f"{(time.monotonic() - llm_started) * 1000:.0f} ms")
                record = {"path": "llm_error", "reason": llm_error}
            else:
                stats.llm_timeouts += 1
                waited = (f"[tick {tick}] LLM timeout after {(time.monotonic() - llm_started) * 1000:.0f} ms "
                          f"(limit {limit_ms:.0f} ms)")
                record = {"path": "llm", "llm_timeout": True}
            # llm-only: no decider choice exists, so treat the decider as down (-> wait on timeout)
            action = timeout_action(chosen, options, gate.decider_down or llm_only, already_acted)
            if action == "none":
                print(f"{waited} -> LLM already executed an action, not sending another")
                metrics.write(TickRecord(**{**base, **record}, **llm_fields))
                check_breaker(breaker_tripped, llm_error, llm_error_streak)
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
            metrics.write(TickRecord(**{**base, **record}, **llm_fields))
            check_breaker(breaker_tripped, llm_error, llm_error_streak)
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
        shadow_fields: dict[str, Any] = {}
        if shadow:  # AFTER the LLM acted: no latency for the LLM decision, the simulator is lock-step
            label, kind = executed_choice(call, turn, plan_at_tick, all_options)
            shadow_fields = {"executed_label": label, "executed_kind": kind}
            if shadow_down:
                shadow_fields["shadow_error"] = "decider unavailable"
            else:
                shadow_fields |= shadow_probe(decider, state, options, filtered.reason)
                shadow_down = "shadow_error" in shadow_fields
        metrics.write(TickRecord(**base, path="llm", **llm_fields, **shadow_fields))


def check_breaker(tripped: bool, error: str | None, streak: int) -> None:
    if tripped:
        raise LlmErrorLimitReached(f"{streak} consecutive transient LLM errors (last: {(error or '').split(':', 1)[-1]}); "
                                   "stopping with no further model calls. Check your Bedrock quotas.")


DEFAULT_NAME = "decider-agent"


def parse_args(default_name: str = DEFAULT_NAME) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTF example agent: a code-owned loop where a Strands Decider selects the action tool.")
    parser.add_argument("--model", choices=sorted(MODELS.keys()), default="haiku",
                        help="Model used for the LLM fallback (default: haiku).")
    parser.add_argument("--match", default="llm-1", help="Match id to join (default: llm-1).")
    parser.add_argument("--name", default=default_name,
                        help=f"Agent name used to join the match (default: {default_name}; lowercased to fit [a-z0-9_-]).")
    parser.add_argument("--team", choices=("red", "blue"), default=None,
                        help="Team to join (default: try red, then blue).")
    parser.add_argument("--agent-index", type=int, default=None,
                        help="Slot within the team to take (default: the server picks).")
    parser.add_argument("--record-turns", metavar="PATH", default=None,
                        help="Append the join/map payload once and every raw wait_for_turn payload to PATH as JSONL "
                             "(no headers or tokens). Write under runs/ and never commit private-server recordings.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Minimum decider confidence to act without the LLM (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--raw-turns", action="store_true",
                        help="Print each raw wait_for_turn payload, the state sent to the decider and its options.")
    parser.add_argument("--llm-budget-ms", type=float, default=DEFAULT_LLM_BUDGET_MS,
                        help=f"Expected LLM fallback time; skipped when less time is left (default: {DEFAULT_LLM_BUDGET_MS}).")
    parser.add_argument("--max-tokens", type=int, default=MODEL_MAX_TOKENS,
                        help=f"Max output tokens per model call (default: {MODEL_MAX_TOKENS}).")
    parser.add_argument("--safety-margin-ms", type=float, default=DEFAULT_SAFETY_MARGIN_MS,
                        help=f"Extra margin added to the LLM budget (default: {DEFAULT_SAFETY_MARGIN_MS}).")
    parser.add_argument("--llm-only", action="store_true",
                        help="Ablation: the LLM fallback agent decides EVERY tick and the decider is never "
                             "consulted (no server needed). The LLM always runs with the time left, no budget "
                             "skip; on timeout the tick is a wait. The filter and combine_answers are not used.")
    parser.add_argument("--shadow-decider", action="store_true",
                        help="Only with --llm-only: after the LLM acted each tick, ask the local Decider the same "
                             "question and record its pick, confidence and the executed label in --metrics. The "
                             "Decider's choice is never executed. Needs a Decider server; errors only record "
                             "shadow_error.")
    parser.add_argument("--max-llm-calls", type=int, default=DEFAULT_MAX_LLM_CALLS,
                        help="Hard cap on model invocations for the whole run; 0 never calls the LLM. Past the cap "
                             f"a tick runs the decider choice (if valid) or waits (default: {DEFAULT_MAX_LLM_CALLS}).")
    parser.add_argument("--max-consecutive-llm-errors", type=int, default=DEFAULT_MAX_CONSECUTIVE_LLM_ERRORS,
                        help="Exit non-zero after this many transient LLM errors (throttling, service unavailable) "
                             "in a row; each such tick falls back to the decider choice or a wait, a successful LLM "
                             f"call resets the streak (default: {DEFAULT_MAX_CONSECUTIVE_LLM_ERRORS}).")
    parser.add_argument("--metrics", metavar="PATH", default=None,
                        help="Write one JSON line of per-tick metrics (path, latencies, tokens) to PATH.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the decider warm-up before joining the match.")
    args = parser.parse_args()
    if args.shadow_decider and not args.llm_only:
        parser.error("--shadow-decider only makes sense with --llm-only")
    return args


def main(system_prompt: str = SYSTEM_PROMPT, default_name: str = DEFAULT_NAME) -> int:
    args = parse_args(default_name)

    decider: Decider | None = None
    decider_down_at_start = False
    if not args.llm_only or args.shadow_decider:  # plain llm-only never touches the decider server
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
    try:
        stats = with_client(client, lambda: play(client, decider, args, system_prompt=system_prompt))
    except LlmErrorLimitReached as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(f"\n{stats.summary()}; {decider if decider is not None else 'decider disabled (llm-only)'}")
    return 0


if __name__ == "__main__":
    exit_code = run_guarded(main)
    sys.stdout.flush()
    sys.stderr.flush()
    exit_if_threads_linger(exit_code)
    sys.exit(exit_code)
