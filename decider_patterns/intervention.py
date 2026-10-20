"""Strands wiring for Decider decisions: an `InterventionHandler` gate plus two loop-level helpers.

`DeciderGateHandler`
    Reviews a proposed tool call with typed yes/no questions *before it runs*. If a `GuideRule` fires, the
    call is cancelled and the model receives feedback and retries; otherwise it proceeds. It fails open
    (an unreachable Decider never stops the agent) and caps consecutive guides so a strict Decider cannot
    stall the agent.

`TickCancelHandler`
    For loops that give an LLM a time budget: once the budget expires, block any late action from the
    abandoned LLM thread, and report whether it had already acted (so you never send a second action).

`StopAfterToolsHook`
    Ends the agent's turn right after an action tool completes, skipping the wasted follow-up model call.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from strands.hooks import AfterToolCallEvent, AfterToolsEvent
from strands.interventions import Deny, Guide, InterventionHandler, Proceed

from decider_patterns.client import DECIDER_ERRORS, DownFlag
from decider_patterns.tool_selection import parse_noul, parse_noul_or_none

DEFAULT_MAX_CONSECUTIVE_GUIDES = 2
MAX_STATE_MESSAGES = 12
MAX_BLOCK_CHARS = 1500

Ask = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class GuideRule:
    """Guide the model with `feedback` when the P(true) of question `question` crosses a bound.

    Set `yes_at_least` to fire on a high probability (e.g. "is the state stale?"), or `yes_below` to fire on
    a low one (e.g. "are the arguments grounded?").
    """

    question: str
    feedback: str
    yes_at_least: float | None = None
    yes_below: float | None = None

    def fires(self, answers: Mapping[str, Any]) -> bool:
        if self.question not in answers:
            return False
        p = parse_noul_or_none(answers[self.question])
        if p is None:  # an unusable answer is not evidence either way
            return False
        if self.yes_at_least is not None and p >= self.yes_at_least:
            return True
        return self.yes_below is not None and p < self.yes_below


def render_recent_messages(messages: list[dict], limit: int = MAX_STATE_MESSAGES) -> str:
    """Flatten the recent agent messages (text, tool calls, tool results) into plain text."""

    def clip(text: str) -> str:
        return text if len(text) <= MAX_BLOCK_CHARS else text[:MAX_BLOCK_CHARS] + " ...[truncated]"

    lines: list[str] = []
    for message in messages[-limit:]:
        role = message.get("role", "unknown")
        for block in message.get("content", []):
            if not isinstance(block, dict):
                continue
            if "text" in block:
                lines.append(f"{role}: {clip(block['text'])}")
            elif "toolUse" in block:
                use = block["toolUse"]
                lines.append(f"{role} called {use.get('name')}: {json.dumps(use.get('input') or {})}")
            elif "toolResult" in block:
                parts = [c.get("text") or json.dumps(c.get("json")) for c in block["toolResult"].get("content", [])]
                lines.append(f"tool result: {clip(' '.join(p for p in parts if p))}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def _tool_name(event: Any) -> str:
    return (getattr(event, "tool_use", None) or {}).get("name", "")


class DeciderGateHandler(InterventionHandler):
    """Gate selected tool calls with Decider questions (see the module docstring)."""

    name = "decider-gate"

    def __init__(
        self,
        *,
        ask: Ask,
        gated_tools: Collection[str],
        questions: Mapping[str, dict[str, Any]],
        rules: Collection[GuideRule],
        describe: Callable[[Any], str],
        max_consecutive_guides: int = DEFAULT_MAX_CONSECUTIVE_GUIDES,
        tool_prefix: str = "",
        timeout_s: float | None = None,
        retry_after_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ask = ask
        self._gated = frozenset(gated_tools)
        self._questions = dict(questions)
        self._rules = tuple(rules)
        self._describe = describe
        self._max_guides = max_consecutive_guides
        self._prefix = tool_prefix
        self._consecutive_guides = 0
        self._timeout_s = timeout_s
        self._down = DownFlag(retry_after_s, clock)
        self.log: list[dict[str, Any]] = []

    @property
    def decider_down(self) -> bool:
        return self._down.down

    @decider_down.setter
    def decider_down(self, value: bool) -> None:
        self._down.mark() if value else self._down.reset()

    def reset(self) -> None:
        """Re-enable the gate after a Decider failure (it is sticky unless `retry_after_s` is set)."""
        self._down.reset()

    def _short(self, event: Any) -> str:
        name = _tool_name(event)
        return name[len(self._prefix):] if self._prefix and name.startswith(self._prefix) else name

    def _state(self, event: Any) -> str:
        agent = getattr(event, "agent", None)
        recent = render_recent_messages(getattr(agent, "messages", None) or [])
        return f"--- RECENT CONVERSATION ---\n{recent}\n\n--- PROPOSED TOOL CALL ---\n{self._describe(event)}"

    def before_tool_call(self, event: Any, **kwargs: Any) -> Proceed | Guide:
        tool = self._short(event)
        if self.decider_down or tool not in self._gated:
            return Proceed(reason="ungated")
        if self._consecutive_guides >= self._max_guides:
            self._consecutive_guides = 0
            self.log.append({"tool": tool, "verdict": "forced"})
            return Proceed(reason="guide cap reached")
        try:
            kwargs = {} if self._timeout_s is None else {"timeout": self._timeout_s}
            answers = self._ask(self._state(event), self._questions, **kwargs)
        except DECIDER_ERRORS:
            self._down.mark()  # fail open: the agent must not stop because the gate died
            return Proceed(reason="decider unavailable")

        scores = {key: round(parse_noul(answers.get(key)), 2) for key in self._questions}
        rule = next((r for r in self._rules if r.fires(answers)), None)
        self.log.append({"tool": tool, "verdict": "guide" if rule else "proceed", **scores})
        if rule is None:
            self._consecutive_guides = 0
            return Proceed(reason="decider approved")
        self._consecutive_guides += 1
        return Guide(feedback=rule.feedback, reason=f"{rule.question} rule fired")


class TickCancelHandler(InterventionHandler):
    """Blocks action tools once the tick's cancel event is set (the LLM timed out).

    An abandoned LLM thread may still try to act later; its action would be rejected as late or, worse,
    double the one already sent. `executed_action` records that an action tool was allowed through before
    the cancel, so the timeout path can avoid a second action.
    """

    name = "tick-cancel"

    def __init__(self, action_tools: Collection[str], cancel: threading.Event) -> None:
        self.action_tools = frozenset(action_tools)
        self.cancel = cancel
        self.executed_action = False
        self._lock = threading.Lock()

    def before_tool_call(self, event: Any, **kwargs: Any) -> Proceed | Deny:
        with self._lock:  # check-and-record is atomic with respect to the timeout path
            if self.cancel.is_set():
                return Deny(reason="tick timed out")
            if _tool_name(event) in self.action_tools:
                self.executed_action = True
            return Proceed()

    def cancel_tick(self) -> bool:
        """Set the cancel event; True when the LLM had already been allowed to act."""
        with self._lock:
            self.cancel.set()
            return self.executed_action


class StopAfterToolsHook:
    """Ends the Agent's turn right after an action tool completes (no second model call).

    `AfterToolsEvent.end_turn` makes the event loop halt after the tool batch without calling the model again
    (verified on strands-agents 1.59). Interventions cannot do this, and `AfterToolsEvent` carries tool results
    without names, so `AfterToolCallEvent` records that an action tool ran. A cancelled call (denied by
    `TickCancelHandler`) does not count as an action.
    """

    def __init__(self, action_tools: Collection[str]) -> None:
        self.action_tools = frozenset(action_tools)
        self.acted = False

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self.after_tool_call)
        registry.add_callback(AfterToolsEvent, self.after_tools)

    def after_tool_call(self, event: Any) -> None:
        if _tool_name(event) in self.action_tools and not getattr(event, "cancel_message", None):
            self.acted = True

    def after_tools(self, event: Any) -> None:
        if self.acted:
            event.end_turn = "action sent"
