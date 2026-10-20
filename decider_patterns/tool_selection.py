"""Pattern #tool-selection: let the Decider pick the next action tool from the current state.

The pattern in four steps:

1. Turn the current state into a text `state` and a set of *options* (label -> what that option means).
2. Ask ONE typed `choice` question over those options (extra yes/no questions can ride in the same request).
3. Read the answer: a label and a confidence. Confident enough -> call that tool directly, no LLM.
4. Otherwise route the decision: hand it to an LLM (`llm`), run the Decider's pick because there is no time
   left for an LLM (`budget`), or do the safe default (`wait`).

`ToolSelectionGate` packages steps 2-4. Mapping a label to a concrete tool call, and running the LLM, stay in
your agent: the gate only says which route to take.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from decider_patterns.client import DECIDER_ERRORS, Decider, DownFlag

ACTION_KEY = "action"
"""Key of the tool-selection question in the request and in the answers."""

Route = Literal["decider", "budget", "llm", "wait"]

DEFAULT_THRESHOLD = 0.85
DEFAULT_LLM_BUDGET_MS = 4500
DEFAULT_SAFETY_MARGIN_MS = 300


def parse_choice(answer: Any) -> tuple[str | None, float]:
    """`(label, confidence)` from a Decider `choice` answer; `(None, 0.0)` if unusable.

    Shape: `{"type": "choice", "choice": str, "probabilities": {...}, "confidence": float}`. Note that
    `confidence` is not the top probability, so thresholds apply to `confidence`. Falls back to the chosen
    label's probability when `confidence` is absent.
    """
    if not isinstance(answer, dict):
        return None, 0.0
    label = answer.get("choice")
    if not isinstance(label, str):
        return None, 0.0
    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)):
        probs = answer.get("probabilities")
        confidence = probs.get(label) if isinstance(probs, dict) else None
    return label, float(confidence) if isinstance(confidence, (int, float)) else 0.0


def parse_noul_or_none(answer: Any) -> float | None:
    """P(true) from a Decider `noul` answer (`{"type": "noul", "noul": float}`); `None` if unusable."""
    value = answer.get("noul") if isinstance(answer, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_noul(answer: Any) -> float:
    """P(true) from a Decider `noul` answer; 0.0 if unusable (use `parse_noul_or_none` to tell them apart)."""
    value = parse_noul_or_none(answer)
    return 0.0 if value is None else value


def accept_choice(label: str | None, confidence: float, threshold: float) -> bool:
    return label is not None and confidence >= threshold


def choose_path(
    chosen: str | None,
    confidence: float,
    threshold: float,
    options: Mapping[str, str],
    remaining: float,
    llm_budget_ms: float,
    margin_ms: float,
    decider_down: bool,
) -> Route:
    """How to act on this decision (pure).

    decider: confident, offered choice. llm: not confident but there is time for the LLM.
    budget: no time for the LLM and the Decider's choice is a valid option, so run it.
    wait: no time for the LLM and nothing valid to run.
    """
    valid = chosen is not None and chosen in options
    if not decider_down and valid and accept_choice(chosen, confidence, threshold):
        return "decider"
    if remaining >= llm_budget_ms + margin_ms:
        return "llm"
    if not decider_down and valid:
        return "budget"
    return "wait"


def timeout_action(
    chosen: str | None, options: Mapping[str, str], decider_down: bool, already_acted: bool
) -> Literal["decider", "wait", "none"]:
    """What to send when the LLM fallback timed out (pure).

    none: the LLM already executed an action, never send a second one.
    decider: the Decider's choice is a valid offered option. wait: nothing valid to run.
    """
    if already_acted:
        return "none"
    if not decider_down and chosen is not None and chosen in options:
        return "decider"
    return "wait"


@dataclass(frozen=True)
class Decision:
    """The outcome of one tool-selection round."""

    label: str | None
    confidence: float
    route: Route
    answers: Mapping[str, Any] = field(default_factory=dict)


class ToolSelectionGate:
    """Confidence-gated tool selection with a fallback route.

    Usage::

        gate = ToolSelectionGate("Pick the next action for the agent.", threshold=0.85)
        decision = gate.select(decider.ask, state_text, options, remaining_ms=remaining)
        if decision.route in ("decider", "budget"):
            call_tool(decision.label)
        elif decision.route == "llm":
            run_my_llm_agent()          # the fallback hook-in point
        else:
            call_safe_default()
    """

    def __init__(
        self,
        instructions: str,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        llm_budget_ms: float = DEFAULT_LLM_BUDGET_MS,
        safety_margin_ms: float = DEFAULT_SAFETY_MARGIN_MS,
        timeout_s: float | None = None,
        retry_after_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.instructions = instructions
        self.threshold = threshold
        self.llm_budget_ms = llm_budget_ms
        self.safety_margin_ms = safety_margin_ms
        self.timeout_s = timeout_s
        self._down = DownFlag(retry_after_s, clock)

    @property
    def decider_down(self) -> bool:
        return self._down.down

    @decider_down.setter
    def decider_down(self, value: bool) -> None:
        self._down.mark() if value else self._down.reset()

    def reset(self) -> None:
        """Forget a Decider failure so the next `select` asks it again."""
        self._down.reset()

    def questions(
        self, options: Mapping[str, str], extra: Mapping[str, dict[str, Any]] | None = None
    ) -> dict[str, dict[str, Any]]:
        """The request's questions: the tool choice plus any extra questions sharing the same state."""
        extra = dict(extra or {})
        if ACTION_KEY in extra:
            raise ValueError(f"{ACTION_KEY!r} is reserved for the tool-selection question")
        return {ACTION_KEY: Decider.choice(self.instructions, options), **extra}

    def route(
        self, label: str | None, confidence: float, options: Mapping[str, str], remaining_ms: float
    ) -> Route:
        return choose_path(
            label, confidence, self.threshold, options, remaining_ms,
            self.llm_budget_ms, self.safety_margin_ms, self.decider_down,
        )

    def decide(
        self,
        answers: Mapping[str, Any],
        options: Mapping[str, str],
        remaining_ms: float,
        *,
        label: str | None = None,
        confidence: float | None = None,
    ) -> Decision:
        """Turn answers into a `Decision`. Pass `label`/`confidence` to apply your own combining policy."""
        if label is None and confidence is None:
            label, confidence = parse_choice(answers.get(ACTION_KEY))
        confidence = 0.0 if confidence is None else confidence
        return Decision(label, confidence, self.route(label, confidence, options, remaining_ms), answers)

    def select(
        self,
        ask: Callable[..., dict[str, Any]],
        state: Any,
        options: Mapping[str, str],
        remaining_ms: float,
        *,
        extra: Mapping[str, dict[str, Any]] | None = None,
    ) -> Decision:
        """Ask the Decider and decide. A failing Decider is marked down and routes to the fallback.

        Down is sticky unless `retry_after_s` was set (or `reset()` is called). With `timeout_s`, `ask` is
        called with `timeout=min(timeout_s, remaining seconds)`, so the request cannot outlive the window.
        """
        if self.decider_down:
            return Decision(None, 0.0, self.route(None, 0.0, options, remaining_ms))
        kwargs: dict[str, Any] = {}
        if self.timeout_s is not None:
            kwargs["timeout"] = max(0.05, min(self.timeout_s, remaining_ms / 1000))
        try:
            answers = ask(state, self.questions(options, extra), **kwargs)
        except DECIDER_ERRORS:
            self._down.mark()
            return Decision(None, 0.0, self.route(None, 0.0, options, remaining_ms))
        return self.decide(answers, options, remaining_ms)

    def on_llm_timeout(self, label: str | None, options: Mapping[str, str], already_acted: bool):
        return timeout_action(label, options, self.decider_down, already_acted)
