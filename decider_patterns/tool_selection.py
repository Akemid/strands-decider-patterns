"""Pattern #tool-selection: let the Decider pick the next action tool from the current state.

The pattern in four steps:

1. Turn the current state into a text `state` and a set of *options* (label -> what that option means).
2. Narrow the options with code: facts you can compute (blocked, out of range, unsafe) are not worth a model
   call, so drop them (`narrow_options`, or `ToolSelectionGate(option_filter=...)`). The Decider then only
   judges what is left.
3. Ask ONE typed `choice` question over those options (extra yes/no questions can ride in the same request).
4. Read the answer: a label and a confidence. Confident enough -> call that tool directly, no LLM.
5. Otherwise route the decision: hand it to an LLM (`llm`), run the Decider's pick because there is no time
   left for an LLM (`budget`), or do the safe default (`wait`).

`ToolSelectionGate` packages steps 2-5. Mapping a label to a concrete tool call, and running the LLM, stay in
your agent: the gate only says which route to take.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from decider_patterns.client import DECIDER_ERRORS, Decider, DownFlag

ACTION_KEY = "action"
"""Key of the tool-selection question in the request and in the answers."""

Route = Literal["decider", "budget", "llm", "wait"]
FilterReason = Literal["filtered", "lesser_evil", "unfiltered"]
"""Why the offered set is what it is: options were dropped, the filter left nothing so the full set is offered
(the lesser evil), or nothing was dropped."""

DEFAULT_THRESHOLD = 0.85
# Haiku measured ~2.6 s avg (p95 ~4.7 s); 3000 + 300 margin stays reachable after a ~1.3 s Decider call in a 5 s tick.
DEFAULT_LLM_BUDGET_MS = 3000
DEFAULT_SAFETY_MARGIN_MS = 300


@dataclass(frozen=True)
class FilterResult:
    """The options to offer to the Decider and why."""

    options: dict[str, str]
    reason: FilterReason

    @property
    def offered(self) -> int:
        return len(self.options)


def narrow_options(
    options: Mapping[str, str],
    keep: Callable[[str], bool],
    *,
    always_keep: Collection[str] = ("wait",),
) -> FilterResult:
    """Drop the options `keep(label)` rejects (pure, game-agnostic).

    `always_keep` labels (the safe default) survive every filter and do not count as a real alternative:
    when nothing else passes, the FULL set is offered instead ("lesser_evil"), because choosing among bad
    options beats choosing between one bad option and doing nothing. Reasons: `filtered` (something was
    dropped), `lesser_evil` (nothing else passed, full set), `unfiltered` (nothing was dropped).
    """
    full = dict(options)
    passed = {label: text for label, text in full.items() if label in always_keep or keep(label)}
    if not any(label not in always_keep for label in passed) and any(label not in always_keep for label in full):
        return FilterResult(full, "lesser_evil")
    return FilterResult(passed, "filtered" if len(passed) < len(full) else "unfiltered")


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
    filter_reason: FilterReason = "unfiltered"
    options_offered: int = 0


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
        option_filter: Callable[[Mapping[str, str], Any], FilterResult] | None = None,
    ) -> None:
        self.instructions = instructions
        self.option_filter = option_filter
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

    def narrow(self, options: Mapping[str, str], state: Any = None) -> FilterResult:
        """Apply `option_filter(options, state)`; without one, every option is offered."""
        if self.option_filter is None:
            return FilterResult(dict(options), "unfiltered")
        return self.option_filter(options, state)

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
        filter_reason: FilterReason = "unfiltered",
    ) -> Decision:
        """Turn answers into a `Decision`. Pass `label`/`confidence` to apply your own combining policy.

        `options` must be the set that was offered (after narrowing): a pick outside it is not valid.
        """
        if label is None and confidence is None:
            label, confidence = parse_choice(answers.get(ACTION_KEY))
        confidence = 0.0 if confidence is None else confidence
        return Decision(label, confidence, self.route(label, confidence, options, remaining_ms), answers,
                        filter_reason, len(options))

    def select(
        self,
        ask: Callable[..., dict[str, Any]],
        state: Any,
        options: Mapping[str, str],
        remaining_ms: float,
        *,
        extra: Mapping[str, dict[str, Any]] | None = None,
    ) -> Decision:
        """Narrow the options, ask the Decider and decide. A failing Decider is marked down and routes to the fallback.

        The question and the validity of the pick use the narrowed set (`Decision.filter_reason` says how it was
        narrowed, `Decision.options_offered` how many options were offered).

        Down is sticky unless `retry_after_s` was set (or `reset()` is called). With `timeout_s`, `ask` is
        called with `timeout=min(timeout_s, remaining seconds)`, so the request cannot outlive the window.
        """
        narrowed = self.narrow(options, state)
        options = narrowed.options
        if self.decider_down:
            return Decision(None, 0.0, self.route(None, 0.0, options, remaining_ms), {},
                            narrowed.reason, len(options))
        kwargs: dict[str, Any] = {}
        if self.timeout_s is not None:
            kwargs["timeout"] = max(0.05, min(self.timeout_s, remaining_ms / 1000))
        try:
            answers = ask(state, self.questions(options, extra), **kwargs)
        except DECIDER_ERRORS:
            self._down.mark()
            return Decision(None, 0.0, self.route(None, 0.0, options, remaining_ms), {},
                            narrowed.reason, len(options))
        return self.decide(answers, options, remaining_ms, filter_reason=narrowed.reason)

    def on_llm_timeout(self, label: str | None, options: Mapping[str, str], already_acted: bool):
        return timeout_action(label, options, self.decider_down, already_acted)
