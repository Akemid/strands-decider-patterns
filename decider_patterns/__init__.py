"""Reusable Strands Decider decision patterns.

- `client`: a tiny HTTP client for a running Decider server.
- `tool_selection`: ask the Decider which action tool to call, with a confidence-gated LLM fallback.
- `intervention`: Strands `InterventionHandler` recipes that gate a tool call with a Decider question.
- `timing`: per-decision time budget helpers.
- `budget`: a hard cap on model invocations and a one-live-LLM-thread guard.

Nothing here knows about any particular game or domain.
"""

from decider_patterns.budget import (
    LlmBudget,
    LlmCallCapReached,
    LlmCallCounter,
    is_cap_error,
    is_max_tokens_error,
    is_transient_llm_error,
    transient_llm_error_code,
)
from decider_patterns.client import DECIDER_ERRORS, Decider, ServerUnavailable, ask_serialized
from decider_patterns.intervention import (
    DeciderGateHandler,
    GuideRule,
    StopAfterToolsHook,
    TickCancelHandler,
)
from decider_patterns.timing import llm_timeout_s, remaining_ms, run_with_timeout
from decider_patterns.tool_selection import (
    Decision,
    FilterResult,
    ToolSelectionGate,
    narrow_options,
    parse_choice,
    parse_noul,
    parse_noul_or_none,
)

__all__ = [
    "DECIDER_ERRORS",
    "Decider",
    "DeciderGateHandler",
    "Decision",
    "FilterResult",
    "GuideRule",
    "LlmBudget",
    "LlmCallCapReached",
    "LlmCallCounter",
    "ServerUnavailable",
    "StopAfterToolsHook",
    "TickCancelHandler",
    "ToolSelectionGate",
    "ask_serialized",
    "is_cap_error",
    "is_max_tokens_error",
    "is_transient_llm_error",
    "llm_timeout_s",
    "narrow_options",
    "parse_choice",
    "parse_noul",
    "parse_noul_or_none",
    "remaining_ms",
    "run_with_timeout",
    "transient_llm_error_code",
]
