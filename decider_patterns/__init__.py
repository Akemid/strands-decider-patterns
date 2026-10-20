"""Reusable Strands Decider decision patterns.

- `client`: a tiny HTTP client for a running Decider server.
- `tool_selection`: ask the Decider which action tool to call, with a confidence-gated LLM fallback.
- `intervention`: Strands `InterventionHandler` recipes that gate a tool call with a Decider question.
- `timing`: per-decision time budget helpers.

Nothing here knows about any particular game or domain.
"""

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
    ToolSelectionGate,
    parse_choice,
    parse_noul,
    parse_noul_or_none,
)

__all__ = [
    "DECIDER_ERRORS",
    "Decider",
    "DeciderGateHandler",
    "Decision",
    "GuideRule",
    "ServerUnavailable",
    "StopAfterToolsHook",
    "TickCancelHandler",
    "ToolSelectionGate",
    "ask_serialized",
    "llm_timeout_s",
    "parse_choice",
    "parse_noul",
    "parse_noul_or_none",
    "remaining_ms",
    "run_with_timeout",
]
