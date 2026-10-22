"""Hard cap on LLM spend for a loop: count real model invocations and allow one live LLM thread at a time."""

from __future__ import annotations

import threading
from typing import Any

from botocore.exceptions import ClientError
from strands.hooks import BeforeModelCallEvent
from strands.types.exceptions import MaxTokensReachedException, ModelThrottledException

TRANSIENT_ERROR_CODES = frozenset({
    "ThrottlingException", "ServiceUnavailableException", "serviceUnavailableException",
    "ModelNotReadyException", "InternalServerException", "internalServerException", "modelStreamErrorException",
})


class LlmCallCapReached(RuntimeError):
    """Raised inside the Agent when another model invocation would exceed the budget."""


def is_cap_error(exc: BaseException | None) -> bool:
    """True when `exc` or anything in its cause/context chain is `LlmCallCapReached` (Strands wraps hook errors)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, LlmCallCapReached):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def is_max_tokens_error(exc: BaseException | None) -> bool:
    """True when `exc` or anything in its cause/context chain is Strands' `MaxTokensReachedException`."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, MaxTokensReachedException):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False

def transient_llm_error_code(exc: BaseException | None) -> str | None:
    """The Bedrock error code when `exc` or its cause/context chain is a transient model error, else None.

    Transient: `ModelThrottledException` (code "ThrottlingException") or a botocore `ClientError` /
    `EventStreamError` (a `ClientError` subclass) whose code is in `TRANSIENT_ERROR_CODES`.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code")
            if code in TRANSIENT_ERROR_CODES:
                return code
        elif isinstance(exc, ModelThrottledException):
            return "ThrottlingException"
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None

def is_transient_llm_error(exc: BaseException | None) -> bool:
    """True when `exc` (or its cause chain) is a transient Bedrock error: throttling, unavailable, not ready."""
    return transient_llm_error_code(exc) is not None

class LlmBudget:
    """Thread-safe model-invocation budget plus an in-flight marker.

    `record_call` is invoked once per actual model invocation (see `LlmCallCounter`), so a runaway Agent
    cannot exceed `max_calls`. `begin`/`end` bracket one LLM tick: while it is live (an abandoned thread
    after a timeout counts), `skip_reason` says "llm_busy" so no second thread is started.
    """

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self._calls = 0
        self._live = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        return self._calls

    def record_call(self) -> None:
        with self._lock:
            if self._calls >= self.max_calls:
                raise LlmCallCapReached(f"LLM call cap reached ({self.max_calls})")
            self._calls += 1

    def begin(self) -> None:
        with self._lock:
            self._live += 1

    def end(self) -> None:
        with self._lock:
            self._live -= 1

    def skip_reason(self) -> str | None:
        """Why a new LLM call must not start now: "llm_capped", "llm_busy" or None."""
        with self._lock:
            if self._calls >= self.max_calls:
                return "llm_capped"
            if self._live > 0:
                return "llm_busy"
            return None


class LlmCallCounter:
    """Strands hook provider: every `BeforeModelCallEvent` spends one call of the budget."""

    def __init__(self, budget: LlmBudget) -> None:
        self.budget = budget

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self.before_model_call)

    def before_model_call(self, event: Any) -> None:
        self.budget.record_call()
