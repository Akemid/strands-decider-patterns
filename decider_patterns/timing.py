"""Per-decision time budget helpers (pure, plus one thread wrapper)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

DEFAULT_DEADLINE_MS = 5000
MIN_LLM_TIMEOUT_S = 0.05


def remaining_ms(deadline_remaining_ms: float | None, elapsed_ms: float) -> float:
    """Time left in the decision window: the deadline remaining at receipt minus time elapsed since."""
    base = DEFAULT_DEADLINE_MS if deadline_remaining_ms is None else deadline_remaining_ms
    return base - elapsed_ms


def llm_timeout_s(remaining: float, margin_ms: float) -> float:
    """Seconds the LLM fallback may run: time left minus the safety margin, at least 50 ms."""
    return max(MIN_LLM_TIMEOUT_S, (remaining - margin_ms) / 1000)


def run_with_timeout(fn: Callable[[], Any], timeout_s: float) -> tuple[bool, Any]:
    """Run `fn` in a daemon thread and wait at most `timeout_s`. Returns `(timed_out, result)`.

    On timeout the thread is abandoned (it cannot be killed) and `(True, None)` is returned. An exception
    raised by `fn` is re-raised in the caller. A fresh thread per call means an abandoned run never blocks
    the next decision. Pair it with `TickCancelHandler` so the abandoned run cannot act late.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True, name="llm-tick")
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        return True, None
    if "error" in box:
        raise box["error"]
    return False, box.get("value")
