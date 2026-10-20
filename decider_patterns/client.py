"""A small synchronous client for a running decision-model server.

Deliberately `urllib` rather than a client library: these examples should need nothing
beyond `strands-agents`, so that a reader can drop one into their own project without
first agreeing to a dependency.

Synchronous on purpose too. Strands calls a handler method that is a plain `def`
directly, so a blocking request is the simplest thing that works -- and it keeps requests
to the server strictly sequential, which is how it is best driven: one request at a time.

**Renaming.** The product name is spelled out in exactly one place, `NAME` below, and the
things derived from it sit directly underneath. Everything else -- this module's filename,
the `Decider` class, the demos -- is named for what it does rather than what it is called,
so a future rename is an edit to this block and nothing else.
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

try:  # POSIX only; on Windows the cross-process lock is skipped
    import fcntl
except ImportError:  # pragma: no cover - depends on the platform
    fcntl = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- the name

NAME = "strands-decider"
"""What this thing is called today. The only place the name is written."""

ENV_URL = "STRANDS_DECIDER_URL"
"""Environment variable that overrides the endpoint. Derived from NAME, so rename together."""

SERVE_HINT = "strands-decider serve <checkpoint> --port 8099"
"""How to start a server, shown when nothing is listening.

This is the upstream CLI entry point (`[project.scripts]` in `pyproject.toml`), which is a
separate thing from NAME: it only changes if the Python package itself is renamed.
"""

MODEL_ID = "hobson-latest"
"""Sent as `model` on every request. The server ignores it and answers as whatever
checkpoint it loaded, so this is only here to show the full wire shape."""

DEFAULT_PORT = 8099
DEFAULT_URL = os.environ.get(ENV_URL, f"http://127.0.0.1:{DEFAULT_PORT}")


class ServerUnavailable(RuntimeError):
    """The server is not answering. Raised with the command that starts one."""


class Decider:
    """One state, many typed questions, one round trip.

    The questions cannot see each other, so their answers stay independent -- but the
    state is encoded once and shared across them, which is why asking eight things costs
    far less than eight requests. Ask everything you need at once.
    """

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        # Generous: on Apple silicon the *first* request at a given input length pays a
        # one-off MPS shape compile, which is seconds on a long document.
        self.timeout = timeout
        self.calls = 0
        self.input_tokens = 0
        self.last_latency_ms: float | None = None

    # ---- questions -------------------------------------------------------

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Yes/no. The answer is P(true), and with two outcomes that *is* the confidence.

        `criteria` optionally describes what `"true"` and `"false"` mean. It is worth
        writing: a noul renders as a two-option list, so those descriptions are read the
        same way a choice's options are, and they sharpen the boundary considerably.
        """
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            question["criteria"] = dict(criteria)
        return question

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        """Pick one of N. `criteria` maps option name -> what that option means.

        The option names are read from this request, not baked into the model, so you can
        change them without retraining anything.
        """
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        """Rate against an ordered rubric, lowest level first."""
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    # ---- asking ----------------------------------------------------------

    def _unavailable(self, exc: Exception | str) -> ServerUnavailable:
        return ServerUnavailable(
            f"no {NAME} at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}"
        )

    def ask(
        self, state: Any, questions: Mapping[str, dict[str, Any]], timeout: float | None = None
    ) -> dict[str, Any]:
        """Evaluate `state` against `questions`; answers come back under the same keys.

        `timeout` (seconds) overrides the client default for this request only. A reply that is not
        the expected JSON shape raises `ServerUnavailable`, like an unreachable server.
        """
        body = json.dumps(
            {"state": state, "model": MODEL_ID, "questions": dict(questions)}
        ).encode()
        request = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout if timeout is None else timeout
            ) as response:
                payload = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc
        except ValueError as exc:  # invalid JSON (JSONDecodeError, UnicodeDecodeError)
            raise self._unavailable(exc) from exc
        try:
            answers = payload["answers"]
            usage = payload.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            latency_ms = payload.get("latency_ms")
        except (KeyError, TypeError, AttributeError) as exc:
            raise self._unavailable(f"malformed reply: {exc!r}") from exc
        if not isinstance(answers, dict) or not isinstance(input_tokens, (int, float)):
            raise self._unavailable("malformed reply: unexpected shape")
        self.calls += 1
        self.input_tokens += input_tokens
        self.last_latency_ms = latency_ms
        return answers

    def health(self) -> dict[str, Any]:
        """The loaded checkpoint, window and device. Raises if nothing is serving."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc

    def banner(self) -> str:
        h = self.health()
        return (
            f"{NAME}: {h['checkpoint']} on {h['base_model']} "
            f"[{h['device']}, window {h['max_length']}] at {self.url}"
        )

    def __str__(self) -> str:
        return f"{self.calls} decisions, {self.input_tokens} input tokens"

DECIDER_ERRORS = (ServerUnavailable, OSError, http.client.HTTPException)
"""Everything that means "the Decider could not answer". `RemoteDisconnected` is an
`http.client` error rather than a `URLError`, so it is not wrapped in `ServerUnavailable`."""

LOCK_FILE = os.path.join(tempfile.gettempdir(), "strands-decider.lock")


class DownFlag:
    """Remembers that the Decider failed. Sticky unless `retry_after_s` is set.

    With `retry_after_s`, `down` turns False again that many seconds after `mark()`, so the next
    decision tries the Decider again. `clock` is injectable for tests.
    """

    def __init__(
        self, retry_after_s: float | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.retry_after_s = retry_after_s
        self._clock = clock
        self._marked_at: float | None = None

    @property
    def down(self) -> bool:
        if self._marked_at is None:
            return False
        if self.retry_after_s is not None and self._clock() - self._marked_at >= self.retry_after_s:
            self._marked_at = None
        return self._marked_at is not None

    def mark(self) -> None:
        self._marked_at = self._clock()

    def reset(self) -> None:
        self._marked_at = None


def ask_serialized(
    decider: Decider,
    state: Any,
    questions: Mapping[str, dict[str, Any]],
    timeout: float | None = None,
) -> dict[str, Any]:
    """Ask the Decider while holding a cross-process file lock (skipped where `fcntl` is missing).

    A local server runs inference in a thread pool without a lock, so concurrent requests from several
    agent processes can crash it. One request at a time across all processes on this machine. On Windows
    there is no `fcntl`: the lock is skipped and the request is sent directly.
    """
    kwargs = {} if timeout is None else {"timeout": timeout}
    if fcntl is None:
        return decider.ask(state, questions, **kwargs)
    with open(LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return decider.ask(state, questions, **kwargs)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
