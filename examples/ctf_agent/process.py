"""Process-level safety nets: guarded entry point, MCP client lifecycle, forced exit on lingering threads."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from typing import Any

EXIT_GRACE_S = 2.0


def lingering_threads(threads: Any) -> list[Any]:
    """Alive, non-daemon threads other than the main thread: what blocks interpreter exit."""
    main = threading.main_thread()
    return [t for t in threads if t is not main and t.is_alive() and not t.daemon]


def describe_threads(threads: Any) -> str:
    return ", ".join(f"{t.name} (daemon={t.daemon})" for t in threads)


def exit_if_threads_linger(code: int, grace_s: float = EXIT_GRACE_S, *, exit_fn=os._exit,
                           enumerate_fn=threading.enumerate, sleep=time.sleep,
                           clock=time.monotonic, log=None) -> bool:
    """Safety net: if non-daemon threads outlive `grace_s`, log them and `exit_fn(code)` (os._exit).

    Python would otherwise wait forever for them at interpreter exit. Returns True when it forced
    the exit (only observable with a fake `exit_fn`).
    """
    log = log or (lambda message: print(message, file=sys.stderr))
    deadline = clock() + grace_s
    while True:
        remaining = lingering_threads(enumerate_fn())
        if not remaining:
            return False
        if clock() >= deadline:
            break
        sleep(0.05)
    log(f"[exit] non-daemon threads still alive after {grace_s:.1f}s grace, forcing exit {code}: "
        f"{describe_threads(remaining)}; all threads: {describe_threads(enumerate_fn())}")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - never block the forced exit
            pass
    exit_fn(code)
    return True


def run_guarded(fn) -> int:
    """Run `fn` and turn its outcome into an exit code, printing the traceback on failure."""
    try:
        return fn()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BaseException:  # noqa: BLE001 - reported, not swallowed: non-zero exit
        traceback.print_exc()
        return 1


def with_client(client: Any, body) -> Any:
    """Start the MCP client, run `body`, and ALWAYS stop it (even if start fails or on Ctrl-C)."""
    try:
        client.start()
        return body()
    finally:
        try:
            client.stop(None, None, None)
        except Exception as exc:  # noqa: BLE001 - a failing stop must not mask the real error
            print(f"[exit] MCP client stop failed: {exc}", file=sys.stderr)
