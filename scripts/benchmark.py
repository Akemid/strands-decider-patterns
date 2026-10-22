"""Reproducible SIMULATOR-ONLY benchmark: Decider mode vs --llm-only over the same seeds.

    uv run python scripts/benchmark.py --runs 2 --dry-run          # print the commands, run nothing
    uv run python scripts/benchmark.py --seeds 1..5 --ticks 40     # the real thing (uses Bedrock)

For each seed and mode (interleaved: seed 1 decider, seed 1 llm-only, seed 2 ...) it starts a fresh
`simulator/serve.py` on a loopback port, plays one match with the agent, and stops the simulator. Output goes to
`runs/bench-<timestamp>/`: per-run metrics JSONL and logs, plus `summary.json` and `summary.md`.

`--llm-only` runs also pass `--shadow-decider` (disable with `--no-shadow`): after the LLM acted, the local Decider
is asked the same question and its pick is recorded next to the executed label, at no Bedrock cost. That needs the
Decider server for llm-only runs too. Calibrate `--threshold` with `scripts/threshold_sweep.py` on those files.

Cost guards (all overridable, see `--help`): `--max-llm-calls-per-run`, `--max-total-llm-calls`, `--max-cost-usd`,
`--max-consecutive-failures`, `--ticks` <= 200. Estimates come from the metrics files, so keep an AWS Budgets alert.

Only the bundled simulator is supported. The MCP URL must be loopback and never port 8080.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.ctf_agent.metrics import load_records, percentile, summarize  # noqa: E402
from scripts.threshold_sweep import (  # noqa: E402
    SHADOW_NOTE,
    shadow_buckets,
    shadow_pairs,
    shadow_sweep,
    thresholds,
)

MODES = ("decider", "llm-only")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
FORBIDDEN_PORT = 8080
DEFAULT_PORT = 8765
DEFAULT_DECIDER_URL = "http://127.0.0.1:8099"
FILTER_REASONS = ("filtered", "lesser_evil", "unfiltered")
PATHS = ("decider", "llm", "budget", "wait", "llm_capped", "llm_busy", "llm_error")

# Cost guards (defaults). Estimates come from the metrics JSONL, so they lag the real bill: keep an AWS Budgets
# alert as the real backstop.
MAX_TICKS = 200
DEFAULT_MAX_LLM_CALLS_PER_RUN = 60
DEFAULT_MAX_TOTAL_LLM_CALLS = 300
DEFAULT_MAX_COST_USD = 5.0
DEFAULT_MAX_CONSECUTIVE_FAILURES = 2
DEFAULT_MAX_CONSECUTIVE_LLM_ERRORS = 3
# ASSUMPTION: tokens billed for an LLM tick whose metrics carry no usage (e.g. a timeout), so it never counts as free.
FALLBACK_LLM_TOKENS = (4000, 256)

# ASSUMPTION, NOT VERIFIED: USD per 1M tokens used only for the cost estimate. Check the current Amazon Bedrock
# pricing page for your region before quoting any dollar figure. Keys match `MODELS` in examples/ctf_agent/agent.py.
PRICE_TABLE: dict[str, dict[str, float]] = {
    "haiku": {"input_per_1m": 1.00, "output_per_1m": 5.00},
    "opus": {"input_per_1m": 5.00, "output_per_1m": 25.00},
    "nova-lite": {"input_per_1m": 0.06, "output_per_1m": 0.24},
}
PRICE_TABLE_NOTE = (
    "ASSUMPTION: prices (USD per 1M tokens) come from the PRICE_TABLE constant in scripts/benchmark.py. They are "
    "placeholders to be verified against the Amazon Bedrock pricing page, not current pricing. The Decider is local "
    "and costs no tokens."
)
FIDELITY_NOTE = (
    "Simulator caveat (see simulator/README.md, 'Simulator fidelity'): ticks are lock-step, so latency has no game "
    "cost, and the game dynamics are simplified, so win rate is not representative. Use this for per-decision "
    "latency, path mix and LLM usage only."
)


class UnsafeTarget(ValueError):
    """The MCP URL or port is not allowed for a simulator-only benchmark."""


@dataclass(frozen=True)
class RunSpec:
    seed: int
    mode: str


@dataclass(frozen=True)
class Config:
    seeds: list[int]
    modes: list[str]
    ticks: int
    model: str
    port: int
    out: Path
    run_timeout_s: float = 900.0
    ready_timeout_s: float = 30.0
    max_llm_calls_per_run: int = DEFAULT_MAX_LLM_CALLS_PER_RUN
    max_total_llm_calls: int = DEFAULT_MAX_TOTAL_LLM_CALLS
    max_cost_usd: float = DEFAULT_MAX_COST_USD
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    max_consecutive_llm_errors: int = DEFAULT_MAX_CONSECUTIVE_LLM_ERRORS  # passed to the agent as its breaker
    llm_budget_ms: float | None = None  # None: keep the agent default
    threshold: float | None = None  # None: keep the agent default
    shadow: bool = True  # llm-only runs also ask the local Decider (never executed); see --no-shadow


@dataclass(frozen=True)
class RunResult:
    seed: int
    mode: str
    status: str  # ok | failed | timeout | sim-not-ready
    exit_code: int | None
    outcome: str  # informational: last match-over line from the agent log
    duration_s: float
    llm_errors: int = 0  # ticks recorded as path llm_error
    llm_error_reasons: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    exit_code: int | None
    timed_out: bool
    tail: str


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


# ---------------------------------------------------------------- plan

def parse_seeds(spec: str) -> list[int]:
    """`1..5` (inclusive), `2,5,9` or `4`."""
    try:
        if ".." in spec:
            low, high = (int(x) for x in spec.split("..", 1))
            if high < low:
                raise ValueError("empty range")
            return list(range(low, high + 1))
        return [int(x) for x in spec.split(",")]
    except ValueError as exc:
        raise ValueError(f"invalid --seeds {spec!r}: use 1..5 or 1,3,5") from exc


def seeds_from_runs(runs: int, seed_start: int) -> list[int]:
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    return list(range(seed_start, seed_start + runs))


def parse_modes(spec: str) -> list[str]:
    modes = [m.strip() for m in spec.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise ValueError(f"invalid --modes {spec!r}: choose from {', '.join(MODES)}")
    return modes


def build_plan(seeds: Sequence[int], modes: Sequence[str]) -> list[RunSpec]:
    """Same seeds for every mode, interleaved per seed to reduce drift."""
    return [RunSpec(seed, mode) for seed in seeds for mode in modes]


# ---------------------------------------------------------------- guards

def assert_safe_port(port: int) -> None:
    if port == FORBIDDEN_PORT:
        raise UnsafeTarget(f"port {FORBIDDEN_PORT} is refused: this benchmark only talks to the simulator")


def assert_safe_mcp_url(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host not in LOOPBACK_HOSTS:
        raise UnsafeTarget(f"MCP host {host!r} is not loopback: simulator-only benchmark")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    assert_safe_port(port)


def mcp_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/mcp"


# ---------------------------------------------------------------- commands

def metrics_path(cfg: Config, spec: RunSpec) -> Path:
    return cfg.out / f"{spec.mode}-seed{spec.seed}.jsonl"


def simulator_command(cfg: Config, spec: RunSpec, python: str = sys.executable) -> list[str]:
    return [python, "simulator/serve.py", "--seed", str(spec.seed), "--port", str(cfg.port),
            "--ticks", str(cfg.ticks), "--host", "127.0.0.1"]


def agent_command(cfg: Config, spec: RunSpec, python: str = sys.executable) -> list[str]:
    cmd = [python, "-m", "examples.ctf_agent.agent", "--match", f"bench-{spec.mode}-s{spec.seed}",
           "--metrics", str(metrics_path(cfg, spec)), "--model", cfg.model,
           "--max-llm-calls", str(cfg.max_llm_calls_per_run),
           "--max-consecutive-llm-errors", str(cfg.max_consecutive_llm_errors)]
    if cfg.llm_budget_ms is not None:
        cmd += ["--llm-budget-ms", str(cfg.llm_budget_ms)]
    if cfg.threshold is not None:
        cmd += ["--threshold", str(cfg.threshold)]
    if spec.mode == "llm-only":
        cmd.append("--llm-only")
        if cfg.shadow:
            cmd.append("--shadow-decider")
    return cmd


def agent_env(cfg: Config, base: Mapping[str, str]) -> dict[str, str]:
    return {**base, "CTF_MCP_URL": mcp_url(cfg.port)}


def limits(cfg: Config) -> dict[str, Any]:
    return {"ticks": cfg.ticks, "max_ticks": MAX_TICKS, "max_llm_calls_per_run": cfg.max_llm_calls_per_run,
            "max_total_llm_calls": cfg.max_total_llm_calls, "max_cost_usd": cfg.max_cost_usd,
            "max_consecutive_failures": cfg.max_consecutive_failures,
            "max_consecutive_llm_errors": cfg.max_consecutive_llm_errors, "run_timeout_s": cfg.run_timeout_s}


def format_limits(cfg: Config) -> str:
    return (f"max-llm-calls-per-run {cfg.max_llm_calls_per_run}, max-total-llm-calls {cfg.max_total_llm_calls}, "
            f"max-cost-usd {cfg.max_cost_usd}, max-consecutive-failures {cfg.max_consecutive_failures}, "
            f"max-consecutive-llm-errors {cfg.max_consecutive_llm_errors}, "
            f"ticks {cfg.ticks} (max {MAX_TICKS}), run timeout {cfg.run_timeout_s:.0f}s")


def _display(cfg: Config) -> Config:
    """The same config with `out` relative to the repo root when possible (shorter dry-run output)."""
    try:
        return replace(cfg, out=cfg.out.relative_to(ROOT))
    except ValueError:
        return cfg


def _or_default(value: float | None) -> str:
    return "default" if value is None else str(value)


def format_dry_run(cfg: Config, python: str = "uv run python") -> str:
    """The commands a real run would execute (the real run uses this interpreter's own `python`)."""
    cfg = _display(cfg)
    lines = [f"# dry run: nothing is executed. Output would go to {cfg.out}/",
             f"# cost guards: {format_limits(cfg)}",
             f"# agent tuning: llm-budget-ms {_or_default(cfg.llm_budget_ms)}, threshold {_or_default(cfg.threshold)}",
             f"# shadow decider: {'on' if cfg.shadow else 'off'} (llm-only runs also ask the local Decider, never "
             "executed; needs the Decider server)",
             "# the benchmark aborts the remaining runs when a limit is hit; also set an AWS Budgets alert"]
    for spec in build_plan(cfg.seeds, cfg.modes):
        lines.append(f"\n# seed {spec.seed}, mode {spec.mode}")
        lines.append(" ".join(simulator_command(cfg, spec, python)) + "   # started, then stopped after the run")
        lines.append(f"CTF_MCP_URL={mcp_url(cfg.port)} " + " ".join(agent_command(cfg, spec, python))
                     + f"   # timeout {cfg.run_timeout_s:.0f}s")
    return "\n".join(lines)


# ---------------------------------------------------------------- preflight

def check_decider(health: Callable[[], Mapping[str, Any]]) -> Check:
    try:
        h = health()
    except Exception as exc:  # any failure means the Decider is not usable
        return Check("decider", False, f"Decider health check failed: {exc}")
    return Check("decider", True, f"checkpoint {h.get('checkpoint')} on {h.get('device')}")


def check_aws(identity: Callable[[], Mapping[str, Any]]) -> Check:
    try:
        who = identity()
    except Exception as exc:
        return Check("aws", False, f"AWS credentials not usable (sts get_caller_identity): {exc}")
    account = str(who.get("Account") or "")
    return Check("aws", True, f"account ****{account[-4:]}")  # never print the full account id


def run_preflight(modes: Sequence[str], skip_decider: bool, skip_aws: bool,
                  decider_health: Callable[[], Mapping[str, Any]],
                  aws_identity: Callable[[], Mapping[str, Any]], shadow: bool = False) -> list[Check]:
    """`shadow` makes llm-only runs need the Decider server too."""
    checks: list[Check] = []
    if ("decider" in modes or (shadow and "llm-only" in modes)) and not skip_decider:
        checks.append(check_decider(decider_health))
    if not skip_aws:
        checks.append(check_aws(aws_identity))
    return checks


def _decider_health() -> Mapping[str, Any]:
    from decider_patterns.client import Decider

    return Decider().health()


def _aws_identity() -> Mapping[str, Any]:
    import boto3

    return boto3.client("sts").get_caller_identity()


# ---------------------------------------------------------------- orchestration

def wait_for_port(host: str, port: int, timeout_s: float, proc: Any = None) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def run_agent_process(cmd: list[str], env: Mapping[str, str], timeout: float, log_path: Path) -> AgentResult:
    """Run the agent to completion (or timeout), logging combined output to `log_path`.

    The agent starts in its own session so a timeout kills the whole process group (the agent's threads and
    any child), not just the direct child: an orphan would keep calling Bedrock.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, env=dict(env), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
        code, timed_out = proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            output = exc.output
        code, timed_out = None, True
    text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output or "")
    log_path.write_text(text, encoding="utf-8")
    return AgentResult(code, timed_out, outcome_line(text))


def outcome_line(text: str) -> str:
    matches = [ln.strip() for ln in text.splitlines() if ln.startswith("match over")]
    return matches[-1] if matches else ""


def stop_process(proc: Any) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def llm_error_summary(cfg: Config, spec: RunSpec, exit_code: int | None) -> tuple[int, dict[str, int], str]:
    """(llm_error ticks, count per reason, breaker outcome) read from the run's metrics JSONL.

    The outcome is "llm errors: <reason>" only when the agent exited non-zero and its last
    `max_consecutive_llm_errors` ticks are all transient llm_error ticks (the agent's breaker), else "".
    """
    path = metrics_path(cfg, spec)
    if not path.exists():
        return 0, {}, ""
    errors = [r for r in load_records(path) if r.get("path") == "llm_error"]
    reasons: dict[str, int] = {}
    for r in errors:
        key = str(r.get("reason") or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    outcome = ""
    n = cfg.max_consecutive_llm_errors
    if exit_code not in (0, None) and n > 0:
        tail = load_records(path)[-n:]
        if len(tail) == n and all(r.get("path") == "llm_error" and str(r.get("reason", "")).startswith("transient:")
                                  for r in tail):
            outcome = f"llm errors: {tail[-1]['reason']}"
    return len(errors), reasons, outcome


def execute_run(cfg: Config, spec: RunSpec, popen: Callable[..., Any] = subprocess.Popen,
                wait_ready: Callable[..., bool] = wait_for_port,
                run_agent: Callable[..., AgentResult] = run_agent_process) -> RunResult:
    started = time.monotonic()
    cfg.out.mkdir(parents=True, exist_ok=True)
    sim = popen(simulator_command(cfg, spec), cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_ready("127.0.0.1", cfg.port, cfg.ready_timeout_s, sim):
            return RunResult(spec.seed, spec.mode, "sim-not-ready", None, "", time.monotonic() - started)
        result = run_agent(agent_command(cfg, spec), agent_env(cfg, os.environ), cfg.run_timeout_s,
                           cfg.out / f"{spec.mode}-seed{spec.seed}.log")
    finally:
        stop_process(sim)
    if result.timed_out:
        status = "timeout"
    else:
        has_metrics = metrics_path(cfg, spec).exists() and metrics_path(cfg, spec).stat().st_size > 0
        status = "ok" if result.exit_code == 0 and has_metrics else "failed"
    n_errors, reasons, breaker = llm_error_summary(cfg, spec, result.exit_code)
    return RunResult(spec.seed, spec.mode, status, result.exit_code, breaker or result.tail,
                     time.monotonic() - started, n_errors, reasons)


def run_usage(cfg: Config, spec: RunSpec) -> tuple[int, float]:
    """(LLM calls, estimated USD) of one run, read from its metrics JSONL.

    An LLM tick without token counts (a timeout) is billed at `FALLBACK_LLM_TOKENS`, so it is never free.
    """
    path = metrics_path(cfg, spec)
    if not path.exists():
        return 0, 0.0
    price = PRICE_TABLE.get(cfg.model)
    calls, cost = 0, 0.0
    for r in load_records(path):
        if r.get("llm_ms") is None and r.get("path") != "llm":
            continue
        calls += 1
        tokens_in, tokens_out = r.get("llm_input_tokens"), r.get("llm_output_tokens")
        if tokens_in is None and tokens_out is None:
            tokens_in, tokens_out = FALLBACK_LLM_TOKENS
        cost += estimate_cost(tokens_in or 0, tokens_out or 0, price) or 0.0
    return calls, cost


def run_all(cfg: Config, execute: Callable[..., RunResult] = execute_run,
            usage: Callable[[Config, RunSpec], tuple[int, float]] = run_usage) -> tuple[list[RunResult], str | None]:
    """Run the plan in order. Returns (results, abort reason or None); the remaining runs are skipped on abort."""
    results: list[RunResult] = []
    total_calls, total_cost, failures = 0, 0.0, 0
    for spec in build_plan(cfg.seeds, cfg.modes):
        print(f"[bench] seed {spec.seed} mode {spec.mode} ...", flush=True)
        res = execute(cfg, spec)
        print(f"[bench]   -> {res.status} ({res.duration_s:.0f}s) {res.outcome}", flush=True)
        results.append(res)
        calls, cost = usage(cfg, spec)
        total_calls, total_cost = total_calls + calls, total_cost + cost
        failures = 0 if res.status == "ok" else failures + 1
        reason = None
        if failures >= cfg.max_consecutive_failures:
            reason = f"{failures} consecutive failed/timeout runs (max-consecutive-failures {cfg.max_consecutive_failures})"
        elif total_calls > cfg.max_total_llm_calls:
            reason = f"max-total-llm-calls exceeded ({total_calls} > {cfg.max_total_llm_calls})"
        elif total_cost >= cfg.max_cost_usd:
            reason = f"max-cost-usd reached (estimated ${total_cost:.2f} >= ${cfg.max_cost_usd:.2f})"
        if reason:
            print(f"[bench] ABORTING remaining runs: {reason}", flush=True)
            return results, reason
    return results, None


# ---------------------------------------------------------------- aggregation

def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"avg": sum(values) / len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}


def _latency_ms(record: Mapping[str, Any]) -> float | None:
    """Decision latency of a tick, or None when nothing was timed (a tick must not count as 0 ms)."""
    if record.get("decider_ms") is not None or record.get("llm_ms") is not None:
        if record.get("decision_ms") is not None:
            return float(record["decision_ms"])
        return float((record.get("decider_ms") or 0) + (record.get("llm_ms") or 0))
    if record.get("decision_ms"):  # a recorded non-zero total; 0 with no timings is a synthetic default
        return float(record["decision_ms"])
    return None


def aggregate_mode(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool the tick records of one mode (all seeds)."""
    n = len(records)
    base = summarize(list(records))  # reuse compare_runs aggregation (avg, p95, llm calls)
    e2e = [ms for r in records if (ms := _latency_ms(r)) is not None]
    decider_only = [float(r["decider_ms"]) for r in records if r.get("decider_ms") is not None]
    filtered_ticks = [r for r in records if r.get("filter_reason") is not None]
    offered = [r["options_offered"] for r in records if isinstance(r.get("options_offered"), int | float)]
    return {
        "decisions": n,
        "filter_share": ({reason: sum(1 for r in filtered_ticks if r["filter_reason"] == reason) / len(filtered_ticks)
                          for reason in FILTER_REASONS} if filtered_ticks else None),
        "options_offered_avg": sum(offered) / len(offered) if offered else None,
        "path_share": {p: (sum(1 for r in records if r.get("path") == p) / n if n else 0.0) for p in PATHS},
        "end_to_end_ms": _stats(e2e),
        "ticks_without_latency": n - len(e2e),
        "decider_only_ms": _stats(decider_only),
        "llm_calls": base.llm_calls,
        "input_tokens": sum(r.get("llm_input_tokens") or 0 for r in records),
        "output_tokens": sum(r.get("llm_output_tokens") or 0 for r in records),
    }


def shadow_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Agreement between the shadow Decider and the LLM's executed action, or None without shadow data."""
    pairs, excluded = shadow_pairs(records)
    if not pairs:
        return None
    return {"compared": len(pairs), "excluded": excluded,
            "rows": [list(r) for r in shadow_sweep(pairs, thresholds())],
            "buckets": [list(b) for b in shadow_buckets(pairs)]}


def estimate_cost(input_tokens: int, output_tokens: int, price: Mapping[str, float] | None) -> float | None:
    if price is None:
        return None
    return input_tokens / 1e6 * price["input_per_1m"] + output_tokens / 1e6 * price["output_per_1m"]


def llm_calls_avoided(decider_stats: Mapping[str, Any], llm_only_stats: Mapping[str, Any]) -> int:
    """LLM calls the Decider mode did not make, taking the llm-only ablation (one call per decision) as baseline.

    Uses the decider-mode tick count against its own LLM calls, as `compare_runs` does.
    """
    del llm_only_stats  # the baseline is one LLM call per decision by construction
    return int(decider_stats["decisions"] - decider_stats["llm_calls"])


def build_summary(cfg: Config, results: Sequence[RunResult], env: Mapping[str, Any],
                  aborted: str | None = None) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in cfg.modes:
        records: list[dict[str, Any]] = []
        for seed in cfg.seeds:
            path = cfg.out / f"{mode}-seed{seed}.jsonl"
            if path.exists():
                records.extend(load_records(path))
        stats = aggregate_mode(records)
        if mode == "llm-only":
            stats["shadow"] = shadow_summary(records)
        mode_runs = [r for r in results if r.mode == mode]
        stats["runs_ok"] = sum(1 for r in mode_runs if r.status == "ok")
        stats["runs_failed"] = len(mode_runs) - stats["runs_ok"]
        stats["cost_usd"] = estimate_cost(stats["input_tokens"], stats["output_tokens"], PRICE_TABLE.get(cfg.model))
        modes[mode] = stats
    summary: dict[str, Any] = {
        "environment": {**env, "model": cfg.model, "ticks": cfg.ticks, "seeds": cfg.seeds, "modes": cfg.modes,
                        "llm_budget_ms": _or_default(cfg.llm_budget_ms), "threshold": _or_default(cfg.threshold),
                        "shadow_decider": cfg.shadow},
        "modes": modes,
        "runs": [asdict(r) for r in results],
        "limits": limits(cfg),
        "aborted": aborted,
        "price_note": PRICE_TABLE_NOTE,
        "price_used": PRICE_TABLE.get(cfg.model),
        "fidelity_note": FIDELITY_NOTE,
    }
    if "decider" in modes and "llm-only" in modes:
        summary["llm_calls_avoided"] = llm_calls_avoided(modes["decider"], modes["llm-only"])
    return summary


def _ms(stats: Mapping[str, float] | None, key: str) -> str:
    return "n/a" if not stats or stats.get(key) is None else f"{stats[key]:.0f} ms"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def format_shadow_section(shadow: Mapping[str, Any] | None) -> list[str]:
    lines = ["", "## Shadow agreement", ""]
    if not shadow:
        return lines + ["No shadow data (run without `--no-shadow` and with the Decider server up). Then run "
                        "`uv run python scripts/threshold_sweep.py <out>/llm-only-seed*.jsonl`."]
    lines += [f"> {SHADOW_NOTE[0].upper()}{SHADOW_NOTE[1:]}", "",
              f"Compared ticks: {shadow['compared']}. Excluded (no executed label): {shadow['excluded']}.", "",
              "| threshold | covered | agreement among covered | n |", "|---|---|---|---|"]
    lines += [f"| {t:.2f} | {_pct(share)} | {_pct(agree)} | {n} |" for t, share, agree, n in shadow["rows"]]
    lines += ["", "For the per-confidence-bucket table run `scripts/threshold_sweep.py` on the llm-only files."]
    return lines


def format_markdown(summary: Mapping[str, Any]) -> str:
    env = summary["environment"]
    modes = summary["modes"]
    names = list(modes)
    lines = ["# Benchmark: Decider vs LLM-only (simulator)", "", f"> {summary.get('fidelity_note', FIDELITY_NOTE)}", "",
             "## Environment", ""]
    for key in ("git_commit", "decider_checkpoint", "device", "model", "model_id", "ticks", "seeds", "llm_budget_ms",
                "threshold", "shadow_decider"):
        lines.append(f"- {key}: {env.get(key, 'n/a')}")
    lines += ["", "## Cost guards", ""]
    lines += [f"- {key}: {value}" for key, value in summary.get("limits", {}).items()]
    lines.append(f"- aborted: {summary['aborted']}" if summary.get("aborted") else "- aborted: no")
    lines += ["", "## Results", "", "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def row(label: str, fn: Callable[[Mapping[str, Any]], str]) -> None:
        lines.append(f"| {label} | " + " | ".join(fn(modes[m]) for m in names) + " |")

    row("runs ok / failed", lambda s: f"{s['runs_ok']} / {s['runs_failed']}")
    row("decisions", lambda s: str(s["decisions"]))
    row("end-to-end latency avg", lambda s: _ms(s["end_to_end_ms"], "avg"))
    row("end-to-end latency p50", lambda s: _ms(s["end_to_end_ms"], "p50"))
    row("end-to-end latency p95", lambda s: _ms(s["end_to_end_ms"], "p95"))
    row("ticks without latency (excluded)", lambda s: str(s.get("ticks_without_latency", 0)))
    row("Decider-only latency avg", lambda s: _ms(s["decider_only_ms"], "avg"))
    row("Decider-only latency p95", lambda s: _ms(s["decider_only_ms"], "p95"))
    for p in PATHS:
        row(f"% decisions: {p}", lambda s, p=p: f"{s['path_share'][p]:.0%}")
    if any(m.get("filter_share") for m in modes.values()):
        for reason in FILTER_REASONS:
            row(f"% ticks: {reason}", lambda s, reason=reason: "n/a" if not s.get("filter_share")
                else f"{s['filter_share'][reason]:.0%}")
        row("options offered (avg)", lambda s: "n/a" if s.get("options_offered_avg") is None
            else f"{s['options_offered_avg']:.1f}")
    row("LLM calls", lambda s: str(s["llm_calls"]))
    row("Bedrock input / output tokens", lambda s: f"{s['input_tokens']} / {s['output_tokens']}")
    row("estimated cost (USD, assumed prices)",
        lambda s: "n/a" if s["cost_usd"] is None else f"{s['cost_usd']:.4f}")
    if "llm_calls_avoided" in summary:
        lines += ["", f"LLM calls avoided by the Decider vs llm-only: {summary['llm_calls_avoided']}"]
    if "llm-only" in modes:
        lines += format_shadow_section(modes["llm-only"].get("shadow"))
    lines += ["", f"_{summary['price_note']}_", "", "## Runs (informational)", "",
              "Outcome (winner/score) is listed only for transparency; win rate is not a metric here.", "",
              "| seed | mode | status | exit | llm errors | outcome |", "|---|---|---|---|---|---|"]
    for r in summary["runs"]:
        reasons = ", ".join(f"{k} x{v}" for k, v in (r.get("llm_error_reasons") or {}).items())
        errors = f"{r.get('llm_errors', 0)} ({reasons})" if r.get("llm_errors") else "0"
        lines.append(f"| {r['seed']} | {r['mode']} | {r['status']} | {r['exit_code']} | {errors} | "
                     f"{r['outcome'] or '-'} |")
    return "\n".join(lines) + "\n"


def write_outputs(out: Path, summary: Mapping[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(format_markdown(summary), encoding="utf-8")


# ---------------------------------------------------------------- environment / CLI

def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def collect_environment(cfg: Config) -> dict[str, Any]:
    from examples.ctf_agent.agent import MODELS

    env: dict[str, Any] = {"git_commit": git_commit(), "model_id": MODELS.get(cfg.model, cfg.model),
                           "decider_checkpoint": None, "device": None}
    if "decider" in cfg.modes or (cfg.shadow and "llm-only" in cfg.modes):
        try:
            health = _decider_health()
            env["decider_checkpoint"], env["device"] = health.get("checkpoint"), health.get("device")
        except Exception as exc:
            env["decider_checkpoint"] = f"unavailable ({exc})"
    return env


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    seeds = p.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", help="Seeds: 1..5 (inclusive) or a comma list.")
    seeds.add_argument("--runs", type=int, help="Number of seeds, starting at --seed-start.")
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--modes", default=",".join(MODES), help="Comma list of: decider, llm-only (default both).")
    p.add_argument("--ticks", type=int, default=40)
    p.add_argument("--model", default="haiku", choices=sorted(PRICE_TABLE))
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--out", type=Path, default=None, help="Output dir (default runs/bench-<timestamp>/).")
    p.add_argument("--run-timeout", type=float, default=900.0, help="Wall-clock seconds per agent run.")
    p.add_argument("--max-llm-calls-per-run", type=int, default=DEFAULT_MAX_LLM_CALLS_PER_RUN,
                   help="Passed to each agent run as --max-llm-calls (hard cap on model invocations).")
    p.add_argument("--max-total-llm-calls", type=int, default=DEFAULT_MAX_TOTAL_LLM_CALLS,
                   help="Abort the remaining runs once the LLM calls recorded so far exceed this.")
    p.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD,
                   help="Abort the remaining runs once the ESTIMATED cost (assumed prices) reaches this.")
    p.add_argument("--max-consecutive-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
                   help="Abort after this many failed/timeout runs in a row.")
    p.add_argument("--max-consecutive-llm-errors", type=int, default=DEFAULT_MAX_CONSECUTIVE_LLM_ERRORS,
                   help="Passed to each agent run: it exits non-zero after this many transient LLM errors in a row.")
    p.add_argument("--llm-budget-ms", type=float, default=None,
                   help="Passed to each agent run as --llm-budget-ms (default: the agent's own default).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Passed to each agent run as --threshold (default: the agent's own default).")
    p.add_argument("--no-shadow", action="store_true",
                   help="Do not pass --shadow-decider to llm-only runs (then llm-only needs no Decider server).")
    p.add_argument("--dry-run", action="store_true", help="Print the commands and run nothing.")
    p.add_argument("--skip-decider-check", action="store_true", help="Skip the Decider health preflight.")
    p.add_argument("--skip-aws-check", action="store_true", help="Skip the AWS credentials preflight.")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    if args.seeds:
        seeds = parse_seeds(args.seeds)
    else:
        seeds = seeds_from_runs(args.runs or 3, args.seed_start)
    if not 1 <= args.ticks <= MAX_TICKS:
        raise ValueError(f"--ticks must be between 1 and {MAX_TICKS} (got {args.ticks}): it bounds the LLM spend")
    assert_safe_port(args.port)
    assert_safe_mcp_url(mcp_url(args.port))
    out = args.out or ROOT / "runs" / f"bench-{datetime.now():%Y%m%d-%H%M%S}"
    return Config(seeds=seeds, modes=parse_modes(args.modes), ticks=args.ticks, model=args.model,
                  port=args.port, out=Path(out), run_timeout_s=args.run_timeout,
                  max_llm_calls_per_run=args.max_llm_calls_per_run, max_total_llm_calls=args.max_total_llm_calls,
                  max_cost_usd=args.max_cost_usd, max_consecutive_failures=args.max_consecutive_failures,
                  max_consecutive_llm_errors=args.max_consecutive_llm_errors,
                  llm_budget_ms=args.llm_budget_ms, threshold=args.threshold, shadow=not args.no_shadow)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = config_from_args(args)
    except ValueError as exc:  # includes UnsafeTarget
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(format_dry_run(cfg))
        return 0
    checks = run_preflight(cfg.modes, args.skip_decider_check, args.skip_aws_check, _decider_health, _aws_identity,
                           shadow=cfg.shadow)
    for c in checks:
        print(f"[preflight] {c.name}: {'ok' if c.ok else 'FAILED'} - {c.detail}")
    if not all(c.ok for c in checks):
        print("error: preflight failed (use --skip-decider-check / --skip-aws-check to bypass a check)",
              file=sys.stderr)
        return 3
    results, aborted = run_all(cfg)
    summary = build_summary(cfg, results, collect_environment(cfg), aborted=aborted)
    write_outputs(cfg.out, summary)
    print(f"[bench] wrote {cfg.out}/summary.md and summary.json")
    return 0 if aborted is None and all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
