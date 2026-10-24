# decider-patterns

A reusable **tool-selection** pattern for [Strands](https://strandsagents.com) agents built on
**Strands Decider 2B**, with Amazon Bedrock as the LLM fallback, plus a Capture the Flag (CTF) example agent
and a local simulator so everything runs end to end on a laptop.

Hackathon: *Agent Experience Hackathon: Strands Decider*. Declared pattern: **#tool-selection**.

Demo video (under 3 minutes): https://youtu.be/BX6Ps0Rnb8o

## The idea

Most agent ticks do not need an LLM. The agent already knows its options (which tools make sense right now);
it only has to *choose*. The Decider is a small model that answers typed questions about a state in one
request, with a calibrated confidence. So:

1. Turn the state into text, and the possible actions into an **option set** (`label -> meaning`).
2. Ask the Decider one typed `choice` question over the options (yes/no questions can ride in the same request).
3. Confident answer: call that tool directly. No LLM call.
4. Unsure or unreachable: fall back to a Strands `Agent` on Bedrock, restricted to the action tools, with a time budget.

Where the decision lives (`decider_patterns/`, game-agnostic, no CTF imports):

| Module | What it gives you |
| --- | --- |
| `tool_selection.py` | `ToolSelectionGate`: builds the question, parses the answer, applies a confidence threshold and routes to `decider`, `llm`, `budget` or `wait`. |
| `intervention.py` | `DeciderGateHandler`: a Strands `InterventionHandler` that reviews a proposed tool call with Decider questions and guides the model when a rule fires. Also `TickCancelHandler` and `StopAfterToolsHook` for loop-level control. |
| `client.py` | A small `urllib` client for the Decider server (`Decider.noul/choice/score`) and a cross-process lock helper. |
| `timing.py` | Per-decision time budget helpers and `run_with_timeout`. |

See [decider_patterns/README.md](decider_patterns/README.md) for the recipe and usage.

## The example: a CTF agent

`examples/ctf_agent/` is a code-owned game loop: `wait_for_turn` -> render state -> filter the options in code
(drop blocked or risky routes, or keep only tag/defend options when the flag is threatened) -> one Decider request
with a single `action` choice over what is left -> call the tool.
Below the confidence threshold it runs a Strands `Agent` on Bedrock for that tick. `--llm-only` is the ablation
where the LLM decides every tick. See [examples/ctf_agent/README.md](examples/ctf_agent/README.md).

The real game server is private and is not part of this repo. `simulator/` is an in-memory test double of its MCP
interface (turn contract v2) with a tiny dynamic game (`simulator/engine.py`), served over streamable HTTP so the
agent plays a short match to `match_over`.

## Quick start

Requirements: Python 3.13+, [uv](https://docs.astral.sh/uv/), and for the LLM fallback AWS credentials with
access to the Bedrock model you pick (default: Claude Haiku 4.5).

```bash
uv sync

# 1. Decider server (needs the model, e.g. on Apple silicon or a GPU)
uv run strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8099
#    No model handy? A scripted stand-in (NOT the model) exercises the wiring:
#    uv run python simulator/fake_decider.py --port 8099

# 2. The game simulator (one process = one match)
uv run python simulator/serve.py --ticks 60

# 3. The agent, writing per-tick metrics
uv run python -m examples.ctf_agent.agent --match demo --metrics runs/decider.jsonl
```

Environment: `STRANDS_DECIDER_URL` (default `http://127.0.0.1:8099`), `CTF_MCP_URL` (default
`http://127.0.0.1:8765/mcp`, the simulator).

The simulator uses port 8765 on purpose, to stay clear of common dev servers on 8080.

## Measure it: Decider vs LLM-only

```bash
# restart the simulator between runs
uv run python -m examples.ctf_agent.agent --match demo --metrics runs/decider.jsonl
uv run python -m examples.ctf_agent.agent --match demo --metrics runs/llm-only.jsonl --llm-only
uv run python scripts/compare_runs.py runs/decider.jsonl runs/llm-only.jsonl
```

Each tick writes one JSON line: path taken (`decider` / `llm` / `budget` / `wait`), Decider and LLM latency, token
counts when available. The comparison reports average and p95 decision latency, the share of decisions handled
without the LLM, and LLM calls avoided.

A single short local run (25 ticks in decider mode on the 2B model on Apple silicon vs 12 ticks `--llm-only` with
Claude Haiku 4.5) gave about 970 ms average per decision against about 2600 ms, with no LLM call in any decider-mode
tick. Treat it as an illustration, not a benchmark: one match, one machine, tiny samples.

## Benchmark

`scripts/benchmark.py` repeats the Decider vs `--llm-only` comparison over several seeds, simulator only (it refuses
any non-loopback MCP host and port 8080). For each seed it plays both modes in turn on a fresh simulator, then writes
`summary.json` and `summary.md`.

Prerequisites: a Decider server on `STRANDS_DECIDER_URL` (default `http://127.0.0.1:8099`) for decider mode, and AWS
credentials with Bedrock access (both modes can call the LLM). Preflight checks both; skip one with
`--skip-decider-check` / `--skip-aws-check`.

```bash
# see exactly what would run, executes nothing
uv run python scripts/benchmark.py --runs 2 --dry-run

# the real run (calls Bedrock, costs money)
uv run python scripts/benchmark.py --seeds 1..5 --ticks 40 --model haiku
```

Output lands in `runs/bench-<timestamp>/`: one `<mode>-seed<N>.jsonl` and `.log` per run, plus `summary.md` and
`summary.json` (latency avg/p50/p95 over ticks that were timed, Decider-only vs end-to-end, path mix, LLM calls avoided, tokens, estimated cost,
environment). The cost uses a price table in the script that is an assumption: verify it before quoting dollars.

### Cost guards

LLM spend is bounded at several layers. All defaults are conservative and overridable:

- Agent (`--max-llm-calls`, default 60, `0` = never call the LLM): counts real model invocations, so a runaway
  turn cannot exceed it. Past the cap a tick runs the Decider choice (if valid) or waits (`llm_capped` in the metrics).
  Only one LLM thread is allowed at a time (`llm_busy`), and a tick that timed out ends its Agent turn instead of
  letting an abandoned thread call the model again.
- Model: `max_tokens=1024` (`--max-tokens`; 256 truncated tool-call turns), one attempt (no botocore or Strands retries), 10 s read and 5 s connect timeouts.
- Benchmark: `--max-llm-calls-per-run 60`, `--max-total-llm-calls 300`, `--max-cost-usd 5.0` (an estimate from the
  metrics files at assumed prices), `--max-consecutive-failures 2` and `--ticks` at most 200. A hit limit aborts
  the remaining runs and `summary.md` records `aborted: <reason>`. A timed-out run kills the whole agent process group.

If the model still hits `max_tokens`, the tick is recorded as `llm_error` (reason `max_tokens`) and runs the Decider
choice (if valid) or waits; other unexpected LLM errors crash the agent on purpose, so nothing can loop and multiply calls.

Transient Bedrock errors (throttling, `serviceUnavailableException`, model not ready, internal server error) are handled
the same way: the tick is recorded as `llm_error` (reason `transient:<code>`) and falls back to the Decider choice or a
wait, with no app-level retry (still one model call per tick). `--max-consecutive-llm-errors` (default 3, agent and
benchmark) makes the agent exit non-zero after that many in a row, with no further model calls; a successful LLM call
resets the streak. `summary.md` lists the `llm_error` ticks and reasons per run, and a run stopped by that breaker is
`failed` with outcome `llm errors: <reason>`. Any other error still crashes immediately.

If you see these errors often, check your Bedrock quotas (Service Quotas -> Amazon Bedrock: on-demand requests per
minute and tokens per minute for the model you use) and request an increase if needed. This is an account-side action.

Tuning flags (agent, also passed through by the benchmark and shown in `--dry-run` and `summary.md`):
`--llm-budget-ms` (default 3000: the LLM only runs if that plus the 300 ms margin is left, so a ~1.3 s Decider call in a
5 s tick still leaves room for Bedrock), `--threshold` (Decider confidence needed to skip the LLM, default 0.85) and
`--max-tokens` (agent only). Pick a threshold offline and for free with the sweep, which only reads a metrics file:

```
uv run python scripts/threshold_sweep.py runs/bench-<ts>/decider-seed1.jsonl [--out sweep.md] [--min 0.3 --max 0.95 --step 0.05]
```

It prints, per threshold, the share of ticks that would be `decider` vs need the LLM, plus the confidence
distribution. It shows coverage only, not decision quality.

These are estimates, not billing. As the real backstop, create an AWS Budgets alert (and action) on the account
before running the benchmark; that is a step for you, the script cannot do it.

Caveat: the simulator is lock-step, so latency has no game cost, and win rate is not representative. See
[simulator/README.md](simulator/README.md#simulator-fidelity).

## Tests

```bash
uv run pytest
```

No network, AWS credentials or Decider server are needed: the client is tested against a local HTTP server, the
intervention against a real Strands `Agent` with a scripted model, and the agent loop against an in-memory MCP server.

## Layout

```
decider_patterns/   reusable pattern (client, tool_selection, intervention, timing)
examples/ctf_agent/ the CTF example agent and its metrics
simulator/          MCP test double (fidelity notes: simulator/README.md), tiny game engine, HTTP server, scripted stand-in Decider
scripts/            compare_runs.py, benchmark.py
tests/
```

## License

MIT
