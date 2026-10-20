# decider-patterns

A reusable **tool-selection** pattern for [Strands](https://strandsagents.com) agents built on
**Strands Decider 2B**, with Amazon Bedrock as the LLM fallback, plus a Capture the Flag (CTF) example agent
and a local simulator so everything runs end to end on a laptop.

Hackathon: *Agent Experience Hackathon: Strands Decider*. Declared pattern: **#tool-selection**.

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

`examples/ctf_agent/` is a code-owned game loop: `wait_for_turn` -> render state -> one Decider request with three
questions (`action` choice over the server's routes, plus `threat` and `route_bad` yes/no) -> call the tool.
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
scripts/            compare_runs.py
tests/
```

## License

MIT
