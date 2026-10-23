# CTF example agent

A Capture the Flag player whose loop is plain Python and whose per-tick decision is a Strands Decider
**tool selection** (see [../../decider_patterns/README.md](../../decider_patterns/README.md)).

```
wait_for_turn -> render state -> filter options in code -> ONE Decider request -> call the tool directly
                                        |
                       unsure / unreachable: Strands Agent on Bedrock, once, restricted to the action tools
```

## Modules

| File | Role |
| --- | --- |
| `agent.py` | The loop (`play`), CLI, MCP join, LLM fallback agent, per-tick metrics. |
| `state.py` | Pure parsing of turns and the static map, plus the loop-owned path `Plan`. |
| `options.py` | Routes to options, label to tool call, and `filter_options` (code decides facts first), `combine_answers`. |
| `render.py` | Turn to text for the Decider and the LLM. |
| `metrics.py` | JSONL per-tick records and the run comparison used by `scripts/compare_runs.py`. |
| `process.py` | Guarded entry point, MCP client lifecycle, forced exit if threads linger. |

## Filter in code, then one question

Code decides facts, the Decider decides judgment (the "narrow, then choose" pattern in
[../../decider_patterns/README.md](../../decider_patterns/README.md)). `filter_options` runs before the request:

- `flag_threatened`: my flag is carried by an enemy, or an enemy inside my field is within the defend radius (BFS
  steps around walls) of my flag base. The radius approximates the game's defend radius: 4 steps on a 15-wide map, scaled by the map width taken from
  the MCP map (no map: the radius check is off, logged once, and only a carried flag counts).
- Threatened and an unblocked `tag:*` / `tag_*` / `defend_flag` option exists: offer only those plus `wait`.
- Otherwise drop the bad routes (blocked now, or with risks). `wait` always stays.
- If nothing but `wait` would remain, the full set is offered (`lesser_evil`).

The request then holds ONE question, the `action` choice over what is left, with short instructions and compact
option texts. The decision is the choice and its confidence; the per-tick `reason` is the filter reason
(`filtered`, `lesser_evil`, `unfiltered`). The metrics JSONL also records `filter_reason`, `options_offered` and
`decider_input_tokens`. Shadow mode (`--llm-only --shadow-decider`) asks the same filtered single question.

## Run

```bash
uv run python simulator/serve.py                      # terminal 1
uv run python -m examples.ctf_agent.agent --match demo  # terminal 2
```

Useful flags: `--threshold` (Decider confidence needed to skip the LLM, default 0.85; the reported real-server
results use `--threshold 0.70`, picked from coverage on real turns with `scripts/threshold_sweep.py`),
`--llm-only` (ablation, no Decider server needed), `--shadow-decider` (with `--llm-only`: also ask the Decider after
the LLM acted, never executed, to calibrate the threshold), `--model haiku|opus|nova-lite`, `--metrics PATH`,
`--raw-turns` (print the state and options sent to the Decider), `--no-warmup`, `--team red|blue`,
`--agent-index N`, `--record-turns PATH`.

Cost guards: `--max-llm-calls N` (real model invocations per run, default 60, `0` = never call Bedrock),
`--max-tokens N` (Bedrock `max_tokens`, default 1024), `--max-consecutive-llm-errors N` (stop after N transient
Bedrock errors in a row, such as throttling, default 3; each one is handled per tick first), `--llm-budget-ms`
(see below).

Environment: `CTF_MCP_URL` (default `http://127.0.0.1:8765/mcp`), `STRANDS_DECIDER_URL`.

## Time budget

Each tick has a deadline (5 s by default). The LLM fallback only runs when `llm_budget_ms + safety_margin_ms`
(3000 + 300 by default) is still left. With less time left, the Decider's own pick runs (`budget`), or `wait` if
there is none. With a 5 s deadline and a Decider that takes about a second, most unsure ticks end up on `budget`;
raise the deadline or lower `--llm-budget-ms` to let the LLM decide more of them.

## Against a real game server

The agent speaks the MCP turn contract v2 (`join_match`, `wait_for_turn`, `move`, `tag`, `grab_flag`, `wait`). Point
`CTF_MCP_URL` at any server implementing it. The simulator is a test double, not the real game. The agent never
imports simulator code: the map (width, walls, fields, flag bases) comes only from `join_match.map` or `get_map`.

### Run against your own game server

Zero Bedrock calls (`--max-llm-calls 0`; the Decider server is still used unless you add `--llm-only`):

```bash
CTF_MCP_URL=http://127.0.0.1:8080/mcp uv run python -m examples.ctf_agent.agent --match <id> \
  --max-llm-calls 0 --metrics runs/real/<id>.jsonl --record-turns runs/real/<id>-turns.jsonl
```

Add `--team red|blue` and `--agent-index N` if the server needs a specific seat (default: red, then blue).
`--record-turns` appends the join/map payload once and each raw `wait_for_turn` payload as JSONL; no headers or
tokens are written. Recordings of a private server must not be committed: `runs/` is gitignored, so write there.
