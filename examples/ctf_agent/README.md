# CTF example agent

A Capture the Flag player whose loop is plain Python and whose per-tick decision is a Strands Decider
**tool selection** (see [../../decider_patterns/README.md](../../decider_patterns/README.md)).

```
wait_for_turn -> render state -> ONE Decider request -> call the tool directly
                                        |
                       unsure / unreachable: Strands Agent on Bedrock, once, restricted to the action tools
```

## Modules

| File | Role |
| --- | --- |
| `agent.py` | The loop (`play`), CLI, MCP join, LLM fallback agent, per-tick metrics. |
| `state.py` | Pure parsing of turns and the static map, plus the loop-owned path `Plan`. |
| `options.py` | Routes to options, label to tool call, and `combine_answers` (merges the three answers). |
| `render.py` | Turn to text for the Decider and the LLM. |
| `metrics.py` | JSONL per-tick records and the run comparison used by `scripts/compare_runs.py`. |
| `process.py` | Guarded entry point, MCP client lifecycle, forced exit if threads linger. |

## One request, three questions

The state is encoded once and shared:

- `action`: choice over the routes the server offers (the tool-selection question).
- `threat`: yes/no, is my flag threatened. If it fires and a tag or defend option exists, that wins.
- `route_bad`: yes/no, is the route I would continue blocked or risky. If it fires, switch to a safe same-goal route.

`combine_answers` is a small pure function, so the policy is easy to read and test.

## Run

```bash
uv run python simulator/serve.py                      # terminal 1
uv run python -m examples.ctf_agent.agent --match demo  # terminal 2
```

Useful flags: `--threshold` (Decider confidence needed to skip the LLM, default 0.85), `--threat-threshold`,
`--llm-only` (ablation, no Decider server needed), `--model haiku|opus|nova-lite`, `--metrics PATH`,
`--raw-turns` (print the state and options sent to the Decider), `--no-warmup`.

Environment: `CTF_MCP_URL` (default `http://127.0.0.1:8765/mcp`), `STRANDS_DECIDER_URL`.

## Time budget

Each tick has a deadline (5 s by default). The LLM fallback only runs when `llm_budget_ms + safety_margin_ms`
(4500 + 300 by default) is still left. With less time left, the Decider's own pick runs (`budget`), or `wait` if
there is none. With a 5 s deadline and a Decider that takes about a second, most unsure ticks end up on `budget`;
raise the deadline or lower `--llm-budget-ms` to let the LLM decide more of them.

## Against a real game server

The agent speaks the MCP turn contract v2 (`join_match`, `wait_for_turn`, `move`, `tag`, `grab_flag`, `wait`). Point
`CTF_MCP_URL` at any server implementing it. The simulator is a test double, not the real game.
