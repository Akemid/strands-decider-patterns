# Simulator

An in-memory (and streamable-HTTP) test double of the CTF game's MCP interface, with a tiny dynamic game behind it.
`engine.py` is the game, `fake_ctf_mcp.py` the MCP surface, `serve.py` the local server, `fake_decider.py` a scripted
stand-in for the Decider.

## Simulator fidelity

**Contract: matches the real server for every field the agent reads.** Turn payloads (`routes` with their ids,
ordering, capped `path`, `risks[].cell`, `closest_threat`, `blocked_reason`), `moves`, `teammates`, `agents[].carrying`,
the `join_match` result and the action tool schemas (`rationale`, `direction` enum) follow the real wire shapes.
`tests/test_fake_ctf_contract.py` and `tests/test_sim_engine.py` pin them.

**Dynamics are simplified on purpose:**

- capturing the flag ends the match;
- ticks are lock-step, so latency has no game cost (a slow decision does not lose the tick);
- the map is small and fixed;
- enemies are simple chasers, and teammates stand still;
- there is no event stream (`events` is empty, so `blocked_reason` can only be `occupied`).

`closest_threat.step` counts along the full route, so it may index past the 8 cells shown in `path`.

Use the simulator to check wiring and to measure per-decision latency. Do not draw win-rate or strength
conclusions from it.
