# decider_patterns: tool selection with Strands Decider

A drop-in decision pattern: **a typed question plus an option set, wired into the agent's path**, with an LLM
fallback when the Decider is not sure. Nothing here depends on any particular domain.

## 1. The typed question and the option set

You provide the options; the Decider picks one. Option names are read from the request, so you can change them
without retraining.

```python
options = {
    "search_docs": "look the answer up in the documentation",
    "create_ticket": "open a support ticket for a human",
    "reply": "answer the user directly",
    "wait": "do nothing this turn",
}
```

Write each description as a neutral fact, not a recommendation: the Decider reads them like a list of choices.
Include a safe default (`wait`) so there is always something valid to run.

## 2. Narrow, then choose

Code decides FACTS, the Decider decides JUDGMENT. Anything your code can compute from the state (a tool is
blocked, a target is out of range, an action is unsafe) should not be a question for the Decider: drop those
options first, then ask ONE `choice` over what is left. The request gets shorter (latency grows with input
tokens: about 1 ms per token on an M3 Pro, measured) and the Decider only spends its judgment where it is needed.

```python
from decider_patterns import ToolSelectionGate, narrow_options

def option_filter(options, state):
    # facts you can compute; keep(label) -> True means "still offer it"
    return narrow_options(options, lambda label: label not in blocked_tools(state))

gate = ToolSelectionGate("Pick the next action.", option_filter=option_filter)
decision = gate.select(ask, state_text, options, remaining_ms=time_left_ms)
decision.filter_reason, decision.options_offered   # log them next to the confidence
```

Rules of `narrow_options(options, keep, always_keep=("wait",))`:

- `always_keep` labels (the safe default) survive every filter.
- If nothing but the always-kept options would remain, the FULL set is offered instead (the lesser evil: picking the
  least bad real option beats choosing between one bad option and doing nothing). Reason: `lesser_evil`.
- `filter_reason` is `filtered` (something was dropped), `lesser_evil`, or `unfiltered` (nothing was dropped).
- The route and the validity of the pick are checked against the narrowed set, so a pick outside it never runs.

Facts that depend on more than the option label (a rule that switches on a threat) belong in your own filter
function; see `filter_options` in `examples/ctf_agent/options.py`. Also keep the request text short: one-sentence
instructions and compact option descriptions (goal, size, first step, flags such as BLOCKED or risk).

## 3. Wire it into the agent loop (`ToolSelectionGate`)

```python
from strands import Agent, tool
from decider_patterns import Decider, ToolSelectionGate, ask_serialized

decider = Decider()  # STRANDS_DECIDER_URL or http://127.0.0.1:8099
gate = ToolSelectionGate("Pick the next action for the support agent.", threshold=0.85, timeout_s=4.0)

decision = gate.select(
    lambda state, questions: ask_serialized(decider, state, questions),
    state_text,
    options,
    remaining_ms=time_left_ms,
)

if decision.route in ("decider", "budget"):   # confident, or no time left for an LLM
    run_tool(decision.label)
elif decision.route == "llm":                 # the fallback hook-in point
    run_llm_fallback(state_text)              # see the Bedrock example below
else:                                         # "wait": nothing valid and no time
    run_safe_default()
```

The fallback is a Strands `Agent` on Bedrock whose tools are only the action set (same approach as
`examples/ctf_agent`, which passes the model id string to `Agent`):

```python
@tool
def reply(text: str) -> str:
    """Answer the user directly."""
    return send_reply(text)

@tool
def wait() -> str:
    """Do nothing this turn."""
    return "waited"

def run_llm_fallback(state_text: str):
    agent = Agent(
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",  # Bedrock model id
        tools=[reply, wait],                                      # only the option set
        system_prompt="Pick exactly one action tool for the current state.",
        callback_handler=None,
    )
    return agent(state_text)
```

Routes, from `choose_path`:

| Route | When |
| --- | --- |
| `decider` | the pick is an offered option and `confidence >= threshold` |
| `llm` | not confident (or Decider down) and at least `llm_budget_ms + safety_margin_ms` is left |
| `budget` | not confident, no time for the LLM, but the pick is valid: run it |
| `wait` | nothing valid to run and no time |

Notes:

- `confidence` is the answer's `confidence` field, not the top probability. Tune `threshold` on your own data:
  lower means fewer LLM calls and more risk.
- A failing Decider (unreachable, or a malformed reply, both surfaced as `ServerUnavailable`) flips
  `gate.decider_down` and later decisions route to the fallback; the agent keeps going. The flag is sticky by
  default: pass `retry_after_s=30` to try the Decider again after a cooldown, or call `gate.reset()`.
- `Decider()` defaults to a 120 s timeout (the first request at a new input length can compile for seconds), far
  above a 5 s decision window. Pass `timeout_s=` to the gate (or `timeout_s=` to `DeciderGateHandler`): each request
  then uses `min(timeout_s, time left)`. Use a separate `decider.ask(...)` warm-up for the slow first request.
- Extra yes/no questions can ride in the same request (`gate.select(..., extra={"risky": Decider.noul(...)})`) and
  be combined in your own policy (`gate.decide(..., label=, confidence=)`), but every question adds input tokens
  and so latency. Prefer filtering facts in code (section 2); the CTF example moved from three questions to one.
- If your fallback LLM has a time budget, run it with `run_with_timeout` and guard it with `TickCancelHandler`, so a
  late action from an abandoned run is blocked and you never send a second action. Add `StopAfterToolsHook` to end
  the turn right after the action tool, saving a model call.

## 4. Or gate individual tool calls (`DeciderGateHandler`)

When the LLM stays in charge and you only want a cheap safety review before risky tools, use the Strands
intervention. It asks yes/no questions about the proposed call and, if a rule fires, cancels it and sends feedback so
the model retries.

```python
from strands import Agent
from decider_patterns import Decider, DeciderGateHandler, GuideRule, ask_serialized

decider = Decider()
gate = DeciderGateHandler(
    ask=lambda state, questions: ask_serialized(decider, state, questions),
    gated_tools=("transfer_funds",),                         # only review the costly ones
    questions={
        "grounded": Decider.noul(
            "Are all argument values backed by the conversation?",
            {"true": "every value appears in the conversation", "false": "a value was guessed"},
        ),
    },
    rules=(GuideRule("grounded", "Use only values from the conversation.", yes_below=0.45),),
    describe=lambda ev: f"{ev.tool_use['name']}: {ev.tool_use['input']}",
)
agent = Agent(model=..., tools=[transfer_funds], interventions=[gate])
```

Behavior worth knowing:

- Ungated tools proceed without a Decider call.
- It fails open: if the Decider is unreachable or replies with garbage the tool proceeds and the gate disables
  itself (sticky unless `retry_after_s` is set; `gate.reset()` re-enables it).
- A rule never fires on an unusable answer (`parse_noul_or_none` returns `None`), even for `yes_below`.
- Consecutive guides are capped (`max_consecutive_guides`, default 2), then the call proceeds, so a strict gate cannot
  stall the agent.
- `gate.log` records each decision with its scores.

## Running a Decider

```bash
uv run strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8099
```

Send requests one at a time: use `ask_serialized` when several agent processes share one server. It takes a
cross-process file lock where `fcntl` exists (macOS/Linux) and skips the lock on Windows.
