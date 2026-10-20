"""DeciderGateHandler wired into a real Strands Agent with a scripted model (no AWS, no Decider server)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable

from strands import Agent, tool
from strands.models import Model

from decider_patterns import DeciderGateHandler, GuideRule


class ScriptedModel(Model):
    """First call proposes the `transfer` tool; any later call just answers with text."""

    def __init__(self) -> None:
        self.calls = 0

    def update_config(self, **model_config): ...

    def get_config(self):
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncIterable[dict]:
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 1:
            start = {"toolUse": {"toolUseId": "t1", "name": "transfer"}}
            yield {"contentBlockStart": {"start": start}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"amount": 5})}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}


def run_agent(p_grounded: float):
    executed = []

    @tool
    def transfer(amount: int) -> str:
        """Transfer money."""
        executed.append(amount)
        return "ok"

    gate = DeciderGateHandler(
        ask=lambda state, questions: {"grounded": {"type": "noul", "noul": p_grounded}},
        gated_tools=("transfer",),
        questions={"grounded": {"type": "noul", "instructions": "are the arguments grounded?"}},
        rules=(GuideRule("grounded", "Use values you have seen.", yes_below=0.45),),
        describe=lambda ev: f"{ev.tool_use['name']} {json.dumps(ev.tool_use['input'])}",
    )
    agent = Agent(model=ScriptedModel(), tools=[transfer], interventions=[gate], callback_handler=None)
    agent("move five units")
    return executed, gate


def test_an_approved_call_runs_the_tool():
    executed, gate = run_agent(0.9)
    assert executed == [5]
    assert gate.log[0]["verdict"] == "proceed"


def test_a_guided_call_never_reaches_the_tool():
    executed, gate = run_agent(0.1)
    assert executed == []
    assert gate.log[0]["verdict"] == "guide"
