"""Cost guards: orphan-turn stop, LLM call budget and bounded model config (scripted model, no AWS)."""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterable

import pytest
from botocore.exceptions import ClientError, EventStreamError
from strands import Agent, tool
from strands.models import Model
from strands.types.exceptions import ModelThrottledException

from decider_patterns import (
    LlmBudget,
    LlmCallCapReached,
    LlmCallCounter,
    StopAfterToolsHook,
    TickCancelHandler,
    is_cap_error,
    is_transient_llm_error,
    transient_llm_error_code,
)


class ToolLoopModel(Model):
    """Always proposes the `move` tool, so only a hook can end the turn. `before_call` runs inside stream."""

    def __init__(self, before_call=None) -> None:
        self.calls = 0
        self.before_call = before_call

    def update_config(self, **model_config): ...

    def get_config(self):
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncIterable[dict]:
        self.calls += 1
        if self.before_call is not None:
            self.before_call(self.calls)
        yield {"messageStart": {"role": "assistant"}}
        start = {"toolUse": {"toolUseId": f"t{self.calls}", "name": "move"}}
        yield {"contentBlockStart": {"start": start}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"tick": 1})}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}


def make_move(executed):
    @tool
    def move(tick: int) -> str:
        """Move."""
        executed.append(tick)
        return "ok"

    return move


class TestOrphanStop:
    def test_a_cancelled_tick_ends_the_turn_after_the_denied_batch(self):
        cancel = threading.Event()
        executed: list[int] = []
        model = ToolLoopModel(before_call=lambda n: cancel.set())  # the timeout fires while the model answers
        handler = TickCancelHandler(("move",), cancel)
        agent = Agent(model=model, tools=[make_move(executed)], callback_handler=None, interventions=[handler],
                      hooks=[StopAfterToolsHook(("move",), cancel=cancel)])
        agent("go")
        assert executed == []  # the late action was denied
        assert model.calls == 1  # and the model was NOT called again

    def test_an_uncancelled_action_still_ends_the_turn(self):
        executed: list[int] = []
        model = ToolLoopModel()
        agent = Agent(model=model, tools=[make_move(executed)], callback_handler=None,
                      hooks=[StopAfterToolsHook(("move",), cancel=threading.Event())])
        agent("go")
        assert executed == [1] and model.calls == 1


class TestLlmBudget:
    def test_counts_and_exhausts(self):
        budget = LlmBudget(2)
        assert budget.skip_reason() is None
        budget.record_call()
        budget.record_call()
        assert budget.skip_reason() == "llm_capped"
        with pytest.raises(LlmCallCapReached):
            budget.record_call()
        assert budget.calls == 2  # a refused call is not counted

    def test_zero_means_the_llm_is_never_called(self):
        assert LlmBudget(0).skip_reason() == "llm_capped"

    def test_a_live_llm_thread_blocks_the_next_start(self):
        budget = LlmBudget(10)
        budget.begin()
        assert budget.skip_reason() == "llm_busy"
        budget.end()
        assert budget.skip_reason() is None

    def test_the_counter_hook_counts_real_model_invocations(self):
        budget = LlmBudget(1)
        model = ToolLoopModel()
        agent = Agent(model=model, tools=[make_move([])], callback_handler=None,
                      hooks=[LlmCallCounter(budget)])  # no stop hook: the second model call is refused
        with pytest.raises(Exception) as caught:  # noqa: B017 - Strands wraps the hook error
            agent("go")
        assert is_cap_error(caught.value)
        assert model.calls == 1 and budget.calls == 1

    def test_is_cap_error_walks_the_cause_chain(self):
        wrapped = RuntimeError("outer")
        wrapped.__cause__ = LlmCallCapReached("x")
        assert is_cap_error(wrapped) and not is_cap_error(RuntimeError("other"))


def client_error(code, stream=False):
    cls = EventStreamError if stream else ClientError
    return cls({"Error": {"Code": code, "Message": "boom"}}, "ConverseStream")


def wrapped(exc):
    outer = RuntimeError("event loop failed")
    outer.__cause__ = exc
    return outer


class TestTransientLlmError:
    @pytest.mark.parametrize("code", ["ThrottlingException", "ServiceUnavailableException",
                                      "serviceUnavailableException", "ModelNotReadyException",
                                      "InternalServerException", "internalServerException",
                                      "modelStreamErrorException"])
    @pytest.mark.parametrize("stream", [False, True])
    def test_transient_codes(self, code, stream):
        assert is_transient_llm_error(client_error(code, stream))
        assert is_transient_llm_error(wrapped(client_error(code, stream)))
        assert transient_llm_error_code(wrapped(client_error(code, stream))) == code

    def test_model_throttled_exception_plain_and_wrapped(self):
        assert is_transient_llm_error(ModelThrottledException("slow down"))
        assert transient_llm_error_code(wrapped(ModelThrottledException("slow down"))) == "ThrottlingException"

    def test_throttled_wrapping_a_client_error_reports_its_code(self):
        exc = ModelThrottledException("slow down")
        exc.__cause__ = client_error("ThrottlingException")
        assert transient_llm_error_code(exc) == "ThrottlingException"

    @pytest.mark.parametrize("exc", [client_error("AccessDeniedException"), client_error("ValidationException", True),
                                     client_error("ResourceNotFoundException"), RuntimeError("x"), ValueError("y"),
                                     wrapped(client_error("AccessDeniedException")), None])
    def test_non_transient(self, exc):
        assert not is_transient_llm_error(exc)
        assert transient_llm_error_code(exc) is None
