"""Agent-level cost guards: LLM call cap, one live LLM thread, bounded model config (no AWS, no network)."""

from __future__ import annotations

import pytest
from botocore.exceptions import EventStreamError
from strands import ModelRetryStrategy
from strands.types.exceptions import MaxTokensReachedException, ModelThrottledException

from decider_patterns import LlmCallCounter, StopAfterToolsHook, is_max_tokens_error
from examples.ctf_agent import agent as dl
from tests.test_ctf_agent import SequenceClient, route, turn_with
from tests.test_ctf_metrics_loop import ScriptedDecider, parse, read


class Harness:
    """Runs `play` with a fake Agent whose model invocations are counted through the real LlmCallCounter."""

    def __init__(self, timed_out=False, raises=None):
        self.timed_out = timed_out
        self.raises = raises  # an exception, or a list of exceptions/None (None = success) consumed per call
        self.llm_ticks: list[int] = []
        self.agent_kwargs: list[dict] = []

    def run(self, args, decider=None, ticks=4):
        turns = [{**turn_with([route("enemy_flag_direct", "enemy_flag")], tick=t), "deadline_remaining_ms": 4000}
                 for t in range(1, ticks + 1)] + [{"status": "match_over"}]
        client = SequenceClient(turns)

        def fake_agent(**kwargs):
            self.agent_kwargs.append(kwargs)
            return object()

        def fake_llm_tick(agent_factory, state, tick, handler=None, usage=None):
            agent_factory(handler)
            counter = next(h for h in self.agent_kwargs[-1]["hooks"] if isinstance(h, LlmCallCounter))
            counter.before_model_call(None)  # one model invocation, as Strands would fire it
            self.llm_ticks.append(tick)
            raises = self.raises.pop(0) if isinstance(self.raises, list) else self.raises
            if raises is not None:
                raise raises
            return 1, ("move", {"direction": "left", "tick": tick})

        def fake_timeout(fn, timeout_s):
            return (True, None) if self.timed_out else (False, fn())

        saved = dl.run_llm_tick, dl.run_with_timeout, dl.Agent, dl.build_model
        dl.run_llm_tick, dl.run_with_timeout, dl.Agent = fake_llm_tick, fake_timeout, fake_agent
        dl.build_model = lambda key, max_tokens=None: key
        try:
            dl.play(client, decider, args)
        finally:
            dl.run_llm_tick, dl.run_with_timeout, dl.Agent, dl.build_model = saved
        return client


class TestCli:
    def test_default_cap_is_60_and_zero_is_accepted(self):
        assert parse().max_llm_calls == 60
        assert parse("--max-llm-calls", "0").max_llm_calls == 0


class TestCap:
    def test_the_llm_stops_being_called_once_the_cap_is_reached(self, tmp_path):
        target = tmp_path / "m.jsonl"
        harness = Harness()
        client = harness.run(parse("--llm-only", "--max-llm-calls", "2", "--metrics", str(target)))
        assert harness.llm_ticks == [1, 2]
        assert [r["path"] for r in read(target)] == ["llm", "llm", "llm_capped", "llm_capped"]
        waits = [c for c in client.calls if c[0] == "wait"]
        assert waits == [("wait", {"tick": 3}), ("wait", {"tick": 4})]  # llm-only has no Decider choice

    def test_zero_never_calls_the_llm_and_the_decider_choice_still_runs(self, tmp_path):
        target = tmp_path / "m.jsonl"
        harness = Harness()
        decider = ScriptedDecider([("enemy_flag_direct", 0.2)] * 2)  # low confidence: would go to the LLM
        client = harness.run(parse("--no-warmup", "--max-llm-calls", "0", "--llm-budget-ms", "100",
                                    "--safety-margin-ms", "0", "--metrics", str(target)), decider, ticks=2)
        assert harness.llm_ticks == []
        assert [r["path"] for r in read(target)] == ["llm_capped", "llm_capped"]
        moves = [c for c in client.calls if c[0] in ("move", "grab_flag")]
        assert len(moves) == 2  # the valid Decider pick was executed instead of waiting


class TestOneLiveThread:
    def test_a_timed_out_llm_thread_blocks_the_next_llm_start(self, tmp_path):
        target = tmp_path / "m.jsonl"
        harness = Harness(timed_out=True)  # the fake never finishes fn, like an abandoned thread
        client = harness.run(parse("--llm-only", "--metrics", str(target)), ticks=3)
        paths = [(r["path"], r["llm_timeout"]) for r in read(target)]
        assert paths == [("llm", True), ("llm_busy", False), ("llm_busy", False)]
        assert [c for c in client.calls if c[0] == "wait"] == [("wait", {"tick": t}) for t in (1, 2, 3)]


def max_tokens_error(wrapped=False):
    exc = MaxTokensReachedException("max tokens")
    if not wrapped:
        return exc
    outer = RuntimeError("event loop failed")
    outer.__cause__ = exc
    return outer

class TestMaxTokens:
    def test_cli_override_and_default(self):
        assert parse().max_tokens == dl.MODEL_MAX_TOKENS == 1024
        assert parse("--max-tokens", "512").max_tokens == 512

    def test_build_model_takes_max_tokens(self, monkeypatch):
        seen: dict = {}
        monkeypatch.setattr(dl, "BedrockModel", lambda **kw: seen.update(kw) or "model")
        dl.build_model("haiku", max_tokens=333)
        assert seen["max_tokens"] == 333

    def test_is_max_tokens_error_walks_the_cause_chain(self):
        assert is_max_tokens_error(max_tokens_error())
        assert is_max_tokens_error(max_tokens_error(wrapped=True))
        assert not is_max_tokens_error(RuntimeError("other"))

    def test_llm_only_tick_waits_instead_of_crashing(self, tmp_path):
        target = tmp_path / "m.jsonl"
        harness = Harness(raises=max_tokens_error())
        client = harness.run(parse("--llm-only", "--metrics", str(target)), ticks=2)
        records = read(target)
        assert [(r["path"], r["reason"]) for r in records] == [("llm_error", "max_tokens")] * 2
        assert [c for c in client.calls if c[0] == "wait"] == [("wait", {"tick": 1}), ("wait", {"tick": 2})]

    def test_wrapped_error_falls_back_to_the_valid_decider_choice(self, tmp_path):
        target = tmp_path / "m.jsonl"
        harness = Harness(raises=max_tokens_error(wrapped=True))
        decider = ScriptedDecider([("enemy_flag_direct", 0.2)] * 2)  # low confidence: goes to the LLM
        client = harness.run(parse("--no-warmup", "--llm-budget-ms", "100", "--safety-margin-ms", "0",
                                    "--metrics", str(target)), decider, ticks=2)
        assert [r["path"] for r in read(target)] == ["llm_error", "llm_error"]
        assert len([c for c in client.calls if c[0] in ("move", "grab_flag")]) == 2
        assert harness.llm_ticks == [1, 2]  # one model call per tick, no retry loop

    def test_unknown_llm_errors_still_crash_and_never_retry(self):
        harness = Harness(raises=ValueError("boom"))
        try:
            harness.run(parse("--llm-only"), ticks=3)
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown error must propagate")
        assert harness.llm_ticks == [1]  # no further LLM call after the crash

class TestAgentBuild:
    def test_the_factory_wires_cancel_counter_and_a_single_attempt_retry(self):
        import threading

        from decider_patterns import LlmBudget, TickCancelHandler

        captured: dict = {}
        saved = dl.Agent
        dl.Agent = lambda **kw: captured.update(kw)
        try:
            budget = LlmBudget(5)
            handler = TickCancelHandler(dl.ACTION_TOOLS, threading.Event())
            dl.build_agent_factory(object(), [], "sys", budget)(handler)
        finally:
            dl.Agent = saved
        stop = next(h for h in captured["hooks"] if isinstance(h, StopAfterToolsHook))
        assert stop.cancel is handler.cancel
        assert any(isinstance(h, LlmCallCounter) and h.budget is budget for h in captured["hooks"])
        assert isinstance(captured["retry_strategy"], ModelRetryStrategy)
        assert captured["retry_strategy"]._max_attempts == 1


class TestModelLimits:
    def test_bedrock_model_is_bounded(self, monkeypatch):
        seen: dict = {}
        monkeypatch.setattr(dl, "BedrockModel", lambda **kw: seen.update(kw) or "model")
        assert dl.build_model("haiku") == "model"
        assert seen["model_id"] == dl.MODELS["haiku"]
        assert seen["max_tokens"] == dl.MODEL_MAX_TOKENS == 1024
        cfg = seen["boto_client_config"]
        assert cfg.retries == {"max_attempts": 1, "mode": "standard"}
        assert cfg.read_timeout == 10 and cfg.connect_timeout == 5


def transient_error(code="serviceUnavailableException"):
    return EventStreamError({"Error": {"Code": code, "Message": "busy"}}, "ConverseStream")

class TestTransientLlmErrors:
    def test_cli_default_and_override(self):
        assert parse().max_consecutive_llm_errors == dl.DEFAULT_MAX_CONSECUTIVE_LLM_ERRORS == 3
        assert parse("--max-consecutive-llm-errors", "5").max_consecutive_llm_errors == 5

    def test_transient_error_falls_back_and_continues(self, tmp_path):
        target = tmp_path / "m.jsonl"
        harness = Harness(raises=[transient_error(), None, transient_error("ThrottlingException"), None])
        client = harness.run(parse("--llm-only", "--metrics", str(target)), ticks=4)
        records = read(target)
        assert [(r["path"], r.get("reason")) for r in records] == [
            ("llm_error", "transient:serviceUnavailableException"), ("llm", None),
            ("llm_error", "transient:ThrottlingException"), ("llm", None)]
        assert [c for c in client.calls if c[0] == "wait"] == [("wait", {"tick": 1}), ("wait", {"tick": 3})]
        assert harness.llm_ticks == [1, 2, 3, 4]  # exactly one model call per tick, no app-level retry

    def test_wrapped_throttling_uses_the_decider_choice(self, tmp_path):
        target = tmp_path / "m.jsonl"
        outer = RuntimeError("loop")
        outer.__cause__ = ModelThrottledException("ThrottlingException (reached max retries: 1)")
        harness = Harness(raises=outer)
        decider = ScriptedDecider([("enemy_flag_direct", 0.2)] * 2)
        client = harness.run(parse("--no-warmup", "--llm-budget-ms", "100", "--safety-margin-ms", "0",
                                    "--metrics", str(target)), decider, ticks=2)
        assert [(r["path"], r["reason"]) for r in read(target)] == [("llm_error", "transient:ThrottlingException")] * 2
        assert len([c for c in client.calls if c[0] in ("move", "grab_flag")]) == 2

    def test_consecutive_transient_errors_stop_the_agent(self, tmp_path):
        target = tmp_path / "m.jsonl"
        harness = Harness(raises=transient_error())
        with pytest.raises(dl.LlmErrorLimitReached, match="3 consecutive"):
            harness.run(parse("--llm-only", "--metrics", str(target)), ticks=6)
        assert harness.llm_ticks == [1, 2, 3]  # no model call after the breaker trips
        assert len(read(target)) == 3

    def test_a_success_resets_the_streak(self, tmp_path):
        errs = [transient_error(), transient_error(), None, transient_error(), transient_error(), None]
        harness = Harness(raises=errs)
        harness.run(parse("--llm-only", "--metrics", str(tmp_path / "m.jsonl")), ticks=6)
        assert harness.llm_ticks == [1, 2, 3, 4, 5, 6]

    def test_limit_is_configurable(self):
        harness = Harness(raises=transient_error())
        with pytest.raises(dl.LlmErrorLimitReached):
            harness.run(parse("--llm-only", "--max-consecutive-llm-errors", "1"), ticks=4)
        assert harness.llm_ticks == [1]

    def test_non_transient_error_crashes_after_exactly_one_call(self):
        from botocore.exceptions import ClientError

        harness = Harness(raises=ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}},
                                             "ConverseStream"))
        with pytest.raises(ClientError):
            harness.run(parse("--llm-only"), ticks=3)
        assert harness.llm_ticks == [1]

    def test_main_exits_non_zero_with_a_clear_message(self, monkeypatch, capsys):
        args = parse("--llm-only")
        monkeypatch.setattr(dl, "parse_args", lambda name=dl.DEFAULT_NAME: args)
        monkeypatch.setattr(dl.MCPClient, "load_servers", staticmethod(lambda servers: [object()]))

        def boom(client, body):
            raise dl.LlmErrorLimitReached("3 consecutive transient LLM errors (last: ThrottlingException)")

        monkeypatch.setattr(dl, "with_client", boom)
        assert dl.main() == 3
        assert "3 consecutive transient LLM errors" in capsys.readouterr().err
