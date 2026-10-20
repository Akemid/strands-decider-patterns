"""Tests for the Strands intervention recipes (no Strands Agent, no network)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from strands.interventions import Deny, Guide, Proceed

from decider_patterns.client import ServerUnavailable
from decider_patterns.intervention import (
    DeciderGateHandler,
    GuideRule,
    StopAfterToolsHook,
    TickCancelHandler,
)

GATED = ("transfer",)


def event(tool="transfer", **arguments):
    return SimpleNamespace(
        tool_use={"name": tool, "input": arguments},
        agent=SimpleNamespace(messages=[{"role": "user", "content": [{"text": "hello"}]}]),
    )


def noul(p):
    return {"type": "noul", "noul": p}


QUESTIONS = {"grounded": {"type": "noul", "instructions": "are the arguments grounded?"}}
RULES = (GuideRule("grounded", "Use values you have seen.", yes_below=0.45),)


def handler(ask, **kw):
    return DeciderGateHandler(
        ask=ask,
        gated_tools=GATED,
        questions=QUESTIONS,
        rules=RULES,
        describe=lambda ev: f"call {ev.tool_use['name']}",
        **kw,
    )


class TestGuideRule:
    def test_fires_below_and_above(self):
        below = GuideRule("k", "f", yes_below=0.5)
        above = GuideRule("k", "f", yes_at_least=0.5)
        assert below.fires({"k": noul(0.4)}) and not below.fires({"k": noul(0.5)})
        assert above.fires({"k": noul(0.5)}) and not above.fires({"k": noul(0.4)})

    def test_unusable_answer_never_fires_a_yes_below_rule(self):
        rule = GuideRule("k", "f", yes_below=0.5)
        for garbage in (None, "x", {}, {"noul": "high"}, {"noul": True}):
            assert not rule.fires({"k": garbage})

    def test_missing_answer_never_fires(self):
        assert not GuideRule("k", "f", yes_below=0.5).fires({})


class TestDeciderGateHandler:
    def test_ungated_tools_proceed_without_asking(self):
        def ask(state, questions):
            raise AssertionError("must not ask")

        assert isinstance(handler(ask).before_tool_call(event(tool="lookup")), Proceed)

    def test_grounded_call_proceeds(self):
        result = handler(lambda s, q: {"grounded": noul(0.9)}).before_tool_call(event(amount=5))
        assert isinstance(result, Proceed)

    def test_ungrounded_call_is_guided_with_the_rule_feedback(self):
        result = handler(lambda s, q: {"grounded": noul(0.1)}).before_tool_call(event(amount=5))
        assert isinstance(result, Guide) and result.feedback == "Use values you have seen."

    def test_state_text_carries_the_call_description(self):
        seen = []

        def ask(state, questions):
            seen.append(state)
            return {"grounded": noul(0.9)}

        handler(ask).before_tool_call(event())
        assert "call transfer" in seen[0] and "hello" in seen[0]

    def test_guide_cap_lets_the_call_through_so_the_agent_never_stalls(self):
        gate = handler(lambda s, q: {"grounded": noul(0.0)}, max_consecutive_guides=2)
        kinds = [type(gate.before_tool_call(event())).__name__ for _ in range(4)]
        assert kinds == ["Guide", "Guide", "Proceed", "Guide"]

    def test_unavailable_decider_fails_open_and_stays_off(self):
        calls = []

        def ask(state, questions):
            calls.append(1)
            raise ServerUnavailable("down")

        gate = handler(ask)
        assert isinstance(gate.before_tool_call(event()), Proceed)
        assert isinstance(gate.before_tool_call(event()), Proceed)
        assert len(calls) == 1

    def test_garbage_answers_do_not_guide(self):
        result = handler(lambda s, q: {"grounded": {"type": "noul"}}).before_tool_call(event())
        assert isinstance(result, Proceed)

    def test_malformed_reply_from_a_real_client_fails_open(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from decider_patterns.client import Decider

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["content-length"]))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"garbage")

        httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            decider = Decider(f"http://127.0.0.1:{httpd.server_port}")
            gate = handler(decider.ask)
            assert isinstance(gate.before_tool_call(event()), Proceed)
            assert gate.decider_down
        finally:
            httpd.shutdown()

    def test_timeout_s_is_passed_to_ask(self):
        seen = []

        def ask(state, questions, timeout=None):
            seen.append(timeout)
            return {"grounded": noul(0.9)}

        handler(ask, timeout_s=2.0).before_tool_call(event())
        assert seen == [2.0]

    def test_retry_after_reenables_the_gate_and_reset_clears(self):
        now = [0.0]
        calls = []

        def ask(state, questions):
            calls.append(1)
            if len(calls) == 1:
                raise ServerUnavailable("down")
            return {"grounded": noul(0.1)}

        gate = handler(ask, retry_after_s=5, clock=lambda: now[0])
        assert isinstance(gate.before_tool_call(event()), Proceed)
        assert isinstance(gate.before_tool_call(event()), Proceed) and len(calls) == 1
        now[0] = 6.0
        assert isinstance(gate.before_tool_call(event()), Guide)
        gate.decider_down = True
        gate.reset()
        assert not gate.decider_down

    def test_decisions_are_logged(self):
        gate = handler(lambda s, q: {"grounded": noul(0.1)})
        gate.before_tool_call(event())
        assert gate.log == [{"tool": "transfer", "verdict": "guide", "grounded": 0.1}]


class TestTickCancelHandler:
    def test_records_an_action_and_blocks_after_cancel(self):
        cancel = threading.Event()
        gate = TickCancelHandler(("move", "wait"), cancel)
        assert isinstance(gate.before_tool_call(event(tool="move")), Proceed)
        assert gate.cancel_tick() is True
        assert isinstance(gate.before_tool_call(event(tool="move")), Deny)

    def test_non_action_tools_do_not_count_as_acted(self):
        gate = TickCancelHandler(("move",), threading.Event())
        gate.before_tool_call(event(tool="get_map"))
        assert gate.cancel_tick() is False


class TestStopAfterToolsHook:
    def test_ends_the_turn_only_after_an_action_tool(self):
        hook = StopAfterToolsHook(("move",))
        done = SimpleNamespace(end_turn=None)
        hook.after_tools(done)
        assert done.end_turn is None
        hook.after_tool_call(SimpleNamespace(tool_use={"name": "move"}, cancel_message=None))
        hook.after_tools(done)
        assert done.end_turn == "action sent"

    def test_a_cancelled_action_does_not_count(self):
        hook = StopAfterToolsHook(("move",))
        hook.after_tool_call(SimpleNamespace(tool_use={"name": "move"}, cancel_message="denied"))
        done = SimpleNamespace(end_turn=None)
        hook.after_tools(done)
        assert done.end_turn is None
