"""Tests for the reusable tool-selection gate (pure logic, no network)."""

from __future__ import annotations

import pytest

from decider_patterns.client import ServerUnavailable
from decider_patterns.tool_selection import (
    ACTION_KEY,
    ToolSelectionGate,
    accept_choice,
    choose_path,
    parse_choice,
    parse_noul,
    parse_noul_or_none,
    timeout_action,
)

OPTIONS = {"search": "look something up", "answer": "reply to the user", "wait": "do nothing"}


def answer(label: str | None, confidence: float | None = 0.9, probabilities=None):
    payload: dict = {"type": "choice", "choice": label}
    if confidence is not None:
        payload["confidence"] = confidence
    if probabilities is not None:
        payload["probabilities"] = probabilities
    return {ACTION_KEY: payload}


class TestParsing:
    def test_parse_choice_reads_label_and_confidence(self):
        assert parse_choice({"choice": "search", "confidence": 0.7}) == ("search", 0.7)

    def test_parse_choice_falls_back_to_the_label_probability(self):
        assert parse_choice({"choice": "search", "probabilities": {"search": 0.4}}) == ("search", 0.4)

    @pytest.mark.parametrize("raw", [None, "x", {}, {"choice": 3}, {"choice": None}])
    def test_parse_choice_unusable_is_none(self, raw):
        assert parse_choice(raw) == (None, 0.0)

    def test_parse_choice_without_any_confidence_is_zero(self):
        assert parse_choice({"choice": "search"}) == ("search", 0.0)

    def test_parse_noul(self):
        assert parse_noul({"type": "noul", "noul": 0.25}) == 0.25
        assert parse_noul({"noul": True}) == 0.0
        assert parse_noul(None) == 0.0

    def test_parse_noul_or_none_distinguishes_unusable(self):
        assert parse_noul_or_none({"noul": 0.0}) == 0.0
        assert parse_noul_or_none({"noul": True}) is None
        assert parse_noul_or_none(None) is None
        assert parse_noul_or_none({}) is None

    def test_accept_choice_is_inclusive_at_the_threshold(self):
        assert accept_choice("a", 0.85, 0.85)
        assert not accept_choice("a", 0.84, 0.85)
        assert not accept_choice(None, 1.0, 0.1)


class TestChoosePath:
    def route(self, chosen="search", confidence=0.9, remaining=5000.0, down=False, threshold=0.85):
        return choose_path(chosen, confidence, threshold, OPTIONS, remaining, 4500, 300, down)

    def test_confident_valid_choice_goes_to_the_decider(self):
        assert self.route() == "decider"

    def test_low_confidence_with_time_goes_to_the_llm(self):
        assert self.route(confidence=0.3) == "llm"

    def test_low_confidence_without_time_executes_the_choice_on_budget(self):
        assert self.route(confidence=0.3, remaining=100.0) == "budget"

    def test_unknown_label_without_time_waits(self):
        assert self.route(chosen="nope", remaining=100.0) == "wait"

    def test_decider_down_uses_the_llm_then_waits(self):
        assert self.route(down=True) == "llm"
        assert self.route(down=True, remaining=100.0) == "wait"


class TestTimeoutAction:
    def test_never_a_second_action(self):
        assert timeout_action("search", OPTIONS, False, already_acted=True) == "none"

    def test_valid_choice_runs_otherwise_wait(self):
        assert timeout_action("search", OPTIONS, False, False) == "decider"
        assert timeout_action("nope", OPTIONS, False, False) == "wait"
        assert timeout_action("search", OPTIONS, True, False) == "wait"


class TestToolSelectionGate:
    def gate(self, **kw):
        return ToolSelectionGate("Pick the next action.", **kw)

    def test_questions_hold_one_choice_over_the_options(self):
        questions = self.gate().questions(OPTIONS)
        assert set(questions) == {ACTION_KEY}
        assert questions[ACTION_KEY]["type"] == "choice"
        assert questions[ACTION_KEY]["criteria"] == OPTIONS

    def test_extra_questions_ride_in_the_same_request(self):
        extra = {"risky": {"type": "noul", "instructions": "risky?"}}
        questions = self.gate().questions(OPTIONS, extra=extra)
        assert set(questions) == {ACTION_KEY, "risky"}

    def test_extra_cannot_replace_the_action_question(self):
        with pytest.raises(ValueError):
            self.gate().questions(OPTIONS, extra={ACTION_KEY: {}})

    def test_confident_answer_is_decided_by_the_decider(self):
        decision = self.gate().decide(answer("search", 0.95), OPTIONS, remaining_ms=5000)
        assert (decision.label, decision.route) == ("search", "decider")
        assert decision.confidence == pytest.approx(0.95)

    def test_threshold_is_configurable(self):
        low = self.gate(threshold=0.5).decide(answer("search", 0.6), OPTIONS, remaining_ms=5000)
        high = self.gate(threshold=0.9).decide(answer("search", 0.6), OPTIONS, remaining_ms=5000)
        assert (low.route, high.route) == ("decider", "llm")

    def test_unsure_with_time_hands_over_to_the_llm(self):
        decision = self.gate().decide(answer("search", 0.2), OPTIONS, remaining_ms=5000)
        assert decision.route == "llm"

    def test_unsure_without_time_runs_on_budget(self):
        decision = self.gate().decide(answer("search", 0.2), OPTIONS, remaining_ms=100)
        assert decision.route == "budget"

    def test_garbage_answers_do_not_crash(self):
        decision = self.gate().decide({}, OPTIONS, remaining_ms=100)
        assert (decision.label, decision.route) == (None, "wait")

    def test_label_and_confidence_can_be_overridden_by_a_combining_policy(self):
        decision = self.gate().decide(
            answer("search", 0.2), OPTIONS, remaining_ms=5000, label="answer", confidence=0.99
        )
        assert (decision.label, decision.route) == ("answer", "decider")

    def test_select_asks_once_and_decides(self):
        calls = []

        def ask(state, questions):
            calls.append((state, questions))
            return answer("answer", 0.97)

        decision = self.gate().select(ask, "the state", OPTIONS, remaining_ms=5000)
        assert decision.route == "decider" and decision.label == "answer"
        assert len(calls) == 1 and calls[0][0] == "the state"

    def test_select_marks_the_decider_down_and_stops_asking(self):
        gate = self.gate()
        calls = []

        def ask(state, questions):
            calls.append(1)
            raise ServerUnavailable("nothing listening")

        first = gate.select(ask, "s", OPTIONS, remaining_ms=5000)
        second = gate.select(ask, "s", OPTIONS, remaining_ms=5000)
        assert (first.route, second.route) == ("llm", "llm")
        assert gate.decider_down and len(calls) == 1

    def test_select_without_time_and_down_waits(self):
        def ask(state, questions):
            raise OSError("boom")

        decision = self.gate().select(ask, "s", OPTIONS, remaining_ms=10)
        assert decision.route == "wait"

    def test_timeout_action_delegates_to_the_pure_policy(self):
        gate = self.gate()
        assert gate.on_llm_timeout("search", OPTIONS, already_acted=False) == "decider"
        gate.decider_down = True
        assert gate.on_llm_timeout("search", OPTIONS, already_acted=False) == "wait"


class TestGateResilience:
    def make(self, **kw):
        return ToolSelectionGate("pick", **kw)

    def test_malformed_server_reply_falls_back_to_the_llm(self):
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
                self.wfile.write(b"{}")

        httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            decider = Decider(f"http://127.0.0.1:{httpd.server_port}")
            gate = self.make()
            assert gate.select(decider.ask, "s", OPTIONS, remaining_ms=5000).route == "llm"
            assert gate.decider_down
            assert self.make().select(decider.ask, "s", OPTIONS, remaining_ms=10).route == "wait"
        finally:
            httpd.shutdown()

    def test_timeout_s_is_passed_to_ask_bounded_by_the_time_left(self):
        seen = []

        def ask(state, questions, timeout=None):
            seen.append(timeout)
            return answer("answer", 0.97)

        gate = self.make(timeout_s=3.0)
        gate.select(ask, "s", OPTIONS, remaining_ms=5000)
        gate.select(ask, "s", OPTIONS, remaining_ms=1000)
        assert seen == [3.0, 1.0]

    def test_two_arg_ask_still_works_without_timeout_s(self):
        decision = self.make().select(lambda s, q: answer("answer", 0.97), "s", OPTIONS, remaining_ms=5000)
        assert decision.route == "decider"

    def test_retry_after_lets_the_gate_ask_again_with_an_injected_clock(self):
        now = [0.0]
        outcomes = [ServerUnavailable("down"), answer("answer", 0.97)]
        calls = []

        def ask(state, questions):
            calls.append(1)
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        gate = self.make(retry_after_s=30, clock=lambda: now[0])
        assert gate.select(ask, "s", OPTIONS, remaining_ms=5000).route == "llm"
        assert gate.select(ask, "s", OPTIONS, remaining_ms=5000).route == "llm" and len(calls) == 1
        now[0] = 31.0
        assert gate.select(ask, "s", OPTIONS, remaining_ms=5000).route == "decider"
        assert not gate.decider_down

    def test_reset_clears_a_sticky_down_flag(self):
        gate = self.make()
        gate.decider_down = True
        gate.reset()
        assert not gate.decider_down
