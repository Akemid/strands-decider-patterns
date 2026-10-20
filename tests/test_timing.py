import time

import pytest

from decider_patterns.timing import llm_timeout_s, remaining_ms, run_with_timeout


def test_remaining_uses_the_default_deadline_when_unknown():
    assert remaining_ms(None, 1000) == 4000
    assert remaining_ms(3000, 500) == 2500


def test_llm_timeout_keeps_a_floor():
    assert llm_timeout_s(4000, 300) == pytest.approx(3.7)
    assert llm_timeout_s(10, 300) == 0.05


def test_run_with_timeout_returns_the_value():
    assert run_with_timeout(lambda: 7, 1.0) == (False, 7)


def test_run_with_timeout_abandons_a_slow_call():
    started = time.monotonic()
    assert run_with_timeout(lambda: time.sleep(2), 0.05) == (True, None)
    assert time.monotonic() - started < 1


def test_run_with_timeout_reraises():
    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        run_with_timeout(boom, 1.0)
