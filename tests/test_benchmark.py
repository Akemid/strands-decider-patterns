"""scripts/benchmark.py: pure planning/guards/aggregation plus orchestration with injected fakes.

Nothing here spawns the real agent or simulator, calls AWS or opens a real server.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import benchmark as bm


class TestPlan:
    def test_seed_range_and_list(self):
        assert bm.parse_seeds("1..3") == [1, 2, 3]
        assert bm.parse_seeds("2,5,9") == [2, 5, 9]
        assert bm.parse_seeds("4") == [4]

    @pytest.mark.parametrize("bad", ["", "3..1", "a..b", "1,,2"])
    def test_bad_seed_spec(self, bad):
        with pytest.raises(ValueError):
            bm.parse_seeds(bad)

    def test_runs_and_seed_start(self):
        assert bm.seeds_from_runs(3, 10) == [10, 11, 12]

    def test_modes_are_interleaved_per_seed(self):
        plan = bm.build_plan([1, 2], ["decider", "llm-only"])
        assert [(p.seed, p.mode) for p in plan] == [
            (1, "decider"), (1, "llm-only"), (2, "decider"), (2, "llm-only")]

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            bm.parse_modes("decider,nope")
        assert bm.parse_modes("llm-only") == ["llm-only"]


class TestGuards:
    @pytest.mark.parametrize("url", ["http://127.0.0.1:8765/mcp", "http://localhost:9000/mcp", "http://[::1]:9/mcp"])
    def test_loopback_urls_pass(self, url):
        bm.assert_safe_mcp_url(url)

    @pytest.mark.parametrize("url", ["http://127.0.0.1:8080/mcp", "http://localhost:8080/mcp",
                                     "http://10.0.0.5:8765/mcp", "https://game.example.com/mcp"])
    def test_port_8080_and_remote_hosts_refused(self, url):
        with pytest.raises(bm.UnsafeTarget):
            bm.assert_safe_mcp_url(url)

    def test_port_arg_refused_for_8080(self):
        with pytest.raises(bm.UnsafeTarget):
            bm.assert_safe_port(8080)
        bm.assert_safe_port(8765)


class TestCommands:
    def make(self, mode="decider", out=Path("runs/x")):
        cfg = bm.Config(seeds=[3], modes=[mode], ticks=20, model="haiku", port=8765, out=out)
        return cfg, bm.RunSpec(seed=3, mode=mode)

    def test_simulator_command(self):
        cfg, spec = self.make()
        cmd = bm.simulator_command(cfg, spec, python="py")
        assert cmd == ["py", "simulator/serve.py", "--seed", "3", "--port", "8765", "--ticks", "20", "--host", "127.0.0.1"]

    def test_agent_command_decider(self):
        cfg, spec = self.make("decider")
        cmd = bm.agent_command(cfg, spec, python="py")
        assert cmd[:3] == ["py", "-m", "examples.ctf_agent.agent"]
        assert "--llm-only" not in cmd
        assert cmd[cmd.index("--metrics") + 1] == str(Path("runs/x/decider-seed3.jsonl"))
        assert cmd[cmd.index("--model") + 1] == "haiku"

    def test_agent_command_llm_only(self):
        cfg, spec = self.make("llm-only")
        cmd = bm.agent_command(cfg, spec, python="py")
        assert "--llm-only" in cmd
        assert cmd[cmd.index("--metrics") + 1].endswith("llm-only-seed3.jsonl")

    def test_agent_env_points_at_the_local_simulator(self):
        cfg, _ = self.make()
        assert bm.agent_env(cfg, {"PATH": "p"})["CTF_MCP_URL"] == "http://127.0.0.1:8765/mcp"

    def test_dry_run_lists_every_command(self):
        cfg = bm.Config(seeds=[1, 2], modes=["decider", "llm-only"], ticks=5, model="haiku", port=8765,
                        out=Path("runs/x"))
        text = bm.format_dry_run(cfg, python="py")
        assert text.count("simulator/serve.py") == 4
        assert text.count("examples.ctf_agent.agent") == 4
        assert "CTF_MCP_URL=http://127.0.0.1:8765/mcp" in text


class TestPreflight:
    def test_decider_ok_returns_health(self):
        res = bm.check_decider(lambda: {"checkpoint": "ck", "device": "mps"})
        assert res.ok and "ck" in res.detail

    def test_decider_down(self):
        def boom():
            raise OSError("refused")
        res = bm.check_decider(boom)
        assert not res.ok and "refused" in res.detail

    def test_aws_ok_and_failure(self):
        assert bm.check_aws(lambda: {"Account": "123", "Arn": "arn:x"}).ok
        def boom():
            raise RuntimeError("no creds")
        assert not bm.check_aws(boom).ok

    def test_aws_check_masks_the_account_id(self):
        res = bm.check_aws(lambda: {"Account": "111122223333", "Arn": "arn:x"})
        assert "111122223333" not in res.detail and "****3333" in res.detail

    def test_run_preflight_skips_and_collects(self):
        fails = bm.run_preflight(modes=["decider"], skip_decider=False, skip_aws=True,
                                 decider_health=lambda: (_ for _ in ()).throw(OSError("x")),
                                 aws_identity=lambda: {})
        assert [c.name for c in fails if not c.ok] == ["decider"]

    def test_decider_not_checked_for_llm_only(self):
        checks = bm.run_preflight(modes=["llm-only"], skip_decider=False, skip_aws=False,
                                  decider_health=lambda: (_ for _ in ()).throw(OSError("x")),
                                  aws_identity=lambda: {"Account": "1"})
        assert [c.name for c in checks] == ["aws"] and all(c.ok for c in checks)


def tick(path, decider_ms=None, llm_ms=None, mode="decider", **kw):
    r = {"tick": 1, "mode": mode, "path": path, "decider_ms": decider_ms, "llm_ms": llm_ms,
         "decision_ms": (decider_ms or 0) + (llm_ms or 0)}
    r.update(kw)
    return r


class TestAggregation:
    def test_mode_stats_shares_and_latency(self):
        recs = [tick("decider", 40), tick("decider", 60), tick("llm", 50, 1000, llm_input_tokens=100,
                llm_output_tokens=10), tick("wait")]
        s = bm.aggregate_mode(recs)
        assert s["decisions"] == 4
        assert s["path_share"] == {"decider": 0.5, "llm": 0.25, "budget": 0.0, "wait": 0.25,
                                  "llm_capped": 0.0, "llm_busy": 0.0, "llm_error": 0.0}
        assert s["decider_only_ms"]["avg"] == pytest.approx(50.0)
        assert s["end_to_end_ms"]["p50"] == 60.0  # nearest rank over 40, 60, 1050: the wait tick has no latency
        assert s["ticks_without_latency"] == 1
        assert s["llm_calls"] == 1
        assert s["input_tokens"] == 100 and s["output_tokens"] == 10

    def test_filter_reason_share_and_options_offered_are_aggregated(self):
        recs = [{**tick("decider", 40), "filter_reason": "filtered", "options_offered": 3},
                {**tick("decider", 40), "filter_reason": "filtered", "options_offered": 5},
                {**tick("decider", 40), "filter_reason": "lesser_evil", "options_offered": 7},
                tick("llm", None, 900, mode="llm-only")]  # no filter fields: not counted
        s = bm.aggregate_mode(recs)
        assert s["filter_share"] == {"filtered": pytest.approx(2 / 3), "lesser_evil": pytest.approx(1 / 3),
                                     "unfiltered": 0.0}
        assert s["options_offered_avg"] == pytest.approx(5.0)
        empty = bm.aggregate_mode([tick("llm", None, 900, mode="llm-only")])
        assert empty["filter_share"] is None and empty["options_offered_avg"] is None

    def test_markdown_shows_the_filter_rows_when_present(self):
        summary = TestReport().summary()
        summary["modes"]["decider"].update(filter_share={"filtered": 0.5, "lesser_evil": 0.25, "unfiltered": 0.25},
                                           options_offered_avg=4.0)
        md = bm.format_markdown(summary)
        assert "% ticks: filtered" in md and "50%" in md and "options offered (avg)" in md

    def test_decider_only_latency_absent_when_never_recorded(self):
        s = bm.aggregate_mode([tick("llm", None, 900, mode="llm-only")])
        assert s["decider_only_ms"] is None

    def test_empty(self):
        s = bm.aggregate_mode([])
        assert s["decisions"] == 0 and s["end_to_end_ms"] is None

    def test_cost_uses_price_table(self):
        price = {"input_per_1m": 1.0, "output_per_1m": 5.0}
        assert bm.estimate_cost(1_000_000, 200_000, price) == pytest.approx(2.0)
        assert bm.estimate_cost(10, 10, None) is None

    def test_price_table_is_labeled_assumption(self):
        assert "ASSUMPTION" in bm.PRICE_TABLE_NOTE.upper()
        assert "haiku" in bm.PRICE_TABLE

    def test_llm_calls_avoided(self):
        decider = bm.aggregate_mode([tick("decider", 10)] * 9 + [tick("llm", 10, 500)])
        llm = bm.aggregate_mode([tick("llm", None, 900, mode="llm-only")] * 10)
        assert bm.llm_calls_avoided(decider, llm) == 9


class TestReport:
    def summary(self):
        decider = bm.aggregate_mode([tick("decider", 40), tick("llm", 40, 900, llm_input_tokens=1000,
                                                              llm_output_tokens=100)])
        llm = bm.aggregate_mode([tick("llm", None, 900, mode="llm-only", llm_input_tokens=1000,
                                      llm_output_tokens=100)])
        return {
            "environment": {"git_commit": "abc", "decider_checkpoint": "ck", "device": "mps",
                            "model": "haiku", "model_id": "mid", "ticks": 5, "seeds": [1]},
            "modes": {"decider": {**decider, "runs_ok": 1, "runs_failed": 0, "cost_usd": 0.01},
                      "llm-only": {**llm, "runs_ok": 1, "runs_failed": 1, "cost_usd": 0.02}},
            "llm_calls_avoided": bm.llm_calls_avoided(decider, llm),
            "runs": [{"seed": 1, "mode": "decider", "status": "ok", "exit_code": 0, "outcome": "match over"}],
            "price_note": bm.PRICE_TABLE_NOTE,
        }

    def test_markdown_has_disclaimer_env_and_no_win_rate_headline(self):
        md = bm.format_markdown(self.summary())
        assert "simulator/README.md" in md and "lock-step" in md
        assert "abc" in md and "ck" in md and "mid" in md
        assert "ASSUMPTION" in md.upper()
        assert "win rate" in md.lower() and "not" in md.lower()
        assert md.count("# ") >= 1
        assert "informational" in md.lower()


class FakeProc:
    def __init__(self, exits_at=None):
        self.terminated = False
        self.killed = False
        self._poll = exits_at

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class TestOrchestration:
    def cfg(self, tmp_path, modes=("decider", "llm-only"), seeds=(1,)):
        return bm.Config(seeds=list(seeds), modes=list(modes), ticks=5, model="haiku", port=8765, out=tmp_path)

    def test_run_ok_writes_metrics_and_always_terminates_simulator(self, tmp_path):
        cfg = self.cfg(tmp_path)
        procs = []

        def popen(cmd, **kw):
            p = FakeProc()
            procs.append(p)
            return p

        def run_agent(cmd, env, timeout, log_path):
            assert env["CTF_MCP_URL"].endswith(":8765/mcp")
            Path(cmd[cmd.index("--metrics") + 1]).write_text(json.dumps(tick("decider", 5)) + "\n")
            return bm.AgentResult(exit_code=0, timed_out=False, tail="match over (x): winner red")

        res = bm.execute_run(cfg, bm.RunSpec(1, "decider"), popen=popen, wait_ready=lambda *a, **k: True,
                             run_agent=run_agent)
        assert res.status == "ok" and res.exit_code == 0 and "match over" in res.outcome
        assert procs[0].terminated

    def test_agent_timeout_is_recorded_and_simulator_terminated(self, tmp_path):
        procs = []

        def popen(cmd, **kw):
            procs.append(FakeProc())
            return procs[-1]

        res = bm.execute_run(self.cfg(tmp_path), bm.RunSpec(1, "decider"), popen=popen,
                             wait_ready=lambda *a, **k: True,
                             run_agent=lambda *a, **k: bm.AgentResult(None, True, ""))
        assert res.status == "timeout" and procs[0].terminated

    def test_simulator_not_ready_skips_agent(self, tmp_path):
        called = []
        procs = []

        def popen(cmd, **kw):
            procs.append(FakeProc())
            return procs[-1]

        res = bm.execute_run(self.cfg(tmp_path), bm.RunSpec(1, "decider"), popen=popen,
                             wait_ready=lambda *a, **k: False,
                             run_agent=lambda *a, **k: called.append(1))
        assert res.status == "sim-not-ready" and not called and procs[0].terminated

    def test_agent_exception_still_terminates_simulator(self, tmp_path):
        procs = []

        def popen(cmd, **kw):
            procs.append(FakeProc())
            return procs[-1]

        def boom(*a, **k):
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            bm.execute_run(self.cfg(tmp_path), bm.RunSpec(1, "decider"), popen=popen,
                           wait_ready=lambda *a, **k: True, run_agent=boom)
        assert procs[0].terminated

    def test_nonzero_exit_is_failed(self, tmp_path):
        res = bm.execute_run(self.cfg(tmp_path), bm.RunSpec(1, "llm-only"), popen=lambda c, **k: FakeProc(),
                             wait_ready=lambda *a, **k: True,
                             run_agent=lambda *a, **k: bm.AgentResult(1, False, "boom"))
        assert res.status == "failed"

    def test_run_all_is_sequential_in_plan_order(self, tmp_path):
        order = []

        def fake_execute(cfg, spec, **kw):
            order.append((spec.seed, spec.mode))
            return bm.RunResult(spec.seed, spec.mode, "ok", 0, "o", 1.0)

        results, aborted = bm.run_all(self.cfg(tmp_path, seeds=(1, 2)), execute=fake_execute)
        assert aborted is None
        assert order == [(1, "decider"), (1, "llm-only"), (2, "decider"), (2, "llm-only")]
        assert len(results) == 4


class TestSummarizeOutput:
    def test_build_summary_reads_metrics_and_writes_files(self, tmp_path):
        cfg = bm.Config(seeds=[1], modes=["decider", "llm-only"], ticks=5, model="haiku", port=8765, out=tmp_path)
        (tmp_path / "decider-seed1.jsonl").write_text(
            json.dumps(tick("decider", 10)) + "\n"
            + json.dumps(tick("llm", 10, 500, llm_input_tokens=1000, llm_output_tokens=100)) + "\n")
        (tmp_path / "llm-only-seed1.jsonl").write_text(
            json.dumps(tick("llm", None, 900, mode="llm-only", llm_input_tokens=1000, llm_output_tokens=100)) + "\n")
        results = [bm.RunResult(1, "decider", "ok", 0, "match over", 1.0),
                   bm.RunResult(1, "llm-only", "failed", 1, "", 1.0)]
        env = {"git_commit": "abc", "decider_checkpoint": "ck", "device": "cpu"}
        summary = bm.build_summary(cfg, results, env)
        assert summary["modes"]["decider"]["runs_ok"] == 1
        assert summary["modes"]["llm-only"]["runs_failed"] == 1
        assert summary["modes"]["decider"]["decisions"] == 2
        assert summary["llm_calls_avoided"] == 1
        bm.write_outputs(tmp_path, summary)
        assert json.loads((tmp_path / "summary.json").read_text())["environment"]["git_commit"] == "abc"
        assert "simulator/README.md" in (tmp_path / "summary.md").read_text()


class TestRealRunners:
    def test_agent_runner_reports_timeout_and_kills_the_process_group(self, tmp_path, monkeypatch):
        killed = []

        class FakePopen:
            pid = 4242
            returncode = None
            calls = 0

            def __init__(self, cmd, **kw):
                assert kw["start_new_session"] is True

            def communicate(self, timeout=None):
                FakePopen.calls += 1
                if FakePopen.calls == 1:
                    raise subprocess.TimeoutExpired("x", timeout, output=b"partial\n")
                return b"partial\n", None

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setattr(bm.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        res = bm.run_agent_process(["x"], {}, 1, tmp_path / "log.txt")
        assert res.timed_out and res.exit_code is None
        assert killed == [(4242, bm.signal.SIGKILL)]
        assert (tmp_path / "log.txt").read_text() == "partial\n"

    def test_a_grandchild_dies_with_the_timed_out_agent(self, tmp_path):
        code = ("import subprocess, sys, time\n"
                "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "print(c.pid, flush=True)\n"
                "time.sleep(60)\n")
        res = bm.run_agent_process([sys.executable, "-c", code], dict(os.environ), 1.5, tmp_path / "log.txt")
        assert res.timed_out
        pid = int((tmp_path / "log.txt").read_text().split()[0])
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("the grandchild survived the timeout")


class TestCostGuards:
    def cfg(self, tmp_path, **kw):
        base = dict(seeds=[1, 2, 3], modes=["llm-only"], ticks=5, model="haiku", port=8765, out=tmp_path)
        return bm.Config(**{**base, **kw})

    def test_defaults_are_conservative(self, tmp_path):
        cfg = self.cfg(tmp_path)
        assert (cfg.max_llm_calls_per_run, cfg.max_total_llm_calls) == (60, 300)
        assert (cfg.max_cost_usd, cfg.max_consecutive_failures) == (5.0, 2)

    def test_the_agent_gets_the_per_run_llm_cap(self, tmp_path):
        cfg = self.cfg(tmp_path, max_llm_calls_per_run=7)
        for mode in bm.MODES:
            cmd = bm.agent_command(cfg, bm.RunSpec(1, mode), python="py")
            assert cmd[cmd.index("--max-llm-calls") + 1] == "7"

    def test_ticks_above_the_cap_are_refused(self):
        args = bm.parse_args(["--runs", "1", "--ticks", str(bm.MAX_TICKS + 1)])
        with pytest.raises(ValueError, match="--ticks"):
            bm.config_from_args(args)
        assert bm.config_from_args(bm.parse_args(["--runs", "1", "--ticks", str(bm.MAX_TICKS)])).ticks == bm.MAX_TICKS

    def test_cli_flags_reach_the_config(self):
        args = bm.parse_args(["--runs", "1", "--max-llm-calls-per-run", "9", "--max-total-llm-calls", "20",
                              "--max-cost-usd", "1.5", "--max-consecutive-failures", "3"])
        cfg = bm.config_from_args(args)
        assert (cfg.max_llm_calls_per_run, cfg.max_total_llm_calls) == (9, 20)
        assert (cfg.max_cost_usd, cfg.max_consecutive_failures) == (1.5, 3)

    def test_dry_run_and_markdown_show_the_limits(self, tmp_path):
        cfg = self.cfg(tmp_path)
        text = bm.format_dry_run(cfg, python="py")
        assert "--max-llm-calls 60" in text and "max-total-llm-calls 300" in text and "max-cost-usd 5.0" in text
        summary = bm.build_summary(cfg, [], {}, aborted="max-cost-usd exceeded")
        md = bm.format_markdown(summary)
        assert "aborted: max-cost-usd exceeded" in md and "max_total_llm_calls" in md

    def write_run(self, cfg, spec, llm_ticks, tokens=(1000, 100)):
        records = [tick("llm", None, 500, mode=spec.mode, llm_input_tokens=tokens[0], llm_output_tokens=tokens[1])
                   for _ in range(llm_ticks)]
        bm.metrics_path(cfg, spec).write_text("".join(json.dumps(r) + "\n" for r in records))

    def runner(self, cfg, status="ok", llm_ticks=1, tokens=(1000, 100)):
        def execute(c, spec, **kw):
            self.write_run(c, spec, llm_ticks, tokens)
            return bm.RunResult(spec.seed, spec.mode, status, 0 if status == "ok" else 1, "", 1.0)
        return execute

    def test_total_llm_calls_cap_aborts_the_remaining_runs(self, tmp_path):
        cfg = self.cfg(tmp_path, max_total_llm_calls=5)
        results, aborted = bm.run_all(cfg, execute=self.runner(cfg, llm_ticks=3))
        assert len(results) == 2 and "max-total-llm-calls" in aborted

    def test_cost_cap_aborts_using_the_token_estimate(self, tmp_path):
        cfg = self.cfg(tmp_path, max_cost_usd=0.01)
        results, aborted = bm.run_all(cfg, execute=self.runner(cfg, tokens=(100_000, 1_000)))
        assert len(results) == 1 and "max-cost-usd" in aborted

    def test_calls_without_token_counts_use_a_conservative_estimate(self, tmp_path):
        # A timed-out LLM tick reports no tokens: it must still cost something.
        cfg = self.cfg(tmp_path, max_cost_usd=0.001)
        results, aborted = bm.run_all(cfg, execute=self.runner(cfg, tokens=(None, None)))
        assert len(results) == 1 and "max-cost-usd" in aborted

    def test_consecutive_failures_abort(self, tmp_path):
        cfg = self.cfg(tmp_path, seeds=[1, 2, 3, 4])
        results, aborted = bm.run_all(cfg, execute=self.runner(cfg, status="timeout"))
        assert len(results) == 2 and "consecutive" in aborted

    def test_a_success_resets_the_failure_streak(self, tmp_path):
        cfg = self.cfg(tmp_path, seeds=[1, 2, 3, 4])
        statuses = iter(["failed", "ok", "failed", "ok"])

        def execute(c, spec, **kw):
            self.write_run(c, spec, 0)
            status = next(statuses)
            return bm.RunResult(spec.seed, spec.mode, status, 0, "", 1.0)

        results, aborted = bm.run_all(cfg, execute=execute)
        assert len(results) == 4 and aborted is None


class TestLatencyBias:
    def test_ticks_without_any_timing_are_not_counted_as_zero_ms(self):
        recs = [{"tick": 1, "path": "wait", "mode": "decider"},  # no decision_ms, decider_ms or llm_ms
                tick("decider", 40), tick("decider", 60)]
        s = bm.aggregate_mode(recs)
        assert s["end_to_end_ms"]["avg"] == pytest.approx(50.0)
        assert s["end_to_end_ms"]["p50"] == 40.0
        assert s["ticks_without_latency"] == 1
        assert s["decisions"] == 3

    def test_component_timings_still_count_without_decision_ms(self):
        s = bm.aggregate_mode([{"path": "llm", "decider_ms": 10, "llm_ms": 90}])
        assert s["end_to_end_ms"]["avg"] == pytest.approx(100.0) and s["ticks_without_latency"] == 0


class TestAgentTuningFlags:
    def cfg(self, tmp_path, **kw):
        base = dict(seeds=[1], modes=["decider", "llm-only"], ticks=5, model="haiku", port=8765, out=tmp_path)
        return bm.Config(**{**base, **kw})

    def test_defaults_do_not_override_the_agent(self, tmp_path):
        cmd = bm.agent_command(self.cfg(tmp_path), bm.RunSpec(1, "decider"), python="py")
        assert "--llm-budget-ms" not in cmd and "--threshold" not in cmd

    def test_flags_are_passed_to_every_agent_run(self, tmp_path):
        cfg = self.cfg(tmp_path, llm_budget_ms=2500.0, threshold=0.6)
        for mode in bm.MODES:
            cmd = bm.agent_command(cfg, bm.RunSpec(1, mode), python="py")
            assert cmd[cmd.index("--llm-budget-ms") + 1] == "2500.0"
            assert cmd[cmd.index("--threshold") + 1] == "0.6"

    def test_cli_dry_run_and_markdown_show_them(self, tmp_path):
        args = bm.parse_args(["--runs", "1", "--llm-budget-ms", "2500", "--threshold", "0.6"])
        cfg = bm.config_from_args(args)
        assert (cfg.llm_budget_ms, cfg.threshold) == (2500.0, 0.6)
        text = bm.format_dry_run(cfg, python="py")
        assert "--llm-budget-ms 2500.0" in text and "--threshold 0.6" in text
        md = bm.format_markdown(bm.build_summary(cfg, [], {}))
        assert "- llm_budget_ms: 2500.0" in md and "- threshold: 0.6" in md

    def test_summary_shows_agent_defaults_when_unset(self, tmp_path):
        md = bm.format_markdown(bm.build_summary(self.cfg(tmp_path), [], {}))
        assert "- llm_budget_ms: default" in md and "- threshold: default" in md


class TestShadowDecider:
    def cfg(self, tmp_path, **kw):
        base = dict(seeds=[1], modes=["decider", "llm-only"], ticks=5, model="haiku", port=8765, out=tmp_path)
        return bm.Config(**{**base, **kw})

    def test_llm_only_runs_get_the_shadow_flag_by_default_decider_runs_never(self, tmp_path):
        cfg = self.cfg(tmp_path)
        assert "--shadow-decider" in bm.agent_command(cfg, bm.RunSpec(1, "llm-only"), python="py")
        assert "--shadow-decider" not in bm.agent_command(cfg, bm.RunSpec(1, "decider"), python="py")

    def test_no_shadow_flag_disables_it(self, tmp_path):
        cfg = bm.config_from_args(bm.parse_args(["--runs", "1", "--no-shadow", "--out", str(tmp_path)]))
        assert cfg.shadow is False
        assert "--shadow-decider" not in bm.agent_command(cfg, bm.RunSpec(1, "llm-only"), python="py")
        assert bm.config_from_args(bm.parse_args(["--runs", "1"])).shadow is True

    def test_preflight_requires_the_decider_for_llm_only_with_shadow(self):
        def down():
            raise OSError("x")
        checks = bm.run_preflight(["llm-only"], False, True, down, lambda: {}, shadow=True)
        assert [(c.name, c.ok) for c in checks] == [("decider", False)]
        assert bm.run_preflight(["llm-only"], False, True, down, lambda: {}, shadow=False) == []

    def test_dry_run_shows_shadow(self, tmp_path):
        assert "shadow decider: on" in bm.format_dry_run(self.cfg(tmp_path), python="py")
        assert "shadow decider: off" in bm.format_dry_run(self.cfg(tmp_path, shadow=False), python="py")

    def test_summary_has_a_shadow_agreement_section_when_data_exists(self, tmp_path):
        rows = [tick("llm", None, 900, mode="llm-only", shadow_label="a", shadow_confidence=0.9, executed_label="a"),
                tick("llm", None, 900, mode="llm-only", shadow_label="a", shadow_confidence=0.9, executed_label="b"),
                tick("llm", None, 900, mode="llm-only", shadow_label="a", shadow_confidence=0.4, executed_label=None)]
        (tmp_path / "llm-only-seed1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        summary = bm.build_summary(self.cfg(tmp_path, modes=["llm-only"]), [], {})
        shadow = summary["modes"]["llm-only"]["shadow"]
        assert (shadow["compared"], shadow["excluded"]) == (2, 1)
        md = bm.format_markdown(summary)
        assert "## Shadow agreement" in md and "proxy for quality, not ground truth" in md and "| 0.90 | 100% | 50% | 2 |" in md

    def test_summary_points_to_the_sweep_without_shadow_data(self, tmp_path):
        md = bm.format_markdown(bm.build_summary(self.cfg(tmp_path), [], {}))
        assert "## Shadow agreement" in md and "threshold_sweep.py" in md


class TestLlmErrorReporting:
    def cfg(self, tmp_path, **kw):
        base = dict(seeds=[1], modes=["llm-only"], ticks=5, model="haiku", port=8765, out=tmp_path)
        return bm.Config(**{**base, **kw})

    def write_metrics(self, cfg, records):
        bm.metrics_path(cfg, bm.RunSpec(1, "llm-only")).write_text("".join(json.dumps(r) + "\n" for r in records))

    def execute(self, cfg, exit_code):
        return bm.execute_run(cfg, bm.RunSpec(1, "llm-only"), popen=lambda c, **k: FakeProc(),
                              wait_ready=lambda *a, **k: True,
                              run_agent=lambda *a, **k: bm.AgentResult(exit_code, False, "tail"))

    def test_default_and_flag_reach_the_agent_and_dry_run(self, tmp_path):
        cfg = self.cfg(tmp_path)
        assert cfg.max_consecutive_llm_errors == 3
        cmd = bm.agent_command(cfg, bm.RunSpec(1, "llm-only"), python="py")
        assert cmd[cmd.index("--max-consecutive-llm-errors") + 1] == "3"
        parsed = bm.config_from_args(bm.parse_args(["--runs", "1", "--max-consecutive-llm-errors", "5"]))
        assert parsed.max_consecutive_llm_errors == 5
        assert "--max-consecutive-llm-errors 3" in bm.format_dry_run(cfg, python="py")
        assert "max_consecutive_llm_errors: 3" in bm.format_markdown(bm.build_summary(cfg, [], {}))

    def test_llm_error_ticks_and_reasons_are_counted_from_metrics(self, tmp_path):
        cfg = self.cfg(tmp_path)
        self.write_metrics(cfg, [tick("llm_error", reason="transient:ThrottlingException"),
                                 tick("llm", None, 5), tick("llm_error", reason="max_tokens"),
                                 tick("llm_error", reason="transient:ThrottlingException")])
        res = self.execute(cfg, 0)
        assert res.status == "ok" and res.llm_errors == 3
        assert res.llm_error_reasons == {"transient:ThrottlingException": 2, "max_tokens": 1}

    def test_breaker_exit_is_failed_with_the_reason(self, tmp_path):
        cfg = self.cfg(tmp_path)
        self.write_metrics(cfg, [tick("llm", None, 5)] + [
            tick("llm_error", reason="transient:serviceUnavailableException")] * 3)
        res = self.execute(cfg, 3)
        assert res.status == "failed"
        assert res.outcome == "llm errors: transient:serviceUnavailableException"

    def test_a_crash_with_a_shorter_error_streak_is_not_blamed_on_the_breaker(self, tmp_path):
        cfg = self.cfg(tmp_path)
        self.write_metrics(cfg, [tick("llm_error", reason="transient:ThrottlingException")] * 2)
        assert self.execute(cfg, 1).outcome == "tail"

    def test_summary_lists_llm_errors_per_run(self, tmp_path):
        cfg = self.cfg(tmp_path)
        run = bm.RunResult(1, "llm-only", "failed", 3, "llm errors: transient:ThrottlingException", 1.0, 3,
                           {"transient:ThrottlingException": 3})
        md = bm.format_markdown(bm.build_summary(cfg, [run], {}))
        assert "| llm errors |" in md
        assert "3 (transient:ThrottlingException x3)" in md
        assert "llm errors: transient:ThrottlingException" in md
