"""Phase 5: multithreaded job scheduler tests.

These deliberately use tiny configs (few episodes) -- we are testing the
concurrency machinery, not RL quality. Slow integration checks live in
test_sweep_integration below and are marked with a generous but bounded
runtime (~1-2s).
"""
from __future__ import annotations

import threading
import time

import pytest

from rl_lab.config.schema import ExperimentConfig
from rl_lab.core.fsm import IllegalTransitionError, State
from rl_lab.orchestration.controller import ExperimentController
from rl_lab.orchestration.scheduler import JobScheduler, run_sweep


def tiny_config(name="t", seed=0, **kw) -> ExperimentConfig:
    # build through the real validator (same as test_controller) so we
    # exercise DEFAULTS instead of hand-writing every required field.
    from rl_lab.config.loader import build_config
    raw = {"name": name, "seed": seed, "env": "etch_chamber",
           "agent": "random", "episodes": 2, "max_steps": 20,
           "eval_episodes": 2, "loggers": ["csv", "json"]}
    raw.update(kw)
    return build_config(raw)


def write_yaml(tmp_path, cfg: ExperimentConfig, fname=None) -> str:
    import yaml
    from rl_lab.config.schema import DEFAULTS
    p = tmp_path / (fname or f"{cfg.name}.yaml")
    raw = {k: v for k, v in cfg.to_dict().items() if k in DEFAULTS}
    extra = {k: v for k, v in cfg.to_dict().items() if k not in DEFAULTS}
    if extra:
        raw["extra"] = extra
    p.write_text(yaml.safe_dump(raw))
    return str(p)


class TestSeedExpansion:
    def test_one_config_becomes_n_jobs(self, tmp_path):
        path = write_yaml(tmp_path, tiny_config(name="exp"))
        sched = JobScheduler(num_workers=2)
        jobs = sched.expand_seed_jobs(path, seeds=[7, 8, 9])
        assert len(jobs) == 3
        assert [j["config"].seed for j in jobs] == [7, 8, 9]
        # names get seed suffix so output dirs don't collide
        assert jobs[0]["config"].name == "exp_seed7"

    def test_output_dirs_are_disjoint(self, tmp_path):
        path = write_yaml(tmp_path, tiny_config(name="exp"))
        sched = JobScheduler(num_workers=2)
        jobs = sched.expand_seed_jobs(path, seeds=[1, 2])
        dirs = {}
        for j in jobs:
            c = ExperimentController(j["config"])
            dirs[j["config"].seed] = c.output_dir
        assert dirs[1] != dirs[2]


class TestWorkerPool:
    def test_all_jobs_complete_and_collect(self, tmp_path):
        sched = JobScheduler(num_workers=4)
        for i in range(8):
            sched.submit({"config": tiny_config(name=f"j{i}", seed=i,
                                                output_dir=str(tmp_path))})
        sched.start()
        assert sched.wait(timeout=60), "workers did not finish in time"
        results = sched.results
        assert len(results) == 8
        assert all(r.state is State.DONE for r in results)
        # every worker ran on its own thread name at some point
        assert len(sched._threads) == 4

    def test_results_list_is_a_copy(self, tmp_path):
        sched = JobScheduler(num_workers=1)
        sched.submit({"config": tiny_config(output_dir=str(tmp_path))})
        sched.start(); sched.wait(timeout=30)
        snapshot = sched.results
        snapshot.clear()
        assert len(sched.results) == 1  # internal list untouched

    def test_failure_in_one_job_does_not_kill_pool(self, tmp_path):
        """A broken config must FAIL one result, workers keep serving."""
        sched = JobScheduler(num_workers=2)
        good = tiny_config(name="good", output_dir=str(tmp_path))
        bad = tiny_config(name="bad", output_dir=str(tmp_path))
        bad.env = "no_such_env"   # factory raises at training phase
        sched.submit({"config": bad})
        for i in range(3):
            sched.submit({"config": tiny_config(name=f"g{i}",
                                                output_dir=str(tmp_path))})
        sched.start()
        assert sched.wait(timeout=60)
        res = sched.results
        assert len(res) == 4
        failed = [r for r in res if r.state is State.FAILED]
        assert len(failed) == 1
        assert "no_such_env" in failed[0].error

    def test_aggregate_stats_across_seeds(self, tmp_path):
        sched = JobScheduler(num_workers=3)
        for s in (0, 1, 2):
            sched.submit({"config": tiny_config(name=f"a{s}", seed=s,
                                                output_dir=str(tmp_path))})
        sched.start(); sched.wait(timeout=60)
        agg = sched.aggregate()
        assert agg["n"] == 3 and agg["n_scored"] == 3
        assert agg["mean_eval_reward"] is not None
        # std across seeds uses ddof=1; three identical runs would be 0
        assert agg["std_across_seeds"] >= 0.0
        assert 0.0 <= agg["success_rate"] <= 1.0


class TestGracefulShutdown:
    def test_stop_flag_bails_at_phase_boundary(self, tmp_path):
        """Pre-set stop flag: training happens, evaluation skipped, FAILED
        recorded with the stop reason -- no mid-episode corruption."""
        sched = JobScheduler(num_workers=1)
        sched._stop_flag.set()          # as if Ctrl-C arrived earlier
        sched.submit({"config": tiny_config(output_dir=str(tmp_path))})
        sched.start()
        assert sched.wait(timeout=30)
        r = sched.results[0]
        assert r.state is State.FAILED
        assert "stopped" in r.error.lower()
        # honest partial artifact: training records exist even though eval
        # never ran -- cooperative cancel loses nothing already computed
        assert r.records

    def test_stop_wakes_idle_workers(self):
        """Sentinel shutdown: start pool with zero jobs, stop() must let
        wait() return instead of hanging forever (classic deadlock trap)."""
        sched = JobScheduler(num_workers=3)
        sched.start()
        sched.stop()
        assert sched.wait(timeout=10)
        assert all(not t.is_alive() for t in sched._threads)

    def test_double_ctrlc_force_exit_path(self, tmp_path):
        # second wait(timeout=...) returning False is what CLI maps to 130;
        # here we just verify wait() reports honestly while jobs run long
        sched = JobScheduler(num_workers=1)
        slow = tiny_config(name="slow", episodes=200, max_steps=50,
                           output_dir=str(tmp_path))
        sched.submit({"config": slow})
        sched.start()
        assert not sched.wait(timeout=0.01)  # still running -> False
        sched.stop()
        assert sched.wait(timeout=60)


class TestSharedStateSafety:
    def test_concurrent_snapshot_reads_never_crash(self, tmp_path):
        """Hammer test: monitor thread reads snapshot()/results while
        workers append. With the RLock removed this fails within seconds
        (see DEBUGGING.md entry 5)."""
        sched = JobScheduler(num_workers=4)
        errors: list[Exception] = []
        stop_monitor = threading.Event()

        def monitor():
            try:
                while not stop_monitor.is_set():
                    snap = sched.snapshot()
                    assert set(snap) >= {"finished", "pending", "by_state"}
                    _ = sched.results
                    _ = sched.aggregate()
            except Exception as exc:      # noqa: BLE001 - test harness
                errors.append(exc)

        for i in range(12):
            sched.submit({"config": tiny_config(name=f"h{i}", seed=i,
                                                episodes=4, max_steps=30,
                                                output_dir=str(tmp_path))})
        mon = threading.Thread(target=monitor)
        sched.start()
        mon.start()
        assert sched.wait(timeout=120)
        stop_monitor.set()
        mon.join(timeout=10)
        assert not errors, f"shared-state race: {errors}"
        assert len(sched.results) == 12

    def test_json_artifacts_valid_after_parallel_run(self, tmp_path):
        """Each job writes its own summary.json; parallelism must not
        interleave or truncate them."""
        import json
        sched = JobScheduler(num_workers=4)
        for i in range(6):
            sched.submit({"config": tiny_config(name=f"v{i}", seed=i,
                                                output_dir=str(tmp_path))})
        sched.start(); sched.wait(timeout=60)
        for r in sched.results:
            doc = json.loads((r.output_dir / "summary.json").read_text())
            assert len(doc["episodes"]) == 2
            assert doc["config"]["seed"] == r.config.seed


class TestRunSweepHelper:
    def test_end_to_end_sweep(self, tmp_path):
        path = write_yaml(tmp_path, tiny_config(name="sw",
                                                output_dir=str(tmp_path)))
        seen = []
        results, agg = run_sweep([path], seeds=[0, 1], num_workers=2,
                                 on_progress=lambda snap: seen.append(snap))
        assert len(results) == 2
        assert all(r.state is State.DONE for r in results)
        assert agg["n"] == 2
        assert agg["mean_eval_reward"] is not None


class TestControllerRerunGuard:
    def test_rerun_raises_instead_of_clobbering_done(self, tmp_path):
        """Regression (DEBUGGING.md entry 6): before Phase 5's error
        handling rework, calling run() twice raised IllegalTransitionError
        INSIDE the try block, got swallowed by `except Exception`, and the
        controller rewrote its own state DONE -> FAILED with a confusing
        'experiment error'. Programming errors must propagate."""
        c = ExperimentController(tiny_config(output_dir=str(tmp_path)))
        first = c.run()
        assert first.state is State.DONE
        with pytest.raises(IllegalTransitionError, match="RESET"):
            c.run()
        # the original success record was NOT mutated by the rejected rerun
        assert c.result.state is State.DONE
