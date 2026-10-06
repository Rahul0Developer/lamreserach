"""End-to-end controller tests: FSM drives real phases on tiny configs.

Kept small (few episodes) on purpose -- these are wiring tests, not
learning benchmarks; Phase 3 already covers learning quality.
"""
import csv

import pytest

from rl_lab.config.loader import build_config
from rl_lab.core.fsm import Event, IllegalTransitionError, State
from rl_lab.orchestration.controller import ExperimentController


def tiny_config(**over):
    raw = {"name": "tiny", "env": "etch_chamber", "agent": "random",
           "episodes": 3, "max_steps": 20, "eval_episodes": 2,
           "loggers": ["csv"], "output_dir": str(over.pop("output_dir", "/tmp/rl_lab_test"))}
    raw.update(over)
    return build_config(raw)


def test_full_lifecycle_reaches_done(tmp_path):
    c = ExperimentController(tiny_config(output_dir=tmp_path / "runs"))
    result = c.run()
    assert result.state is State.DONE
    assert result.error is None
    assert len(result.records) == 3
    assert result.eval_report is not None and result.eval_report.episodes == 2
    # states actually traversed in order
    visited = [s for (s, _e, _d) in c._fsm.history] + [c._fsm.state]
    assert visited == [State.IDLE, State.CONFIGURED, State.TRAINING,
                       State.EVALUATING, State.REPORTING, State.DONE]


def test_csv_artifact_written(tmp_path):
    c = ExperimentController(tiny_config(output_dir=tmp_path / "runs",
                                         name="tiny-exp"))
    c.run()
    csv_path = c.output_dir / "episodes.csv"
    assert csv_path.exists()
    rows = list(csv.reader(csv_path.open()))
    assert rows[0] == ["episode", "total_reward", "steps", "truncated"]
    assert len(rows) == 1 + 3


def test_phase_failure_routes_to_failed_state(tmp_path, monkeypatch):
    """Inject a crash in the EVALUATING phase; controller must land in
    FAILED with the reason captured, never DONE."""
    c = ExperimentController(tiny_config(output_dir=tmp_path / "runs"))
    monkeypatch.setattr(c, "_enter_evaluating",
                        lambda o, n: (_ for _ in ()).throw(ValueError("eval boom")))
    result = c.run()
    assert result.state is State.FAILED
    assert "ValueError: eval boom" in result.error
    assert result.records  # training artifacts survive the failure


def test_cannot_run_twice():
    c = ExperimentController(tiny_config())
    c.run()
    with pytest.raises(IllegalTransitionError):
        c.run()   # DONE -> START_TRAINING is not in the table


def test_pause_resume_public_api(tmp_path, monkeypatch):
    """pause()/resume() must be legal mid-TRAINING.

    Subtle design point we hit while writing this (DEBUGGING.md entry 4):
    RESUME re-enters TRAINING, which RE-FIRES the on_enter hook. A naive
    wrapper that pauses/resumes inside itself therefore recurses until
    RecursionError. Real monitors avoid this by driving pause via a flag
    checked *inside* the loop, not by wrapping hooks -- so the product
    API is safe; only hook-wrapping tests need a one-shot guard.
    """
    c = ExperimentController(tiny_config(output_dir=tmp_path / "runs"))

    orig_hook = c._fsm._on_enter[State.TRAINING]
    done_once = []
    def wrapped(old, new):
        orig_hook(old, new)
        if not done_once:            # one-shot: don't recurse on RESUME
            done_once.append(True)
            c.pause()                # TRAINING -> PAUSED
            c.resume()               # PAUSED -> TRAINING (re-fires hook,
                                     # but guard above makes it a no-op cycle)
    c._fsm._on_enter[State.TRAINING] = wrapped

    result = c.run()
    assert result.state is State.DONE
    events = [e for (_s, e, _d) in c._fsm.history]
    assert events.count(Event.PAUSE) == 1 and events.count(Event.RESUME) == 1


def test_reentrant_hook_recursion_is_a_test_bug_not_product_bug():
    """Real debugging lesson (DEBUGGING.md entry 4): RESUME re-enters
    TRAINING, which re-fires the on_enter hook. If a test wraps the hook
    and the wrapper itself triggers RESUME, the wrapper recurses until
    RecursionError -- and because hooks run INSIDE the FSM lock, the
    traceback was ~2000 frames deep with no obvious culprit. The product
    behaves correctly; re-entrant state changes need a guard."""
    c = ExperimentController(tiny_config())
    calls = []

    orig_hook = c._fsm._on_enter[State.TRAINING]
    def wrapping_hook(old, new):
        calls.append((old.name, new.name))
        if len(calls) > 3:          # explicit recursion guard
            return
        orig_hook(old, new)
        c.pause(); c.resume()       # resume re-enters TRAINING -> re-fires us
    c._fsm._on_enter[State.TRAINING] = wrapping_hook

    result = c.run()
    assert result.state is State.DONE
    assert len(calls) <= 4          # guard worked; no stack overflow


def test_pause_from_wrong_state_raises():
    c = ExperimentController(tiny_config())
    with pytest.raises(IllegalTransitionError, match="PAUSE"):
        c.pause()   # CONFIGURED -> PAUSE not in table


def test_monitor_thread_pause_is_safe_under_contention(tmp_path):
    """Real concurrency smoke test: a monitor thread spams pause/resume
    while the controller runs. Each individual trigger is atomic, but a
    naive monitor can catch the FSM at a moment when PAUSE is illegal
    (e.g. already PAUSED or past TRAINING) -- so the *controller* must
    never crash; only the monitor's best-effort request may be refused.
    This is exactly how Phase 5 will wire Ctrl-C handling."""
    import threading
    c = ExperimentController(tiny_config(output_dir=tmp_path / "runs",
                                         episodes=8, max_steps=40))
    refused = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                c.pause()
                c.resume()
            except Exception as exc:   # IllegalTransitionError is expected noise
                refused.append(str(exc))
                break                  # state moved on; nothing more to do

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    result = c.run()
    stop.set(); t.join(timeout=2)

    assert result.state is State.DONE      # the RUN itself never breaks
    assert result.error is None
    # history stays consistent no matter what the monitor did
    states = {s for (s, _e, _d) in c._fsm.history} | {c._fsm.state}
    assert State.FAILED not in states


def test_qlearning_config_end_to_end(tmp_path):
    c = ExperimentController(tiny_config(agent="qlearning", episodes=5,
                                         output_dir=tmp_path / "runs"))
    result = c.run()
    assert result.state is State.DONE
    assert result.eval_report.mean_reward < 100  # sanity bound, not quality
