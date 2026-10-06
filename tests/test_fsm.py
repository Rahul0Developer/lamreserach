"""FSM tests: table-driven legality, error quality, hooks, thread safety."""
import threading

import pytest

from rl_lab.core.fsm import (Event, ExperimentFSM, IllegalTransitionError,
                             State, TRANSITIONS)


def drive(fsm, events):
    for e in events:
        fsm.trigger(e)
    return fsm.state


HAPPY = [Event.CONFIGURE, Event.START_TRAINING, Event.FINISH_TRAINING,
         Event.FINISH_EVAL, Event.FINISH_REPORT]


def test_happy_path_reaches_done():
    fsm = ExperimentFSM()
    assert drive(fsm, HAPPY) is State.DONE
    # history records every hop for post-mortems
    assert len(fsm.history) == 5
    assert fsm.history[0] == (State.IDLE, Event.CONFIGURE, State.CONFIGURED)


def test_initial_state_is_idle_and_terminal_flags():
    fsm = ExperimentFSM()
    assert fsm.state is State.IDLE
    assert not fsm.is_terminal
    drive(fsm, HAPPY)
    assert fsm.is_terminal


@pytest.mark.parametrize("event", list(Event))
def test_illegal_from_idle_raises_clear_error(event):
    if event is Event.CONFIGURE:
        pytest.skip("this one IS legal")
    fsm = ExperimentFSM()
    with pytest.raises(IllegalTransitionError) as exc:
        fsm.trigger(event)
    msg = str(exc.value)
    # Error message quality is a feature: names state, event, legal options.
    assert "IDLE" in msg and event.name in msg and "CONFIGURE" in msg


def test_cannot_evaluate_before_training():
    fsm = ExperimentFSM()
    drive(fsm, [Event.CONFIGURE])
    with pytest.raises(IllegalTransitionError, match="FINISH_TRAINING"):
        fsm.trigger(Event.FINISH_TRAINING)


def test_pause_resume_cycle():
    fsm = ExperimentFSM()
    drive(fsm, [Event.CONFIGURE, Event.START_TRAINING, Event.PAUSE])
    assert fsm.state is State.PAUSED
    # PAUSE again from PAUSED is illegal -- no double-pause states.
    with pytest.raises(IllegalTransitionError):
        fsm.trigger(Event.PAUSE)
    drive(fsm, [Event.RESUME, Event.FINISH_TRAINING])
    assert fsm.state is State.EVALUATING


def test_fail_from_every_active_state():
    active = [State.CONFIGURED, State.TRAINING, State.PAUSED,
              State.EVALUATING, State.REPORTING]
    for st in active:
        fsm = ExperimentFSM(initial=st)
        fsm.trigger(Event.FAIL)
        assert fsm.state is State.FAILED


def test_reset_from_failed_enables_retry():
    fsm = ExperimentFSM()
    drive(fsm, [Event.CONFIGURE, Event.FAIL])
    drive(fsm, [Event.RESET, Event.CONFIGURE])
    assert fsm.state is State.CONFIGURED


def test_transition_table_invariants():
    """Guard against someone editing the table into nonsense."""
    for (src, ev), dst in TRANSITIONS.items():
        assert isinstance(src, State) and isinstance(ev, Event)
        assert isinstance(dst, State)
        assert dst is not src, "self-loops hide bugs; use explicit states"
    # DONE must be a dead end except RESET
    done_events = [e for (s, e) in TRANSITIONS if s is State.DONE]
    assert done_events == [Event.RESET]


def test_hooks_fire_on_enter_exit_in_order():
    log = []
    fsm = ExperimentFSM(
        on_enter={State.TRAINING: lambda o, n: log.append(("enter", n.name))},
        on_exit={State.TRAINING: lambda o, n: log.append(("exit", o.name))})
    fsm.trigger(Event.CONFIGURE)
    fsm.trigger(Event.START_TRAINING)   # enter TRAINING
    fsm.trigger(Event.PAUSE)            # exit TRAINING
    assert log == [("enter", "TRAINING"), ("exit", "TRAINING")]


def test_hook_exception_propagates_and_state_rollback_question():
    """Document deliberate behavior: if an enter-hook raises AFTER the
    state moved, the FSM stays in the new state (caller can fire FAIL).
    We assert it so a future 'fix' that silently rolls back is caught."""
    def boom(_o, _n):
        raise RuntimeError("phase failed")
    fsm = ExperimentFSM(on_enter={State.TRAINING: boom})
    fsm.trigger(Event.CONFIGURE)
    with pytest.raises(RuntimeError):
        fsm.trigger(Event.START_TRAINING)
    assert fsm.state is State.TRAINING
    fsm.trigger(Event.FAIL)   # recovery path exists from TRAINING
    assert fsm.state is State.FAILED


def test_concurrent_triggers_only_one_wins():
    """Two threads race FINISH_TRAINING from TRAINING: exactly one legal,
    the other must get IllegalTransitionError (lock makes check+set atomic)."""
    fsm = ExperimentFSM(initial=State.TRAINING)
    errors, oks = [], []

    def worker():
        try:
            fsm.trigger(Event.FINISH_TRAINING)
            oks.append(1)
        except IllegalTransitionError as e:
            errors.append(e)

    ts = [threading.Thread(target=worker) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=5)
    assert len(oks) == 1 and len(errors) == 1
    assert fsm.state is State.EVALUATING
