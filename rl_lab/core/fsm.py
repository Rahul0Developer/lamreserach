"""Experiment lifecycle Finite State Machine.

Why an FSM instead of boolean flags (is_training, is_done...): flag soup
lets you reach impossible combinations (training=True, done=True) and
scatters legality checks across every method. A transition *table* makes
the legal graph one explicit data structure -- reviewable, testable, and
impossible to "fix" by adding one more if.

Design notes:
  - States are an Enum; transitions are (from_state, event) -> to_state.
  - Illegal transitions raise IllegalTransitionError naming state, event,
    and the legal alternatives -- a stack trace that reads like documentation.
  - on_enter / on_exit hooks are small callables (Template Method style):
    ExperimentController plugs its phases in here, so THIS class stays
    pure control logic with zero RL dependencies (clean-architecture rule:
    policies don't import mechanisms).
  - History is recorded for post-mortems: "why did the run FAIL?" is
    answerable from the log alone.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    CONFIGURED = auto()
    TRAINING = auto()
    PAUSED = auto()
    EVALUATING = auto()
    REPORTING = auto()
    DONE = auto()
    FAILED = auto()


class Event(Enum):
    CONFIGURE = auto()     # IDLE       -> CONFIGURED
    START_TRAINING = auto()  # CONFIGURED -> TRAINING
    PAUSE = auto()         # TRAINING   -> PAUSED
    RESUME = auto()        # PAUSED     -> TRAINING
    FINISH_TRAINING = auto()   # TRAINING   -> EVALUATING
    FINISH_EVAL = auto()   # EVALUATING -> REPORTING
    FINISH_REPORT = auto()  # REPORTING  -> DONE
    FAIL = auto()          # any active -> FAILED
    RESET = auto()         # terminal   -> IDLE (retry without new process)


# (State, Event) -> State. Single source of truth for legality.
TRANSITIONS: dict[tuple[State, Event], State] = {
    (State.IDLE, Event.CONFIGURE): State.CONFIGURED,
    (State.CONFIGURED, Event.START_TRAINING): State.TRAINING,
    (State.TRAINING, Event.PAUSE): State.PAUSED,
    (State.PAUSED, Event.RESUME): State.TRAINING,
    (State.TRAINING, Event.FINISH_TRAINING): State.EVALUATING,
    (State.EVALUATING, Event.FINISH_EVAL): State.REPORTING,
    (State.REPORTING, Event.FINISH_REPORT): State.DONE,
}

# FAIL is legal from every non-terminal state; expanded into the table
# programmatically so the table above stays readable.
_ACTIVE_STATES = [State.CONFIGURED, State.TRAINING, State.PAUSED,
                  State.EVALUATING, State.REPORTING]
for _s in _ACTIVE_STATES:
    TRANSITIONS[(_s, Event.FAIL)] = State.FAILED

# Terminal states can only be re-entered via RESET.
_TERMINAL_STATES = [State.DONE, State.FAILED]
for _s in _TERMINAL_STATES:
    TRANSITIONS[(_s, Event.RESET)] = State.IDLE


class IllegalTransitionError(RuntimeError):
    """Raised when an event cannot fire in the current state."""


HookFn = Callable[[State, State], None]


class ExperimentFSM:
    """Table-driven lifecycle machine. Thread-safe: Phase 5's parallel
    runner will read .state from monitor threads while workers drive
    events, so every check-and-set happens under one lock (no TOCTOU)."""

    def __init__(self, initial: State = State.IDLE,
                 on_enter: dict[State, HookFn] | None = None,
                 on_exit: dict[State, HookFn] | None = None) -> None:
        self._state = initial
        self._lock = threading.RLock()
        self._on_enter = dict(on_enter or {})
        self._on_exit = dict(on_exit or {})
        self.history: list[tuple[State, Event, State]] = []

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def legal_events(self) -> list[Event]:
        with self._lock:
            return [ev for (st, ev) in TRANSITIONS if st is self._state]

    def trigger(self, event: Event) -> State:
        """Fire an event. Returns the new state or raises IllegalTransitionError.

        Hooks run inside the lock on purpose: a hook firing while another
        thread triggers a second event could observe half-transitioned
        state. Keep hooks fast (they only record data / flip flags).
        """
        with self._lock:
            target = TRANSITIONS.get((self._state, event))
            if target is None:
                legal = [e.name for e in self.legal_events()]
                raise IllegalTransitionError(
                    f"cannot fire {event.name} in state {self._state.name}; "
                    f"legal events here: {legal or 'NONE (terminal)'}")
            old = self._state
            exit_hook = self._on_exit.get(old)
            enter_hook = self._on_enter.get(target)
            self._state = target
            self.history.append((old, event, target))
            if exit_hook:
                exit_hook(old, target)
            if enter_hook:
                enter_hook(old, target)
            return target
