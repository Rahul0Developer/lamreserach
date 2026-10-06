"""ExperimentController: orchestrates one full experiment run.

Why this layer exists (and why the FSM lives in core/ but the controller
does not): the Trainer knows episodes, the Evaluator knows statistics --
nobody owns "load config -> train -> evaluate -> report". Orchestration is
a separate responsibility; keeping it out of core/ means core/ stays free
of RL imports and can be unit-tested with fakes.

The FSM drives phase execution through on_enter hooks. The controller's
run() only fires events; if a phase raises, we fire FAIL. That structure
makes it impossible to mark an experiment DONE while a phase failed --
the exact class of bug flag-based code hides.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rl_lab.config.schema import ExperimentConfig
from rl_lab.core.fsm import Event, ExperimentFSM, IllegalTransitionError, State
from rl_lab.evaluation.evaluator import EvalReport, Evaluator


from rl_lab.factory import create_agent, create_env
from rl_lab.logging.console_logger import ConsoleLogger
from rl_lab.logging.csv_logger import CSVLogger
from rl_lab.logging.json_logger import JSONLogger
from rl_lab.training.trainer import EpisodeRecord, Trainer


class _StopRequested(RuntimeError):
    """Raised inside run() when the scheduler's stop_flag is set at a
    phase boundary. Separate type so we can label the result honestly."""


@dataclass
class RunResult:
    """Everything Phase 6's ReportGenerator will consume."""
    config: ExperimentConfig
    state: State                      # DONE or FAILED
    records: list[EpisodeRecord] = field(default_factory=list)
    eval_report: EvalReport | None = None
    error: str | None = None
    wall_time_s: float = 0.0
    output_dir: Path | None = None


def build_loggers(config: ExperimentConfig, output_dir: Path) -> list:
    """Instantiate configured loggers (Observer attach happens here)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    loggers: list = []
    for name in config.loggers:
        if name == "console":
            loggers.append(ConsoleLogger())
        elif name == "csv":
            loggers.append(CSVLogger(str(output_dir / "episodes.csv")))
        elif name == "json":
            loggers.append(JSONLogger(str(output_dir / "summary.json")))
    return loggers


class ExperimentController:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        stem = Path(config.name).stem          # names with '/' would escape runs/
        self.output_dir = Path(config.output_dir) / stem
        self.result = RunResult(config=config, state=State.IDLE,
                                output_dir=self.output_dir)
        self._fsm = ExperimentFSM(
            on_enter={
                # Late-bound dispatch (getattr at call time): hooks resolve
                # through the instance so tests can monkeypatch the phase
                # methods, and _hook() records which phase blew up.
                State.TRAINING: self._hook("_enter_training"),
                State.EVALUATING: self._hook("_enter_evaluating"),
                State.REPORTING: self._hook("_enter_reporting"),
            })
        # Phase bookkeeping: FINISH_* events fire *before* the phase they
        # are named after runs (FINISH_TRAINING enters EVALUATING). run()
        # checks this flag to catch a late crash that arrives after the
        # FSM already reached DONE -- without it, a failed phase could
        # still report success. (Found by test_phase_failure_...; see
        # DEBUGGING.md entry 3.)
        self._phase_error: str | None = None
        # Config arrives via the FSM too -- no back door into CONFIGURED.
        self._fsm.trigger(Event.CONFIGURE)

    # -- wrapped hooks: record which phase blew up ----------------------------

    def _hook(self, method_name: str):
        """Wrap a phase method so its exceptions are attributed to the
        phase. getattr at CALL time (late binding) keeps monkeypatching
        working in tests."""
        def wrapper(old, new):
            try:
                getattr(self, method_name)(old, new)
            except Exception as exc:
                self._phase_error = f"{type(exc).__name__}: {exc}"
                raise
        return wrapper

    @property
    def state(self) -> State:
        return self._fsm.state

    # -- public lifecycle ------------------------------------------------------

    def run(self, stop_flag: "threading.Event | None" = None) -> RunResult:
        # Guard the re-run case explicitly. Without this line, a second
        # run() on a DONE/FAILED controller raised IllegalTransitionError
        # from START_TRAINING -- and the broad `except Exception` below
        # swallowed it, recording IT as the experiment's error and
        # silently FAILing a perfectly good finished run. Programming
        # errors must not masquerade as experiment failures (DEBUGGING.md
        # entry 6). Terminal controllers need RESET before reuse.
        if self._fsm.state in (State.DONE, State.FAILED):
            raise IllegalTransitionError(
                f"controller already {self._fsm.state.name}; "
                f"trigger Event.RESET before running again")
        """Happy path: TRAINING -> EVALUATING -> REPORTING -> DONE.

        Each _fsm.trigger() executes that phase's work via its on_enter
        hook. Any exception routes to FAIL so the result always carries
        the state *and* the reason.

        stop_flag (Phase 5): cooperative cancellation, checked at phase
        boundaries only. We never abort mid-episode -- partial training
        state would poison the Q-table; finishing the current phase keeps
        every artifact consistent. A stopped run reports FAILED with a
        'stopped by user' error string, which is honest for exit codes.
        """
        start = time.perf_counter()
        try:
            self._fsm.trigger(Event.START_TRAINING)   # -> TRAINING (trains)
            if stop_flag is not None and stop_flag.is_set():
                raise _StopRequested("stopped before evaluation")
            self._fsm.trigger(Event.FINISH_TRAINING)  # -> EVALUATING (evals)
            if stop_flag is not None and stop_flag.is_set():
                raise _StopRequested("stopped before reporting")
            self._fsm.trigger(Event.FINISH_EVAL)      # -> REPORTING (stubs)
            self._fsm.trigger(Event.FINISH_REPORT)    # -> DONE
            # Late failure: a phase raised while the machine had already
            # moved past it (see _phase_error comment above). DONE is not
            # a legal FAIL source, so we surface it via the result object.
            if self._phase_error is not None:
                self.result.error = self._phase_error
                self.result.state = State.FAILED
                self.result.wall_time_s = time.perf_counter() - start
                return self.result
        except Exception as exc:
            # In-flight failure: FSM sits in the state whose hook raised;
            # FAIL is legal from every active state by construction.
            # If FAIL itself raises (illegal source state), still record
            # the ORIGINAL error -- losing it was bug 3b.
            self.result.error = self._phase_error or f"{type(exc).__name__}: {exc}"
            try:
                self._fsm.trigger(Event.FAIL)
            except IllegalTransitionError:
                pass
        self.result.wall_time_s = time.perf_counter() - start
        self.result.state = self._fsm.state
        return self.result

    def pause(self) -> None:
        """Manual PAUSE from outside (e.g. signal handler in Phase 5)."""
        self._fsm.trigger(Event.PAUSE)

    def resume(self) -> None:
        self._fsm.trigger(Event.RESUME)

    # -- FSM hooks: the actual phases ------------------------------------------

    def _enter_training(self, _old: State, _new: State) -> None:
        cfg = self.config
        env = create_env(cfg.env, seed=cfg.seed, extra=cfg.extra)
        agent = create_agent(cfg.agent, env, cfg, seed=cfg.seed)
        loggers = build_loggers(cfg, self.output_dir)
        trainer = Trainer(env, agent, loggers=loggers, max_steps=cfg.max_steps)
        self._agent = agent          # handed to the evaluation phase
        self._train_env = env
        self.result.records = trainer.train(episodes=cfg.episodes,
                                            config=cfg.to_dict(), seed=cfg.seed)

    def _enter_evaluating(self, _old: State, _new: State) -> None:
        cfg = self.config
        evaluator = Evaluator(self._train_env, max_steps=cfg.max_steps,
                              success_threshold=cfg.success_threshold)
        # Frozen policy: training=False inside Evaluator keeps exploration off.
        self.result.eval_report = evaluator.evaluate(
            self._agent, episodes=cfg.eval_episodes, seed=cfg.seed + 1)

    def _enter_reporting(self, _old: State, _new: State) -> None:
        # Phase 6: real report generation. Deferred import so core/reporting
        # never hard-depends on matplotlib (headless CI without plotting can
        # still run train+eval; the ImportError is honest and local).
        from rl_lab.reporting.report_generator import ReportGenerator
        gen = ReportGenerator()
        self._report_path = gen.write_run_report(
            self.result, self.output_dir / "report.md")
