"""Parallel experiment runner: thread-safe job queue + worker pool.

Why threads and not processes (for now): our workload is tabular Q-learning
inside Python loops plus file IO -- and the heavy inner step actually calls
numpy/pandas C code that releases the GIL, so real work overlaps. The
backend is deliberately hidden behind submit()/run_all() so swapping in
ProcessPoolExecutor later is a one-file change. If we ever
add DQN-torch sweeps, processes become mandatory: torch spawns its own
intra-op threads and N python processes x M torch threads = thrashing.

Patterns used:
  - Producer/consumer with queue.Queue (already lock-based internally;
    we do NOT reinvent it with deque+Lock -- stdlib queues are tested).
  - Sentinel shutdown: putting None per worker wakes idle consumers cleanly.
    Alternative considered: an Event checked inside get() -- rejected because
    blocking get() can't poll events without timeout-polling loops.
  - Result collection under ONE lock (RLock) shared with progress reads, so
    the CLI monitor never sees a half-updated results list.

Graceful shutdown contract (mirrors equipment control software):
  stop_flag is an Event polled at *phase boundaries* by each controller run.
  Ctrl-C / scheduler.stop() sets it; in-flight episodes finish their current
  phase, jobs then exit with state PAUSED instead of being killed mid-step.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

from rl_lab.config.loader import load_config
from rl_lab.core.fsm import State
from rl_lab.orchestration.controller import ExperimentController, RunResult


class JobScheduler:
    """Fan YAML configs out over seed-variant jobs on a worker pool."""

    def __init__(self, num_workers: int | None = 4, max_qsize: int = 0) -> None:
        # None = auto: one worker per physical-ish core, capped by nothing
        # here (job count caps real parallelism anyway). CLI passes None
        # when the user gives --workers 0.
        if num_workers is None:
            num_workers = min(8, (os.cpu_count() or 4))
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.num_workers = num_workers
        # Bounded queue = backpressure. With maxsize=0 (unbounded) a huge
        # sweep just eats RAM; a bound forces producers to slow down, which
        # matters when jobs carry big config payloads. Kept configurable.
        self._queue: queue.Queue = queue.Queue(maxsize=max_qsize)
        self._results: list[RunResult] = []
        # RLock (not Lock): _collect() holds it while calling snapshot(),
        # which re-acquires it. A plain Lock would deadlock on itself.
        self._lock = threading.RLock()
        self._stop_flag = threading.Event()
        # Lazy start: threads spawn on the first submit(). The previous
        # design had callers remember sched.start() AFTER submitting --
        # and workers started later never wake for items already sitting
        # in the queue (queue.Queue only notifies on put()). That ordering
        # bug wedged every pre-start submission (DEBUGGING.md #7). Making
        # "submit implies pool alive" an invariant removes the footgun
        # entirely; explicit start() stays legal and idempotent.
        # _start_lock guards lazy start against a submit/enter race: two
        # producers must never both see _started == False.
        self._started = False
        self._start_lock = threading.Lock()
        # DEBUGGING.md #7, second layer (REWORKED): earlier we promised an
        # atexit drain, but it was never actually wired up -- leaked pools
        # still wedged the interpreter in threading._shutdown at exit.
        # Workers are now DAEMON threads: they can never block process
        # exit. We keep join() as our normal completion proof (`with
        # sched:` / wait-after-close); daemon-ness is only the last-resort
        # guarantee that a forgotten close() costs a silent exit instead of
        # a hung CI job. Tradeoff documented: daemons die mid-job if the
        # main thread exits without closing -- acceptable for sweeps whose
        # results only matter to the still-alive parent.
        self._threads: list[threading.Thread] = []
        # join()-based wait() needs to know the pool was closed, otherwise
        # workers blocked on an empty queue never exit and "all done" is
        # indistinguishable from "idle waiting for more work".
        self._closed = False
        self._submitted = 0

    # -- submission -----------------------------------------------------------

    def expand_seed_jobs(self, path: str | Path, seeds: list[int]) -> list[dict]:
        """One config file x N seeds -> N jobs (the 'sweep' unit).

        Seed variants write to runs/<name>_seed<k>/ so CSV artifacts from
        parallel workers never collide -- separate output dirs are cheaper
        than a shared-file lock and keep every run independently inspectable.
        """
        base = load_config(path)
        jobs = []
        for s in seeds:
            stem = Path(base.name).stem
            cfg = replace(base, seed=s, name=f"{stem}_seed{s}",
                          output_dir=base.output_dir)
            jobs.append({"config": cfg})
        return jobs

    def submit(self, job: dict) -> None:
        """Enqueue {'config': ExperimentConfig} or {'path': yaml}. Blocks if
        the queue is bounded and full -- intentional backpressure.

        Lazily starts the worker pool on first submit so callers cannot
        create the put-before-start ordering bug (see __init__ comment).
        """
        if "path" in job:
            cfg = load_config(job["path"])
            job = {"config": cfg}
        self.start()                    # idempotent
        self._submitted += 1
        self._queue.put(job)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker pool. Idempotent -- safe to call explicitly or
        rely on submit()'s lazy start (both paths end up here once).

        Double-checked locking: fast path reads _started without a lock;
        the check-and-set happens atomically under _start_lock so two
        concurrent producers can never each spawn a full pool."""
        if self._started or self._closed:      # fast path, no lock needed
            return
        with self._start_lock:
            if self._started or self._closed:  # re-check: another thread won
                return
            self._started = True
            for i in range(self.num_workers):
                t = threading.Thread(target=self._worker_loop,
                                     name=f"rl-worker-{i}",
                                     daemon=True)  # DEBUGGING.md #7: leaked pools must never block interpreter exit
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        """Signal shutdown and wake every worker with one sentinel.

        Order matters: set the flag FIRST so workers that grab a real job
        between stop() and join() still bail at the next phase boundary;
        sentinels drain the queue afterwards. If the pool never started
        (zero jobs submitted), there is nothing to wake -- just mark closed.
        """
        self._stop_flag.set()
        with self._lock:
            if not self._closed:
                self._closed = True
                if not self._started:
                    # No threads exist, but any queued jobs would never be
                    # consumed -> wait() would hang on its count predicate.
                    # Drain now so "closed" always means "quiesced".
                    try:
                        while True:
                            self._queue.get_nowait()
                            self._submitted -= 1   # honest accounting
                            self._queue.task_done()
                    except queue.Empty:
                        pass
                    return
                for _ in self._threads:
                    self._queue.put(None)      # sentinel per worker

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return                 # sentinel: orderly exit
                self._run_job(job)
            finally:
                self._queue.task_done()

    def _run_job(self, job: dict) -> None:
        cfg = job["config"]
        try:
            controller = ExperimentController(cfg)
            # Cooperative cancellation hook: controller checks this Event
            # between FSM phases (see controller.run()).
            result = controller.run(stop_flag=self._stop_flag)
        except Exception as exc:  # noqa: BLE001
            # A worker thread must NEVER die from one bad job: without this
            # net, an exception before _collect() loses the result AND the
            # worker, so wait() hangs and the whole sweep wedges. Record it
            # as a FAILED run so aggregate()/reports still see the job.
            result = RunResult(config=cfg, state=State.FAILED,
                               error=f"{type(exc).__name__}: {exc}")
        self._collect(result)

    def _collect(self, result: RunResult) -> None:
        with self._lock:
            self._results.append(result)

    # -- reporting ------------------------------------------------------------

    def snapshot(self) -> dict:
        """Thread-safe progress view for monitors/CLI (called while running)."""
        with self._lock:
            done = len(self._results)
            states = {}
            for r in self._results:
                states[r.state.name] = states.get(r.state.name, 0) + 1
            return {
                "finished": done,
                # unfinished_tasks counts sentinels too once stop() posted
                # them; callers treat 'pending' as advisory progress only.
                "pending": self._queue.unfinished_tasks,
                "by_state": states,
            }

    def wait(self, timeout: float | None = None) -> bool:
        """True when every submitted job has been collected.

        BUG FOUND THE HARD WAY (DEBUGGING.md #7): the first version waited
        on queue.unfinished_tasks, but run_sweep submits BEFORE start(), so
        unfinished_tasks counts jobs no worker has even seen yet -- with
        workers idle-blocked in get() nothing new ever wakes them and the
        poll loop wedged until timeout. Correct "done" predicate is simply:
        every thread alive OR every job collected. When the pool is still
        open we never join() (workers parked in get() have no exit path
        until close()); join is only meaningful after stop()/close().
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._count_results() < self._submitted:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        if self._closed:
            # After shutdown was requested, "done" also means the threads
            # actually exited -- that's what proves sentinels were consumed.
            for t in self._threads:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                t.join(timeout=remaining)
                if t.is_alive():
                    return False
        return True

    # -- context manager ------------------------------------------------------
    # __enter__/__exit__ make "forgot to shut down" impossible by structure:
    # exiting the block ALWAYS posts sentinels and joins, even on exception.
    # This is the same discipline as `with open()` -- resources you acquire
    # should be released without relying on caller memory.

    def __enter__(self) -> "JobScheduler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # If an exception is propagating, stop_flag makes in-flight jobs bail
        # at their next phase boundary instead of piling up work nobody will
        # collect; then drain cleanly either way. stop() handles the
        # never-started case and posts exactly one sentinel per live worker.
        if exc_type is not None:
            self._stop_flag.set()
        self.stop()
        for t in self._threads:
            t.join(timeout=30)
        return False   # never swallow exceptions

    def _count_results(self) -> int:
        with self._lock:
            return len(self._results)

    @property
    def results(self) -> list[RunResult]:
        with self._lock:
            return list(self._results)   # copy: callers can't mutate our list

    def aggregate(self) -> dict:
        """Cross-seed stats -- what Phase 6's comparison table consumes.

        mean/std across SEEDS of eval-mean-reward answers the question a
        hiring manager actually asks: is the improvement robust to seed,
        or did we get lucky? (std across seeds != std_reward within a seed.)
        """
        with self._lock:
            ok = [r for r in self._results if r.eval_report is not None]
        if not ok:
            return {"n": len(self._results), "mean_eval_reward": None,
                    "std_across_seeds": None, "success_rate": None}
        means = [r.eval_report.mean_reward for r in ok]
        successes = [r.eval_report.success_rate for r in ok]
        n = len(means)
        mean = sum(means) / n
        # sample std (ddof=1); guard n==1 -> 0.0 (statistics.stdev raises)
        var = sum((m - mean) ** 2 for m in means) / (n - 1) if n > 1 else 0.0
        return {
            "n": len(self._results),
            "n_scored": n,
            "mean_eval_reward": mean,
            "std_across_seeds": var ** 0.5,
            "success_rate": sum(successes) / n,
            "failed": sum(1 for r in self._results if r.state is State.FAILED),
        }


def run_sweep(paths: list[str | Path], seeds: list[int], num_workers: int = 4,
              on_progress=None) -> tuple[list[RunResult], dict]:
    """Convenience entry: fan out, run, join. Used by `rl-lab sweep`.

    on_progress(snapshot_dict) is called from the CALLING thread only
    (no logger-thread surprises -- see DEBUGGING.md entry on Observer
    thread-safety).
    """
    sched = JobScheduler(num_workers=num_workers)
    total = 0
    for p in paths:
        jobs = sched.expand_seed_jobs(p, seeds)
        total += len(jobs)
        for j in jobs:
            sched.submit(j)
    last = 0
    # with-block guarantees sentinel shutdown + join even if a progress
    # callback raises (see DEBUGGING.md #7: leaked non-daemon threads made
    # the interpreter hang at exit).
    with sched:
        while not sched.wait(timeout=0.5):
            snap = sched.snapshot()
            if on_progress and snap["finished"] != last:
                last = snap["finished"]
                on_progress(snap)
            if total and snap["finished"] >= total:
                break
    snap = sched.snapshot()
    if on_progress and snap["finished"] != last:
        on_progress(snap)
    return sched.results, sched.aggregate()
