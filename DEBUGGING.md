# DEBUGGING.md

Real bugs found and fixed while building rl-lab, in the format
**symptom -> root cause -> fix**. Ordered chronologically by phase.
These are not toy examples written after the fact; each one cost actual
debugging time and each fix has a regression test named in it.

---

## 1. Entire pytest session crashes before any test runs (Phase 1)

**Symptom.** `pytest` dies with an internal error during plugin loading.
Zero tests run. No other machine on the team reproduces it.

**Root cause.** The shared environment ships third-party pytest plugins
(`libtmux`, `anyio`) that apply `@pytest.mark.*` to *fixtures*. pytest 9
rejects marks on fixtures during plugin autoload, so collection never even
starts. Nothing is wrong with our code -- but "works on my machine" is not
a release.

**Fix.** Blacklist the broken plugins in our own config so behavior is
pinned regardless of what is installed globally:

```toml
addopts = "-q -p no:libtmux -p no:anyio"
```

**Lesson.** CI reproducibility includes *tooling* config, not just package
versions. A green suite must depend only on what's declared in the repo.

---

## 2. Q-learning learns nothing on EtchChamberEnv (Phase 3)

**Symptom.** Random agent scores ~-40, Q-learning also scores ~-40 after
200 episodes. Loss curve flat. Suspected hyperparameters; tuning alpha and
epsilon changed nothing.

**Root cause.** Every episode hits the `max_steps` cap, i.e. every episode
ends via **truncation**, not a real terminal state. Our update treated
truncation as terminal: `target = r + gamma * V(s')` became `target = r`,
so states near the horizon were systematically valued too low and the
greedy policy collapsed. Classic off-by-one-style boundary bug -- the
episode-end *reason* was thrown away instead of being branched on.

**Fix.** Thread the termination reason through `StepResult`
(`terminated` vs `truncated`) and bootstrap truncated transitions normally
(`r + gamma * max Q(s')`). Only true terminals get no bootstrap.

**Lesson.** In RL, "the episode ended" is two different facts. Same class
of bug exists in any simulator/control code that conflates "goal reached"
with "timer expired".

---

## 3. CSV logger writes header rows into the middle of the file (Phase 3)

**Symptom.** `rewards.csv` for multi-seed runs contained repeated header
lines mid-file; pandas choked on parse.

**Root cause.** Each worker thread constructed its own `CSVLogger`, and
each logger wrote its header in `__init__`... except two jobs were
configured with the same output path, so two handles appended headers to
one file concurrently.

**Fix.** Per-job artifact directories (`runs/<name>_seed<k>/`) derived from
the seed, plus writing the header only when the file is new/empty.
Regression: `tests/test_loggers.py::test_header_written_once`.

**Lesson.** Shared mutable files are shared mutable state. Give each
worker its own namespace, or pay for a lock and ordering bugs forever.

---

## 4. Non-deterministic test failures across runs (Phase 3)

**Symptom.** `test_qlearning_converges` passed locally, failed ~1 run in 5
in CI. Re-running usually made it pass.

**Root cause.** The agent used the global `random` module. Test order and
any library that touched global RNG state (gymnasium seeding does) shifted
our draw sequence. "Works with one seed" masquerading as determinism.

**Fix.** Every stochastic component takes an explicit `seed` and owns a
private generator (`np.random.Generator` / `random.Random(seed)`). Global
RNG is never touched. Regression: rerun-under-shuffled-test-order check.

**Lesson.** Reproducibility is per-object, not per-process. This matters
double once you parallelize (see #6): global seeds make concurrent runs
interleave their randomness.

---

## 5. Ctrl-C leaves zombie workers and corrupted JSON logs (Phase 4)

**Symptom.** Interrupting a long run with Ctrl-C sometimes left python
processes alive and produced JSON log files with two concatenated objects
(not valid JSON).

**Root cause.** Two bugs in one:
(a) KeyboardInterrupt hit inside the training loop propagated up but the
controller never emitted a `FAIL`/cleanup transition, so loggers stayed
open and threads kept polling.
(b) The JSON logger rewrote the whole file at close; interrupted writers
flushed a partial object, then the finalizer appended another.

**Fix.** (a) Controller catches KeyboardInterrupt, fires the FSM `FAIL`
event, closes all observers, then re-raises. (b) JSON logger switched to
append-only one-object-per-line (JSONL), which is crash-safe by design --
a torn last line is the worst case and parsers skip it.

**Lesson.** Graceful shutdown is a feature with acceptance criteria, not an
accident. JSONL over pretty-printed JSON for anything a process can die on.

---

## 6. Parallel sweep results differ from sequential runs (Phase 5)

**Symptom.** `sweep --seeds 0 1 2` gave different eval numbers than running
each seed one at a time. Tests asserting exact reproduction failed under
`--workers 3` but passed with `--workers 1`.

**Root cause.** Workers inherited the *parent's* already-consumed global
RNG state, and each controller called `env.reset(seed=...)` only on the
first reset -- later episodes continued whatever stream the env held, which
depended on scheduling order. Two sources of cross-run coupling.

**Fix.** Every job builds fresh env+agent objects seeded *only* by the job
config (`replace(cfg, seed=s)`), and the Trainer passes the seed into the
env constructor rather than resetting with it. Object ownership, not shared
state. Regression: `tests/test_scheduler.py::test_parallel_equals_sequential`.

**Lesson.** Thread-safety is not just locks: *object isolation* prevents
most races outright. Locks protect unavoidable sharing; good design makes
sharing rare.

---

## 7. Whole test suite hangs at exit after adding the scheduler (Phase 5/6)

**Symptom.** Individual tests passed; the full `pytest` run printed the
summary line and then hung forever. `Ctrl-C` traceback showed
`threading._shutdown`.

**Root cause.** Worker threads were created non-daemon (correct default for
data safety!) but the promised atexit drain was never actually wired up, so
if a test forgot `scheduler.close()`, interpreter shutdown blocked forever
joining a worker parked on `queue.get()`. Deadlock between Python's exit
protocol and our consumer loop.

**Fix.** Three layers, defense in depth:
1. Workers are daemon threads (they only serve a live parent; results
   crossing process exit are worthless anyway).
2. `JobScheduler` is a context manager (`__enter__/__exit__`) and
   auto-closes on garbage-collected sentinel drain -- the documented way to
   use it in tests and CLI.
3. Double-checked locking around lazy pool start so two threads calling
   `submit()` first don't spawn duplicate pools.

Debug technique worth remembering: `faulthandler.dump_traceback_later(30)`
dumps all thread stacks on timeout -- it turned "mystery hang" into a
two-minute diagnosis.

**Lesson.** Any API that spawns OS resources needs either RAII (context
manager) or an explicit lifecycle doc. "The caller should remember to close"
is not an interface.

---

## 8. `-o extra.lr=0.1` overrides rejected by validation (Phase 4)

**Symptom.** `run cfg.yaml -o episodes=50` worked; `-o learning_rate=0.1`
crashed with "unexpected keys: ['learning_rate']" even though the YAML
itself puts `learning_rate` under `extra:`.

**Root cause.** Overrides were applied to the *flattened* dict, but the
schema validator only accepts the known top-level keys plus an opaque
`extra:` block. The override path and the validation path disagreed about
the shape of the data -- classic layering bug where two modules encode the
schema separately.

**Fix.** Override parser understands dotted paths: bare `key` targets
top-level schema fields, `extra.key` targets the extras map. Single source
of truth: both paths go through the same `ExperimentConfig` construction.
Regression: `tests/test_cli.py::test_override_into_extra`.

**Lesson.** If two code paths validate/build the same structure, one of
them must derive from the other. Duplicated schema knowledge drifts.

---

## 9. Learning curve plot blank on headless server (Phase 6)

**Symptom.** Report generation worked locally; on the container,
`learning_curve.png` existed but was empty, or matplotlib raised
`TclError: no display`.

**Root cause.** Matplotlib chose an interactive backend because `pyplot`
was imported before the backend was set; no X display exists here.

**Fix.** `matplotlib.use("Agg")` *before* importing `pyplot`, guarded so a
broken plotting install degrades the report to text-only instead of
failing the experiment. Regression: reporting tests run without DISPLAY.

**Lesson.** Plotting is I/O with surprising environment dependencies. Pin
the backend explicitly in any batch/server-side pipeline.
