# feat: rl-lab v0.1.0 — modular RL benchmark framework (Phases 1–7)

Closes #1 (initial release) · Branch: `feature/phase6-reporting` → `main` · Tag: `v0.1.0`

---

## Summary

Complete implementation of **rl-lab**, an OOP-first framework to train,
evaluate, log and report reinforcement-learning experiments. Built in seven
incremental phases with tests at every step. Ships a custom
semiconductor-flavored environment (`EtchChamberEnv`: chamber pressure / gas
flow control to hit a target etch rate, with physics-inspired dynamics and
sensor noise) alongside standard Gymnasium CartPole to prove the framework is
generic.

**Final metrics:** 123 tests passing (1 skipped), **91% coverage overall /
96% on core modules** (target was >=70%), end-to-end CLI verified for
single runs, seed sweeps, config overrides, and Markdown+PNG report
generation.

## What's included (by phase)

| Phase | Deliverable | Key files |
|-------|-------------|-----------|
| 1 | Package scaffold, ABCs, YAML config loader + validation | `rl_lab/core/base.py`, `rl_lab/config/` |
| 2 | `EtchChamberEnv`, `RandomAgent`, Trainer loop, Factory, Gym adapter | `rl_lab/envs/`, `rl_lab/training/` |
| 3 | Tabular Q-Learning agent, exploration Strategy (epsilon-greedy + decay), Evaluator, Console/CSV/JSON loggers (Observer) | `rl_lab/agents/`, `rl_lab/core/exploration.py`, `rl_lab/logging/` |
| 4 | Experiment FSM (table-driven transitions, illegal moves raise clear errors), controller facade, argparse CLI (`run`, `validate`, `list`, `sweep`) | `rl_lab/core/fsm.py`, `rl_lab/orchestration/controller.py`, `rl_lab/cli.py` |
| 5 | Multithreaded sweep runner: thread-safe job queue, worker pool, lock-protected results, graceful shutdown via `Event` flag | `rl_lab/orchestration/scheduler.py` |
| 6 | Report generator: learning curves (matplotlib/Agg), per-run Markdown reports, cross-run comparison tables, sweep aggregate — wired into the FSM's REPORTING state | `rl_lab/reporting/report_generator.py` |
| 7 | Docs: README, Mermaid architecture + FSM + sequence diagrams, DEBUGGING.md (9 real bugs), extension guide | `README.md`, `docs/architecture.md`, `DEBUGGING.md` |

## Design patterns (and where else they apply)

- **Strategy** — agents and exploration policies are swappable objects.
  Same shape as payment providers or compression codecs behind one interface.
- **Factory** — configs name agents/envs; nothing imports concrete classes
  in the training path. Standard for plugin systems (pytest plugins, DB drivers).
- **Adapter** — `GymEnvWrapper` maps the Gymnasium API onto `BaseEnvironment`;
  exactly how you'd wrap a vendor protocol (SECS/GEM, OPC-UA) behind a stable internal API.
- **Observer** — loggers subscribe to trainer events; decouples *what* is
  recorded from *how* training proceeds (same as event buses / message brokers).
- **Table-driven FSM** — lifecycle states (`IDLE -> CONFIGURED -> TRAINING ->
  EVALUATING -> REPORTING -> DONE`, plus `PAUSED`/`FAILED`) defined as a
  transition dict, not nested ifs. This is how equipment job managers model
  wafer/process lifecycles.
- **Facade** — `ExperimentController` hides wiring behind `run(config_path)`.

## Threading model & correctness notes

- Workers pull from a `queue.Queue`; shared result list guarded by a `Lock`;
  shutdown coordinated through a `threading.Event` sentinel.
- Reproducibility: every experiment gets its **own** `numpy.random.Generator`
  seeded per-(config, seed) — no global-RNG coupling, so parallel sweeps are
  bit-identical to sequential ones (regression-tested).
- Truncation vs termination handled correctly in Q-learning updates
  (bootstrap time-limited episodes instead of treating them as absorbing —
  see DEBUGGING.md #3).

## Debugging log highlights (`DEBUGGING.md`, 9 entries)

Each entry is symptom -> root cause -> fix, with a named regression test:
pytest-plugin autoload crash, namespace leak between envs, truncated-vs-terminated
learning bug, global-RNG nondeterminism, parallel-vs-sequential divergence,
FSM late-crash reporting, re-entrant recursion, interpreter-exit thread hang
(daemon workers + context-manager close), and `-o extra.key` validation layering.

## Test plan

- [x] `python -m pytest` — 123 passed, 1 skipped, clean exit (no thread leaks)
- [x] Coverage: `python -m pytest --cov=rl_lab` — 91% total, 96% core
- [x] `python -m rl_lab.cli run configs/cartpole_random.yaml --report auto -o episodes=50`
- [x] `python -m rl_lab.cli sweep configs/etch_qlearning.yaml --seeds 0 1 2 --workers 3 --report`
- [x] `python -m rl_lab.cli validate configs/etch_qlearning.yaml` and bad-config error paths
- [x] FSM illegal-transition tests (e.g. TRAINING -> REPORTING raises with actionable message)
- [x] Scheduler determinism test: same seeds, threads=1 vs threads=3 -> identical results

## Reviewer reading order

README -> `docs/architecture.md` (diagrams) -> `DEBUGGING.md` -> `tests/`.

## Known limitations (documented honestly in README)

- DQN agent is stubbed (PyTorch optional); tabular Q-learning only.
- JSON logger writes a summary file, not a true JSONL event stream yet.
- Threads chosen over processes; GIL fine for this workload, backend swap
  documented under "How I'd extend this".

---

Co-authored during a mentored build-out; happy to walk through any module in review.
