# rl-lab: Reinforcement Learning Benchmark Lab

A modular, OOP-first framework to **train, evaluate, log and report**
reinforcement-learning experiments -- built as a portfolio project for
reviewers at semiconductor-equipment software teams. It ships a custom
process-control environment (`EtchChamberEnv`: chamber pressure + gas flow
vs. target etch rate, with sensor noise) *and* runs standard Gymnasium
environments (CartPole) through the identical pipeline, proving the
framework is generic rather than one-off glue code.

The point of this repo is not SOTA agents. The point is **engineering**:
clean architecture, tested state machines, thread-safe parallelism,
reproducible configs, crash-safe logging, and honest debugging records.

```
$ python -m rl_lab.cli sweep configs/etch_qlearning.yaml --seeds 0 1 2 --workers 3 --report
  progress: 3/3 done
  etch-qlearning-v1_seed0   DONE  eval mean=  -4.56 success=100%
  etch-qlearning-v1_seed1   DONE  eval mean=  -3.79 success=100%
  etch-qlearning-v1_seed2   DONE  eval mean=  -3.02 success=100%
aggregate over 3/3 scored runs: mean=-3.79 std_across_seeds=0.77
sweep report: runs/sweep_report.md
```

---

## Why this exists

RL code in the wild tends to be notebook soup: globals, hidden RNG state,
training loops that cannot be stopped cleanly, results nobody can reproduce.
This project treats an experiment runner the way equipment software treats a
process recipe:

- **Illegal states are impossible, not discouraged.** The experiment
  lifecycle is a table-driven FSM (`IDLE -> CONFIGURED -> TRAINING ->
  EVALUATING -> REPORTING -> DONE`, plus `PAUSED`/`FAILED`). An illegal
  transition raises an error naming the state, event, and legal options --
  the same discipline a fab tool applies to recipe sequencing.
- **Reproducibility is per-object.** Every env/agent owns a seeded private
  RNG; global RNG is never touched. Parallel sweeps are bit-identical to
  sequential runs (there is a regression test for exactly that).
- **Shutdown is a feature.** Ctrl-C or `scheduler.stop()` finishes the
  current phase, marks jobs `PAUSED`, and closes every logger -- no torn
  files, no zombie threads. See DEBUGGING.md #5 and #7 for the bugs that
  taught us this the hard way.

## Quick start

Requires Python 3.11+ (developed on 3.12).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pyyaml numpy gymnasium matplotlib pytest coverage

# one experiment, from YAML config
python -m rl_lab.cli run configs/etch_qlearning.yaml

# quick knob tweaks without editing files
python -m rl_lab.cli run configs/cartpole_random.yaml -o episodes=50

# multi-seed parallel sweep + aggregated markdown report
python -m rl_lab.cli sweep configs/etch_qlearning.yaml --seeds 0 1 2 --workers 3 --report

# check a config without burning GPU-hours on a typo
python -m rl_lab.cli validate configs/etch_qlearning.yaml
python -m rl_lab.cli list            # registered envs and agents

# tests + coverage
python -m pytest                      # 123 passed, 1 skipped
python -m coverage run -m pytest && python -m coverage report --include="rl_lab/*"
                                      # 91% total, core modules >=85%
```

Artifacts land in `runs/<experiment>/`: `episodes.csv`, `summary.json`,
`report.md`, `learning_curve.png`. Sweeps add `runs/sweep_report.md`.

## Architecture at a glance

```
rl_lab/
  core/          abstract contracts (BaseAgent/BaseEnvironment/BaseLogger),
                 StepResult, ExplorationPolicy, ExperimentFSM   <- zero deps
  agents/        RandomAgent, QLearningAgent (tabular)
  envs/          EtchChamberEnv (custom physics), GymEnvWrapper (Adapter)
  training/      Trainer episode loop
  evaluation/    Evaluator (mean, std, success rate)
  logging/       Console / CSV / JSONL observers
  reporting/     ReportGenerator (Markdown + matplotlib curves)
  orchestration/ ExperimentController (Facade) + JobScheduler (threads)
  config/        YAML loader + dataclass schema, fail-fast validation
  cli.py         argparse entry: run | sweep | validate | list
tests/           124 tests, 91% coverage
docs/            Mermaid component diagram, FSM diagram, run sequence
DEBUGGING.md     9 real bugs: symptom -> root cause -> fix
```

Full diagrams (component graph, FSM state diagram, end-to-end sequence):
[`docs/architecture.md`](docs/architecture.md).

### Design patterns, and where else you'd see them

| Pattern | Here | Elsewhere |
|---|---|---|
| **Strategy** | `BaseAgent` + `ExplorationPolicy` are swappable mid-loop; Trainer depends only on interfaces | sort comparators, payment providers, compression codecs |
| **Factory** | `factory.py` maps `"qlearning"` / `"gym:CartPole-v1"` strings to objects; new type = one registration line | DB connection pools, serializer registries |
| **Observer** | loggers subscribe to training events; adding InfluxDB later touches zero training code | event buses, webhooks, pub-sub brokers |
| **Adapter** | `GymEnvWrapper` converts Gymnasium's `(obs, rew, term, trunc, info)` into our frozen `StepResult` | legacy-API gateways, logging shims |
| **Table-driven FSM** | legality lives in one dict `(state, event) -> state`, not scattered ifs | protocol parsers, workflow engines, **tool recipe sequencers** |
| **Producer/Consumer** | `queue.Queue` + worker pool with sentinel shutdown and `Event` stop-flag | job queues, message workers |
| **Facade / Template Method** | `ExperimentController` wires phases via FSM `on_enter` hooks | compiler drivers, framework lifecycle callbacks |

### EtchChamberEnv -- the domain hook

Deliberately simple but mass-balance honest:

- Pressure follows a first-order lag toward the pump throttle setpoint.
- Etch rate follows a Langmuir-Hinshelwood-like law:
  `R = k_e * Q / (1 + Q/Q_sat) * f(P)` -- gas flow saturates and pressure
  has an optimum, so "max both knobs" is not a winning policy.
- Reward penalises `|etch_rate - target|` **plus actuator movement** (fast
  valve changes wear hardware in real tools).
- Gaussian sensor noise on readings; continuous physics underneath, bucketed
  discrete view on top for tabular agents -- the same split real equipment
  software makes between process simulation and recipe logic.

## Configuration

One YAML per experiment, validated before a single step runs:

```yaml
name: etch-qlearning-v1
env: etch_chamber            # or "gym:CartPole-v1"
agent: qlearning
seed: 0                      # reproducibility knob
episodes: 300
max_steps: 60                # truncation limit per episode
eval_episodes: 20
log_dir: runs/etch-qlearning-v1
extra:                       # agent-specific knobs pass through untouched
  learning_rate: 0.2
  epsilon_start: 1.0
  epsilon_end: 0.05
```

Unknown keys and out-of-range values raise errors that name the offending
field (`config error: 'learnig_rate' is not a known key; did you mean
'learning_rate'?` style) instead of failing 20 minutes into training.

## Testing & quality

- **123 passed + 1 skipped, 91% coverage** (core modules ≥85%, several at
  100%). Property-style checks for the FSM (every absent table entry must
  raise), determinism tests (same seed → byte-identical CSV), and a
  parallel-equals-sequential scheduler test.
- Thread-safety verified under `-p no:randomly`-style shuffled ordering and
  stress iterations.
- `pyproject.toml` pins pytest plugin blacklisting so the suite runs green
  on polluted shared environments (DEBUGGING.md #1).

## Debugging log

[`DEBUGGING.md`](DEBUGGING.md) records nine bugs actually hit during
development, each as **symptom → root cause → fix**, including:

- Q-learning flatlining because **truncation was treated as termination** (#2)
- Non-reproducible CI failures from **global RNG sharing** (#4)
- Parallel sweeps diverging from sequential runs (#6)
- The full pytest session **hanging at interpreter exit** on leaked worker
  threads, diagnosed with `faulthandler.dump_traceback_later` (#7)

## Git workflow

- `main` – stable, reviewed commits only
- `dev` – integration branch
- `feature/phaseN-*` – one branch per milestone, merged with descriptive
  commit messages ("add table-driven experiment FSM", "fix truncated-vs-
  terminated bootstrap in Q-learning")
- Local pre-commit hook runs the suite; red tests don't get committed.

## How I'd extend this

Ordered by what a real team would need next:

1. **New agent (DQN, PyTorch).** Implement `BaseAgent` in
   `agents/dqn_agent.py`, register it in `factory.py` (one line), add a
   `replay_buffer` to `extra:` config. The Trainer needs zero changes --
   that's the Strategy/Factory contract paying rent. Then make the
   scheduler backend switchable: threads today, `ProcessPoolExecutor` for
   torch sweeps (torch intra-op threads × N processes = thrashing; the
   submit()/run_all() interface already isolates this decision).
2. **New logger (InfluxDB / Prometheus pushgateway).** Subclass
   `BaseLogger`, implement `on_episode_end`/`close`, subscribe it in the
   controller's configure hook. Observer pattern means no existing file
   changes. Add batching + backpressure here, since metrics endpoints are
   slower than disk.
3. **New protocol (SECS/GEM-style messaging).** This is where the RL lab
   meets real fab software: wrap the env's `step()` behind a message-based
   client so `EtchChamberEnv` could be replaced by a simulator speaking a
   SECS-II subset over TCP. The `BaseEnvironment` ABC is the seam; an
   adapter like `GymEnvWrapper` slots in beside it. Bonus: exercise the
   graceful-shutdown path against timeouts and partial messages.
4. **Hyperparameter search.** Sweep already fans seeds across workers; the
   same `JobScheduler` grid-searches `extra:` knobs with a cartesian
   product, reporting via `comparison_table()` sorted best-first.
5. **Checkpoint/resume.** FSM already has PAUSED; persist Q-tables per job
   and let RESET accept a checkpoint path. (Honest note: currently unimple‐
   mented; listed because the lifecycle model supports it.)

## Known limitations (stated, not hidden)

- Tabular Q-learning only; DQN stubbed out deliberately (optional dep).
- Threads chosen over processes with a documented swap path (see above).
- Reports are Markdown-only; HTML template output is TODO.
- `success` metric is env-defined (CartPole solve == score ≥ 475; etch env
  counts episodes within tolerance band).
- JSON logger writes a per-run `summary.json` (JSONL streaming is on the
to-do list once runs outlive processes).

## License / contact

MIT. Built as a hiring-manager-readable engineering sample: read
`docs/architecture.md` first, then `DEBUGGING.md` -- they carry more signal
than any README sentence I could write about myself.
