# Architecture

## Component diagram

```mermaid
flowchart TB
    subgraph entry ["Entry"]
        CLI["cli.py<br/>(run / sweep / validate / list)"]
    end

    subgraph orchestration ["Orchestration"]
        SCH["JobScheduler<br/>thread pool + queue.Queue<br/>graceful stop (Event)"]
        CTRL["ExperimentController<br/>(Facade: owns FSM + phases)"]
        FSM["ExperimentFSM<br/>transition TABLE, hooks"]
    end

    subgraph config ["Config"]
        LOAD["loader.py<br/>YAML -> dict"]
        SCHEMA["schema.py<br/>ExperimentConfig dataclass<br/>fail-fast validation"]
    end

    subgraph training ["Training core"]
        FAC["Factory<br/>(create env/agent by name)"]
        TR["Trainer<br/>episode loop"]
        EV["Evaluator<br/>mean/std/success"]
    end

    subgraph domain ["Domain interfaces (core/)"]
        BASE["BaseEnvironment<br/>BaseAgent<br/>BaseLogger<br/>StepResult (frozen)"]
        EXPL["ExplorationPolicy<br/>(Strategy: eps-greedy...)"]
    end

    subgraph impls ["Implementations"]
        ETCH["EtchChamberEnv<br/>physics + sensor noise"]
        GYM["GymEnvWrapper<br/>(Adapter)"]
        RAND["RandomAgent"]
        QL["QLearningAgent"]
        LOGS["Console / CSV / JSONL loggers<br/>(Observer)"]
        REP["ReportGenerator<br/>Markdown + PNG plots"]
    end

    CLI --> SCH --> CTRL
    CLI --> CTRL
    CTRL --> FSM
    CTRL --> LOAD --> SCHEMA --> FAC
    CTRL --> TR --> EV
    CTRL --> REP
    FAC --> ETCH & GYM & RAND & QL
    TR --> EXPL
    CTRL -. "subscribe" .-> LOGS
    ETCH -.implements.-> BASE
    GYM -.implements.-> BASE
    RAND -.implements.-> BASE
    QL -.implements.-> BASE
    LOGS -.implements.-> BASE
    QL --> EXPL
```

Dependency rule (clean architecture): arrows point inward. `core/` knows
nothing about YAML, threads, matplotlib or gymnasium; those are mechanisms
injected from the outside.

## Experiment lifecycle FSM

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> CONFIGURED: CONFIGURE
    CONFIGURED --> TRAINING: START_TRAINING
    TRAINING --> PAUSED: PAUSE
    PAUSED --> TRAINING: RESUME
    TRAINING --> EVALUATING: FINISH_TRAINING
    EVALUATING --> REPORTING: FINISH_EVAL
    REPORTING --> DONE: FINISH_REPORT
    DONE --> IDLE: RESET
    PAUSED --> IDLE: RESET
    IDLE --> FAILED: FAIL
    CONFIGURED --> FAILED: FAIL
    TRAINING --> FAILED: FAIL
    PAUSED --> FAILED: FAIL
    EVALUATING --> FAILED: FAIL
    REPORTING --> FAILED: FAIL
    FAILED --> IDLE: RESET
    DONE --> [*]
```

Any (state, event) pair absent from the table raises
`IllegalTransitionError` naming the state, the event, and the legal
alternatives. Note there is no arrow out of DONE except RESET -- you cannot
"resume" a finished experiment; you start a new run object.

## One full experiment run (sequence)

```mermaid
sequenceDiagram
    participant U as User/CLI
    participant C as Controller
    participant F as FSM
    participant T as Trainer
    participant E as Env (Etch/CartPole)
    participant A as Agent (Q-learning)
    participant L as Loggers (Observer)

    U->>C: run(config.yaml)
    C->>F: CONFIGURE (loads+validates YAML)
    F-->>C: CONFIGURED (on_enter: build env/agent via Factory, attach loggers)
    C->>F: START_TRAINING
    F-->>C: TRAINING (on_enter: Trainer.fit)
    loop each episode
        T->>E: reset()
        T->>A: act(state)  [eps-greedy Strategy]
        T->>E: step(action)
        E-->>T: StepResult(reward, done, truncated)
        T->>A: observe(...)  [bootstrap iff not true terminal]
        T->>L: on_episode_end(stats)
    end
    C->>F: FINISH_TRAINING
    F-->>C: EVALUATING (on_enter: Evaluator: greedy policy, N episodes)
    C->>F: FINISH_EVAL
    F-->>C: REPORTING (on_enter: ReportGenerator -> md + png)
    C->>F: FINISH_REPORT
    F-->>C: DONE
    C-->>U: RunResult(state, eval stats, artifact paths)

    Note over C,F: any exception -> FAIL event -> FAILED,<br/>loggers closed, error recorded in result
```

## Design patterns used

| Pattern | Where | Why here / elsewhere |
|---|---|---|
| Strategy | `BaseAgent`, `ExplorationPolicy` | swap random/eps-greedy/softmax without touching Trainer. Same pattern as sorting comparators, payment providers. |
| Factory | `rl_lab/factory.py` | build env/agent from config strings; new types = one registration line. Like DB connection factories, serializer registries. |
| Adapter | `GymEnvWrapper` | third-party Gymnasium API -> our StepResult contract without forking it. Like logging shims, legacy-API gateways. |
| Observer | `BaseLogger` subscribers on controller events | CSV/JSON/console decoupled from training loop. Like event buses, webhooks, pub-sub brokers. |
| FSM (table-driven) | `core/fsm.py` | legality as data, not scattered ifs. Recipe sequencers in wafer fabs use exactly this to prevent illegal tool states. |
| Facade | `ExperimentController` | hides FSM+trainer+evaluator+report wiring behind one `run()`. Like a compiler driver API. |
| Template Method | FSM `on_enter/on_exit` hooks | control flow fixed, phase work injected. Like framework lifecycle callbacks (React hooks, servlet init/destroy). |
| Producer/Consumer | `JobScheduler` queue + workers | stdlib-verified queue instead of hand-rolled deque+Lock. Job queues everywhere in equipment automation. |
