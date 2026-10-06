"""Tabular Q-Learning agent (Watkins 1989), off-policy.

Update rule:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * V(s') - Q(s,a) ]
    V(s')  = max_a' Q(s',a')   for real terminals,
             Q(s,a) target uses NO future value when the episode was
             TRUNCATED -- truncation is an artifact of our time limit, not
             a true absorbing state. Treating it as terminal biases values
             low and (on EtchChamberEnv, where every episode hits the step
             cap) cripples learning. This distinction is handled in observe().

Storage: dict-of-dicts instead of a dense numpy array. Rationale: bucketed
state spaces can be large (CartPole discretisation ~ hundreds of bins) and
mostly unvisited early on; sparse storage also makes "which states have we
actually seen?" debuggable with a single len() call.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from rl_lab.core.base import BaseAgent, StepResult
from rl_lab.core.exploration import EpsilonGreedy, ExplorationPolicy


class QLearningAgent(BaseAgent):
    name = "qlearning"

    def __init__(self, n_actions: int, learning_rate: float = 0.2,
                 discount: float = 0.95, seed: int | None = None,
                 policy: ExplorationPolicy | None = None) -> None:
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount
        # Own RNG instance: never touch the global random module, or two
        # agents in one process would contaminate each other's streams
        # (this exact bug bites us again in the multithreaded runner).
        self._rng = random.Random(seed)
        self.policy = policy or EpsilonGreedy()
        # Optimistic initialisation: start values at 0 (rewards here are
        # <= 0), so every state looks attractive and exploration falls out
        # of greediness itself, not just epsilon.
        self.q_table: dict[Any, list[float]] = {}

    # -- helpers --------------------------------------------------------------

    def _values(self, state: Any) -> list[float]:
        """Lazy row creation keeps the table sparse; default 0.0."""
        return self.q_table.setdefault(state, [0.0] * self.n_actions)

    def best_action(self, state: Any) -> int:
        row = self._values(state)
        return max(range(self.n_actions), key=lambda i: row[i])

    # -- BaseAgent (Strategy) API ---------------------------------------------

    def act(self, observation: Any, training: bool = True) -> int:
        row = self._values(observation)
        return self.policy.select(row, self._rng, training)

    def observe(self, observation: Any, action: int, result: StepResult) -> None:
        row = self._values(observation)
        q_old = row[action]
        if result.terminated:
            target = result.reward                    # true end: no bootstrap
        else:
            # Truncation OR in-between step: both bootstrap V(s'). A time
            # limit is not an absorbing state -- see module docstring.
            target = result.reward + self.gamma * max(self._values(result.observation))
        row[action] = q_old + self.alpha * (target - q_old)

    def on_episode_end(self, episode: int) -> None:
        self.policy.on_episode_end(episode)

    # -- persistence (nice-to-have for long runs) -----------------------------

    def save(self, path: str) -> None:
        import json
        # tuple keys -> strings; JSON has no tuple type.
        serialisable = {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): v
                        for k, v in self.q_table.items()}
        with open(path, "w") as fh:
            json.dump({"alpha": self.alpha, "gamma": self.gamma,
                       "q_table": serialisable}, fh)

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "QLearningAgent":
        """Restore a checkpoint saved by save().

        Schema-checked on purpose: a table trained with a different action
        count or state-bucketing would otherwise load "successfully" and
        silently corrupt learning -- the worst kind of bug to chase in a
        long run. Fail loudly at load time instead.
        """
        import json
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"checkpoint {path} is not readable JSON: {exc}") from exc
        for key in ("alpha", "gamma", "q_table"):
            if key not in data:
                raise ValueError(
                    f"checkpoint {path} missing required field '{key}'")
        q_table = data["q_table"]
        if not isinstance(q_table, dict):
            raise ValueError(f"checkpoint {path}: q_table must be a mapping")
        rows = set(len(v) for v in q_table.values() if isinstance(v, list))
        if len(rows) > 1:
            raise ValueError(
                f"checkpoint {path}: ragged Q-table (row widths {sorted(rows)})"
                " -- likely written by an incompatible build")
        n_actions = rows.pop() if rows else 0
        agent = cls(n_actions=n_actions, learning_rate=data["alpha"],
                   discount=data["gamma"], seed=seed)
        # save() stringifies tuple keys as "a|b" (JSON has no tuples); restore
        # them so loaded states match the env's native tuple observations.
        # Components are int-cast where possible because EtchChamberEnv hands
        # us integer bucket tuples -- ("0","1") would never match (0,1).
        def _restore_key(k: str):
            parts = k.split("|")
            if len(parts) == 1:
                return k
            return tuple(int(p) if p.lstrip("-").isdigit() else p
                         for p in parts)
        agent.q_table = {_restore_key(k): list(v) for k, v in q_table.items()}
        return agent
