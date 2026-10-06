"""Trainer: the episode loop. Knows interfaces only, never concrete classes.

Strategy pattern in action: agent and env are injected as BaseAgent /
BaseEnvironment, so this file will not change when Q-learning or DQN land.
Logger list is the Observer: we broadcast events, loggers decide what to do.

Episode semantics (deliberate, matches Gymnasium):
  - terminated  = the MDP ended by its own rules (agent won/lost)
  - truncated   = we cut it off at max_steps; NOT a real terminal state.
    Agents must bootstrap V(s') on truncation instead of treating r as final,
    otherwise value estimates are biased low ("optimistic bias" inverted).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rl_lab.core.base import BaseAgent, BaseEnvironment, BaseLogger


@dataclass
class EpisodeRecord:
    episode: int
    total_reward: float
    steps: int
    truncated: bool
    extras: dict = field(default_factory=dict)


class Trainer:
    def __init__(self, env: BaseEnvironment, agent: BaseAgent,
                 loggers: list[BaseLogger] | None = None,
                 max_steps: int = 200) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.env = env
        self.agent = agent
        # Defensive copy: callers mutating their list afterwards must not
        # change a running broadcast (Observer detach-mid-notification bug).
        self.loggers: list[BaseLogger] = list(loggers or [])
        self.max_steps = max_steps

    # -- observer helpers: one place that touches every logger ---------------

    def _broadcast(self, event: str, *args, **kwargs) -> None:
        for logger in self.loggers:
            getattr(logger, event)(*args, **kwargs)

    def add_logger(self, logger: BaseLogger) -> None:
        """Subscribe after construction (Observer attach at runtime)."""
        self.loggers.append(logger)

    def train(self, episodes: int, config: dict | None = None,
              seed: int | None = None) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        self._broadcast("on_train_start", config or {})

        per_ep = bool((config or {}).get("per_episode_seeding", False))
        for ep in range(1, episodes + 1):
            # Default: seed only flows when per_episode_seeding is on; envs
            # keep one noise stream across episodes otherwise (standard RL).
            ep_seed = seed * 1000 + ep if (seed is not None and per_ep) else None
            obs, _info = self.env.reset(seed=ep_seed)
            total_reward = 0.0
            steps = 0
            truncated = False

            for _ in range(self.max_steps):
                action = self.agent.act(obs, training=True)
                result = self.env.step(action)
                # Agent learns from the transition BEFORE we rebind obs.
                self.agent.observe(obs, action, result)
                total_reward += result.reward
                steps += 1
                obs = result.observation
                if result.done:
                    truncated = result.truncated
                    break

            self.agent.on_episode_end(ep)
            record = EpisodeRecord(episode=ep, total_reward=total_reward,
                                   steps=steps, truncated=truncated)
            records.append(record)
            self._broadcast("on_episode_end", ep, total_reward, steps,
                            {"truncated": truncated})

        try:
            summary = self._summarise(records)
            self._broadcast("on_train_end", summary)
        finally:
            # Template-method cleanup: loggers with a close() (CSV file
            # handles) get it even if an episode raised mid-run.
            for logger in self.loggers:
                closer = getattr(logger, "close", None)
                if callable(closer):
                    closer()
        return records

    @staticmethod
    def _summarise(records: list[EpisodeRecord]) -> dict:
        if not records:
            return {"episodes": 0}
        rewards = [r.total_reward for r in records]
        n = len(rewards)
        mean = sum(rewards) / n
        var = sum((x - mean) ** 2 for x in rewards) / n
        last10 = rewards[-10:]
        return {
            "episodes": n,
            "mean_reward": mean,
            "std_reward": var ** 0.5,
            "mean_last10": sum(last10) / len(last10),
            "mean_steps": sum(r.steps for r in records) / n,
        }
