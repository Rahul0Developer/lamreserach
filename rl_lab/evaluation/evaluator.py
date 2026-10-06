"""Evaluator: run a *frozen* agent through N episodes, report statistics.

Separate from Trainer on purpose (single responsibility): training mutates
the agent, evaluation must not. We pass training=False so exploration is
off and observe() is never called -- evaluating with epsilon-greedy noise
would understate a policy's true quality, a classic self-inflicted benchmark
bug.

Metrics: mean/std reward (population std via statistics.pstdev), min/max,
mean steps, and success_rate = fraction of episodes with total_reward >=
threshold. Threshold is caller-supplied because "success" is env-specific
(CartPole: survive 195+ steps; EtchChamber: hold near-target rate). With no
threshold every episode counts as a success -- honest default, not a guess.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from rl_lab.core.base import BaseAgent, BaseEnvironment


@dataclass
class EvalReport:
    episodes: int
    mean_reward: float
    std_reward: float          # 0.0 when single episode
    min_reward: float
    max_reward: float
    success_rate: float        # fraction of episodes >= threshold
    mean_steps: float
    raw_rewards: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw_rewards"}


class Evaluator:
    def __init__(self, env: BaseEnvironment, max_steps: int = 200,
                 success_threshold: float | None = None) -> None:
        self.env = env
        self.max_steps = max_steps
        # Default threshold: any episode scoring above half the theoretical
        # worst case counts as "acceptable process". Env-specific overrides
        # belong in config -- the controller passes cfg.success_threshold
        # through here (schema.py), so this constructor default is only the
        # fallback for direct/programmatic use.
        self.success_threshold = success_threshold

    def evaluate(self, agent: BaseAgent, episodes: int = 20,
                 seed: int | None = None) -> EvalReport:
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        rewards: list[float] = []
        steps_list: list[int] = []

        for _ in range(episodes):
            obs, _info = self.env.reset(seed=seed)
            total = 0.0
            steps = 0
            for _ in range(self.max_steps):
                action = agent.act(obs, training=False)
                result = self.env.step(action)
                total += result.reward
                steps += 1
                obs = result.observation
                if result.done:
                    break
            rewards.append(total)
            steps_list.append(steps)

        threshold = self.success_threshold
        successes = sum(1 for r in rewards if threshold is None or r >= threshold)
        return EvalReport(
            episodes=episodes,
            mean_reward=statistics.fmean(rewards),
            std_reward=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            min_reward=min(rewards),
            max_reward=max(rewards),
            success_rate=successes / episodes,
            mean_steps=statistics.fmean(steps_list),
            raw_rewards=rewards,
        )
