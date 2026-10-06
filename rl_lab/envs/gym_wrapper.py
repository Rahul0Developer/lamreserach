"""GymEnvWrapper: Adapter between Gymnasium's API and BaseEnvironment.

Why wrap instead of subclassing gym.Env: our whole framework (Trainer,
Evaluator, loggers) speaks BaseEnvironment/StepResult. The adapter is the
single translation point -- if Gymnasium changes its step() tuple shape
again, only this file breaks. That is the isolation an adapter buys you.
"""
from __future__ import annotations

import numpy as np

from rl_lab.core.base import BaseEnvironment, StepResult


class GymEnvWrapper(BaseEnvironment):
    def __init__(self, spec: str, seed: int | None = None) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "gymnasium is required for 'gym:' envs. "
                "pip install gymnasium") from exc
        self._env = gym.make(spec)
        self.name = f"gym:{spec}"
        self._seed = seed

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(seed=seed if seed is not None else self._seed)
        return obs, info

    def step(self, action: int) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        return StepResult(observation=obs, reward=float(reward),
                          terminated=bool(terminated), truncated=bool(truncated),
                          info=dict(info))

    def action_space_size(self) -> int:
        return int(self._env.action_space.n)

    def observation_space_size(self) -> int:
        # Only meaningful for discrete spaces; tabular agents should not be
        # pointed at continuous gym envs (guard raises, not silently lies).
        space = self._env.observation_space
        if hasattr(space, "n"):
            return int(space.n)
        raise TypeError(f"{self.name} has a continuous observation space; "
                        "use a function-approximation agent (DQN) instead")
