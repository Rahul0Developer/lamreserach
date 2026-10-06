"""RandomAgent: the control baseline every RL result must beat.

Also the simplest possible implementation of the BaseAgent strategy, used in
tests to prove the Trainer loop is correct before any learning is involved.
"""
from __future__ import annotations

import random
from typing import Any

from rl_lab.core.base import BaseAgent, StepResult


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, action_space_size: int, seed: int | None = None) -> None:
        if action_space_size < 1:
            raise ValueError("action_space_size must be >= 1")
        self.n_actions = action_space_size
        self._rng = random.Random(seed)

    def act(self, observation: Any, training: bool = True) -> int:
        # Uniform over actions; `training` flag ignored (nothing to learn).
        return self._rng.randrange(self.n_actions)

    def observe(self, observation: Any, action: Any, result: StepResult) -> None:
        # No state to update -- interface compliance is intentional here so
        # the Trainer can treat all agents uniformly (Liskov substitution).
        pass
