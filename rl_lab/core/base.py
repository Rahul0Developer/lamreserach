"""Abstract contracts shared by every component in rl_lab.

Why ABCs instead of duck typing: the Trainer only talks to these interfaces,
so a new agent or environment plugs in without touching training code
(dependency inversion). A hiring manager reading this repo should be able to
add an agent by implementing exactly three methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepResult:
    """Immutable result of one environment step.

    frozen=True so accidental mutation inside the trainer is caught at
    runtime rather than silently corrupting logged data.
    """
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class BaseEnvironment(ABC):
    """Gymnasium-compatible interface. GymEnvWrapper adapts real gym envs."""

    name: str = "unnamed-env"

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[Any, dict]:
        """Start a new episode, return (initial_observation, info)."""

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """Advance one tick and return a StepResult."""

    @abstractmethod
    def action_space_size(self) -> int:
        """Discrete view of the action space, used by tabular agents."""

    @abstractmethod
    def observation_space_size(self) -> int:
        """Discrete (bucketed) view of the observation space."""


class BaseAgent(ABC):
    """Strategy pattern: the agent IS the decision strategy the Trainer uses.

    Swapping RandomAgent for QLearningAgent changes behaviour with zero
    changes to the training loop -- that is the whole point of the pattern.
    """

    name: str = "unnamed-agent"

    @abstractmethod
    def act(self, observation: Any, training: bool = True) -> Any:
        """Pick an action for the given observation."""

    @abstractmethod
    def observe(self, observation: Any, action: Any, result: StepResult) -> None:
        """Learn from a transition the environment just produced."""

    def on_episode_end(self, episode: int) -> None:
        """Hook for end-of-episode bookkeeping (e.g. anneal exploration).

        Default no-op: template method so subclasses only override what
        they care about.
        """


class BaseLogger(ABC):
    """Observer pattern: loggers subscribe to training events.

    The Trainer holds a list of BaseLogger and broadcasts events to all of
    them. Adding CSV output later never modifies the Trainer.
    """

    @abstractmethod
    def on_train_start(self, config: dict) -> None: ...

    @abstractmethod
    def on_episode_end(self, episode: int, total_reward: float, steps: int,
                       extra: dict) -> None: ...

    @abstractmethod
    def on_train_end(self, summary: dict) -> None: ...
