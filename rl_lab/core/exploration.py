"""Exploration policies: a second Strategy-pattern axis inside agents.

Why split this out of the agent: "how do I pick an action" (exploration) and
"how do I update my value estimates" (learning) change independently. A DQN
later reuses EpsilonGreedy with a network instead of a table; an upper-UCB
agent swaps only the policy. Testing each in isolation is also trivial.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod


class ExplorationPolicy(ABC):
    """Chooses between exploit (greedy over Q values) and explore."""

    @abstractmethod
    def select(self, q_values: list[float], rng: random.Random,
               training: bool) -> int:
        """Return the action index to take."""

    def on_episode_end(self, episode: int) -> None:
        """Hook for schedules that depend on episode count. Default no-op."""


class EpsilonGreedy(ExplorationPolicy):
    """With prob epsilon sample uniformly, else argmax of q_values.

    Linear decay from epsilon_start to epsilon_end over decay_episodes:
    simple, predictable, and debuggable -- you can compute the exact epsilon
    at any episode when a run behaves strangely. (Exponential decay is the
    common alternative; TODO: expose schedule as config if we need it.)
    """

    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 decay_episodes: int = 100) -> None:
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("need 0 <= epsilon_end <= epsilon_start <= 1")
        if decay_episodes < 1:
            raise ValueError("decay_episodes must be >= 1")
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_episodes = decay_episodes
        self._episode = 0

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._episode / self.decay_episodes)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select(self, q_values: list[float], rng: random.Random,
               training: bool) -> int:
        if training and rng.random() < self.epsilon:
            return rng.randrange(len(q_values))
        # Deterministic tie-break: first max wins (reproducibility matters).
        best = max(range(len(q_values)), key=lambda i: q_values[i])
        return best

    def on_episode_end(self, episode: int) -> None:
        self._episode = episode


class Softmax(ExplorationPolicy):
    """Boltzmann exploration: sample actions proportional to exp(Q/T).

    Useful where epsilon-greedy wastes trials on clearly-bad arms. Kept tiny
    and dependency-free so tabular tests don't need numpy here.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def select(self, q_values: list[float], rng: random.Random,
               training: bool) -> int:
        if not training:
            return max(range(len(q_values)), key=lambda i: q_values[i])
        t = max(q_values)
        weights = [pow(2.718281828, (q - t) / self.temperature) for q in q_values]
        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if pick <= acc:
                return i
        return len(q_values) - 1  # float-drift fallback, never negative idx
