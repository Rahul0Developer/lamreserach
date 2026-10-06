"""Typed experiment configuration.

Why dataclasses instead of raw dicts: a typo like "learnig_rate" should fail
loudly at load time with a pointer to the offending key, not silently fall
back to a default halfway through a 20-minute training run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

# Explicit defaults per field so error messages can name the fallback used.
DEFAULTS = {
    "name": "unnamed-experiment",
    "seed": 42,
    "env": None,          # required
    "agent": None,        # required
    "episodes": 100,
    "max_steps": 200,
    "learning_rate": 0.1,
    "discount": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 50,
    "loggers": ["console"],
    "output_dir": "runs",
    # Deterministic env noise per episode index. Off by default: standard RL
    # practice varies the noise stream between episodes; on = full run-for-run
    # reproducibility of every single transition (good for regression tests).
    "per_episode_seeding": False,
    # Evaluation phase knobs (used by the experiment FSM, not the Trainer).
    "eval_episodes": 20,
    # "success" is env-specific; None keeps Evaluator's honest default.
    "success_threshold": None,
}


class ConfigError(Exception):
    """Raised for any invalid/missing configuration. Message tells the user
    exactly which key and what was expected."""


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    env: str
    agent: str
    episodes: int
    max_steps: int
    learning_rate: float
    discount: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_episodes: int
    loggers: list = field(default_factory=lambda: ["console"])
    output_dir: str = "runs"
    per_episode_seeding: bool = False
    eval_episodes: int = 20
    success_threshold: float | None = None
    # Free-form extras (e.g. DQN hidden_size) pass through untouched.
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Range checks that physics/RL actually care about -- an out-of-range
        # epsilon or negative lr produces "works but learns nothing" runs
        # that are miserable to debug later.
        if not 0.0 <= self.epsilon_end <= 1.0:
            raise ConfigError(f"epsilon_end={self.epsilon_end} must be in [0, 1]")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ConfigError(f"epsilon_start={self.epsilon_start} must be in [0, 1]")
        if self.epsilon_start < self.epsilon_end:
            raise ConfigError("epsilon_start must be >= epsilon_end (decay only)")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate={self.learning_rate} must be > 0")
        if not 0.0 <= self.discount <= 1.0:
            raise ConfigError(f"discount={self.discount} must be in [0, 1]")
        if self.episodes < 1:
            raise ConfigError(f"episodes={self.episodes} must be >= 1")
        if self.max_steps < 1:
            raise ConfigError(f"max_steps={self.max_steps} must be >= 1")
        if self.eval_episodes < 1:
            raise ConfigError(f"eval_episodes={self.eval_episodes} must be >= 1")
        if not isinstance(self.seed, int):
            raise ConfigError(f"seed={self.seed!r} must be an integer")
        valid_loggers = {"console", "csv", "json"}
        bad = set(self.loggers) - valid_loggers
        if bad:
            raise ConfigError(
                f"loggers {sorted(bad)} unknown; valid: {sorted(valid_loggers)}")

    def to_dict(self) -> dict:
        """Plain dict for loggers/serialization (Observer payloads)."""
        d = {f.name: getattr(self, f.name) for f in fields(self)
             if f.name != "extra"}
        d.update(self.extra)
        return d
