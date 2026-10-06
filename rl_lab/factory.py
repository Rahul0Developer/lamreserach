"""Factory: turn config strings ('etch_chamber', 'gym:CartPole-v1') into objects.

Why a registry dict instead of if/elif chains: adding a new env or agent is
one @register line in its own module -- the factory file never changes
(open/closed). This is also how we'll wire DQN in later without touching
the Trainer.
"""
from __future__ import annotations

import random
from typing import Callable

from rl_lab.core.base import BaseAgent, BaseEnvironment
from rl_lab.envs.etch_chamber import EtchChamberEnv

_ENV_REGISTRY: dict[str, Callable[..., BaseEnvironment]] = {}
_AGENT_REGISTRY: dict[str, Callable[..., BaseAgent]] = {}


def register_env(name: str):
    def deco(cls):
        _ENV_REGISTRY[name] = cls
        return cls
    return deco


def register_agent(name: str):
    def deco(cls):
        _AGENT_REGISTRY[name] = cls
        return cls
    return deco


# -- uniform construction contract -------------------------------------------
# Every agent factory receives (env, config, seed). Concrete agents take what
# they need; this adapter keeps the call signature identical for all of them.

from rl_lab.agents.random_agent import RandomAgent
from rl_lab.agents.qlearning_agent import QLearningAgent
from rl_lab.core.exploration import EpsilonGreedy, Softmax


class _RandomFactory(RandomAgent):
    """Adapter ctor: RandomAgent only needs n_actions."""

    def __init__(self, env: BaseEnvironment, config, seed: int | None = None):
        super().__init__(action_space_size=env.action_space_size(), seed=seed)


class _QLearningFactory(QLearningAgent):
    """Wires config -> agent, including the exploration policy (Strategy
    inside a Strategy): extra.policy picks epsilon-greedy or softmax."""

    def __init__(self, env: BaseEnvironment, config, seed: int | None = None):
        extra = getattr(config, "extra", {}) or {}
        eps_kwargs = dict(epsilon_start=config.epsilon_start,
                          epsilon_end=config.epsilon_end,
                          decay_episodes=config.epsilon_decay_episodes)
        policy_name = extra.get("policy", "epsilon_greedy")
        if policy_name == "epsilon_greedy":
            policy = EpsilonGreedy(**eps_kwargs)
        elif policy_name == "softmax":
            # Temperature decays like epsilon would; reuse the schedule knobs.
            policy = Softmax(temperature=max(0.05, config.epsilon_start))
        else:
            raise KeyError(f"unknown exploration policy '{policy_name}'. "
                           "Valid: epsilon_greedy, softmax")
        super().__init__(n_actions=env.action_space_size(),
                         learning_rate=config.learning_rate,
                         discount=config.discount, seed=seed, policy=policy)


# Built-ins. Import side effects register future modules (see __init__.py).
register_env("etch_chamber")(EtchChamberEnv)
register_agent("random")(_RandomFactory)
register_agent("qlearning")(_QLearningFactory)


# Keys in config.extra that belong to agents/envs respectively. The YAML
# "extra:" block is one flat namespace for user convenience; the framework
# must not forward agent knobs (policy, temperature...) into env ctors --
# that produced a real TypeError we just caught in integration tests.
_ENV_EXTRA_KEYS = {"target_etch_rate", "process_time_s", "sensor_noise"}


def create_env(spec: str, seed: int | None = None,
               extra: dict | None = None) -> BaseEnvironment:
    extra = {k: v for k, v in (extra or {}).items() if k in _ENV_EXTRA_KEYS}
    if spec.startswith("gym:"):
        # Local import: gymnasium is an optional dependency (~30 MB); the
        # core framework must work without it. Adapter pattern in envs/gym_wrapper.
        from rl_lab.envs.gym_wrapper import GymEnvWrapper
        return GymEnvWrapper(spec[4:], seed=seed)
    if spec not in _ENV_REGISTRY:
        raise KeyError(f"unknown env '{spec}'. Registered: {sorted(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[spec](seed=seed, **extra)


def create_agent(spec: str, env: BaseEnvironment, config,
                 seed: int | None = None) -> BaseAgent:
    if spec not in _AGENT_REGISTRY:
        raise KeyError(f"unknown agent '{spec}'. Registered: {sorted(_AGENT_REGISTRY)}")
    rng_seed = seed if seed is not None else random.randrange(2**31)
    return _AGENT_REGISTRY[spec](env=env, config=config, seed=rng_seed)
