"""Factory + adapter tests. Gymnasium-dependent test skips if not installed."""
import pytest

from rl_lab.agents.random_agent import RandomAgent
from rl_lab.envs.etch_chamber import EtchChamberEnv
from rl_lab.factory import create_agent, create_env


def test_create_env_from_spec():
    env = create_env("etch_chamber", seed=1)
    assert isinstance(env, EtchChamberEnv)


def test_unknown_env_lists_options_in_error():
    with pytest.raises(KeyError) as exc:
        create_env("nope")
    assert "etch_chamber" in str(exc.value)  # error tells you what IS valid


def test_unknown_agent_raises_keyerror():
    env = create_env("etch_chamber", seed=1)
    with pytest.raises(KeyError):
        create_agent("magic", env, config=None, seed=1)


def test_create_random_agent_wired_to_env_action_space():
    env = create_env("etch_chamber", seed=1)
    agent = create_agent("random", env, config=None, seed=5)
    assert isinstance(agent, RandomAgent)
    assert agent.n_actions == env.action_space_size()


def test_factory_seed_reproducibility():
    env = create_env("etch_chamber", seed=2)
    a1 = create_agent("random", env, config=None, seed=8)
    a2 = create_agent("random", env, config=None, seed=8)
    obs, _ = env.reset()
    assert [a1.act(obs) for _ in range(10)] == [a2.act(obs) for _ in range(10)]


def test_gym_wrapper_cartpole():
    pytest.importorskip("gymnasium")
    env = create_env("gym:CartPole-v1", seed=3)
    obs, info = env.reset()
    assert hasattr(obs, "__len__") and len(obs) == 4
    assert env.action_space_size() == 2
    result = env.step(1)
    assert result.reward == pytest.approx(1.0)  # CartPole gives +1 per step
    assert not result.terminated
    # continuous-space guard: Box obs space must refuse tabular sizing
    from rl_lab.core.base import BaseEnvironment
    import gymnasium as gym
    from rl_lab.envs.gym_wrapper import GymEnvWrapper
    w = GymEnvWrapper("Pendulum-v1", seed=1)
    with pytest.raises(TypeError):
        w.observation_space_size()
