"""EtchChamberEnv contract + dynamics tests."""
import numpy as np
import pytest

from rl_lab.core.base import StepResult
from rl_lab.envs.etch_chamber import ACTIONS, EtchChamberEnv


def test_reset_returns_bucketed_observation():
    env = EtchChamberEnv(seed=1)
    obs, info = env.reset()
    assert isinstance(obs, tuple) and len(obs) == 3
    assert all(0 <= level < 3 for level in obs)
    assert info["target"] == 50.0


def test_step_returns_stepreresult_and_valid_reward_range():
    env = EtchChamberEnv(seed=2)
    env.reset()
    result = env.step(4)  # no-op action
    assert isinstance(result, StepResult)
    # normalised error penalty keeps per-step reward sane; effort term is
    # small; if this ever blows past bounds a scaling bug slipped in
    assert -2.0 < result.reward <= 0.0
    assert result.info["etch_rate"] >= 0.0


@pytest.mark.parametrize("bad_action", [-1, 9, "up", None])
def test_invalid_action_raises(bad_action):
    env = EtchChamberEnv(seed=3)
    env.reset()
    with pytest.raises(ValueError):
        env.step(bad_action)


def test_truncation_at_process_time():
    env = EtchChamberEnv(process_time_s=10.0, seed=4)
    env.reset()
    results = [env.step(4) for _ in range(12)]
    assert not any(r.truncated for r in results[:9])
    assert results[9].truncated          # exactly on tick 10
    assert results[9].terminated is False  # truncation != termination


def test_same_seed_reproduces_trajectory():
    def rollout(seed):
        env = EtchChamberEnv(seed=seed)
        env.reset()
        return [env.step(a % len(ACTIONS)).reward for a in range(30)]
    assert rollout(7) == rollout(7)
    # different seeds must diverge (noise actually applied)
    assert rollout(7) != rollout(8)


def test_dynamics_respond_to_flow_input():
    """Physics sanity: maxing gas flow should raise etch rate; zero flow kills it."""
    env = EtchChamberEnv(process_time_s=1e9, seed=10)
    env.reset()
    high = [env.step(5) for _ in range(60)]      # dp=0, dq=+1 repeated
    assert env.etch_rate > 20.0
    env.reset()
    for _ in range(60):
        env.step(3)                              # hold pressure, flow down
    assert env.etch_rate < 5.0                   # near-zero flow -> near-zero rate


def test_space_sizes_consistent_with_action_table():
    env = EtchChamberEnv()
    assert env.action_space_size() == len(ACTIONS)
    assert env.observation_space_size() == 27
