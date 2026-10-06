"""Tests for the tabular Q-Learning agent."""
import math

import pytest

from rl_lab.agents.qlearning_agent import QLearningAgent
from rl_lab.core.base import StepResult
from rl_lab.core.exploration import EpsilonGreedy


def make_step(obs, reward, terminated=False, truncated=False):
    return StepResult(observation=obs, reward=reward, terminated=terminated,
                      truncated=truncated, info={})


def test_rejects_bad_hyperparams():
    with pytest.raises(ValueError):
        QLearningAgent(n_actions=0)
    with pytest.raises(ValueError):
        QLearningAgent(n_actions=2, learning_rate=0)
    with pytest.raises(ValueError):
        QLearningAgent(n_actions=2, discount=1.5)


def test_q_update_one_step_hand_computed():
    a = QLearningAgent(n_actions=2, learning_rate=0.5, discount=0.9, seed=0)
    # s=(0,) take action 1, get r=1.0, land in s'=(1,) whose best Q is 0 (init).
    a.observe((0,), 1, make_step((1,), 1.0))
    assert a.q_table[(0,)][1] == pytest.approx(0.5 * (1.0 + 0.9 * 0.0 - 0.0))


def test_terminated_step_does_not_bootstrap():
    a = QLearningAgent(n_actions=2, learning_rate=1.0, discount=0.99, seed=0)
    a._values((9,))[0] = 500.0  # poisoned successor value (lazy rows!)
    a.observe((0,), 1, make_step((9,), 2.0, terminated=True))
    # alpha=1 -> Q becomes exactly r; if we had bootstrapped we'd see 2+0.99*500
    assert a.q_table[(0,)][1] == pytest.approx(2.0)


def test_truncated_step_bootstraps_successor_value():
    a = QLearningAgent(n_actions=2, learning_rate=1.0, discount=0.5, seed=0)
    a._values((9,))[0] = 4.0
    a.observe((0,), 1, make_step((9,), 2.0, truncated=True))
    # truncation != terminal: target = 2 + 0.5 * max(Q(s')) = 2 + 0.5*4 = 4
    assert a.q_table[(0,)][1] == pytest.approx(4.0)


def test_learned_policy_beats_random_on_two_state_chain():
    """Toy deterministic MDP: from state 0, action 1 leads to goal (+1);
    action 0 self-loops with 0 reward. A correct Q-learning agent must
    discover action 1 without being told."""
    a = QLearningAgent(n_actions=2, learning_rate=0.3, discount=0.95, seed=1)
    a.policy = EpsilonGreedy(epsilon_start=0.3, epsilon_end=0.05,
                             decay_episodes=100)
    for ep in range(1, 301):
        state = 0
        for _ in range(10):
            act = a.act(state, training=True)
            if act == 1:
                a.observe(state, act, make_step("goal", 1.0, terminated=True))
                break
            a.observe(state, act, make_step(0, 0.0))
        else:
            a.observe(state, 0, make_step(0, 0.0, truncated=True))
        a.on_episode_end(ep)
    assert a.best_action(0) == 1


def test_act_is_deterministic_in_eval_mode():
    a = QLearningAgent(n_actions=3, seed=5)
    a._values((0,)).__setitem__(slice(None), [0.1, 0.9, 0.2])
    picks = {a.act((0,), training=False) for _ in range(30)}
    assert picks == {1}


def test_save_writes_valid_json(tmp_path):
    import json
    a = QLearningAgent(n_actions=2, seed=0)
    a.observe((0, 1), 0, make_step((1, 1), 0.5))
    path = tmp_path / "q.json"
    a.save(str(path))
    data = json.loads(path.read_text())
    assert data["q_table"]["0|1"][0] > 0.0  # updated from 0.0 init
    assert data["alpha"] == a.alpha
