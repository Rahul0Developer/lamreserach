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


def test_load_roundtrip_restores_behaviour(tmp_path):
    """save() -> load() must reproduce identical greedy actions (checkpoint
    correctness, not just valid JSON)."""
    a = QLearningAgent(n_actions=3, seed=1)
    for s in [(0, 0), (1, 2), (2, 1)]:
        a.q_table[s] = [float(s[0]), float(s[1]), 0.5]
    path = tmp_path / "ckpt.json"
    a.save(str(path))
    b = QLearningAgent.load(str(path), seed=1)
    assert b.n_actions == 3
    # tuple keys must round-trip back to tuples, not stay as "a|b" strings
    assert set(b.q_table) == set(a.q_table)
    for s in a.q_table:
        assert b.best_action(s) == a.best_action(s)


def test_load_rejects_missing_fields(tmp_path):
    import json
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"alpha": 0.2}))   # no q_table/gamma
    with pytest.raises(ValueError, match="missing required field"):
        QLearningAgent.load(str(path))


def test_load_rejects_ragged_table(tmp_path):
    import json
    path = tmp_path / "ragged.json"
    path.write_text(json.dumps({"alpha": 0.2, "gamma": 0.9,
                                "q_table": {"a|b": [0.1, 0.2], "c|d": [0.3]}}))
    with pytest.raises(ValueError, match="ragged"):
        QLearningAgent.load(str(path))


def test_load_rejects_non_json(tmp_path):
    path = tmp_path / "junk.json"
    path.write_text("not json at all")
    with pytest.raises(ValueError, match="not readable JSON"):
        QLearningAgent.load(str(path))
