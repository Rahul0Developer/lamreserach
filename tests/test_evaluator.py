"""Tests for Evaluator: frozen-policy statistics."""
import pytest

from rl_lab.agents.qlearning_agent import QLearningAgent
from rl_lab.core.base import StepResult
from rl_lab.evaluation.evaluator import Evaluator


class FixedRewardEnv:
    """Deterministic env: every episode scores exactly -n steps of -0.1."""
    name = "fixed"

    def __init__(self, episode_len=5):
        self.episode_len = episode_len

    def reset(self, *, seed=None):
        self._t = 0
        return (0, {})

    def step(self, action):
        self._t += 1
        done = self._t >= self.episode_len
        return StepResult(observation=0, reward=-0.1, terminated=done,
                          truncated=False, info={})

    def action_space_size(self):
        return 2

    def observation_space_size(self):
        return 1


class RecordingAgent(QLearningAgent):
    """Proves evaluation does not learn: counts observe() calls."""
    def __init__(self):
        super().__init__(n_actions=2, seed=0)
        self.observe_calls = 0
        self.train_flags = []

    def observe(self, observation, action, result):
        self.observe_calls += 1

    def act(self, observation, training=True):
        self.train_flags.append(training)
        return 0


def test_evaluator_reports_exact_stats_on_deterministic_env():
    agent = RecordingAgent()
    ev = Evaluator(FixedRewardEnv(episode_len=5), max_steps=10,
                   success_threshold=-0.6)
    report = ev.evaluate(agent, episodes=4)
    assert report.mean_reward == pytest.approx(-0.5)
    assert report.std_reward == pytest.approx(0.0)
    assert report.min_reward == pytest.approx(-0.5)
    assert report.max_reward == pytest.approx(-0.5)
    assert report.success_rate == 1.0          # -0.5 >= -0.6
    assert report.mean_steps == pytest.approx(5)


def test_success_threshold_respected():
    ev = Evaluator(FixedRewardEnv(episode_len=5), max_steps=10,
                   success_threshold=0.0)      # nothing reaches 0.0
    report = ev.evaluate(RecordingAgent(), episodes=3)
    assert report.success_rate == 0.0


def test_evaluation_does_not_learn_and_asks_for_greedy_actions():
    agent = RecordingAgent()
    ev = Evaluator(FixedRewardEnv(), max_steps=10)
    ev.evaluate(agent, episodes=2)
    assert agent.observe_calls == 0            # frozen policy
    assert set(agent.train_flags) == {False}   # exploration off


def test_rejects_nonsense_episode_count():
    with pytest.raises(ValueError):
        Evaluator(FixedRewardEnv()).evaluate(RecordingAgent(), episodes=0)


def test_max_steps_bounds_long_episodes():
    ev = Evaluator(FixedRewardEnv(episode_len=1000), max_steps=7)
    report = ev.evaluate(RecordingAgent(), episodes=1)
    assert report.mean_steps == 7              # cut off, no hang
