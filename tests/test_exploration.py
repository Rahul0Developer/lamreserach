"""Tests for exploration policies (Strategy axis inside agents)."""
import random

import pytest

from rl_lab.core.exploration import EpsilonGreedy, Softmax


def test_epsilon_starts_high_and_decays_linearly():
    p = EpsilonGreedy(epsilon_start=1.0, epsilon_end=0.0, decay_episodes=10)
    assert p.epsilon == 1.0
    p.on_episode_end(5)
    assert p.epsilon == pytest.approx(0.5)
    p.on_episode_end(10)
    assert p.epsilon == pytest.approx(0.0)
    # Past the decay horizon it must clamp, not go negative.
    p.on_episode_end(50)
    assert p.epsilon == pytest.approx(0.0)


def test_invalid_schedule_rejected():
    with pytest.raises(ValueError):
        EpsilonGreedy(epsilon_start=0.2, epsilon_end=0.8)  # ramp-up, not decay
    with pytest.raises(ValueError):
        EpsilonGreedy(decay_episodes=0)


def test_greedy_at_zero_epsilon_picks_argmax_first_tie():
    p = EpsilonGreedy(epsilon_start=0.0, epsilon_end=0.0)
    rng = random.Random(0)
    assert p.select([1.0, 5.0, 5.0, 2.0], rng, training=True) == 1


def test_full_exploration_visits_every_action():
    p = EpsilonGreedy(epsilon_start=1.0, epsilon_end=1.0)
    rng = random.Random(7)
    seen = {p.select([0.0, 0.0, 0.0], rng, training=True) for _ in range(200)}
    assert seen == {0, 1, 2}


def test_eval_mode_is_always_greedy_even_with_high_epsilon():
    # This is what stops "my evaluated policy is random" benchmark bugs.
    p = EpsilonGreedy(epsilon_start=1.0, epsilon_end=1.0)
    rng = random.Random(1)
    for _ in range(50):
        assert p.select([0.0, 3.0, 1.0], rng, training=False) == 1


def test_softmax_prefers_best_but_samples_others():
    p = Softmax(temperature=0.5)
    rng = random.Random(3)
    counts = [0, 0, 0]
    for _ in range(600):
        counts[p.select([0.0, 1.0, 0.2], rng, training=True)] += 1
    assert counts[1] > counts[0] and counts[1] > counts[2]
    assert counts[0] > 0  # still explores clearly-worse arms sometimes


def test_softmax_eval_is_greedy():
    p = Softmax()
    rng = random.Random(4)
    assert all(p.select([-1.0, 0.5, 0.1], rng, training=False) == 1
               for _ in range(20))


def test_softmax_rejects_bad_temperature():
    with pytest.raises(ValueError):
        Softmax(temperature=0.0)
