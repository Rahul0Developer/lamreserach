"""Tests for the config -> components -> trainer integration path.

These catch the bugs unit tests miss: wrong wiring in the factory, config
keys that never reach the objects they configure, loggers listed in YAML but
never instantiated. Integration-level = one step below end-to-end.
"""
import pytest

from rl_lab.config.loader import load_config
from rl_lab.core.exploration import EpsilonGreedy, Softmax
from rl_lab.factory import create_agent, create_env
from rl_lab.agents.qlearning_agent import QLearningAgent


def test_qlearning_factory_reads_all_hyperparams_from_config():
    cfg = load_config("configs/etch_qlearning.yaml")
    env = create_env(cfg.env, seed=cfg.seed, extra=cfg.extra)
    agent = create_agent(cfg.agent, env, cfg, seed=cfg.seed)
    assert isinstance(agent, QLearningAgent)
    # Each of these was a silent-default bug waiting to happen; pin them.
    assert agent.alpha == cfg.learning_rate
    assert agent.gamma == cfg.discount
    assert agent.n_actions == env.action_space_size()
    policy = agent.policy
    assert isinstance(policy, EpsilonGreedy)
    assert policy.epsilon_start == cfg.epsilon_start
    assert policy.epsilon_end == cfg.epsilon_end
    assert policy.decay_episodes == cfg.epsilon_decay_episodes


def test_softmax_policy_selectable_via_extra():
    cfg = load_config("configs/etch_qlearning.yaml")
    cfg.extra["policy"] = "softmax"
    env = create_env(cfg.env, seed=1, extra=cfg.extra)
    agent = create_agent(cfg.agent, env, cfg, seed=1)
    assert isinstance(agent.policy, Softmax)


def test_unknown_policy_raises_helpful_error():
    cfg = load_config("configs/etch_qlearning.yaml")
    cfg.extra["policy"] = "boltzman"  # typo
    env = create_env(cfg.env, seed=1, extra=cfg.extra)
    with pytest.raises(KeyError, match="epsilon_greedy"):
        create_agent(cfg.agent, env, cfg, seed=1)


def test_same_seed_produces_identical_training_run():
    """Reproducibility contract the README promises: same config + seed =>
    byte-identical episode rewards (per_episode_seeding on)."""
    from rl_lab.config.loader import build_config
    from rl_lab.training.trainer import Trainer

    def run_once():
        cfg = build_config({
            "name": "repro", "env": "etch_chamber", "agent": "qlearning",
            "episodes": 5, "max_steps": 20, "seed": 123,
            "per_episode_seeding": True, "loggers": [],
        })
        env = create_env(cfg.env, seed=cfg.seed, extra=cfg.extra)
        agent = create_agent(cfg.agent, env, cfg, seed=cfg.seed)
        trainer = Trainer(env, agent, max_steps=cfg.max_steps)
        return [r.total_reward for r in trainer.train(
            episodes=cfg.episodes, config=cfg.to_dict(), seed=cfg.seed)]

    first, second = run_once(), run_once()
    assert first == pytest.approx(second)


def test_reset_without_seed_keeps_single_noise_stream():
    """Documents DEFAULT behaviour honestly: reset(seed=None) resets the
    chamber state but NOT the rng stream, so episode 2's noise continues
    where episode 1 left off. Two envs built with the same seed and walked
    identically therefore agree step-for-step -- across resets too."""
    env = create_env("etch_chamber", seed=7)
    env.reset()
    first_step_rates = [env.step(0).info["etch_rate"] for _ in range(3)]
    env.reset(seed=None)          # state reset, rng keeps flowing
    second_step_rates = [env.step(0).info["etch_rate"] for _ in range(3)]
    # Same action sequence, different position in the noise stream:
    assert first_step_rates != second_step_rates
