"""Trainer loop correctness tests using only the RandomAgent baseline.

If these pass, the loop mechanics (episode counting, reward accumulation,
observer broadcast) are right -- any bad numbers later are the agent's
fault, not the harness'. Classic test-double strategy.
"""
import pytest

from rl_lab.agents.random_agent import RandomAgent
from rl_lab.core.base import BaseLogger
from rl_lab.envs.etch_chamber import EtchChamberEnv
from rl_lab.training.trainer import EpisodeRecord, Trainer


class RecordingLogger(BaseLogger):
    """Test observer: captures event order for assertions."""

    def __init__(self):
        self.events: list[tuple] = []

    def on_train_start(self, config):
        self.events.append(("start", config))

    def on_episode_end(self, episode, total_reward, steps, extra):
        self.events.append(("episode", episode, total_reward, steps))

    def on_train_end(self, summary):
        self.events.append(("end", summary))


def make_trainer(episodes_env_time=20.0, max_steps=25):
    env = EtchChamberEnv(process_time_s=episodes_env_time, seed=1)
    agent = RandomAgent(action_space_size=env.action_space_size(), seed=1)
    return Trainer(env, agent, max_steps=max_steps)


def test_train_returns_one_record_per_episode():
    records = make_trainer().train(episodes=5)
    assert len(records) == 5
    assert [r.episode for r in records] == [1, 2, 3, 4, 5]  # 1-based, no off-by-one


def test_steps_bounded_by_max_steps_and_process_time():
    # env truncates at t=20 before trainer's cap of 25
    records = make_trainer(episodes_env_time=20.0, max_steps=25).train(episodes=3)
    assert all(r.steps == 20 and r.truncated for r in records)
    # trainer cap binds when env never ends
    records = make_trainer(episodes_env_time=1e9, max_steps=7).train(episodes=2)
    assert all(r.steps == 7 for r in records)


def test_logger_receives_full_event_sequence():
    logger = RecordingLogger()
    trainer = make_trainer()
    trainer.add_logger(logger)
    trainer.train(episodes=3, config={"name": "t"})
    kinds = [e[0] for e in logger.events]
    assert kinds == ["start", "episode", "episode", "episode", "end"]
    assert logger.events[-1][1]["episodes"] == 3


def test_total_reward_matches_manual_rollout():
    """Same seeds, same action sequence -> identical totals (reproducibility)."""
    env = EtchChamberEnv(process_time_s=15.0, seed=9)
    agent = RandomAgent(env.action_space_size(), seed=3)
    trainer = Trainer(env, agent, max_steps=50)
    rec = trainer.train(episodes=1)[0]

    env2 = EtchChamberEnv(process_time_s=15.0, seed=9)
    agent2 = RandomAgent(env2.action_space_size(), seed=3)
    obs, _ = env2.reset()
    manual = 0.0
    for _ in range(50):
        a = agent2.act(obs)
        r = env2.step(a)
        manual += r.reward
        if r.done:
            break
    assert rec.total_reward == pytest.approx(manual)


def test_invalid_max_steps_rejected():
    env = EtchChamberEnv()
    with pytest.raises(ValueError):
        Trainer(env, RandomAgent(env.action_space_size()), max_steps=0)


def test_summary_stats_correct():
    trainer = make_trainer()
    records = [EpisodeRecord(i, float(i), 10, False) for i in range(1, 5)]
    s = trainer._summarise(records)
    assert s["mean_reward"] == pytest.approx(2.5)
    assert s["std_reward"] == pytest.approx(1.1180, abs=1e-3)
    assert s["mean_last10"] == pytest.approx(2.5)
