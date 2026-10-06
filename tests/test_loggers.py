"""Tests for CSV/JSON loggers and the Observer wiring inside Trainer."""
import csv
import json

import pytest

from rl_lab.core.base import BaseLogger, StepResult
from rl_lab.logging.csv_logger import CSVLogger
from rl_lab.logging.json_logger import JSONLogger
from rl_lab.training.trainer import Trainer


class FakeEnv:
    """Two-episode, two-step env; enough to exercise logger plumbing."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    def reset(self, *, seed=None):
        self.calls = 0
        return ("s0", {})

    def step(self, action):
        self.calls += 1
        done = self.calls >= 2
        return StepResult(observation="s0", reward=-0.5 * self.calls,
                          terminated=done, truncated=False, info={})

    def action_space_size(self):
        return 2

    def observation_space_size(self):
        return 1


class FakeAgent:
    name = "fake"

    def act(self, obs, training=True):
        return 0

    def observe(self, obs, action, result):
        pass

    def on_episode_end(self, episode):
        pass


def test_csv_logger_writes_header_and_rows(tmp_path):
    path = tmp_path / "run.csv"
    logger = CSVLogger(str(path))
    trainer = Trainer(FakeEnv(), FakeAgent(), loggers=[logger], max_steps=5)
    trainer.train(episodes=3, config={"name": "t"})

    with open(path) as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["episode", "total_reward", "steps", "truncated"]
    assert len(rows) == 4  # header + 3 episodes
    assert rows[1][0] == "1"
    # each episode: -0.5 + -1.0 = -1.5
    assert float(rows[1][1]) == pytest.approx(-1.5)


def test_csv_logger_flushes_per_row(tmp_path):
    """Read the file mid-run: a crash must not lose committed episodes."""
    path = tmp_path / "partial.csv"
    seen_during_run = []

    class PeekLogger(BaseLogger):
        def on_train_start(self, config): pass
        def on_episode_end(self, ep, r, s, extra):
            with open(path) as fh:
                seen_during_run.append(len(fh.readlines()))
        def on_train_end(self, summary): pass

    logger = CSVLogger(str(path))
    trainer = Trainer(FakeEnv(), FakeAgent(), loggers=[logger, PeekLogger()],
                      max_steps=5)
    trainer.train(episodes=3)
    # After ep1 the peeker should already see header+1 row (flushed, not buffered).
    assert seen_during_run[0] == 2


def test_csv_close_is_idempotent(tmp_path):
    logger = CSVLogger(str(tmp_path / "x.csv"))
    logger.on_train_start({})
    logger.close()
    logger.close()  # second close must not raise


def test_json_logger_roundtrip(tmp_path):
    path = tmp_path / "run.json"
    logger = JSONLogger(str(path))
    trainer = Trainer(FakeEnv(), FakeAgent(), loggers=[logger], max_steps=5)
    trainer.train(episodes=2, config={"name": "jtest"})

    data = JSONLogger.load(str(path))
    assert data["config"]["name"] == "jtest"
    assert len(data["episodes"]) == 2
    assert data["episodes"][0]["total_reward"] == pytest.approx(-1.5)
    assert data["summary"]["episodes"] == 2


def test_trainer_copies_logger_list_at_construction():
    """Observer detach-mid-notification guard: mutating the caller's list
    after construction must not change broadcast behaviour."""
    logger = JSONLogger("/tmp/should_not_be_written_phase3.json")
    loggers = [logger]
    trainer = Trainer(FakeEnv(), FakeAgent(), loggers=loggers, max_steps=5)
    loggers.clear()  # hostile caller
    assert trainer.loggers == [logger]  # trainer kept its own copy
