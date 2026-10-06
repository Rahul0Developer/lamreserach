"""JSONLogger: whole-run record as one JSON document.

Trade-off vs CSVLogger: JSON gives us nested payloads (per-episode extras,
config snapshot) but must be rewritten or accumulated in memory. We keep
episodes in memory (fine for <= tens of thousands of rows) and write once
on train end -- plus a best-effort write on close so interrupted runs still
produce something readable.

TODO(phase5): with the multithreaded runner many jobs share runs/; make the
filename include job id + seed so results never overwrite each other.
"""
from __future__ import annotations

import json
import os
import threading

from rl_lab.core.base import BaseLogger


class JSONLogger(BaseLogger):
    def __init__(self, path: str) -> None:
        self.path = path
        self._config: dict = {}
        self._episodes: list[dict] = []
        self._summary: dict = {}
        # Phase 5: logger instances may be shared across worker threads.
        # list.append is atomic under CPython but _write() serializes the
        # episodes list mid-append from another thread -> RuntimeError.
        # The lock makes accumulate+dump one critical section.
        self._lock = threading.Lock()

    def on_train_start(self, config: dict) -> None:
        with self._lock:
            self._config = dict(config)
            self._episodes = []

    def on_episode_end(self, episode: int, total_reward: float, steps: int,
                       extra: dict) -> None:
        with self._lock:
            self._episodes.append({"episode": episode,
                                   "total_reward": round(total_reward, 4),
                                   "steps": steps, **extra})

    def on_train_end(self, summary: dict) -> None:
        with self._lock:
            self._summary = dict(summary)
            self._write()

    def _write(self) -> None:
        # NOTE: caller holds self._lock (or is close()); no re-lock here.
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump({"config": self._config,
                       "summary": self._summary,
                       "episodes": self._episodes}, fh, indent=2)

    @staticmethod
    def load(path: str) -> dict:
        """Read back a run for the Evaluator/report side."""
        with open(path) as fh:
            return json.load(fh)
