"""CSVLogger: one row per episode, flush-per-row.

Why not buffer aggressively: training runs get Ctrl-C'd and killed on
clusters; a half-written CSV you can still read beats a perfect one that
does not exist. Buffering is the classic "data loss on crash" bug class.
"""
from __future__ import annotations

import csv
import os
import threading

from rl_lab.core.base import BaseLogger


class CSVLogger(BaseLogger):
    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = None
        self._writer = None
        # Phase 5 made loggers shared across threads (one logger instance
        # can be attached to several controllers). Lock per-logger, not
        # per-file-handle: the lock must guard lazy open() too.
        self._lock = threading.Lock()

    def _ensure_open(self) -> None:
        if self._writer is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            # newline="" is required by the csv module docs; without it
            # Windows writes \r\r\n and Excel chokes (real cross-platform bug).
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.writer(self._fh)
            self._writer.writerow(["episode", "total_reward", "steps",
                                   "truncated"])

    def on_train_start(self, config: dict) -> None:
        with self._lock:
            self._ensure_open()

    def on_episode_end(self, episode: int, total_reward: float, steps: int,
                       extra: dict) -> None:
        with self._lock:
            self._ensure_open()  # tolerate trainers that skip on_train_start
            self._writer.writerow([episode, f"{total_reward:.4f}", steps,
                                   int(bool(extra.get("truncated")))])
            self._fh.flush()

    def on_train_end(self, summary: dict) -> None:
        self.close()

    def close(self) -> None:
        """Idempotent; call from finally-blocks so interrupted runs flush."""
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
                self._writer = None
