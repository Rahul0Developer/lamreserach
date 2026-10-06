"""ConsoleLogger: human-readable progress while a run is in flight.

Observer pattern concrete subscriber #1. Kept dependency-free (no tqdm) so
it works in CI logs and notebooks alike.
"""
from __future__ import annotations

from rl_lab.core.base import BaseLogger


class ConsoleLogger(BaseLogger):
    def __init__(self, every: int = 10, quiet: bool = False) -> None:
        self.every = max(1, every)
        self.quiet = quiet

    def on_train_start(self, config: dict) -> None:
        if not self.quiet:
            name = config.get("name", "run")
            print(f"[rl_lab] starting '{name}' "
                  f"(env={config.get('env')} agent={config.get('agent')})")

    def on_episode_end(self, episode: int, total_reward: float, steps: int,
                       extra: dict) -> None:
        if not self.quiet and episode % self.every == 0:
            print(f"[rl_lab] ep {episode:>4}  reward {total_reward:>8.2f}  "
                  f"steps {steps}")

    def on_train_end(self, summary: dict) -> None:
        if not self.quiet:
            print(f"[rl_lab] done: mean_reward={summary.get('mean_reward', 0):.3f} "
                  f"std={summary.get('std_reward', 0):.3f} "
                  f"last10={summary.get('mean_last10', 0):.3f}")
