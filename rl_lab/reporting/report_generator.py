"""ReportGenerator: markdown reports + reward-curve plots (Phase 6).

Why it consumes RunResult objects and NOT the CSV logs: the CSV is an
external, append-only audit trail (for pandas/dashboards); the report is a
derived view. Re-reading our own CSVs would couple the formatter to a file
format and lose structured fields (eval stats, FSM state, error strings)
that don't round-trip cleanly through text. Single source of truth stays
the in-memory result objects; serialization is one-way.

Design choices:
  - matplotlib with explicit Agg backend, created lazily inside methods.
    Import cost only paid when someone actually asks for a plot, and CI
    without a display still works. TODO: if plotting becomes optional at
    install time, guard with try/import and emit tables-only reports.
  - Rolling-mean smoothing instead of raw per-episode curves: raw reward
    is spiky noise that hides learning trends; window defaults to ~5% of
    episodes (min 1). We plot BOTH faintly so nobody has to trust the
    smoothed line alone.
  - Markdown output (not HTML): renders natively on GitHub, which is what
    a hiring manager reads first. HTML was considered and rejected --
    template engine dependency for no real gain here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless-safe before pyplot import
import matplotlib.pyplot as plt

from rl_lab.core.fsm import State
from rl_lab.orchestration.controller import RunResult


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Simple trailing moving average. O(n*window), fine for <=10k episodes.
    (A cumsum version exists; premature until profiles say otherwise.)"""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i + 1 - lo))
    return out


class ReportGenerator:
    def plot_learning_curve(self, result: RunResult, path: Path,
                            window: int | None = None) -> Path:
        """Training reward curve (raw + rolling mean) -> PNG. Returns path."""
        records = result.records
        if not records:
            raise ValueError(f"run '{result.config.name}' has no episode "
                             "records to plot (did training fail early?)")
        rewards = [r.total_reward for r in records]
        if window is None:
            window = max(1, len(rewards) // 20)   # ~5% smoothing window
        episodes = [r.episode for r in records]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
        ax.plot(episodes, rewards, color="steelblue", alpha=0.30,
                linewidth=1, label="raw reward")
        ax.plot(episodes, _rolling_mean(rewards, window), color="navy",
                linewidth=2, label=f"rolling mean (w={window})")
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        title = f"{result.config.name}  [{result.state.name}]"
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)         # figures leak memory if left open in loops
        return path

    def write_run_report(self, result: RunResult, md_path: Path) -> Path:
        """One-page markdown report per run: config, eval stats, curve."""
        cfg = result.config
        lines = [f"# Experiment report: {cfg.name}", ""]
        lines += ["## Configuration", "",
                  "| key | value |", "|---|---|",
                  f"| env | `{cfg.env}` |",
                  f"| agent | `{cfg.agent}` |",
                  f"| seed | {cfg.seed} |",
                  f"| episodes | {cfg.episodes} |",
                  f"| max_steps | {cfg.max_steps} |",
                  f"| eval_episodes | {cfg.eval_episodes} |",
                  "", "## Result", "",
                  f"- final state: **{result.state.name}**",
                  f"- wall time: {result.wall_time_s:.1f}s"]
        if result.error:
            lines.append(f"- error: `{result.error}`")
        er = result.eval_report
        if er is not None:
            lines += [f"- eval mean reward: **{er.mean_reward:.2f}** "
                      f"(std {er.std_reward:.2f})",
                      f"- eval min/max: {er.min_reward:.2f} / "
                      f"{er.max_reward:.2f}",
                      f"- success rate: {er.success_rate:.0%}"]
        if result.records:
            first = result.records[0].total_reward
            last = result.records[-1].total_reward
            trend = "improved" if last > first else "flat/declined"
            lines.append(f"- training reward {first:.1f} -> {last:.1f} "
                         f"({trend})")
        # Plot next to the md file so relative link works in GitHub viewers.
        png_name = "learning_curve.png"
        try:
            self.plot_learning_curve(result, md_path.parent / png_name)
            lines += ["", "## Learning curve", "",
                      f"![learning curve]({png_name})"]
        except Exception as exc:      # noqa: BLE001 - report must still land
            # A broken plot should never destroy the textual report; note
            # the failure inline so post-mortems see it happened.
            lines += ["", f"## Learning curve", "", f"(plot failed: {exc})"]
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines) + "\n")
        return md_path

    def comparison_table(self, results: list[RunResult]) -> str:
        """Markdown table across runs/seeds -- the 'is it robust?' view.

        Sorted by mean eval reward descending so the best config floats up.
        Runs without an eval report (FAILED early) are kept but show '-';
        hiding failures from the table would be dishonest reporting.
        """
        scored = [r for r in results if r.eval_report is not None]
        unscored = [r for r in results if r.eval_report is None]
        scored.sort(key=lambda r: r.eval_report.mean_reward, reverse=True)
        rows = ["| run | state | eval mean | std(within) | success | wall(s) |",
                "|---|---|---|---|---|---|"]
        for r in scored:
            er = r.eval_report
            rows.append(f"| {r.config.name} | {r.state.name} | "
                        f"{er.mean_reward:.2f} | {er.std_reward:.2f} | "
                        f"{er.success_rate:.0%} | {r.wall_time_s:.1f} |")
        for r in unscored:
            rows.append(f"| {r.config.name} | {r.state.name} | - | - | - | "
                        f"{r.wall_time_s:.1f} |")
        # Cross-seed spread: std ACROSS seeds answers reproducibility,
        # distinct from within-run std above.
        if len(scored) > 1:
            means = [r.eval_report.mean_reward for r in scored]
            m = sum(means) / len(means)
            var = sum((x - m) ** 2 for x in means) / (len(means) - 1)
            rows.append("")
            rows.append(f"Aggregate over {len(scored)} scored runs: "
                        f"mean={m:.2f}, std_across_seeds={var ** 0.5:.2f}")
        return "\n".join(rows)

    def write_sweep_report(self, results: list[RunResult],
                           md_path: Path, agg: dict | None = None) -> Path:
        """Multi-run sweep report: comparison table + overlaid curves."""
        lines = ["# Sweep report", "", self.comparison_table(results), ""]
        png_name = "sweep_curves.png"
        try:
            self._overlay_curves(results, md_path.parent / png_name)
            lines += [f"![sweep curves]({png_name})", ""]
        except Exception as exc:      # noqa: BLE001
            lines.append(f"(overlay plot failed: {exc})")
        if agg and agg.get("mean_eval_reward") is not None:
            lines += ["## Aggregate", "",
                      f"- scored runs: {agg['n_scored']}/{agg['n']}",
                      f"- mean eval reward: {agg['mean_eval_reward']:.2f}",
                      f"- std across seeds: {agg['std_across_seeds']:.2f}",
                      f"- mean success rate: {agg['success_rate']:.0%}",
                      f"- failed runs: {agg.get('failed', 0)}"]
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines) + "\n")
        return md_path

    def _overlay_curves(self, results: list[RunResult], path: Path) -> Path:
        """One smoothed curve per run, same axes -- visual seed-spread check."""
        usable = [r for r in results if r.records]
        if not usable:
            raise ValueError("no runs with training records to plot")
        fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
        for r in usable:
            rewards = [rec.total_reward for rec in r.records]
            w = max(1, len(rewards) // 20)
            ax.plot(range(len(rewards)), _rolling_mean(rewards, w),
                    linewidth=1.5, label=r.config.name)
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward (smoothed)")
        ax.set_title("Sweep: training curves per run")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
