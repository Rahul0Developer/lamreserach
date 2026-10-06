"""Phase 6: report generator tests.

matplotlib is exercised for real (Agg backend, tiny figures) -- if the
backend ever stops being headless-safe these tests fail loudly in CI,
which is exactly what we want to know before a demo does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rl_lab.config.loader import build_config
from rl_lab.core.fsm import State
from rl_lab.evaluation.evaluator import EvalReport
from rl_lab.orchestration.controller import RunResult
from rl_lab.reporting.report_generator import ReportGenerator, _rolling_mean
from rl_lab.training.trainer import EpisodeRecord


def make_result(name="r0", seed=0, n_records=10, with_eval=True) -> RunResult:
    cfg = build_config({"name": name, "seed": seed, "env": "etch_chamber",
                        "agent": "random", "episodes": n_records,
                        "max_steps": 20, "eval_episodes": 4})
    records = [EpisodeRecord(episode=i, total_reward=-40.0 + 2.0 * i,
                             steps=20, truncated=False)
               for i in range(n_records)]
    er = EvalReport(episodes=4, mean_reward=-10.0, std_reward=1.5,
                    min_reward=-13.0, max_reward=-8.0, success_rate=0.5,
                    mean_steps=19.0, raw_rewards=[-13, -11, -9, -8]) \
        if with_eval else None
    return RunResult(config=cfg, state=State.DONE, records=records,
                     eval_report=er, wall_time_s=0.5)


class TestRollingMean:
    def test_window_one_is_identity(self):
        vals = [1.0, -2.0, 3.5]
        assert _rolling_mean(vals, 1) == vals

    def test_smoothing_reduces_variance(self):
        spiky = [0.0, 10.0] * 20
        smooth = _rolling_mean(spiky, 4)
        var_raw = max(spiky) - min(spiky)
        var_sm = max(smooth) - min(smooth)
        assert var_sm < var_raw

    def test_edge_first_element_uses_partial_window(self):
        # classic off-by-one trap: window=3 but only 1 sample available
        assert _rolling_mean([6.0, 4.0], 3)[0] == 6.0
        assert _rolling_mean([6.0, 4.0], 3)[1] == 5.0


class TestLearningCurve:
    def test_plot_written_and_nonempty(self, tmp_path):
        gen = ReportGenerator()
        png = gen.plot_learning_curve(make_result(), tmp_path / "c.png")
        assert png.exists() and png.stat().st_size > 1000

    def test_empty_records_raises_clearly(self, tmp_path):
        r = make_result()
        r.records = []
        with pytest.raises(ValueError, match="no episode"):
            ReportGenerator().plot_learning_curve(r, tmp_path / "c.png")


class TestRunReport:
    def test_markdown_contains_key_facts(self, tmp_path):
        path = ReportGenerator().write_run_report(make_result(),
                                                  tmp_path / "report.md")
        text = path.read_text()
        assert "# Experiment report: r0" in text
        assert "DONE" in text
        assert "-10.00" in text          # eval mean
        assert "learning_curve.png" in text
        assert (tmp_path / "learning_curve.png").exists()

    def test_failed_run_still_gets_a_report(self, tmp_path):
        r = make_result(with_eval=False)
        r.state = State.FAILED
        r.error = "ValueError: boom"
        path = ReportGenerator().write_run_report(r, tmp_path / "report.md")
        text = path.read_text()
        assert "FAILED" in text and "boom" in text

    def test_plot_failure_degrades_to_text_only(self, tmp_path, monkeypatch):
        """Report must survive a broken plotter (we ship reports even when
        matplotlib chokes on exotic data)."""
        gen = ReportGenerator()
        monkeypatch.setattr(gen, "plot_learning_curve",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("mpl down")))
        path = gen.write_run_report(make_result(), tmp_path / "report.md")
        assert "plot failed" in path.read_text()
        assert path.exists()


class TestComparisonAndSweep:
    def test_table_sorts_best_first_and_keeps_failures(self):
        good = make_result(name="good")
        bad = make_result(name="bad", with_eval=False)
        bad.state = State.FAILED
        table = ReportGenerator().comparison_table([bad, good])
        rows = [ln for ln in table.splitlines() if ln.startswith("|")]
        # header, separator, good, bad -- failures visible, not hidden
        assert "good" in rows[2] and "bad" in rows[3]
        assert "std_across_seeds" not in table   # single scored run: no spread

    def test_aggregate_line_when_multiple_scored(self):
        rs = [make_result(name=f"a{i}", with_eval=True) for i in range(3)]
        rs[1].eval_report.mean_reward = 5.0
        table = ReportGenerator().comparison_table(rs)
        assert "std_across_seeds" in table

    def test_sweep_report_full_artifact_set(self, tmp_path):
        results = [make_result(name=f"s{i}", seed=i, n_records=25)
                   for i in range(3)]
        agg = {"n": 3, "n_scored": 3, "mean_eval_reward": -10.0,
               "std_across_seeds": 0.7, "success_rate": 0.5, "failed": 0}
        path = ReportGenerator().write_sweep_report(
            results, tmp_path / "sweep.md", agg=agg)
        text = path.read_text()
        assert "Sweep report" in text
        assert "Aggregate" in text
        assert (tmp_path / "sweep_curves.png").exists()
