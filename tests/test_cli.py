"""CLI tests: exit codes, overrides re-validation, subcommands.

We call main(argv) directly (no subprocess): faster, and capsys keeps it
readable. The one subprocess test guards the `python -m rl_lab.cli` path
that README will tell users about.
"""
import subprocess
import sys

from rl_lab.cli import main


def write_cfg(tmp_path, **over):
    raw = {"name": "cli-test", "env": "etch_chamber", "agent": "random",
           "episodes": 2, "max_steps": 10, "eval_episodes": 2,
           "loggers": ["csv"], "output_dir": str(tmp_path / "runs")}
    raw.update(over)
    import yaml
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_run_ok_exit_zero(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    rc = main(["run", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "final state: DONE" in out
    assert "eval:" in out


def test_bad_config_exit_two(tmp_path, capsys):
    cfg = write_cfg(tmp_path, episodes=0)   # fails validation
    rc = main(["run", str(cfg)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "episodes" in err                # message names the offending key


def test_override_on_extra_key_works(tmp_path, capsys):
    """Regression: -o used to reject any key from the config's extra block.

    Root cause: cmd_run round-tripped cfg.to_dict() (which FLATTENS extras
    to top level) back through build_config (which expects them nested).
    The fix re-nests unknown keys before re-validating. This test would
    have caught it at the time -- that's why we keep round-trip tests.
    """
    cfg = write_cfg(tmp_path, extra={"target_etch_rate": 50.0})
    rc = main(["run", str(cfg), "-o", "target_etch_rate=55"])
    out = capsys.readouterr().out
    assert rc == 0, "extra-key override must not fail validation"
    assert "final state: DONE" in out


def test_override_still_rejects_real_typos(tmp_path, capsys):
    """The fix must not have made validation laxer: brand-new keys still fail."""
    cfg = write_cfg(tmp_path)
    rc = main(["run", str(cfg), "-o", "learnig_rate=0.1"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "not set by" in err


def test_extra_prefix_override_adds_new_key(tmp_path, capsys):
    """extra.<key>=value is the deliberate way to add an undeclared knob."""
    cfg = write_cfg(tmp_path)
    rc = main(["run", str(cfg), "-o", "extra.pressure_limit=2.5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "final state: DONE" in out


def test_override_is_validated_too(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    # epsilon_start=2.0 is out of range; must fail exactly like YAML would.
    rc = main(["run", str(cfg), "-o", "epsilon_start=2.0"])
    assert rc == 2
    assert "epsilon_start" in capsys.readouterr().err


def test_override_changes_value(tmp_path, capsys):
    cfg = write_cfg(tmp_path, episodes=2)
    rc = main(["run", str(cfg), "-o", "episodes=4"])
    assert rc == 0
    # 4 episode rows + header in the CSV proves the override took effect.
    csv_file = next(tmp_path.rglob("episodes.csv"))
    assert len(csv_file.read_text().strip().splitlines()) == 5


def test_override_malformed_rejected(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    rc = main(["run", str(cfg), "-o", "oops-no-equals"])
    assert rc == 2


def test_missing_config_file_exit_two(tmp_path, capsys):
    rc = main(["run", str(tmp_path / "nope.yaml")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_list_shows_registries(capsys):
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    for name in ("etch_chamber", "qlearning", "random", "gym:"):
        assert name in out


def test_validate_subcommand(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    rc = main(["validate", str(cfg)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_module_invocation_works(tmp_path):
    """README promises `python -m rl_lab.cli`; keep that promise."""
    cfg = write_cfg(tmp_path)
    proc = subprocess.run([sys.executable, "-m", "rl_lab.cli", "validate",
                           str(cfg)], capture_output=True, text=True,
                          cwd="/workspace", timeout=60)
    assert proc.returncode == 0 and "OK" in proc.stdout


def test_run_reports_failed_experiment_exit_one(tmp_path, capsys):
    """A crashing env must exit 1 with the error printed -- FAILED runs are
    exactly what CI needs to catch (regression for DEBUGGING.md #3)."""
    cfg = write_cfg(tmp_path)
    # Sabotage through the factory seam: an env spec that constructs fine
    # but blows up at reset time is simulated by monkeypatching the registry.
    from rl_lab import factory

    class BoomEnv:
        action_space_size = 1   # what the factory/trainer actually reads

        def reset(self, seed=None):
            raise RuntimeError("chamber interlock")

        def step(self, a):
            raise RuntimeError("unreachable")

    orig = factory._ENV_REGISTRY["etch_chamber"]
    factory._ENV_REGISTRY["etch_chamber"] = lambda **kw: BoomEnv()
    try:
        rc = main(["run", str(cfg)])
    finally:
        factory._ENV_REGISTRY["etch_chamber"] = orig
    err = capsys.readouterr().err
    assert rc == 1
    assert "interlock" in err


def test_sweep_with_report_writes_files(tmp_path, capsys):
    """End-to-end sweep + report path exercised through main(), not just
    unit-level scheduler calls."""
    raw = {"name": "cli-sweep", "env": "etch_chamber", "agent": "random",
           "episodes": 2, "max_steps": 10, "eval_episodes": 2,
           "loggers": ["csv"], "output_dir": str(tmp_path / "runs")}
    import yaml
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    out_md = tmp_path / "sweep_report.md"
    rc = main(["sweep", str(p), "--seeds", "0", "1", "--workers", "2",
               "--report", str(out_md)])
    assert rc == 0
    text = out_md.read_text()
    assert "cli-sweep_seed0" in text and "cli-sweep_seed1" in text
    assert "Aggregate" in text or "aggregate" in text.lower()
