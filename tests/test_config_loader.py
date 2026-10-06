"""First tests: config loading and validation contracts."""
import textwrap

import pytest

from rl_lab.config.loader import build_config, load_config
from rl_lab.config.schema import ConfigError


MINIMAL = {"env": "etch_chamber", "agent": "random"}


def test_minimal_config_gets_defaults():
    cfg = build_config(MINIMAL)
    assert cfg.seed == 42            # documented default
    assert cfg.episodes == 100
    assert cfg.loggers == ["console"]


def test_missing_required_env_raises_with_key_name():
    with pytest.raises(ConfigError, match="required key"):
        build_config({"agent": "random"})


def test_typo_in_key_is_rejected_not_silently_ignored():
    # This is the bug class config validation exists to catch.
    with pytest.raises(ConfigError, match="learnig_rate"):
        build_config({**MINIMAL, "learnig_rate": 0.1})


def test_epsilon_out_of_range_fails_at_load_time():
    with pytest.raises(ConfigError, match="epsilon_end"):
        build_config({**MINIMAL, "epsilon_end": 1.5})


def test_unknown_logger_name_fails():
    with pytest.raises(ConfigError, match="influxdb"):
        build_config({**MINIMAL, "loggers": ["influxdb"]})


def test_extra_block_passes_through_unvalidated():
    cfg = build_config({**MINIMAL, "extra": {"target_etch_rate": 50.0}})
    assert cfg.extra["target_etch_rate"] == 50.0
    assert cfg.to_dict()["target_etch_rate"] == 50.0


def test_load_config_from_yaml_file(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(textwrap.dedent("""
        env: etch_chamber
        agent: qlearning
        episodes: 5
        seed: 3
    """))
    cfg = load_config(p)
    assert cfg.episodes == 5 and cfg.seed == 3


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
