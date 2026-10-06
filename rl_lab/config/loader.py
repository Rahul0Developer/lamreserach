"""YAML -> ExperimentConfig with defaults and precise error messages."""
from __future__ import annotations

from pathlib import Path

import yaml

from rl_lab.config.schema import DEFAULTS, ConfigError, ExperimentConfig

_KNOWN_KEYS = set(DEFAULTS)


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: YAML syntax error: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{p}: file is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(raw).__name__}")
    return build_config(raw, source=str(p))


def build_config(raw: dict, source: str = "<dict>") -> ExperimentConfig:
    """Validate a raw dict (from YAML or CLI overrides) into a config object."""
    missing = [k for k in ("env", "agent") if not raw.get(k)]
    if missing:
        raise ConfigError(f"{source}: required key(s) missing: {missing}")

    # Unknown keys: warn-by-error is friendlier than silent typos. Nested
    # "extra:" block is the escape hatch for agent-specific options.
    unknown = set(raw) - _KNOWN_KEYS - {"extra"}
    if unknown:
        raise ConfigError(
            f"{source}: unknown key(s) {sorted(unknown)}. "
            f"Known keys: {sorted(_KNOWN_KEYS)} "
            f"(put agent-specific options under 'extra:')")

    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in raw.items() if k in _KNOWN_KEYS})
    extra = dict(raw.get("extra") or {})

    try:
        return ExperimentConfig(**merged, extra=extra)
    except TypeError as exc:  # e.g. wrong type for a known key
        raise ConfigError(f"{source}: {exc}") from exc
    except ConfigError:
        raise
