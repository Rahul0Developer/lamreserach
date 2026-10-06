"""rl-lab command line entry point.

Why argparse over click/typer: zero dependencies, and the subcommand set
(run / sweep / validate / list) is all we need. If this ever grows past a
handful of commands I'd revisit.

Exit codes matter in CI and for equipment automation (our SEMI EAP
analogs parse them): 0 ok, 1 experiment FAILED, 2 bad config/usage,
130 interrupted sweep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rl_lab.config.loader import build_config, load_config
from rl_lab.config.schema import DEFAULTS, ConfigError
from rl_lab.core.fsm import State
from rl_lab.factory import _AGENT_REGISTRY, _ENV_REGISTRY
from rl_lab.orchestration.controller import ExperimentController

_KNOWN_KEYS = set(DEFAULTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl-lab",
        description="Modular RL benchmark lab (train / evaluate / log).")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one experiment from a YAML config")
    run_p.add_argument("config", type=Path, help="path to experiment YAML")
    # Repeatable -o key=value overrides; cheap reproducibility knob for
    # quick sweeps without cloning config files.
    run_p.add_argument("-o", "--set", dest="overrides", action="append",
                       default=[], metavar="KEY=VALUE",
                       help="override a config key, e.g. -o episodes=50")
    # Phase 6: report path. "auto" (bare --report) keeps the file in the
    # run dir; any other value is treated as an output path.
    run_p.add_argument("--report", nargs="?", const="auto", default=None,
                       metavar="PATH",
                       help="write markdown report + learning curve "
                            "(default: runs/<name>/report.md)")

    sub.add_parser("list", help="show registered envs and agents")

    val_p = sub.add_parser("validate", help="check a config without running")
    val_p.add_argument("config", type=Path)

    # Phase 5: parallel seed sweep. --workers 0 means "auto": min(jobs, cpu).
    sw_p = sub.add_parser("sweep", help="run one config x N seeds in parallel")
    sw_p.add_argument("configs", type=Path, nargs="+",
                      help="one or more experiment YAML files")
    sw_p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                      metavar="K", help="seed list (default: 0 1 2)")
    sw_p.add_argument("--workers", type=int, default=0,
                      help="worker threads; 0 = auto (default)")
    # Phase 6: sweep report. Same "auto" convention as run --report.
    sw_p.add_argument("--report", nargs="?", const="auto", default=None,
                      metavar="PATH",
                      help="write sweep report with comparison table and "
                           "overlaid curves (default: runs/sweep_report.md)")
    return parser


def _parse_overrides(pairs: list[str]) -> dict:
    """-o key=value -> dict with best-effort scalar typing (YAML rules).

    'extra.<name>=value' is the explicit escape hatch for keys a config
    doesn't declare yet; plain keys must already exist (checked by caller).
    """
    import yaml
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"override '{pair}' is not KEY=VALUE")
        key, _, raw = pair.partition("=")
        key = key.strip()
        # safe_load gives ints/floats/bools/lists from their text form.
        out[key] = yaml.safe_load(raw)
    return out


def cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "report", False):
        # Import here, not at module top: `rl-lab run` must work on machines
        # without matplotlib (headless CI); only --report pays that import.
        from rl_lab.reporting.report_generator import ReportGenerator
    cfg = load_config(args.config)   # validates file-level errors first
    if args.overrides:
        # Rebuild through the same validator instead of poking attributes:
        # an override like epsilon_start=2.0 must fail exactly like a bad
        # YAML value would, not sneak through.
        # NOTE: to_dict() flattens extra keys to top level (logger payload
        # shape), but build_config() expects them nested under "extra:".
        # Round-tripping without this split made `-o target_etch_rate=55`
        # crash on configs that use the extra block -- see DEBUGGING.md.
        raw = cfg.to_dict()
        nested = {k: raw.pop(k) for k in list(raw) if k not in _KNOWN_KEYS}
        overrides = _parse_overrides(args.overrides)
        # Strict rule, matching the YAML loader: -o may only touch keys that
        # already exist (core schema or config's extra block). A brand-new
        # key is a typo until proven otherwise -- silently routing it into
        # "extra" would let `learnig_rate=0.1` run with defaults instead of
        # failing loudly. Use `-o extra.new_key=value` to add one on purpose.
        allowed = set(raw) | set(nested)
        for key, value in overrides.items():
            if key.startswith("extra."):
                # Explicit escape hatch: user declares "this is an extra".
                nested[key[len("extra."):]] = value
                continue
            if key not in allowed:
                raise ConfigError(
                    f"'{key}' is not set by {args.config}; known keys: "
                    f"{sorted(allowed)} (add it under 'extra:' in the YAML "
                    f"or override as extra.{key}=value)")
            target = raw if key in raw else nested
            target[key] = value
        if nested:
            raw["extra"] = nested
        cfg = build_config(raw, source=f"{args.config} (with -o overrides)")

    controller = ExperimentController(cfg)
    result = controller.run()

    print(f"\n[{cfg.name}] final state: {result.state.name}"
          f"  wall: {result.wall_time_s:.1f}s")
    if result.eval_report:
        er = result.eval_report
        print(f"  eval: mean={er.mean_reward:.2f} std={er.std_reward:.2f} "
              f"success={er.success_rate:.0%} over {er.episodes} episodes")
    if result.output_dir:
        print(f"  artifacts: {result.output_dir}/")
    if getattr(args, "report", False):
        # --report regenerates report.md at a user-chosen path (the FSM's
        # reporting phase already wrote one into the run dir; this is the
        # convenience override). FAILED runs still get a report -- reports
        # of failures are exactly the ones people need to read.
        out = Path(args.report) if args.report != "auto" else \
            result.output_dir / "report.md"
        path = ReportGenerator().write_run_report(result, out)
        print(f"  report: {path}")
    if result.error:
        print(f"  ERROR: {result.error}", file=sys.stderr)
        return 1
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    print("environments:")
    for name in sorted(_ENV_REGISTRY):
        print(f"  {name}")
    print("  gym:<EnvSpec>   (gymnasium adapter, e.g. gym:CartPole-v1)")
    print("agents:")
    for name in sorted(_AGENT_REGISTRY):
        print(f"  {name}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"OK: {args.config} -> {cfg.name} "
          f"(env={cfg.env}, agent={cfg.agent}, episodes={cfg.episodes})")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Parallel seed sweep with Ctrl-C graceful shutdown.

    KeyboardInterrupt path: first Ctrl-C sets the scheduler stop flag and
    waits for workers to finish their current phase (data stays consistent);
    second Ctrl-C gives up waiting and exits 130 (standard 128+SIGINT).
    """
    from rl_lab.orchestration.scheduler import JobScheduler

    sched = JobScheduler(num_workers=args.workers or None)
    total = 0
    for p in args.configs:
        jobs = sched.expand_seed_jobs(p, args.seeds)
        total += len(jobs)
        for j in jobs:
            sched.submit(j)
    sched.start()
    print(f"sweep: {total} jobs ({len(args.configs)} config(s) x "
          f"{len(args.seeds)} seeds) on {sched.num_workers} workers")
    try:
        while not sched.wait(timeout=0.5):
            snap = sched.snapshot()
            print(f"  progress: {snap['finished']}/{total} done, "
                  f"{snap['pending']} pending", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted: stopping after current phase "
              "(Ctrl-C again to force exit)")
        sched.stop()
        if not sched.wait(timeout=60):
            print("workers did not finish in time; exiting dirty", file=sys.stderr)
            return 130
    results, agg = sched.results, sched.aggregate()
    print()  # newline after progress line
    for r in sorted(results, key=lambda r: r.config.name):
        er = r.eval_report
        tail = (f"eval mean={er.mean_reward:8.2f} std={er.std_reward:6.2f} "
                f"success={er.success_rate:.0%}") if er else ""
        err = f"  error: {r.error}" if r.error else ""
        print(f"  {r.config.name:<28} {r.state.name:<8} {tail}{err}")
    if agg["mean_eval_reward"] is not None:
        print(f"aggregate over {agg['n_scored']}/{agg['n']} scored runs: "
              f"mean={agg['mean_eval_reward']:.2f} "
              f"std_across_seeds={agg['std_across_seeds']:.2f} "
              f"success={agg['success_rate']:.0%}")
    if getattr(args, "report", False):
        from rl_lab.reporting.report_generator import ReportGenerator
        out = Path(args.report) if args.report != "auto" else \
            Path("runs") / "sweep_report.md"
        path = ReportGenerator().write_sweep_report(results, out, agg=agg)
        print(f"sweep report: {path}")
    return 1 if any(r.state.name == "FAILED" for r in results) else 0


_HANDLERS = {"run": cmd_run, "list": cmd_list, "validate": cmd_validate,
             "sweep": cmd_sweep}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _HANDLERS[args.command](args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
