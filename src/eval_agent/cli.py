"""eval-agent: write, test and run evals for tool-using agents.

    eval-agent cases PATH                         show how a golden dataset was read
    eval-agent explore --target MOD:FN --cases PATH
                                                  run the agent on cases with pretend tools; save traces
    eval-agent analyze [RUN]                      find what went wrong and group it into failure types
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from .dataset import load_cases
from .runner import SdkTarget, auth_warnings, run_cases, save_run
from .trace import RunError, Trace, read_traces


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Set variables from a .env file in the current folder. Never overrides ones already set.

    Returns the names it set (never the values).
    """
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_target(spec: str):
    """"examples.support_agent.target:make_target" -> the Target that function returns."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"--target must look like module.path:function, got {spec!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    factory = getattr(importlib.import_module(module_name), attr)
    return factory()


def cmd_cases(args: argparse.Namespace) -> int:
    try:
        _, report = load_cases(args.path)
    except (OSError, ValueError) as exc:
        print(f"Couldn't read {args.path}: {exc}")
        return 1
    print(report.summary())
    return 0


def _line(item: Trace | RunError) -> str:
    if isinstance(item, RunError):
        return f"  ! {item.case_id} (try {item.rep + 1}): couldn't run [{item.kind}] {item.message}"
    tools = " -> ".join(item.tools_called()) or "no tools"
    cost = f" ${item.cost_usd:.3f}" if item.cost_usd is not None else ""
    return f"  {item.case_id} (try {item.rep + 1}): {item.status}; {tools}{cost}"


def cmd_explore(args: argparse.Namespace) -> int:
    try:
        cases, report = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        print(f"Couldn't read {args.cases}: {exc}")
        return 1
    print(report.summary(), end="\n\n")
    if args.only:
        wanted = [i.strip() for i in args.only.split(",") if i.strip()]
        missing = sorted(set(wanted) - {c.id for c in cases})
        if missing:
            print(f"No cases with these ids: {', '.join(missing)}")
            return 1
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    target = load_target(args.target)
    warnings = auth_warnings() if isinstance(target, SdkTarget) else []
    for warning in warnings:
        print(f"Warning: {warning}\n")
    print(f"Running {len(cases)} cases x {args.reps} on {target.name}...")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    traces, errors = asyncio.run(
        run_cases(target, cases, reps=args.reps, concurrency=args.concurrency,
                  timeout_s=args.timeout, on_done=lambda item: print(_line(item), flush=True))
    )
    path = save_run(args.out, traces, errors, {
        "command": "explore", "target": args.target, "target_name": target.name,
        "cases_file": str(args.cases), "reps": args.reps, "started_at": started, "warnings": warnings,
    })
    total = len(cases) * args.reps
    print(f"\n{len(traces)} of {total} runs finished; {len(errors)} couldn't run. Saved to {path}")
    fatal = next((e for e in errors if e.kind in {"cli_not_found", "unfaked_tools", "unsupported"}), None)
    if fatal:
        print(f"Stopped early: {fatal.message}")
        return 2
    return 0 if traces else 1


def _latest_run(runs_dir: Path) -> Path | None:
    runs = sorted(p for p in runs_dir.glob("*") if (p / "traces.jsonl").exists()) if runs_dir.is_dir() else []
    return runs[-1] if runs else None


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import MODEL, analyze_run, describe_target, estimate_cost
    from .report import analysis_markdown

    run_dir = args.run or _latest_run(Path("runs"))
    if run_dir is None or not (run_dir / "traces.jsonl").exists():
        print("No run to analyze. Run `eval-agent explore` first, or pass the run's folder.")
        return 1
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    traces = read_traces(run_dir / "traces.jsonl")
    if not traces:
        print(f"{run_dir.name} has no finished runs to analyze.")
        return 1

    cases_path = args.cases or meta.get("cases_file")
    cases = {}
    if cases_path and Path(cases_path).exists():
        cases = {c.id: c for c in load_cases(cases_path)[0]}
    else:
        print("Couldn't find the run's cases file, so I'll review without expected outcomes.")
    target_spec = args.target or meta.get("target")
    agent = describe_target(load_target(target_spec)) if target_spec else "(No description of the agent.)"
    agent_name = meta.get("target_name") or target_spec or "your agent"

    model = args.model or MODEL
    estimate = estimate_cost(traces, cases, agent, model)
    print(f"Reviewing {len(traces)} runs from {run_dir.name} with {model}. Estimated cost: up to ${estimate:.2f}.")
    if estimate > args.max_cost:
        print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it to go ahead.")
        return 1

    analysis = asyncio.run(analyze_run(traces, cases, agent, model=model, concurrency=args.concurrency))
    (run_dir / "analysis.json").write_text(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    report = analysis_markdown(analysis, agent_name=agent_name, run_name=run_dir.name)
    (run_dir / "analysis.md").write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Saved to {run_dir / 'analysis.md'}")
    return 0 if analysis.reviews else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cases", help="show how a golden dataset was read")
    p.add_argument("path", type=Path)
    p.set_defaults(func=cmd_cases)

    p = sub.add_parser("explore", help="run the agent on cases with pretend tools and save traces")
    p.add_argument("--target", required=True, help="module.path:function returning the target")
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--only", help="comma-separated case ids to run")
    p.add_argument("--limit", type=int, default=30, help="at most this many cases (0 = all)")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300, help="seconds per run")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.set_defaults(func=cmd_explore)

    p = sub.add_parser("analyze", help="find what went wrong in a run and group it into failure types")
    p.add_argument("run", nargs="?", type=Path, help="the run's folder (default: the latest in ./runs)")
    p.add_argument("--cases", type=Path, help="cases file (default: the one the run used)")
    p.add_argument("--target", help="module.path:function (default: the one the run used)")
    p.add_argument("--model", help="Claude model to analyze with (default: claude-opus-5-5)")
    p.add_argument("--max-cost", type=float, default=2.0, help="stop if the estimate is above this, in $")
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    loaded = load_dotenv()
    if loaded:
        print(f"Loaded {', '.join(loaded)} from .env")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
