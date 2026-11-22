"""eval-agent: write, test and run evals for tool-using agents.

    eval-agent setup [REPO]                       read an agent's repo; write its target and pretend tools
    eval-agent cases PATH                         show how a golden dataset was read
    eval-agent explore --target MOD:FN --cases PATH
                                                  run the agent on cases with pretend tools; save traces
    eval-agent analyze [RUN]                      find what went wrong and group it into failure types
    eval-agent generate --target MOD:FN --cases PATH
                                                  write new test cases that cover what yours miss
    eval-agent plan [--runs RUN ...]              write the eval plan; --feedback "..." revises it;
                                                  --approve approves it
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
from .trace import MAX_TURNS, USER_TURN_LIMIT, RunError, Trace, read_traces


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


def cmd_generate(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .analyze import MODEL, describe_target
    from .generate import MAX_COUNT, build_prompt, estimate_cost, fake_tool_names, generate_cases
    from .trace import write_jsonl

    try:
        cases, report = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        print(f"Couldn't read {args.cases}: {exc}")
        return 1
    target = load_target(args.target)
    agent = describe_target(target)
    count = max(1, min(args.count, MAX_COUNT))
    model = args.model or MODEL
    example_world = getattr(target, "example_world", None) or {}
    estimate = estimate_cost(build_prompt(agent, cases, report, count, example_world), count, model)
    print(f"Writing {count} new cases for {target.name} with {model}. Estimated cost: up to ${estimate:.2f}.")
    if estimate > args.max_cost:
        print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it to go ahead.")
        return 1

    kept, dropped, spend = asyncio.run(
        generate_cases(agent, cases, report, fake_tool_names(target), count=count, model=model,
                       example_world=example_world))
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = args.out or args.cases.parent / "added" / f"{args.cases.stem}-generated-{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, [asdict(case) for case in kept])

    print(f"\nI wrote {len(kept)} new cases to {out}:")
    for case in kept:
        task = case.task if len(case.task) <= 70 else case.task[:67] + "..."
        print(f"  {case.id}: \"{task}\" [{', '.join(case.tags)}]\n      {case.notes}")
    if dropped:
        print(f"\nI dropped {len(dropped)} that couldn't run as written:")
        for reason in dropped:
            print(f"  - {reason}")
    print(f"\nCost: about ${spend.cost_usd:.2f}. These are suggestions: read them, delete any you don't want, "
          f"then run them with\n  eval-agent explore --target {args.target} --cases {out}")
    return 0 if kept else 1


def smoke_run(root: Path, spec: str, tool_map: dict, out: str, timeout_s: float):
    """Run the agent once on the setup step's simple task, and notice if any repo file changed (FR-6)."""
    from .check import changed_files, snapshot
    from .dataset import Case
    from .toolmap import SmokeResult

    smoke = SmokeResult(task=(tool_map.get("smoke_task") or "").strip())
    if not smoke.task:
        smoke.error = "The setup step didn't suggest a first task."
        return smoke, ""
    previous = os.getcwd()
    os.chdir(root)  # agents often read config relative to their own folder
    package = spec.split(".", 1)[0]
    for name in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[name]  # an `evals` imported from somewhere else must not shadow this repo's
    try:
        before = snapshot(root)
        try:
            target = load_target(spec)
        except Exception as exc:
            smoke.error = f"Couldn't load the target: {type(exc).__name__}: {exc}"
            return smoke, ""
        case = Case(id="first-run", task=smoke.task, user_replies=list(tool_map.get("smoke_replies") or []),
                    starting_state=dict(getattr(target, "example_world", None) or {}), tags=["setup"],
                    source="synthetic")
        traces, errors = asyncio.run(run_cases(target, [case], reps=1, concurrency=1, timeout_s=timeout_s))
        smoke.changed_files = changed_files(before, snapshot(root))
        path = save_run(root / out / "runs", traces, errors, {
            "command": "setup", "target": spec, "target_name": target.name, "smoke_task": smoke.task})
    finally:
        os.chdir(previous)
    if errors:
        smoke.error = f"[{errors[0].kind}] {errors[0].message}"
    if traces:
        trace = traces[0]
        smoke.ran, smoke.status = True, trace.status
        # Show its answer to the task itself, not to a later scripted reply.
        first = next((m.text for m in trace.messages if m.role == "agent"), trace.final_answer)
        smoke.tools, smoke.answer, smoke.cost_usd = trace.tools_called(), first, trace.cost_usd
        if trace.status in (MAX_TURNS, USER_TURN_LIMIT):
            smoke.error = f"It didn't finish ({trace.status})."
    return smoke, str(path)


def cmd_setup(args: argparse.Namespace) -> int:
    from .intake import MODEL, RepoFiles, estimate_cost, run_intake
    from .toolmap import summary, tool_map_data, tool_map_yaml

    root = args.repo.resolve()
    if not root.is_dir():
        print(f"{args.repo} isn't a folder. Point me at your agent's repo (a local folder).")
        return 1
    out = args.out.strip("/\\")
    existing = [name for name in ("target.py", "fakes.py", "tool_map.yaml") if (root / out / name).exists()]
    if existing and not args.force:
        print(f"{out}/ in {root.name} already has {', '.join(existing)}. Use --force to replace them, or --out to "
              "write somewhere else.")
        return 1
    files = RepoFiles(root, out, tuple(args.exclude or ()), overwrite=args.force)
    model = args.model or MODEL
    estimate = estimate_cost(files)
    print(f"I'll read {root.name} ({len(files.files())} files I'm allowed to read) with {model}, find your agent "
          f"and its tools, and write pretend versions of the tools into {out}/. I won't change your code.")
    print(f"Estimated cost: up to about ${estimate:.2f}. It stops at ${args.max_cost:.2f} (--max-cost).")
    if not args.no_smoke:
        print("Then I'll run your agent once on a simple task, which makes your agent's own model calls.")
    for warning in auth_warnings():
        print(f"Warning: {warning}")
    if estimate > args.max_cost:
        print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it to go ahead.")
        return 1
    print()

    result, files = asyncio.run(run_intake(root, out=out, model=model, max_cost=args.max_cost,
                                           note=args.note or "", emit=lambda line: print(f"  {line}", flush=True),
                                           files=files))
    spec = f"{out.replace('/', '.')}.target:make_target"
    print()
    if not result.ok:
        print("I couldn't finish setting up your agent.")
        if result.stopped:
            print(result.stopped)
        if result.check is not None and not result.check.ok:
            print("The last check said:\n" + result.check.text())
        elif result.map is None:
            print("I didn't get as far as a tool map.")
        if result.final_text:
            print(f"My last words: {result.final_text}")
        for note in result.notes[:5]:
            print(f"Claude Code said: {note}")
        if result.written:
            print(f"Files I wrote (keep or delete them): {', '.join(result.written)}")
        print(f"Cost: ${result.cost_usd:.2f}.")
        return 1

    data = tool_map_data(result.map, result.check, spec)
    (root / out / "tool_map.yaml").write_text(tool_map_yaml(data), encoding="utf-8")
    written = result.written + [f"{out}/tool_map.yaml"]
    gitignore = root / out / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("runs/\n", encoding="utf-8")
        written.append(f"{out}/.gitignore")

    smoke, run_dir = (None, "") if args.no_smoke else smoke_run(root, spec, result.map, out, args.timeout)
    print(summary(data, sorted(written), result.cost_usd, smoke, run_dir))
    print(f"\nNext:\n  1. Read {out}/tool_map.yaml and fix anything that's wrong.\n"
          f"  2. From {root.name}'s folder, run your test cases:\n"
          f"       eval-agent explore --target {spec} --cases YOUR_CASES.jsonl\n"
          "     No test cases yet? A .jsonl file with one {\"task\": ..., \"expected_outcome\": ...} per line "
          "is enough.")
    return 0 if smoke is None or smoke.ok else 2


def _latest_analyzed_run(runs_dir: Path) -> Path | None:
    runs = sorted(p for p in runs_dir.glob("*") if (p / "analysis.json").exists()) if runs_dir.is_dir() else []
    return runs[-1] if runs else None


def _print_plan_check(state) -> None:
    report = state.check_report()
    print(f"Plan check: reading grade {report.grade:g} (target 9 or below). A 15-year-old judge understood "
          f"{report.understood} of {report.judged} checks.")
    if report.issues:
        print(f"Still flagged after {report.rounds} rewrites ({len(report.issues)}):")
        for issue in report.issues[:8]:
            print(f"  - {issue}")


def cmd_plan(args: argparse.Namespace) -> int:
    from .plan import (
        LEVEL_LABEL,
        MODEL,
        add_revision,
        approve,
        estimate_cost,
        gather_facts,
        load_state,
        new_state,
        revise_plan,
        save_state,
        write_plan,
    )
    from .plan_render import markdown, to_html

    out = args.out
    state_path = out / "plan.json"
    model = args.model or MODEL

    def save(state) -> None:
        save_state(state, state_path)
        (out / "EVAL_PLAN.md").write_text(markdown(state), encoding="utf-8")
        (out / "EVAL_PLAN.html").write_text(to_html(state), encoding="utf-8")

    def files() -> str:
        return f"{out / 'EVAL_PLAN.md'} (and EVAL_PLAN.html, one file you can send to anyone)"

    next_steps = (f"\nRead it, then either:\n  eval-agent plan --out {out} --approve\n"
                  f"  eval-agent plan --out {out} --feedback \"what's missing or wrong\"")

    if args.approve or args.feedback:
        if not state_path.exists():
            print(f"There's no plan in {out} yet. Run `eval-agent plan` first.")
            return 1
        state = load_state(state_path)

    if args.approve:
        if state.is_approved:
            print(f"Version {state.version} is already approved.")
            return 0
        approve(state)
        save(state)
        print(f"Approved version {state.version} of the plan. It's recorded in {state_path}, and the next step "
              "builds the tests from exactly this version.")
        return 0

    if args.feedback:
        facts = state.plan_facts()
        estimate = estimate_cost(facts, model, revising=True)
        print(f"Revising the plan with {model}. Estimated cost: up to ${estimate:.2f}.")
        if estimate > args.max_cost:
            print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it to go ahead.")
            return 1
        was_approved = state.is_approved
        result = asyncio.run(revise_plan(facts, state.plan(), args.feedback, model=model))
        add_revision(state, args.feedback, result)
        save(state)
        print(f"\nVersion {state.version}.")
        if result.reply:
            print(f"\n{result.reply}")
        print("\nWhat changed:")
        for change in result.what_changed:
            print(f"  - {change}")
        print()
        _print_plan_check(state)
        if was_approved:
            print(f"\nYou had approved version {state.version - 1}; approve this version when it's right.")
        print(f"\nCost: ${result.spend.cost_usd:.2f}. Saved to {files()}.{next_steps}")
        return 0

    if state_path.exists() and not args.new:
        state = load_state(state_path)
        print(f"There's already a plan in {out} (version {state.version}). Give feedback with --feedback, approve "
              "it with --approve, or start over with --new.")
        return 1
    run_dirs = args.runs or [r for r in [_latest_analyzed_run(Path("runs"))] if r]
    if not run_dirs:
        print("No analyzed runs to plan from. Run `eval-agent explore`, then `eval-agent analyze`, first.")
        return 1
    meta_path = run_dirs[0] / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    target_spec = args.target or meta.get("target") or ""
    target = load_target(target_spec) if target_spec else None
    tool_map_path = args.tool_map or out / "tool_map.yaml"
    tool_map = None
    if tool_map_path.exists():
        import yaml

        tool_map = yaml.safe_load(tool_map_path.read_text(encoding="utf-8"))
    try:
        facts = gather_facts(run_dirs, target_spec=target_spec, target=target, tool_map=tool_map)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    estimate = estimate_cost(facts, model)
    runs_text = ", ".join(r.name for r in facts.runs)
    print(f"Writing a test plan for {facts.agent_name} from {len(facts.runs)} analyzed run(s) ({runs_text}) and "
          f"{len(facts.cases)} examples, with {model}. Estimated cost: up to ${estimate:.2f}.")
    if tool_map is None:
        print(f"(No tool map at {tool_map_path}; I'll use the tools as the target describes them.)")
    if estimate > args.max_cost:
        print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it to go ahead.")
        return 1
    result = asyncio.run(write_plan(facts, model=model))
    state = new_state(facts, result)
    save(state)
    draft = state.plan()
    print(f"\n{draft.short_version}\n\nChecks:")
    for n, check in enumerate(draft.checks, start=1):
        print(f"  {n}. {check.name} ({LEVEL_LABEL[check.level]})")
    print()
    _print_plan_check(state)
    print(f"\nCost: ${result.spend.cost_usd:.2f}. Saved to {files()}.{next_steps}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="read an agent's repo and write the target and pretend tools for it")
    p.add_argument("repo", nargs="?", type=Path, default=Path("."), help="the agent's repo folder (default: here)")
    p.add_argument("--out", default="evals", help="folder inside the repo to write into (default: evals)")
    p.add_argument("--note", help="anything the Eval Agent should know, e.g. where the agent starts")
    p.add_argument("--exclude", action="append", metavar="GLOB", help="files it must not read (repeatable)")
    p.add_argument("--model", help="Claude model to read with (default: claude-opus-5-5)")
    p.add_argument("--max-cost", type=float, default=3.0, help="stop reading at this cost, in $")
    p.add_argument("--force", action="store_true", help="replace target.py, fakes.py and tool_map.yaml if there")
    p.add_argument("--no-smoke", action="store_true", help="don't run the agent once at the end")
    p.add_argument("--timeout", type=float, default=300, help="seconds for that first run")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("plan", help="write the eval plan in plain words; revise it on feedback; approve it")
    p.add_argument("--runs", nargs="+", type=Path, help="analyzed run folders to plan from (default: the latest)")
    p.add_argument("--target", help="module.path:function (default: the one the first run used)")
    p.add_argument("--tool-map", type=Path, help="the setup step's tool map (default: OUT/tool_map.yaml)")
    p.add_argument("--out", type=Path, default=Path("evals"), help="where the plan lives (default: evals)")
    p.add_argument("--feedback", help="what's missing or wrong; writes a new version")
    p.add_argument("--approve", action="store_true", help="approve the current version")
    p.add_argument("--new", action="store_true", help="start a new plan even if one exists")
    p.add_argument("--model", help="Claude model to write with (default: claude-opus-5-5)")
    p.add_argument("--max-cost", type=float, default=2.0, help="stop if the estimate is above this, in $")
    p.set_defaults(func=cmd_plan)

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

    p = sub.add_parser("generate", help="write new test cases that cover what yours miss")
    p.add_argument("--target", required=True, help="module.path:function returning the target")
    p.add_argument("--cases", required=True, type=Path, help="your existing cases (left untouched)")
    p.add_argument("--count", type=int, default=10, help="how many to write (up to 15)")
    p.add_argument("--out", type=Path, help="where to write them (default: added/ next to your cases)")
    p.add_argument("--model", help="Claude model to write with (default: claude-opus-5-5)")
    p.add_argument("--max-cost", type=float, default=1.0, help="stop if the estimate is above this, in $")
    p.set_defaults(func=cmd_generate)

    args = parser.parse_args(argv)
    loaded = load_dotenv()
    if loaded:
        print(f"Loaded {', '.join(loaded)} from .env")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
