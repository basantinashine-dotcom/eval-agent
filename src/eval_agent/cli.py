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
    eval-agent build                              turn the approved plan into tests, proven on earlier runs
    eval-agent score [RUN]                        score a run with the built tests
    eval-agent mutate                             break a copy of the agent on purpose; see if the tests notice
    eval-agent label                              label some answers yourself, to check the judges
    eval-agent calibrate                          see how often each judge agrees with your labels
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


def cmd_build(args: argparse.Namespace) -> int:
    from .build import MAX_JUDGE_CALLS, Builder, build_record, run_build, summary
    from .plan import MODEL, load_state
    from .score import JUDGE_MODEL, judge_estimate

    evals = args.evals
    state_path = evals / "plan.json"
    if not state_path.exists():
        print(f"There's no plan in {evals}. Write one with `eval-agent plan` and approve it first.")
        return 1
    state = load_state(state_path)
    if not state.is_approved:
        print(f"The plan in {evals} isn't approved yet (version {state.version}). Nothing gets built before that. "
              f"Read it, then: eval-agent plan --out {evals} --approve")
        return 1
    existing = [f for f in ("checks.py", "expectations.json", "judges.json") if (evals / f).exists()]
    if existing and not args.force:
        print(f"{evals} already has {', '.join(existing)}. Use --force to build them again.")
        return 1
    facts = state.plan_facts()
    run_dirs = args.runs or [Path(r.path) if r.path else Path("runs") / r.name for r in facts.runs]
    missing = [str(r) for r in run_dirs if not (r / "analysis.json").exists()]
    if missing:
        print(f"Can't find these analyzed runs: {', '.join(missing)}. Pass them with --runs.")
        return 1
    model = args.model or MODEL
    judge_model = args.judge_model or JUDGE_MODEL
    builder = Builder(evals, state, run_dirs, judge_model=judge_model,
                      emit=lambda line: print(f"  {line}", flush=True))
    print(f"Building tests for the {len(builder.checks)} checks in plan version {state.version}, and proving them "
          f"on the {len(builder.items)} tries we already have.")
    print(f"Writing them with {model} stops at ${args.max_cost:.2f}. Judges use {judge_model}: at most "
          f"{MAX_JUDGE_CALLS} questions, up to about ${judge_estimate(MAX_JUDGE_CALLS, judge_model):.2f}.")
    for warning in auth_warnings():
        print(f"Warning: {warning}")
    print()
    result = asyncio.run(run_build(builder, model=model, max_cost=args.max_cost))
    (evals / "build.json").write_text(json.dumps(build_record(builder, result, model), indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    print()
    if result.stopped:
        print(result.stopped)
    if result.problems:
        print("The tests can't run yet:\n" + "\n".join(f"  - {p}" for p in result.problems))
    if result.missing:
        print("Missing:\n" + "\n".join(f"  - {m}" for m in result.missing))
    if result.rows:
        print("Each test, tried on the runs we already have:")
        print(summary(builder, result))
    if result.notes:
        print("\nNotes:\n" + "\n".join(f"  - {n}" for n in result.notes))
    print(f"\nCost: ${result.cost_usd:.2f} writing the tests, ${result.judge_cost_usd:.2f} for "
          f"{result.judge_calls} judge questions. Saved to {evals}: checks.py, expectations.json, judges.json, "
          "build.json.")
    print(f"\nNext: run your agent, then score the run with these tests:\n  eval-agent score RUN --evals {evals}")
    return 0 if result.ok else 2


def cmd_score(args: argparse.Namespace) -> int:
    from .analyze import Spend
    from .plan import load_state
    from .calibrate import trust
    from .score import JUDGE_MODEL, JudgeSpec, apply_trust, combine, judge_estimate, judge_jobs, load_items
    from .score import score_code, score_judges, scores_text

    run_dir = args.run or _latest_run(Path("runs"))
    if run_dir is None or not (run_dir / "traces.jsonl").exists():
        print("No run to score. Run `eval-agent explore` first, or pass the run's folder.")
        return 1
    evals = args.evals
    if not (evals / "expectations.json").exists():
        print(f"There are no tests in {evals} yet. Build them with `eval-agent build --evals {evals}`.")
        return 1
    expectations = json.loads((evals / "expectations.json").read_text(encoding="utf-8"))
    judges = JudgeSpec.load_all(evals / "judges.json")
    items = load_items(run_dir, str(args.cases) if args.cases else None)
    code = score_code(evals, items, expectations)
    if code.problems:
        print("The tests can't run:\n" + "\n".join(f"  - {p}" for p in code.problems))
        return 1
    judge_model = args.judge_model or JUDGE_MODEL
    jobs = judge_jobs(judges, items, expectations, code.results, set(code.registered))
    estimate = judge_estimate(len(jobs), judge_model)
    print(f"Scoring {len(items)} tries from {run_dir.name}: code checks, plus {len(jobs)} judge questions with "
          f"{judge_model} (up to about ${estimate:.2f}).")
    if estimate > args.max_cost:
        print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it to go ahead.")
        return 1
    judged, spend = asyncio.run(score_judges(jobs, model=judge_model)) if jobs else ({}, Spend())
    state = load_state(evals / "plan.json") if (evals / "plan.json").exists() else None
    names = {n: (c.name, c.level) for n, c in enumerate(state.plan().checks, start=1)} if state else \
        {n: (f"check {n}", "should_usually_pass") for v in {**code.results, **judged}.values() for n in v}
    must = {n for n, (_, level) in names.items() if level == "must_never_fail"}
    trusted = trust(evals, judges, judge_model)
    results = combine(code.results, apply_trust(judged, trusted, must))
    (run_dir / "scores.json").write_text(json.dumps(
        {key: {str(n): [v.status, v.reason] for n, v in verdicts.items()} for key, verdicts in results.items()},
        indent=1, ensure_ascii=False), encoding="utf-8")
    report = scores_text(results, names, run_dir.name)
    print()
    print(report)
    print(f"\nJudges cost ${spend.cost_usd:.2f}. Saved to {run_dir / 'scores.json'}")
    failed_must = any(v.status == "fail" and names.get(n, ("", ""))[1] == "must_never_fail"
                      for verdicts in results.values() for n, v in verdicts.items())
    return 1 if failed_must else 0


def cmd_mutate(args: argparse.Namespace) -> int:
    from .dataset import Case
    from .mutate import (
        MutationPlan,
        design,
        estimate,
        from_record,
        judges_per_try,
        record,
        report,
        run_mutations,
        strengthen,
    )
    from .plan import MODEL, load_state
    from .score import JUDGE_MODEL, JudgeSpec

    evals = args.evals
    if not (evals / "plan.json").exists() or not (evals / "expectations.json").exists():
        print(f"There are no built tests in {evals}. Run `eval-agent build --evals {evals}` first.")
        return 1
    state = load_state(evals / "plan.json")
    if not state.is_approved:
        print(f"The plan in {evals} isn't approved, so its tests aren't final. Approve it and build first.")
        return 1
    facts = state.plan_facts()
    checks = state.plan().checks
    names = {n: c.name for n, c in enumerate(checks, start=1)}
    must = {n for n, c in enumerate(checks, start=1) if c.level == "must_never_fail"}
    saved = evals / "mutations.json"
    if args.report_only:
        data = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
        if not data.get("results"):
            print(f"There are no saved breaks in {evals} to report. Run `eval-agent mutate` first.")
            return 1
        plan, run = from_record(data)
        saved.write_text(json.dumps(record(run, plan), indent=1, ensure_ascii=False), encoding="utf-8")
        print(report(run, names, must))
        return 2 if any(r.verdict == "weak" for r in run.results) else 0
    target_spec = args.target or facts.target
    if not target_spec:
        print("I don't know how to run your agent. Pass --target module.path:function.")
        return 1
    target = load_target(target_spec)
    expectations = json.loads((evals / "expectations.json").read_text(encoding="utf-8"))
    judges = JudgeSpec.load_all(evals / "judges.json")
    cases = {c["id"]: Case(**c) for c in facts.cases}
    build = json.loads((evals / "build.json").read_text(encoding="utf-8")) if (evals / "build.json").exists() else {}
    unproven = sorted(row["check"] for row in build.get("checks", []) if not row.get("known_bad") and
                      row["check"] in must)
    baseline_runs = args.runs or [Path(r.path) if r.path else Path("runs") / r.name for r in facts.runs]
    model = args.model or MODEL
    judge_model = args.judge_model or JUDGE_MODEL

    design_cost = 0.0
    reused = saved.exists() and not args.redesign
    if reused:
        plan = MutationPlan(mutations=json.loads(saved.read_text(encoding="utf-8"))["mutations"], notes=[])
        print(f"Using the {len(plan.mutations)} breaks in {saved} (--redesign to write new ones).")
    else:
        print(f"Designing breaks from the plan's list with {model}, plus one for each must-never-fail test that has "
              f"never seen a failure ({len(unproven)}).")
        plan, spend = asyncio.run(design(state, target, expectations, unproven, model=model))
        design_cost = spend.cost_usd
    if args.only:
        wanted = {i.strip() for i in args.only.split(",")}
        plan.mutations = [m for m in plan.mutations if m.id in wanted]
    cost = estimate(plan, reps=args.reps, max_cases=args.max_cases, agent_cost_per_try=facts.cost_per_try(),
                    judges_per_try=judges_per_try(plan, judges), judge_model=judge_model)
    tries = sum(min(len(m.cases), args.max_cases) for m in plan.mutations) * args.reps
    print(f"{len(plan.mutations)} breaks, {tries} tries of your agent in all. Estimated cost: up to ${cost:.2f} "
          "(your agent's model calls plus judges).")
    if cost > args.max_cost:
        print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it, or use --only or --max-cases.")
        return 1
    for warning in auth_warnings() if isinstance(target, SdkTarget) else []:
        print(f"Warning: {warning}")
    print()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = args.out / f"mutations-{stamp}"
    run = asyncio.run(run_mutations(
        plan, target=target, evals=evals, cases=cases, expectations=expectations, judges=judges,
        baseline_runs=baseline_runs, check_numbers=set(names), reps=args.reps, max_cases=args.max_cases,
        judge_model=judge_model, out_dir=out_dir, emit=lambda line: print(f"  {line}", flush=True)))
    resisted = [r for r in run.results if r.verdict == "resisted"]
    if resisted and not reused and not args.no_retry:
        print(f"\n  Your agent resisted {len(resisted)} breaks; designing a stronger try at each.")
        stronger, spend = asyncio.run(strengthen(state, target, expectations, resisted, model=model))
        design_cost += spend.cost_usd
        more = estimate(stronger, reps=args.reps, max_cases=args.max_cases, agent_cost_per_try=facts.cost_per_try(),
                        judges_per_try=judges_per_try(stronger, judges), judge_model=judge_model)
        spent = design_cost + run.agent_cost_usd + run.judge_cost_usd
        if spent + more > args.max_cost:
            print(f"  Skipping them: they'd take the total past --max-cost (${args.max_cost:.2f}).")
        else:
            again = asyncio.run(run_mutations(
                stronger, target=target, evals=evals, cases=cases, expectations=expectations, judges=judges,
                baseline_runs=baseline_runs, check_numbers=set(names), reps=args.reps, max_cases=args.max_cases,
                judge_model=judge_model, out_dir=out_dir, emit=lambda line: print(f"  {line}", flush=True)))
            run.results += again.results
            run.skipped += again.skipped
            run.agent_cost_usd += again.agent_cost_usd
            run.judge_cost_usd += again.judge_cost_usd
            plan.mutations += stronger.mutations
    run.design_cost_usd = round(design_cost, 4)
    saved.write_text(json.dumps(record(run, plan), indent=1, ensure_ascii=False), encoding="utf-8")
    print()
    print(report(run, names, must))
    print(f"\nCost: ${design_cost:.2f} designing, ${run.agent_cost_usd:.2f} for your agent, ${run.judge_cost_usd:.2f} "
          f"for judges. Saved to {saved}; the broken runs are in {out_dir}.")
    weak = any(r.verdict == "weak" for r in run.results)
    return 2 if weak else 0


def _built(evals: Path):
    """The approved plan and built tests in evals, or None after saying what's missing."""
    from .plan import load_state

    if not (evals / "plan.json").exists() or not (evals / "expectations.json").exists():
        print(f"There are no built tests in {evals}. Run `eval-agent build --evals {evals}` first.")
        return None
    return load_state(evals / "plan.json")


def cmd_label(args: argparse.Namespace) -> int:
    from .calibrate import (
        archive_label_set,
        label_in_terminal,
        labeled_before,
        load_label_set,
        mutation_items,
        pick_items,
        write_label_set,
    )
    from .dataset import Case
    from .score import JudgeSpec

    evals = args.evals
    state = _built(evals)
    if state is None:
        return 1
    if not JudgeSpec.load_all(evals / "judges.json"):
        print("Your tests have no judges, so there's nothing to label.")
        return 1
    facts = state.plan_facts()
    if (evals / "labels" / "items.json").exists() and not (args.new or args.fresh):
        name, items = load_label_set(evals)
        page = evals / "labels" / "label.html"
        print(f"Using the label set you already have ({len(items)} tries; --fresh for a new round).")
    else:
        archived = None
        if args.fresh:
            archived, _ = archive_label_set(evals)
        exclude = labeled_before(evals)
        cases = {c["id"]: Case(**c) for c in facts.cases}
        run_dirs = args.runs or [Path(r.path) if r.path else Path("runs") / r.name for r in facts.runs]
        extra = mutation_items(args.runs_dir, cases)
        checks = state.plan().checks
        items = pick_items(evals, run_dirs=run_dirs, extra=extra, cases=cases,
                           checks_based_on={n: c.based_on for n, c in enumerate(checks, start=1)},
                           per_judge=args.per_judge, seed=len(list((evals / "labels").glob("round-*"))),
                           exclude=exclude)
        if not items:
            print("There are no tries left that you haven't labeled. Run your agent on more examples first.")
            return 1
        name, page = write_label_set(evals, items, facts.agent_name)
        if exclude:
            print(f"A fresh round: {len(items)} tries you haven't labeled for these questions."
                  + (f" Your earlier round is kept in {archived}." if archived else ""))
        else:
            print(f"I picked {len(items)} tries for you to label: up to {args.per_judge} for each judge question, "
                  "mixing ones that went wrong before, ones from breaking your agent, and ordinary ones.")
    if args.terminal:
        answers = label_in_terminal(items)
        (evals / "labels" / "labels.json").write_text(json.dumps({"set": name, "answers": answers}, indent=1),
                                                      encoding="utf-8")
        print(f"\nSaved {len(answers)} answers. Next: eval-agent calibrate --evals {evals}")
        return 0
    print(f"\nOpen this page in your browser and answer each question:\n  {page.resolve()}\n"
          "Click Save labels when you're done. Put labels.json in the same folder as the page, or leave it in "
          "Downloads; I'll look in both.\n"
          f"Then: eval-agent calibrate --evals {evals}\n"
          "(Prefer the terminal? eval-agent label --terminal)")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import calibrate, find_labels, load_label_set, read_answers, report, save_calibration
    from .score import JUDGE_MODEL, JudgeSpec, judge_estimate

    evals = args.evals
    state = _built(evals)
    if state is None:
        return 1
    if not (evals / "labels" / "items.json").exists():
        print(f"There's nothing labeled yet. Run `eval-agent label --evals {evals}` first.")
        return 1
    name, items = load_label_set(evals)
    path = find_labels(evals, name, args.labels)
    if path is None:
        print(f"I couldn't find your labels.json for this set ({name}). I looked in {evals / 'labels'} and in "
              "Downloads. Pass it with --labels PATH.")
        return 1
    answers = read_answers(path)
    usable = [i for i in items if answers.get(i.id) in ("yes", "no")]
    print(f"Read {len(answers)} answers from {path} ({len(usable)} yes or no; 'not sure' ones are left out).")
    if not usable:
        return 1
    judge_model = args.judge_model or JUDGE_MODEL
    estimate = judge_estimate(len(usable), judge_model)
    print(f"Asking the judges ({judge_model}) the same questions. Estimated cost: up to ${estimate:.2f}.")
    if estimate > args.max_cost:
        print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it to go ahead.")
        return 1
    judges = JudgeSpec.load_all(evals / "judges.json")
    names = {n: (c.name, c.level) for n, c in enumerate(state.plan().checks, start=1)}
    if args.recheck:
        print("Re-checking the current judge questions on the answers you already gave. If you rewrote a question to "
              "match how you answered, this shows whether it now says what you meant. It doesn't make a judge "
              "trusted, because these are the answers it was rewritten from: label a fresh round for that "
              "(eval-agent label --fresh).")
    results, spend = asyncio.run(calibrate(items, answers, judges, model=judge_model, allow_revised=args.recheck))
    print()
    print(report(results, names))
    if args.recheck:
        print(f"\nCost: ${spend.cost_usd:.2f}. Nothing saved: trust stays as it was.")
        return 0
    save_calibration(evals, results, judge_model, name)
    print(f"\nCost: ${spend.cost_usd:.2f}. Saved to {evals / 'calibration.json'}; `eval-agent score` uses it from now on.")
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

    p = sub.add_parser("build", help="turn the approved plan into tests, and prove them on earlier runs")
    p.add_argument("--evals", type=Path, default=Path("evals"), help="where the plan is (default: evals)")
    p.add_argument("--runs", nargs="+", type=Path, help="analyzed runs to prove the tests on (default: the plan's)")
    p.add_argument("--model", help="Claude model that writes the tests (default: claude-opus-5-5)")
    p.add_argument("--judge-model", help="model for judge questions (default: claude-sonnet-5-5)")
    p.add_argument("--max-cost", type=float, default=5.0, help="stop writing at this cost, in $")
    p.add_argument("--force", action="store_true", help="build again over existing tests")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("score", help="score a run with the built tests")
    p.add_argument("run", nargs="?", type=Path, help="the run's folder (default: the latest in ./runs)")
    p.add_argument("--evals", type=Path, default=Path("evals"), help="where the tests are (default: evals)")
    p.add_argument("--cases", type=Path, help="cases file (default: the one the run used)")
    p.add_argument("--judge-model", help="model for judge questions (default: claude-sonnet-5-5)")
    p.add_argument("--max-cost", type=float, default=2.0, help="stop if the judge estimate is above this, in $")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("mutate", help="break a copy of the agent on purpose and see if the tests notice")
    p.add_argument("--evals", type=Path, default=Path("evals"), help="where the built tests are (default: evals)")
    p.add_argument("--target", help="module.path:function (default: the plan's)")
    p.add_argument("--runs", nargs="+", type=Path, help="earlier runs to compare with (default: the plan's)")
    p.add_argument("--only", help="comma-separated break ids to run")
    p.add_argument("--redesign", action="store_true", help="write new breaks instead of reusing saved ones")
    p.add_argument("--no-retry", action="store_true", help="don't make a stronger try at breaks your agent resisted")
    p.add_argument("--report-only", action="store_true", help="report the saved breaks again; run nothing")
    p.add_argument("--reps", type=int, default=1, help="tries per example per break")
    p.add_argument("--max-cases", type=int, default=4, help="examples per break, at most")
    p.add_argument("--model", help="Claude model that designs the breaks (default: claude-opus-5-5)")
    p.add_argument("--judge-model", help="model for judge questions (default: claude-sonnet-5-5)")
    p.add_argument("--max-cost", type=float, default=3.0, help="stop if the estimate is above this, in $")
    p.add_argument("--out", type=Path, default=Path("runs"), help="where to save the broken runs")
    p.set_defaults(func=cmd_mutate)

    p = sub.add_parser("label", help="label some of your agent's answers yourself, to check the judges")
    p.add_argument("--evals", type=Path, default=Path("evals"), help="where the built tests are (default: evals)")
    p.add_argument("--runs", nargs="+", type=Path, help="analyzed runs to pick from (default: the plan's)")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"), help="where break runs are (default: runs)")
    p.add_argument("--per-judge", type=int, default=3, help="tries per judge question")
    p.add_argument("--terminal", action="store_true", help="answer in the terminal instead of a web page")
    p.add_argument("--new", action="store_true", help="pick the set again (replaces the current one)")
    p.add_argument("--fresh", action="store_true", help="a new round of tries you haven't labeled; keeps the old one")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("calibrate", help="see how often each judge agrees with your labels")
    p.add_argument("--evals", type=Path, default=Path("evals"), help="where the built tests are (default: evals)")
    p.add_argument("--labels", type=Path, help="your labels.json (default: the labels folder, then Downloads)")
    p.add_argument("--judge-model", help="model for judge questions (default: claude-sonnet-5-5)")
    p.add_argument("--max-cost", type=float, default=1.0, help="stop if the estimate is above this, in $")
    p.add_argument("--recheck", action="store_true", help="try rewritten questions on the answers you gave; "
                   "reports only, never makes a judge trusted")
    p.set_defaults(func=cmd_calibrate)

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
