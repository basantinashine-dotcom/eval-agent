"""Run the built tests from start to finish: what evals/run.py (and `eval-agent run`) does.

Runs the agent on every example (3 tries each by default, FR-40), scores each try with the code
checks and the judges (a judge that isn't trusted can only flag a must-never-fail check, FR-39),
and reports in plain words, including which checks gave mixed results across tries.
Exits 1 if a must-never-fail check failed, so CI can block the change (FR-45).

Everything it needs is in the tests folder: plan.json, expectations.json, checks.py,
judges.json, calibration.json and run_config.json (written by `eval-agent export`).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPS = 3


def _load_dotenv(path: Path) -> list[str]:
    """Keys from a .env file, without overriding ones already set. Returns names, never values."""
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def repeats_text(results: dict[str, dict[int, Any]], names: dict[int, tuple[str, str]]) -> str:
    """Per check: how often each example passed across its tries (FR-40), in plain words."""
    lines = []
    for number, (name, _level) in sorted(names.items()):
        by_case: dict[str, list[str]] = defaultdict(list)
        for key, verdicts in results.items():
            v = verdicts.get(number)
            if v is not None and v.status != "skip":
                # A flag is still "something went wrong", just not trusted to fail the run.
                by_case[key.rsplit("/", 1)[-1].rpartition("#")[0]].append("pass" if v.status == "pass" else "fail")
        if not by_case:
            continue
        always = sum(1 for s in by_case.values() if all(x == "pass" for x in s))
        never = sum(1 for s in by_case.values() if all(x == "fail" for x in s))
        mixed = [case for case, s in by_case.items() if len(set(s)) > 1]
        text = f"{name}: passed every time in {always} of {len(by_case)} examples"
        if never:
            text += f"; failed every time in {never}"
        if mixed:
            text += f"; mixed in {len(mixed)} ({', '.join(sorted(mixed)[:3])})"
        lines.append(text + ".")
    if not lines:
        return ""
    return ("Across the repeated tries (your agent doesn't always do the same thing twice):\n"
            + "\n".join(f"  {line}" for line in lines))


def main(argv: list[str] | None = None, *, evals_dir: Path | None = None) -> int:
    previous = os.getcwd()
    try:
        return _main(argv, evals_dir)
    finally:
        os.chdir(previous)


def _main(argv: list[str] | None, evals_dir: Path | None) -> int:
    from .analyze import Spend
    from .calibrate import trust
    from .dataset import load_cases
    from .plan import load_state
    from .runner import SdkTarget, auth_warnings, run_cases
    from .score import (
        JUDGE_MODEL,
        Item,
        JudgeSpec,
        apply_trust,
        combine,
        judge_estimate,
        judge_jobs,
        score_code,
        score_judges,
        scores_text,
        tools_text,
    )
    from .testkit import expectations_for
    from .trace import write_jsonl

    parser = argparse.ArgumentParser(prog="run.py", description="Run your agent's tests and report the results.")
    parser.add_argument("--evals", type=Path, default=evals_dir, help=argparse.SUPPRESS if evals_dir else
                        "the tests folder (default: evals)")
    parser.add_argument("--reps", type=int, default=REPS, help=f"tries per example (default {REPS})")
    parser.add_argument("--quick", action="store_true", help="one try per example, for a fast check")
    parser.add_argument("--only", help="comma-separated example ids")
    parser.add_argument("--no-judges", action="store_true", help="skip the judge questions (code checks only)")
    parser.add_argument("--judge-model", help="model for judge questions (default: the one they were checked with)")
    parser.add_argument("--max-cost", type=float, default=5.0, help="stop if the estimate is above this, in $")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300, help="seconds per try")
    args = parser.parse_args(argv)

    evals = (args.evals or Path("evals")).resolve()
    config_path = evals / "run_config.json"
    if not config_path.exists():
        print(f"There's no run_config.json in {evals}. Run `eval-agent export --evals {evals}` first.")
        return 2
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = (evals / config.get("root", "..")).resolve()
    os.chdir(root)  # the agent's code and the examples are found from its repo's folder
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    loaded = _load_dotenv(root / ".env")
    if loaded:
        print(f"Loaded {', '.join(loaded)} from .env")

    state = load_state(evals / "plan.json")
    if not state.is_approved:
        print("Note: the test plan isn't approved, so these tests may change.")
    checks = state.plan().checks
    names = {n: (c.name, c.level) for n, c in enumerate(checks, start=1)}
    must = {n for n, c in enumerate(checks, start=1) if c.level == "must_never_fail"}
    expectations = json.loads((evals / "expectations.json").read_text(encoding="utf-8"))
    judges = [] if args.no_judges else JudgeSpec.load_all(evals / "judges.json")
    calibration = evals / "calibration.json"
    checked_with = json.loads(calibration.read_text(encoding="utf-8")).get("judge_model") if calibration.exists() else None
    judge_model = args.judge_model or checked_with or JUDGE_MODEL

    cases, seen = [], set()
    for name in config.get("cases", []):
        path = Path(name)
        if not path.exists():
            print(f"Can't find the examples file {name}; skipping it.")
            continue
        for case in load_cases(path)[0]:
            if case.id not in seen and expectations_for(expectations, case.id):  # at least one check applies
                cases.append(case)
                seen.add(case.id)
    if args.only:
        wanted = {i.strip() for i in args.only.split(",")}
        cases = [c for c in cases if c.id in wanted]
    if not cases:
        print("There are no examples to run.")
        return 2

    module_name, _, attr = config["target"].partition(":")
    target = getattr(importlib.import_module(module_name), attr or "make_target")()
    reps = 1 if args.quick else max(1, args.reps)
    tries = len(cases) * reps
    per_try = state.plan_facts().cost_per_try()
    agent_cost = tries * per_try if per_try is not None else None
    judge_cost = judge_estimate(tries * len(judges), judge_model) if judges else 0.0
    estimate = (agent_cost or 0.0) + judge_cost
    print(f"Running {target.name} on {len(cases)} example{'' if len(cases) == 1 else 's'}, "
          f"{reps} {'try' if reps == 1 else 'tries'} each ({tries} {'try' if tries == 1 else 'tries'}).")
    print(f"Estimated cost: up to about ${estimate:.2f}"
          + (" (plus your agent's own model calls, which I can't estimate)" if agent_cost is None else "") + ".")
    if estimate > args.max_cost:
        print(f"That's more than --max-cost (${args.max_cost:.2f}). Raise it, or use --quick or --only.")
        return 2
    for warning in auth_warnings() if isinstance(target, SdkTarget) else []:
        print(f"Warning: {warning}")

    traces, errors = asyncio.run(run_cases(target, cases, reps=reps, concurrency=args.concurrency,
                                           timeout_s=args.timeout))
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = evals / "runs" / stamp
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "traces.jsonl", traces)
    write_jsonl(out / "errors.jsonl", errors)
    by_id = {c.id: c for c in cases}
    items = [Item(f"{stamp}/{t.case_id}#{t.rep}", t, by_id.get(t.case_id)) for t in traces]

    code = score_code(evals, items, expectations)
    if code.problems:
        print("The tests can't run:\n" + "\n".join(f"  - {p}" for p in code.problems))
        return 2
    spend = Spend()
    judged: dict[str, Any] = {}
    jobs = judge_jobs(judges, items, expectations, code.results, set(code.registered))
    if jobs:
        judged, spend = asyncio.run(score_judges(jobs, model=judge_model,
                                                 tools=tools_text(state.plan_facts().tools)))
    results = combine(code.results, apply_trust(judged, trust(evals, judges, judge_model), must))

    report = scores_text(results, names, stamp)
    repeats = repeats_text(results, names) if reps > 1 else ""
    agent_spent = sum(t.cost_usd or 0 for t in traces)
    lines = [report]
    if repeats:
        lines += ["", repeats]
    if errors:
        lines += ["", f"{len(errors)} tries couldn't run (not counted as failures): "
                  + "; ".join(f"{e.case_id} [{e.kind}]" for e in errors[:3])]
    lines += ["", f"Cost: ${agent_spent:.2f} for your agent, ${spend.cost_usd:.2f} for judges."]
    text = "\n".join(lines)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    (out / "scores.json").write_text(json.dumps(
        {key: {str(n): [v.status, v.reason] for n, v in verdicts.items()} for key, verdicts in results.items()},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print()
    print(text)
    print(f"\nSaved to {out}")
    failed_must = any(v.status == "fail" and n in must for verdicts in results.values() for n, v in verdicts.items())
    return 1 if failed_must else 0
