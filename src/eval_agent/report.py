"""Turn an analysis into a report anyone can read (docs/prd.md §7.7: written for a 15-year-old).

Short sentences, everyday words, real examples first, counts instead of percentages,
and technical detail only at the end.
"""

from __future__ import annotations

from collections import Counter

from .analyze import Analysis

SEVERITY_LABEL = {
    "critical": "Must never happen",
    "major": "Serious",
    "minor": "Minor",
}


def _case(key: str) -> str:
    return key.rpartition("#")[0]


def _where(keys: list[str], tries: Counter) -> str:
    """Run keys as case names, with "(2 of 3 tries)" when a case ran more than once."""
    counts = Counter(_case(k) for k in keys)
    parts = []
    for case_id in dict.fromkeys(_case(k) for k in keys):
        total = tries.get(case_id, 1)
        parts.append(case_id if total == 1 else f"{case_id} ({counts[case_id]} of {total} tries)")
    return ", ".join(parts)


def analysis_markdown(analysis: Analysis, *, agent_name: str, run_name: str) -> str:
    total = analysis.total_runs
    reviewed = len(analysis.reviews)
    tries = Counter(_case(key) for key in analysis.reviews)
    repeated = any(n > 1 for n in tries.values())
    clean_runs = set(analysis.clean_runs())
    clean_cases = [c for c in tries if all(k in clean_runs for k in analysis.reviews if _case(k) == c)]
    types = analysis.failure_types
    lines = [f"# What went wrong: {agent_name}", "", f"Run `{run_name}`", ""]

    if repeated:
        most = max(tries.values())
        short = [f"I read {reviewed} of {total} runs of your agent: {len(tries)} cases, "
                 f"up to {most} tries each."]
    else:
        short = [f"I read {reviewed} of {total} runs of your agent."]
    if not types:
        short.append("I didn't find any problems.")
    else:
        fine = (f"{len(clean_cases)} of {len(tries)} cases looked fine every time." if repeated
                else f"{len(clean_runs)} looked fine.")
        short.append(f"{fine} I found {len(types)} kind{'s' if len(types) != 1 else ''} of problem.")
        top = types[0]
        short.append(f'The most serious is "{top.name}" ({len(top.traces)} of {reviewed} runs).')
    if analysis.errors:
        short.append(f"I couldn't review {len(analysis.errors)} run{'s' if len(analysis.errors) != 1 else ''}.")
    lines += ["**The short version:** " + " ".join(short), ""]

    if types:
        lines += ["## Problems I found", ""]
        for n, ft in enumerate(types, start=1):
            check = "a code check" if ft.check_type == "code" else "a yes/no question for an AI judge"
            cases_hit = len({_case(k) for k in ft.traces})
            spread = f", in {cases_hit} of {len(tries)} cases" if repeated else ""
            lines += [
                f"### {n}. {ft.name} · {SEVERITY_LABEL[ft.severity]}",
                "",
                f"**What happens:** {ft.definition}",
                "",
                f"**Example:** {ft.example}",
                "",
                f"**Seen in:** {len(ft.traces)} of {reviewed} runs{spread} ({_where(ft.traces, tries)}).",
                "",
                f"**Where it starts:** {ft.starts_at}",
                "",
                f"**Likely cause:** {ft.likely_cause}",
                "",
                f"**How a test could catch it ({check}):** {ft.check_idea}",
                "",
            ]

    if clean_cases:
        heading = "Cases that looked fine every time" if repeated else "Runs that looked fine"
        lines += [f"## {heading}", "", ", ".join(clean_cases) + ".", ""]

    if analysis.not_failures:
        lines += ["## Things I noticed but didn't count as problems", ""]
        lines += [f"- {item}" for item in analysis.not_failures] + [""]

    if analysis.errors:
        lines += ["## Runs I couldn't review", ""]
        lines += [f"- {_where([key], tries)}: {msg}" for key, msg in analysis.errors.items()] + [""]

    sample = (f"{len(tries)} cases with up to {max(tries.values())} tries each is still a small sample."
              if repeated else f"{reviewed} runs is a small sample. A problem I saw once might be rare or common.")
    lines += [
        "## What this doesn't tell you yet",
        "",
        "- These are my suggestions. You'll confirm, rename or drop each one before any of them becomes a test.",
        f"- {sample}",
        "- I only judged what these runs show. Problems your test cases don't trigger won't appear here.",
        "",
        "---",
        "",
        "*For the technical folks:* per-run reviews are in `analysis.json`. "
        f"Model: {', '.join(sorted(analysis.spend.models)) or analysis.model}. "
        f"Analysis cost: about ${analysis.spend.cost_usd:.2f}.",
        "",
    ]
    return "\n".join(lines)
