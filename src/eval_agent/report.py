"""Turn an analysis into a report anyone can read (docs/prd.md §7.7: written for a 15-year-old).

Short sentences, everyday words, real examples first, counts instead of percentages,
and technical detail only at the end.
"""

from __future__ import annotations

from .analyze import Analysis

SEVERITY_LABEL = {
    "critical": "Must never happen",
    "major": "Serious",
    "minor": "Minor",
}


def _runs(keys: list[str]) -> str:
    """"delete-confirm-yes#0" -> "delete-confirm-yes"; adds "(try 2)" for repeats."""
    out = []
    for key in keys:
        case_id, _, rep = key.rpartition("#")
        out.append(case_id if rep == "0" else f"{case_id} (try {int(rep) + 1})")
    return ", ".join(out)


def analysis_markdown(analysis: Analysis, *, agent_name: str, run_name: str) -> str:
    total = analysis.total_runs
    reviewed = len(analysis.reviews)
    clean = analysis.clean_runs()
    types = analysis.failure_types
    lines = [f"# What went wrong: {agent_name}", "", f"Run `{run_name}`", ""]

    short = [f"I read {reviewed} of {total} runs of your agent."]
    if not types:
        short.append("I didn't find any problems.")
    else:
        short.append(f"{len(clean)} looked fine. I found {len(types)} kind{'s' if len(types) != 1 else ''} of problem.")
        top = types[0]
        short.append(f'The most serious is "{top.name}" ({len(top.traces)} of {reviewed} runs).')
    if analysis.errors:
        short.append(f"I couldn't review {len(analysis.errors)} run{'s' if len(analysis.errors) != 1 else ''}.")
    lines += ["**The short version:** " + " ".join(short), ""]

    if types:
        lines += ["## Problems I found", ""]
        for n, ft in enumerate(types, start=1):
            check = "a code check" if ft.check_type == "code" else "a yes/no question for an AI judge"
            lines += [
                f"### {n}. {ft.name} · {SEVERITY_LABEL[ft.severity]}",
                "",
                f"**What happens:** {ft.definition}",
                "",
                f"**Example:** {ft.example}",
                "",
                f"**Seen in:** {len(ft.traces)} of {reviewed} runs ({_runs(ft.traces)}).",
                "",
                f"**Where it starts:** {ft.starts_at}",
                "",
                f"**Likely cause:** {ft.likely_cause}",
                "",
                f"**How a test could catch it ({check}):** {ft.check_idea}",
                "",
            ]

    if clean:
        lines += ["## Runs that looked fine", "", _runs(clean) + ".", ""]

    if analysis.not_failures:
        lines += ["## Things I noticed but didn't count as problems", ""]
        lines += [f"- {item}" for item in analysis.not_failures] + [""]

    if analysis.errors:
        lines += ["## Runs I couldn't review", ""]
        lines += [f"- {_runs([key])}: {msg}" for key, msg in analysis.errors.items()] + [""]

    lines += [
        "## What this doesn't tell you yet",
        "",
        "- These are my suggestions. You'll confirm, rename or drop each one before any of them becomes a test.",
        f"- {reviewed} runs is a small sample. A problem I saw once might be rare or common.",
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
