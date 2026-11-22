"""Render a plan as evals/EVAL_PLAN.md and as one self-contained HTML file (FR-24, FR-25).

Both come from the same list of blocks, so the two never disagree. The section order and
headings are fixed here (FR-24); the words inside come from the plan.
"""

from __future__ import annotations

import html
from collections import Counter
from typing import Any

from .plan import (
    JUDGE_COST_PER_TRY,
    LEVEL_LABEL,
    REPS,
    SEVERITY_LABEL,
    PlanState,
    run_cost,
    run_intro,
    seen_text,
    should_not_act,
)

# Blocks: ("h1"|"h2"|"p"|"meta", text) · ("lead", label, text) · ("ul"|"ol", [str | (label, text)])
#         ("check", number, PlanCheck) · ("code", [str])
Block = tuple[Any, ...]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _money(value: float) -> str:
    return f"${value:.2f}" if value >= 0.01 else "less than $0.01"


def _date(stamp: str) -> str:
    return stamp[:10]


def blocks(state: PlanState) -> list[Block]:
    draft, facts, report = state.plan(), state.plan_facts(), state.check_report()
    out: list[Block] = [("h1", f"Test plan for {facts.agent_name}")]
    if state.is_approved:
        out.append(("meta", f"Version {state.version} · Approved on {_date(state.approved['at'])}."))
    else:
        out.append(("meta", f"Version {state.version} · Draft, not approved yet. Nothing gets built until you "
                            "approve it."))

    out += [("h2", "1. The short version"), ("p", draft.short_version)]

    out += [("h2", "2. What your agent does (as I understand it)"), ("p", draft.what_it_does)]
    if facts.tools:
        real = sum(1 for t in facts.tools if t["changes_real_world"])
        out.append(("p", f"It has {len(facts.tools)} tools. {real} of them do something real, like sending, "
                         "paying or deleting."))
        out.append(("ul", [(("Does something real:" if t["changes_real_world"] else "Only looks:"),
                            _cap(t["what_it_does"]).rstrip(".") + ".") for t in facts.tools]))

    out.append(("h2", "3. What I tried and what went wrong"))
    stories = {(s.run, s.name): s.story for s in draft.failure_stories}
    for run_no, run in enumerate(facts.runs, start=1):
        out.append(("p", run_intro(run)))
        if not run.failures:
            out.append(("p", "I didn't find any problems in these tries."))
        items = []
        for f in run.failures:
            story = stories.get((run_no, f.name))
            # The writer's plain retelling when there is one; the analyzer's own words otherwise.
            text = (f"{story} Seen in {seen_text(f, run)}." if story
                    else f"{f.definition} Seen in {seen_text(f, run)}. Example: {f.example}")
            items.append((f"{f.name} ({SEVERITY_LABEL.get(f.severity, f.severity)}).", text))
        if items:
            out.append(("ul", items))
        out.append(("p", f"{len(run.clean_cases)} of {run.cases} examples looked fine every time."
                    if run.reps > 1 else f"{len(run.clean_cases)} of {run.cases} examples looked fine."))

    must = sum(1 for c in draft.checks if c.level == "must_never_fail")
    out += [("h2", "4. What I'll test"),
            ("p", f"{len(draft.checks)} checks. {must} of them must never fail.")]
    out += [("check", n, check) for n, check in enumerate(draft.checks, start=1)]

    cases = facts.cases
    yours = sum(1 for c in cases if c.get("source") == "golden")
    out.append(("h2", "5. The examples I'll test with"))
    out.append(("p", f"You have {len(cases)} examples so far: {yours} of yours and {len(cases) - yours} I made "
                     f"up. {sum(1 for c in cases if should_not_act(c))} are ones where your agent should refuse "
                     "or ask first."))
    tags = Counter(t for c in cases for t in (c.get("tags") or []))
    if tags:
        out.append(("lead", "What kinds:", ", ".join(f"{t} ({n})" for t, n in tags.most_common(10)) + "."))
    if draft.cases_to_add:
        out += [("p", "I'll add these, so every check that must never fail has at least 5 examples:"),
                ("ul", list(draft.cases_to_add))]
    samples = _samples(cases)
    if samples:
        out += [("p", f"{len(samples)} examples, so you can see what they look like:"),
                ("ul", [f"“{c['task']}” Done right: {c.get('expected_outcome') or '(not written yet)'}"
                        for c in samples])]

    out += [("h2", "6. What I'll pretend"), ("ul", list(draft.pretend) or ["Nothing."])]

    out += [("h2", "7. How I'll make sure the tests work"),
            ("p", "I'll break a copy of your agent on purpose, and check that the right test notices:"),
            ("ul", list(draft.breaks)), ("lead", "What I need from you:", draft.labeling)]

    out.append(("h2", "8. Cost and time"))
    cost = run_cost(facts, draft)
    lines = []
    if cost.full_cost is not None:
        time = f" and about {cost.full_minutes:g} minutes" if cost.full_minutes is not None else ""
        lines.append(f"A full run: {cost.cases} examples, {REPS} tries each ({cost.tries} tries). "
                     f"About {_money(cost.full_cost)}{time}.")
        quick_time = f" and {cost.quick_minutes:g} minutes" if cost.quick_minutes is not None else ""
        lines.append(f"A quick run on every change (each example once): about {_money(cost.quick_cost)}"
                     f"{quick_time}.")
        lines.append(f"Your agent costs about {_money(cost.agent_per_try)} per try. {cost.judges} checks use a "
                     f"judge, which adds about {_money(JUDGE_COST_PER_TRY)} per check per try.")
    else:
        lines.append(f"A full run: {cost.cases} examples, {REPS} tries each ({cost.tries} tries). I can't "
                     "measure your agent's own cost yet, so I can't price it.")
        lines.append(f"{cost.judges} checks use a judge, which adds about {_money(JUDGE_COST_PER_TRY)} per check "
                     "per try.")
    lines.append(f"Writing this plan cost {_money(state.cost_usd)} so far.")
    out.append(("ul", lines))

    out += [("h2", "9. What I'm not testing, and why"), ("ul", list(draft.not_testing) or ["Nothing I know of."])]
    out += [("h2", "10. Questions for you"), ("ol", list(draft.questions) or ["None right now."])]
    out += [("h2", "11. Words you might not know"),
            ("ul", [(g.word + ":", g.meaning) for g in draft.glossary] or ["None."])]

    out.append(("h2", "12. For the technical folks"))
    tech = [f"Target: `{facts.target}`"]
    tech += [f"Run `{r.name}`: `{r.cases_file}`, {r.reps} {'try' if r.reps == 1 else 'tries'} per case"
             for r in facts.runs]
    out.append(("ul", tech))
    out += [("p", "Checks:"),
            ("ul", [f"Check {n}, {c.name} [{c.check_type}]: {c.technical}" for n, c in enumerate(draft.checks, 1)])]
    if facts.tools:
        out += [("p", "Tools:"), ("ul", [f"`{t['name']}`{' (side effect)' if t['changes_real_world'] else ''}: "
                                         f"{t['how_faked'] or t['what_it_does']}" for t in facts.tools])]
    if facts.patches:
        out += [("p", "Changed for testing:"), ("ul", list(facts.patches))]
    per_case = []
    for r in facts.runs:
        for f in r.failures:
            where = ", ".join(f"`{cid}` ({a}/{b})" for cid, (a, b) in f.per_case.items())
            per_case.append(f"{r.name} · {f.name}: {where}")
    if per_case:
        out += [("p", "Failures by example (tries with it / tries):"), ("ul", per_case)]
    if draft.technical_notes:
        out += [("p", "Notes:"), ("ul", list(draft.technical_notes))]
    check_line = (f"Plan check: reading grade {report.grade:g} (target 9 or below); the 15-year-old judge "
                  f"understood {report.understood} of {report.judged} checks; {report.rounds} rewrite "
                  f"round{'' if report.rounds == 1 else 's'}.")
    out += [("p", check_line)]
    if report.issues:
        out += [("p", "Still flagged after rewriting:"), ("ul", list(report.issues))]
    out.append(("p", "Files, side by side: `EVAL_PLAN.md`, `EVAL_PLAN.html`, and `plan.json` (every version, "
                     "the changelog and the approval)."))

    out.append(("h2", "13. Changes"))
    changes = []
    for entry in state.changelog:
        when = _date(entry["at"])
        if entry.get("approved"):
            changes.append(f"Version {entry['version']} approved on {when}.")
        elif entry.get("feedback"):
            changes.append(f"Version {entry['version']} ({when}). You said: “{entry['feedback']}” "
                           f"What changed: {' '.join(_sentence(c) for c in entry['changes'])}")
        else:
            changes.append(f"Version {entry['version']} ({when}): {' '.join(_sentence(c) for c in entry['changes'])}")
    out.append(("ul", changes))
    return out


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _samples(cases: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
    """Up to n examples covering different kinds: theirs first, at least one where the agent shouldn't act."""
    ordered = sorted(cases, key=lambda c: c.get("source") != "golden")
    picked: list[dict[str, Any]] = []
    stop = next((c for c in ordered if should_not_act(c)), None)
    if stop:
        picked.append(stop)
    seen_tags = {t for c in picked for t in (c.get("tags") or [])}
    for c in ordered:
        if len(picked) >= n:
            break
        if c in picked:
            continue
        tags = set(c.get("tags") or [])
        if not tags or tags - seen_tags:
            picked.append(c)
            seen_tags |= tags
    for c in ordered:
        if len(picked) >= n:
            break
        if c not in picked:
            picked.append(c)
    return picked


# -- Markdown ----------------------------------------------------------------------------------------------

def _md_item(item: Any) -> str:
    return f"**{item[0]}** {item[1]}" if isinstance(item, tuple) else str(item)


def markdown(state: PlanState) -> str:
    lines: list[str] = []
    for block in blocks(state):
        kind = block[0]
        if kind == "h1":
            lines += [f"# {block[1]}", ""]
        elif kind == "meta":
            lines += [f"*{block[1]}*", ""]
        elif kind == "h2":
            lines += [f"## {block[1]}", ""]
        elif kind == "p":
            lines += [block[1], ""]
        elif kind == "lead":
            lines += [f"**{block[1]}** {block[2]}", ""]
        elif kind in ("ul", "ol"):
            for n, item in enumerate(block[1], start=1):
                lines.append(f"{n}. {_md_item(item)}" if kind == "ol" else f"- {_md_item(item)}")
            lines.append("")
        elif kind == "check":
            n, c = block[1], block[2]
            lines += [f"> **Check {n}: {c.name}** · {LEVEL_LABEL[c.level]}", ">",
                      f"> **What we check:** {c.what_we_check}", ">",
                      f"> **Why it matters:** {c.why_it_matters}", ">",
                      f"> **What we saw:** {c.what_we_saw}", ">",
                      f"> **How we test it:** {c.how_we_test}", ""]
    return "\n".join(lines).rstrip() + "\n"


# -- HTML: one file, no server, no account, works offline ----------------------------------------------------

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#5d5c57;--line:#e3e0d8;--card:#ffffff;--accent:#2b5d8a;
--must:#9b2c2c;--must-bg:#fbeaea;--should:#5b6b2a;--should-bg:#eef2e1}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a9a79f;--line:#34332f;
--card:#1f1f1c;--accent:#8db8de;--must:#f0a3a3;--must-bg:#3a1f1f;--should:#c7d69a;--should-bg:#2a3019}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 Georgia,'Iowan Old Style',serif}
main{max-width:46rem;margin:0 auto;padding:2.5rem 1rem 4rem}
h1{font:700 2rem/1.2 system-ui,-apple-system,'Segoe UI',sans-serif;margin:0 0 .4rem}
h2{font:650 1.2rem/1.3 system-ui,-apple-system,'Segoe UI',sans-serif;margin:2.4rem 0 .6rem;
padding-top:1rem;border-top:1px solid var(--line)}
.meta{color:var(--muted);font-style:italic;margin:0 0 1rem}
p{margin:.5rem 0}
ul,ol{padding-left:1.3rem;margin:.5rem 0}
li{margin:.3rem 0}
code{font:.85em ui-monospace,Consolas,monospace;background:var(--line);padding:.05em .3em;border-radius:3px}
.check{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem 1.2rem;margin:1rem 0}
.check h3{font:650 1.05rem/1.3 system-ui,-apple-system,'Segoe UI',sans-serif;margin:0 0 .6rem}
.badge{display:inline-block;font:600 .75rem/1 system-ui,sans-serif;padding:.3em .6em;border-radius:99px;
margin-left:.4rem;vertical-align:middle}
.must{color:var(--must);background:var(--must-bg)}.should{color:var(--should);background:var(--should-bg)}
.check p{margin:.4rem 0}
@media print{body{background:#fff;color:#000}.check{break-inside:avoid}h2{break-after:avoid}}
"""


def _inline(text: str) -> str:
    """Escape, then turn `code` into <code>."""
    parts = html.escape(str(text)).split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def _html_item(item: Any) -> str:
    if isinstance(item, tuple):
        return f"<strong>{_inline(item[0])}</strong> {_inline(item[1])}"
    return _inline(item)


def to_html(state: PlanState) -> str:
    body: list[str] = []
    title = "Test plan"
    for block in blocks(state):
        kind = block[0]
        if kind == "h1":
            title = block[1]
            body.append(f"<h1>{_inline(block[1])}</h1>")
        elif kind == "meta":
            body.append(f'<p class="meta">{_inline(block[1])}</p>')
        elif kind == "h2":
            body.append(f"<h2>{_inline(block[1])}</h2>")
        elif kind == "p":
            body.append(f"<p>{_inline(block[1])}</p>")
        elif kind == "lead":
            body.append(f"<p><strong>{_inline(block[1])}</strong> {_inline(block[2])}</p>")
        elif kind in ("ul", "ol"):
            items = "".join(f"<li>{_html_item(item)}</li>" for item in block[1])
            body.append(f"<{kind}>{items}</{kind}>")
        elif kind == "check":
            n, c = block[1], block[2]
            badge = "must" if c.level == "must_never_fail" else "should"
            body.append(
                f'<section class="check"><h3>Check {n}: {_inline(c.name)}'
                f'<span class="badge {badge}">{LEVEL_LABEL[c.level]}</span></h3>'
                f"<p><strong>What we check:</strong> {_inline(c.what_we_check)}</p>"
                f"<p><strong>Why it matters:</strong> {_inline(c.why_it_matters)}</p>"
                f"<p><strong>What we saw:</strong> {_inline(c.what_we_saw)}</p>"
                f"<p><strong>How we test it:</strong> {_inline(c.how_we_test)}</p></section>")
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(body)}</main></body></html>\n")
