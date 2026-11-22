"""The tool map the setup step hands back: saved as evals/tool_map.yaml and explained in plain words."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .check import CheckResult

FRAMEWORKS = {"claude_agent_sdk": "Claude Agent SDK", "custom": "custom code"}


def map_problems(tool_map: dict[str, Any], check: CheckResult | None) -> list[str]:
    """Where the map and the checked pretend tools disagree."""
    if check is None:
        return []
    checked = {(t.server, t.name): t for t in check.tools}
    mapped = {(t.get("server", ""), t.get("name", "")): t for t in tool_map.get("tools", [])}
    problems = []
    for key in sorted(set(checked) - set(mapped)):
        problems.append(f"{key[0]}/{key[1]} has a pretend version but isn't in the map.")
    for key in sorted(set(mapped) - set(checked)):
        problems.append(f"{key[0]}/{key[1]} is in the map but has no pretend version (check the server name).")
    for key in sorted(set(mapped) & set(checked)):
        if bool(mapped[key].get("changes_real_world")) != checked[key].side_effect:
            problems.append(f"{key[0]}/{key[1]}: the map says changes_real_world="
                            f"{bool(mapped[key].get('changes_real_world'))}, but the pretend tool has "
                            f"side_effect={checked[key].side_effect}. Make them agree.")
    return problems


def tool_map_data(tool_map: dict[str, Any], check: CheckResult | None, spec: str) -> dict[str, Any]:
    checked = {(t.server, t.name) for t in check.tools} if check else set()
    tools = [{
        "name": t.get("name", ""),
        "server": t.get("server", ""),
        "what_it_does": t.get("what_it_does", ""),
        "changes_real_world": bool(t.get("changes_real_world")),
        "faked": (t.get("server", ""), t.get("name", "")) in checked,
        "how_faked": t.get("how_faked", ""),
    } for t in tool_map.get("tools", [])]
    return {
        "agent": tool_map.get("agent_name", ""),
        "framework": FRAMEWORKS.get(tool_map.get("framework", ""), tool_map.get("framework", "")),
        "what_it_does": tool_map.get("summary", ""),
        "entry_point": tool_map.get("entry_point", ""),
        "model": tool_map.get("model", ""),
        "instructions": tool_map.get("instructions_where", ""),
        "target": spec,
        "tools": tools,
        "changed_for_testing": list(tool_map.get("patches", [])),
        "needs_to_run": list(tool_map.get("needs", [])),
        "never_do_proposed": list(tool_map.get("never_do", [])),
        "not_covered": list(tool_map.get("not_covered", [])),
        "questions": list(tool_map.get("questions", [])),
        "check": "passed" if check and check.ok else "failed",
    }


def tool_map_yaml(data: dict[str, Any]) -> str:
    import yaml

    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    header = (f"# Written by `eval-agent setup` on {stamp}. Please check it:\n"
              "# - is every tool that sends, pays, posts, deletes or changes data marked changes_real_world: true?\n"
              "# - is never_do_proposed right? Fix anything that's wrong here; later steps read this file.\n")
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


@dataclass
class SmokeResult:
    ran: bool = False
    status: str = ""
    task: str = ""
    tools: list[str] = field(default_factory=list)
    answer: str = ""
    cost_usd: float | None = None
    error: str = ""
    changed_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.ran and not self.error and not self.changed_files


def _bullets(items: list[str]) -> list[str]:
    return [f"  - {item}" for item in items]


def summary(data: dict[str, Any], written: list[str], intake_cost: float, smoke: SmokeResult | None,
            run_dir: str = "") -> str:
    lines = [f"Your agent: {data['agent']} ({data['framework']}). {data['what_it_does']}".strip()]
    if data["entry_point"]:
        lines.append(f"Where it starts: {data['entry_point']}")
    tools = data["tools"]
    risky = [t for t in tools if t["changes_real_world"]]
    lines += ["", f"Its tools ({len(tools)}; {len(risky)} change things in the real world):"]
    width = max((len(t["name"]) for t in tools), default=0)
    for t in tools:
        effect = "CHANGES THINGS" if t["changes_real_world"] else "only looks"
        faked = "pretend version: yes" if t["faked"] else "pretend version: NO"
        lines.append(f"  {t['name']:<{width}}  {effect:<14}  {faked}  - {t['what_it_does']}")
    if data["changed_for_testing"]:
        lines += ["", "To test it, I change how it runs (your code isn't edited):"] + _bullets(data["changed_for_testing"])
    if data["needs_to_run"]:
        lines += ["", "It needs: " + ", ".join(data["needs_to_run"])]
    if data["never_do_proposed"]:
        lines += ["", "Things it should probably never do (from your code; tell me if I'm wrong):"]
        lines += _bullets(data["never_do_proposed"])
    if data["not_covered"]:
        lines += ["", "What the pretend tools don't cover:"] + _bullets(data["not_covered"])
    if data["questions"]:
        lines += ["", "Questions for you:"] + _bullets(data["questions"])

    lines.append("")
    if smoke is None:
        lines.append("I didn't run your agent yet (--no-smoke).")
    elif not smoke.ran:
        lines.append(f"First run: it couldn't run. {smoke.error}")
    else:
        used = ", ".join(smoke.tools) if smoke.tools else "no tools"
        answer = smoke.answer if len(smoke.answer) <= 200 else smoke.answer[:197] + "..."
        cost = f" (${smoke.cost_usd:.3f})" if smoke.cost_usd is not None else ""
        lines.append(f"First run{cost}: asked \"{smoke.task}\", it used {used} and answered: \"{answer}\"")
        if smoke.error:
            lines.append(f"  But: {smoke.error}")
        if smoke.changed_files:
            lines.append("  Files in your repo changed during the run, so something reached the real code: "
                         + ", ".join(smoke.changed_files[:10]))
        elif not smoke.error:
            lines.append("  Nothing in your repo changed.")
        if smoke.cost_usd is None:
            lines.append("  (I can't measure your agent's own model cost for custom code yet.)")
        if run_dir:
            lines.append(f"  Saved to {run_dir}")
    lines += ["", "I wrote: " + ", ".join(written)]
    smoke_cost = f" + ${smoke.cost_usd:.2f} (your agent's first run)" if smoke and smoke.cost_usd else ""
    lines.append(f"Setup cost: ${intake_cost:.2f} (reading your code){smoke_cost}.")
    return "\n".join(lines)
