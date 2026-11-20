"""Read a golden dataset in whatever shape the user has it, and say how it was read.

Only `task` is required to run a case. `expected_outcome` is the second
recommended field: cases without it can run but can't be graded yet, and the
report says so. See docs/prd.md §9 for the recommended format.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Other names people use for the same fields.
ALIASES = {
    "input": "task", "prompt": "task", "question": "task", "request": "task", "user_message": "task",
    "expected": "expected_outcome", "expected_output": "expected_outcome", "expected_answer": "expected_outcome",
    "answer": "expected_outcome", "outcome": "expected_outcome", "ideal": "expected_outcome",
    "must_not_do": "must_not", "should_not": "must_not",
    "replies": "user_replies", "user_reply": "user_replies",
    "state": "starting_state", "initial_state": "starting_state", "world": "starting_state",
    "tools": "expected_tools", "label": "tags", "category": "tags", "labels": "tags",
}
FIELDS = {
    "id", "task", "expected_outcome", "must_not", "expected_tools", "user_replies",
    "starting_state", "tags", "notes", "source", "inject_failures",
}
JSON_FIELDS = {"expected_tools", "starting_state", "inject_failures"}
SHOULD_NOT_ACT_TAGS = {"should-refuse", "should-ask-first", "should-not-act", "refuse", "ask-first"}


@dataclass
class Case:
    id: str
    task: str
    expected_outcome: str = ""
    must_not: str = ""
    expected_tools: list[dict[str, Any]] = field(default_factory=list)
    user_replies: list[str] = field(default_factory=list)
    starting_state: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    source: str = "golden"   # golden | synthetic
    # Not part of the user-facing format: generated cases use it to make a pretend tool fail.
    inject_failures: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetReport:
    path: str
    rows: int = 0
    cases: int = 0
    skipped: list[str] = field(default_factory=list)          # "row 4: no task"
    problems: list[str] = field(default_factory=list)         # kept, but something looked off
    missing_outcome: list[str] = field(default_factory=list)  # case ids
    duplicate_tasks: list[tuple[str, str]] = field(default_factory=list)
    ignored_columns: set[str] = field(default_factory=set)
    tag_counts: Counter = field(default_factory=Counter)
    should_not_act: int = 0
    with_replies: int = 0

    def gaps(self) -> list[str]:
        gaps = []
        if self.cases and self.should_not_act == 0:
            gaps.append("No cases where the agent should refuse or ask first.")
        if self.cases and not self.tag_counts:
            gaps.append("No tags, so I can't tell which kinds of requests are covered.")
        if self.cases and self.cases < 20:
            gaps.append(f"Only {self.cases} cases. At least 20 is a good start.")
        if self.missing_outcome:
            gaps.append(f"{len(self.missing_outcome)} cases have no expected outcome, so they can run but can't be graded.")
        return gaps

    def summary(self) -> str:
        lines = [f"I read {self.cases} cases from {Path(self.path).name}."]
        if self.skipped:
            lines.append(f"I skipped {len(self.skipped)} rows: " + "; ".join(self.skipped[:5]))
        if self.missing_outcome:
            lines.append(f"{self.cases - len(self.missing_outcome)} have an expected outcome; "
                         f"{len(self.missing_outcome)} don't ({', '.join(self.missing_outcome[:5])}).")
        else:
            lines.append("Every case has an expected outcome.")
        lines.append(f"{self.should_not_act} are cases where the agent should refuse or ask first.")
        if self.with_replies:
            lines.append(f"{self.with_replies} include scripted replies for when the agent asks a question.")
        if self.tag_counts:
            tags = ", ".join(f"{tag} ({n})" for tag, n in self.tag_counts.most_common(8))
            lines.append(f"Tags: {tags}.")
        if self.duplicate_tasks:
            pairs = ", ".join(f"{a} = {b}" for a, b in self.duplicate_tasks[:5])
            lines.append(f"These look like duplicates: {pairs}.")
        if self.ignored_columns:
            lines.append(f"I ignored these columns: {', '.join(sorted(self.ignored_columns))}.")
        for problem in self.problems[:5]:
            lines.append(f"Check: {problem}")
        gaps = self.gaps()
        if gaps:
            lines.append("Gaps: " + " ".join(gaps))
        return "\n".join(lines)


def _norm_key(key: str) -> str:
    key = re.sub(r"[\s\-]+", "_", key.strip().lower())
    return ALIASES.get(key, key)


def _norm_tag(tag: str) -> str:
    return re.sub(r"[\s_]+", "-", tag.strip().lower())


def _as_list(value: Any, sep: str) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(sep) if part.strip()]


def _read_rows(path: Path) -> tuple[list[dict[str, Any] | None], list[str]]:
    """Returns raw rows (None for unreadable ones) and reasons for unreadable rows."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".jsonl":
        rows, errors = [], []
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(None)
                errors.append(f"line {n}: not valid JSON ({exc.msg})")
                continue
            if not isinstance(obj, dict):
                rows.append(None)
                errors.append(f"line {n}: expected an object")
                continue
            rows.append(obj)
        return rows, errors
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("cases", data.get("examples", data.get("data")))
        if not isinstance(data, list):
            raise ValueError("A .json dataset must be a list of cases, or an object with a 'cases' list.")
        return [row if isinstance(row, dict) else None for row in data], []
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        return list(csv.DictReader(text.splitlines(), delimiter=delimiter)), []
    if suffix in (".xlsx", ".xls"):
        raise ValueError("Spreadsheet files can't be read directly yet. Save it as CSV and share that.")
    raise ValueError(f"Don't know how to read {path.name}. Use .jsonl, .json or .csv.")


def load_cases(path: str | Path, *, source: str = "golden") -> tuple[list[Case], DatasetReport]:
    path = Path(path)
    report = DatasetReport(path=str(path))
    raw_rows, read_errors = _read_rows(path)
    report.skipped.extend(read_errors)
    cases: list[Case] = []
    seen_ids: set[str] = set()
    seen_tasks: dict[str, str] = {}

    for index, raw in enumerate(raw_rows, start=1):
        report.rows += 1
        if raw is None:
            continue
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if key is None:
                continue
            norm = _norm_key(key)
            if norm in FIELDS:
                row[norm] = value
            elif value not in (None, ""):
                report.ignored_columns.add(key)

        task = str(row.get("task") or "").strip()
        if not task:
            report.skipped.append(f"row {index}: no task")
            continue

        case_id = str(row.get("id") or "").strip() or f"case-{index:03d}"
        if case_id in seen_ids:
            report.problems.append(f"id {case_id!r} is used twice; renamed the second to {case_id}-{index}")
            case_id = f"{case_id}-{index}"
        seen_ids.add(case_id)

        parsed: dict[str, Any] = {}
        for name in JSON_FIELDS:
            value = row.get(name)
            if isinstance(value, str) and value.strip():
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    report.problems.append(f"{case_id}: {name} isn't valid JSON, ignored it")
                    value = None
            parsed[name] = value or None

        expected_tools = parsed["expected_tools"] or []
        if isinstance(expected_tools, list):
            expected_tools = [{"tool": t} if isinstance(t, str) else t for t in expected_tools]
        else:
            report.problems.append(f"{case_id}: expected_tools should be a list, ignored it")
            expected_tools = []

        case = Case(
            id=case_id,
            task=task,
            expected_outcome=str(row.get("expected_outcome") or "").strip(),
            must_not=str(row.get("must_not") or "").strip(),
            expected_tools=expected_tools,
            user_replies=_as_list(row.get("user_replies"), "|"),
            starting_state=parsed["starting_state"] if isinstance(parsed["starting_state"], dict) else {},
            tags=[_norm_tag(t) for t in _as_list(row.get("tags"), ",")],
            notes=str(row.get("notes") or "").strip(),
            source=str(row.get("source") or source).strip(),
            inject_failures=parsed["inject_failures"] if isinstance(parsed["inject_failures"], dict) else {},
        )

        # The same task with different replies, starting state or failures is a different case.
        setup = json.dumps([case.user_replies, case.starting_state, case.inject_failures], sort_keys=True)
        key = re.sub(r"\W+", " ", task.lower()).strip() + setup
        if key in seen_tasks:
            report.duplicate_tasks.append((seen_tasks[key], case.id))
        else:
            seen_tasks[key] = case.id
        if not case.expected_outcome:
            report.missing_outcome.append(case.id)
        if SHOULD_NOT_ACT_TAGS & set(case.tags):
            report.should_not_act += 1
        if case.user_replies:
            report.with_replies += 1
        report.tag_counts.update(case.tags)
        cases.append(case)

    report.cases = len(cases)
    return cases, report
