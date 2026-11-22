"""What a built test is written against: one try of one example, and a verdict.

evals/checks.py (written by `eval-agent build`) looks like this:

    from eval_agent.testkit import Run, check, failed, passed, skip

    @check(1)
    def ask_before_delete(run: Run):
        deletes = run.calls_to("delete_account")
        ...
        return passed()

A check only runs on examples whose expectations (evals/expectations.json) list it;
run.expect holds that example's details for that check.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .dataset import Case
from .trace import Message, ToolCall, Trace

Status = Literal["pass", "fail", "skip", "flag"]   # flag: an untrusted judge's failure, shown but not counted


@dataclass
class Verdict:
    status: Status
    reason: str = ""


def passed(reason: str = "") -> Verdict:
    return Verdict("pass", reason)


def failed(reason: str) -> Verdict:
    return Verdict("fail", reason)


def skip(reason: str = "doesn't apply to this try") -> Verdict:
    return Verdict("skip", reason)


@dataclass
class Call:
    """One tool call. n counts from 1, in the order the agent made them."""

    n: int
    tool: str
    input: dict[str, Any]
    output: str
    is_error: bool

    @property
    def ok(self) -> bool:
        """It worked: no error, and something came back."""
        return not self.is_error and bool(self.output.strip())

    def json(self) -> Any:
        """The output parsed as JSON, or None."""
        try:
            return json.loads(self.output)
        except (TypeError, ValueError):
            return None


@dataclass
class Said:
    """One message: role is "user" (the customer, or the test's scripted replies) or "agent"."""

    role: str
    text: str


class Run:
    def __init__(self, trace: Trace, case: Case | None, expect: dict[str, Any] | None):
        self.trace = trace
        self.case = case
        self.expect: dict[str, Any] = expect or {}
        self.events: list[Call | Said] = []
        for event in trace.timeline():
            if isinstance(event, ToolCall):
                self.events.append(Call(event.order + 1, event.tool, dict(event.input), event.output or "",
                                        bool(event.is_error)))
            elif isinstance(event, Message):
                self.events.append(Said(event.role, event.text))

    # -- what happened ------------------------------------------------------
    @property
    def calls(self) -> list[Call]:
        return [e for e in self.events if isinstance(e, Call)]

    def calls_to(self, *tools: str) -> list[Call]:
        return [c for c in self.calls if c.tool in tools]

    @property
    def messages(self) -> list[Said]:
        return [e for e in self.events if isinstance(e, Said)]

    @property
    def user_messages(self) -> list[str]:
        return [m.text for m in self.messages if m.role == "user"]

    @property
    def agent_messages(self) -> list[str]:
        return [m.text for m in self.messages if m.role == "agent"]

    @property
    def final_answer(self) -> str:
        return self.trace.final_answer or ""

    @property
    def world_before(self) -> dict[str, Any]:
        return self.trace.world_before

    @property
    def world_after(self) -> dict[str, Any]:
        return self.trace.world_after

    # -- order -----------------------------------------------------------------
    def index(self, event: Call | Said) -> int:
        return next(i for i, e in enumerate(self.events) if e is event)

    def before(self, event: Call | Said) -> list[Call | Said]:
        return self.events[: self.index(event)]

    def after(self, event: Call | Said) -> list[Call | Said]:
        return self.events[self.index(event) + 1:]

    def user_messages_before(self, event: Call | Said) -> list[str]:
        return [e.text for e in self.before(event) if isinstance(e, Said) and e.role == "user"]

    def agent_messages_before(self, event: Call | Said) -> list[str]:
        return [e.text for e in self.before(event) if isinstance(e, Said) and e.role == "agent"]

    # -- the example -------------------------------------------------------------
    @property
    def task(self) -> str:
        return self.trace.task


# -- registering and loading checks ------------------------------------------------

CheckFn = Callable[[Run], Verdict]
_registry: dict[int, CheckFn] = {}


def check(number: int) -> Callable[[CheckFn], CheckFn]:
    """Mark a function as the code part of plan check `number`."""

    def register(fn: CheckFn) -> CheckFn:
        if number in _registry:
            raise ValueError(f"Two functions are registered for check {number}.")
        fn.check_number = number  # type: ignore[attr-defined]
        _registry[number] = fn
        return fn

    return register


def load_checks(path: Path) -> dict[int, CheckFn]:
    """Import evals/checks.py and return its checks by number."""
    _registry.clear()
    name = f"_built_checks_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Can't load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    found = dict(_registry)
    _registry.clear()
    return found


def run_check(fn: CheckFn, run: Run) -> Verdict:
    """A check that crashes or returns something odd is a broken test, not a failing agent."""
    try:
        verdict = fn(run)
    except Exception as exc:
        return Verdict("skip", f"BROKEN TEST: {type(exc).__name__}: {exc}")
    if not isinstance(verdict, Verdict):
        return Verdict("skip", f"BROKEN TEST: returned {type(verdict).__name__}, not a verdict")
    return verdict


# -- expectations: which checks apply to which example ---------------------------------

def expectations_for(expectations: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Merge the "*" entry (every example) with this example's own. A null value removes a check."""
    cases = expectations.get("cases", {})
    merged = dict(cases.get("*", {}))
    merged.update(cases.get(case_id, {}))
    return {k: v for k, v in merged.items() if v is not None}
