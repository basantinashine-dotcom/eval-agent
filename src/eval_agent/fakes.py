"""Pretend versions of an agent's tools.

Every run gets a fresh FakeWorld built from the case's starting state, so one
run can't leak into another. Fake tools read and change that world, record
every call, and can be told to fail so the agent's error handling gets tested.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

FAILURE_MESSAGES = {
    "error": "Internal error: the service could not process this request.",
    "timeout": "Request timed out after 30 seconds.",
    "empty": "",
}


class ToolError(Exception):
    """Raise inside a fake handler to send an error result back to the agent."""


Handler = Callable[[dict[str, Any], "FakeWorld"], Any]


@dataclass
class FakeTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    side_effect: bool = False   # sends, pays, posts or deletes something in the real world


def fake_like(real_tool: Any, handler: Handler, *, side_effect: bool = False) -> FakeTool:
    """A fake with exactly the same name, description and schema as a real SDK tool."""
    return FakeTool(real_tool.name, real_tool.description, real_tool.input_schema, handler, side_effect)


@dataclass
class RecordedCall:
    tool: str
    input: dict[str, Any]
    output: str
    is_error: bool
    injected_failure: str | None = None


class FakeWorld:
    """The state pretend tools act on, e.g. {"orders": [...], "emails": []}."""

    def __init__(self, state: dict[str, Any] | None = None):
        self.state: dict[str, Any] = deepcopy(state) if state else {}
        self.calls: list[RecordedCall] = []


@dataclass
class _FailurePlan:
    mode: str
    remaining: int   # -1 = every call


def _parse_failure(tool: str, spec: Any) -> _FailurePlan:
    """"error" fails once; {"mode": "error", "times": 2} twice; "times": "all" always."""
    if isinstance(spec, str):
        mode, times = spec, 1
    elif isinstance(spec, dict):
        mode, times = spec.get("mode", "error"), spec.get("times", 1)
    else:
        raise ValueError(f"Failure for {tool!r} must be a mode string or a dict, got {spec!r}")
    if mode not in FAILURE_MESSAGES:
        raise ValueError(f"Unknown failure mode {mode!r} for {tool!r}; use one of {sorted(FAILURE_MESSAGES)}")
    return _FailurePlan(mode, -1 if times == "all" else int(times))


class Toolbox:
    def __init__(self, tools: list[FakeTool], world: FakeWorld, failures: dict[str, Any] | None = None):
        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate fake tool names: {names}")
        self.tools = {t.name: t for t in tools}
        self.world = world
        self.crashes: list[str] = []   # fake handlers that raised something other than ToolError
        failures = failures or {}
        self._failures = {name: _parse_failure(name, spec) for name, spec in failures.items() if name in self.tools}

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run a fake tool. Returns (result text, is_error)."""
        tool = self.tools[name]
        plan = self._failures.get(name)
        if plan and plan.remaining != 0:
            if plan.remaining > 0:
                plan.remaining -= 1
            text, is_error = FAILURE_MESSAGES[plan.mode], plan.mode != "empty"
            self.world.calls.append(RecordedCall(name, dict(args), text, is_error, plan.mode))
            return text, is_error
        try:
            result = tool.handler(args, self.world)
            text = result if isinstance(result, str) else json.dumps(result, sort_keys=True, default=str)
            is_error = False
        except ToolError as exc:
            text, is_error = str(exc), True
        except Exception as exc:  # a bug in the fake, not the agent
            self.crashes.append(f"{name}: {type(exc).__name__}: {exc}")
            text, is_error = f"Internal error in pretend tool {name}.", True
        self.world.calls.append(RecordedCall(name, dict(args), text, is_error))
        return text, is_error

    def sdk_tools(self) -> list[Any]:
        """The fakes as Claude Agent SDK tools, with the same names and schemas."""
        from claude_agent_sdk import tool

        def wrap(fake: FakeTool) -> Any:
            @tool(fake.name, fake.description, fake.input_schema)
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                text, is_error = self.call(fake.name, args)
                result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
                if is_error:
                    result["is_error"] = True
                return result

            return handler

        return [wrap(t) for t in self.tools.values()]

    def sdk_server(self, server_name: str) -> Any:
        from claude_agent_sdk import create_sdk_mcp_server

        return create_sdk_mcp_server(server_name, tools=self.sdk_tools())
