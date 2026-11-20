"""What happened in one run of the agent under test.

A Trace records every message, tool call and tool result, plus the fake
world before and after, so any score can be traced back to what the agent
actually did. Runs that never produced a usable result (timeouts, API
errors, harness bugs) are RunErrors instead, kept apart so they are never
scored as agent failures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Trace.status values
COMPLETED = "completed"                       # the agent finished its turn with an answer
ASKED_WITHOUT_ANSWER = "asked_without_answer"  # the agent asked something; no scripted reply was left
MAX_TURNS = "max_turns"                       # the agent hit its own max_turns limit
USER_TURN_LIMIT = "user_turn_limit"           # the back-and-forth went on longer than allowed


@dataclass
class ToolCall:
    order: int
    tool: str                      # short name, e.g. "get_order"
    server: str | None             # MCP server, e.g. "shop"; None for built-in or custom-code tools
    input: dict[str, Any]
    output: str = ""
    is_error: bool = False
    id: str = ""
    parent_id: str | None = None   # set when a subagent made the call


@dataclass
class Message:
    role: str   # "user" or "agent"
    text: str


@dataclass
class Trace:
    case_id: str
    rep: int
    task: str
    status: str = COMPLETED
    final_answer: str = ""
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    world_before: dict[str, Any] = field(default_factory=dict)
    world_after: dict[str, Any] = field(default_factory=dict)
    replies_used: int = 0
    cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    model: str = ""
    notes: list[str] = field(default_factory=list)

    def tools_called(self) -> list[str]:
        return [call.tool for call in self.tool_calls]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        data = dict(data)
        data["messages"] = [Message(**m) for m in data.get("messages", [])]
        data["tool_calls"] = [ToolCall(**c) for c in data.get("tool_calls", [])]
        return cls(**data)


@dataclass
class RunError:
    """A run that produced nothing to score. Never counted as an agent failure."""

    case_id: str
    rep: int
    kind: str      # timeout | api_error | cli_not_found | unfaked_tools | fake_tool_crashed | sdk_error | harness_error
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            data = row.to_dict() if hasattr(row, "to_dict") else row
            fh.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def read_traces(path: str | Path) -> list[Trace]:
    with open(path, encoding="utf-8") as fh:
        return [Trace.from_dict(json.loads(line)) for line in fh if line.strip()]
