"""Run the agent under test on cases, with pretend versions of its risky tools.

Two kinds of target:
- SdkTarget: an agent built on the Claude Agent SDK. Its MCP tool servers are
  swapped for fakes with identical tool names, and every message and tool call
  is read from the SDK's message stream.
- CallableTarget: custom code. A function gets the case and a Toolbox of fakes;
  the tool calls come from what the fakes recorded.

Anything that stops a run from producing a usable result becomes a RunError in
errors.jsonl, never a failed trace.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .dataset import Case
from .fakes import FakeTool, FakeWorld, Toolbox
from .trace import (
    ASKED_WITHOUT_ANSWER,
    COMPLETED,
    MAX_TURNS,
    USER_TURN_LIMIT,
    Message,
    RunError,
    ToolCall,
    Trace,
    write_jsonl,
)

# These errors mean no case can run; stop instead of failing every case the same way.
FATAL_KINDS = {"cli_not_found", "unfaked_tools", "unsupported"}


class HarnessError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class Target(Protocol):
    name: str

    async def run(self, case: Case, rep: int) -> Trace: ...


def looks_like_question(text: str) -> bool:
    """Whether the agent ended its turn by asking the user something.

    A simple rule for now: the last sentence ends with a question mark.
    """
    text = text.strip()
    if not text:
        return False
    last_paragraph = [p for p in re.split(r"\n\s*\n", text) if p.strip()][-1].strip()
    last_paragraph = re.sub(r"[\s*_)\]\"'`]+$", "", last_paragraph)
    return last_paragraph.endswith("?")


def split_tool_name(full_name: str) -> tuple[str | None, str]:
    """"mcp__shop__get_order" -> ("shop", "get_order"); "Bash" -> (None, "Bash")."""
    if full_name.startswith("mcp__"):
        parts = full_name.split("__", 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    return None, full_name


def result_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item, default=str))
        return "\n".join(parts)
    return json.dumps(content, default=str)


def _check_failures(case: Case, toolboxes: dict[str, Toolbox]) -> None:
    known = {name for box in toolboxes.values() for name in box.tools}
    unknown = set(case.inject_failures) - known
    if unknown:
        raise HarnessError("harness_error", f"Case {case.id} makes unknown tools fail: {sorted(unknown)}")


def _default_client(options: Any, toolboxes: dict[str, Toolbox]) -> Any:
    from claude_agent_sdk import ClaudeSDKClient

    return ClaudeSDKClient(options=options)


@dataclass
class SdkTarget:
    """An agent built on the Claude Agent SDK.

    options_factory: returns the agent's own ClaudeAgentOptions.
    fakes: MCP server name -> function returning that server's fake tools.
    real_servers_ok: servers the user said are safe to call for real.
    """

    name: str
    options_factory: Callable[[], Any]
    fakes: dict[str, Callable[[], list[FakeTool]]]
    real_servers_ok: frozenset[str] = frozenset()
    max_user_turns: int = 5
    client_factory: Callable[[Any, dict[str, Toolbox]], Any] = field(default=_default_client)

    def _swap_servers(self, options: Any, toolboxes: dict[str, Toolbox]) -> None:
        servers = options.mcp_servers
        if not isinstance(servers, dict):
            raise HarnessError("unsupported", "MCP servers loaded from a config file aren't supported yet.")
        unfaked = [name for name in servers if name not in toolboxes and name not in self.real_servers_ok]
        if unfaked:
            raise HarnessError(
                "unfaked_tools",
                f"These tool servers have no pretend version, so running them could do real things: {unfaked}. "
                "Add fakes for them, or mark them safe to call for real.",
            )
        options.mcp_servers = {
            name: toolboxes[name].sdk_server(name) if name in toolboxes else config
            for name, config in servers.items()
        }

    async def run(self, case: Case, rep: int) -> Trace:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage

        world = FakeWorld(case.starting_state)
        toolboxes = {server: Toolbox(factory(), world, case.inject_failures) for server, factory in self.fakes.items()}
        _check_failures(case, toolboxes)
        options = self.options_factory()
        self._swap_servers(options, toolboxes)

        trace = Trace(case_id=case.id, rep=rep, task=case.task, world_before=deepcopy(world.state))
        if options.tools is None or options.tools:
            trace.notes.append("Built-in tools (files, shell) were available to the agent.")
        workdir = None
        if options.cwd is None:
            workdir = tempfile.mkdtemp(prefix="eval-agent-")
            options.cwd = workdir

        replies = list(case.user_replies)
        turn_costs: list[float] = []
        pending: dict[str, ToolCall] = {}
        started = time.monotonic()
        try:
            client = self.client_factory(options, toolboxes)
            async with client:
                prompt, user_turns = case.task, 0
                while True:
                    trace.messages.append(Message("user", prompt))
                    await client.query(prompt)
                    result = None
                    turn_text: list[str] = []
                    async for msg in client.receive_response():
                        if isinstance(msg, AssistantMessage):
                            trace.model = trace.model or msg.model
                            for block in msg.content:
                                if isinstance(block, TextBlock) and block.text.strip():
                                    turn_text.append(block.text)
                                    trace.messages.append(Message("agent", block.text))
                                elif isinstance(block, ToolUseBlock):
                                    server, short = split_tool_name(block.name)
                                    call = ToolCall(len(trace.tool_calls), short, server, dict(block.input),
                                                    id=block.id, parent_id=msg.parent_tool_use_id)
                                    pending[block.id] = call
                                    trace.tool_calls.append(call)
                        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
                            for block in msg.content:
                                if isinstance(block, ToolResultBlock) and block.tool_use_id in pending:
                                    call = pending.pop(block.tool_use_id)
                                    call.output = result_text(block.content)
                                    call.is_error = bool(block.is_error)
                        elif isinstance(msg, ResultMessage):
                            result = msg

                    if result is None:
                        raise HarnessError("sdk_error", "The agent's turn ended without a result message.")
                    if result.total_cost_usd is not None:
                        turn_costs.append(result.total_cost_usd)
                    if result.usage:
                        trace.usage = dict(result.usage)
                    if result.is_error:
                        if result.subtype == "error_max_turns":
                            trace.status = MAX_TURNS
                            break
                        detail = f"HTTP {result.api_error_status}" if result.api_error_status else result.subtype
                        raise HarnessError("api_error", f"The agent's API call failed ({detail}): {result.errors or ''}")

                    trace.final_answer = (result.result or "\n".join(turn_text)).strip()
                    if not looks_like_question(trace.final_answer):
                        trace.status = COMPLETED
                        break
                    if not replies:
                        trace.status = ASKED_WITHOUT_ANSWER
                        break
                    user_turns += 1
                    if user_turns > self.max_user_turns:
                        trace.status = USER_TURN_LIMIT
                        break
                    prompt = replies.pop(0)
                    trace.replies_used += 1
        finally:
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)

        trace.duration_ms = int((time.monotonic() - started) * 1000)
        # Assumes the SDK reports a running session total; checked on the first real multi-turn run.
        trace.cost_usd = max(turn_costs) if turn_costs else None
        trace.usage["turn_costs_usd"] = turn_costs
        trace.world_after = deepcopy(world.state)
        _finish(trace, world, toolboxes.values(), replies)
        return trace


@dataclass
class CallableTarget:
    """Custom code: fn(case, toolbox) returns the agent's final answer (sync or async)."""

    name: str
    fn: Callable[[Case, Toolbox], str | Awaitable[str]]
    fakes: Callable[[], list[FakeTool]]

    async def run(self, case: Case, rep: int) -> Trace:
        world = FakeWorld(case.starting_state)
        toolbox = Toolbox(self.fakes(), world, case.inject_failures)
        _check_failures(case, {"tools": toolbox})
        trace = Trace(case_id=case.id, rep=rep, task=case.task, world_before=deepcopy(world.state))
        started = time.monotonic()
        answer = self.fn(case, toolbox)
        if inspect.isawaitable(answer):
            answer = await answer
        trace.duration_ms = int((time.monotonic() - started) * 1000)
        trace.final_answer = str(answer or "")
        trace.messages = [Message("user", case.task), Message("agent", trace.final_answer)]
        trace.tool_calls = [
            ToolCall(i, c.tool, None, c.input, c.output, c.is_error) for i, c in enumerate(world.calls)
        ]
        trace.world_after = deepcopy(world.state)
        _finish(trace, world, [toolbox], [])
        return trace


def _finish(trace: Trace, world: FakeWorld, toolboxes: Any, unused_replies: list[str]) -> None:
    crashes = [crash for box in toolboxes for crash in box.crashes]
    if crashes:
        raise HarnessError("fake_tool_crashed",
                           "A pretend tool crashed, so this run doesn't show what your agent would do: "
                           + "; ".join(crashes))
    for call in world.calls:
        if call.injected_failure:
            trace.notes.append(f"{call.tool} was made to fail on purpose ({call.injected_failure}).")
    if unused_replies:
        trace.notes.append(f"{len(unused_replies)} scripted replies weren't needed.")


def _classify(exc: BaseException) -> tuple[str, str]:
    try:
        from claude_agent_sdk import ClaudeSDKError, CLINotFoundError
    except ImportError:  # pragma: no cover
        ClaudeSDKError = CLINotFoundError = ()  # type: ignore[assignment]
    if isinstance(exc, HarnessError):
        return exc.kind, str(exc)
    if CLINotFoundError and isinstance(exc, CLINotFoundError):
        return "cli_not_found", ("Couldn't find Claude Code (claude.exe), which the Agent SDK needs to run "
                                 "your agent. Install it, or set cli_path in your agent's options.")
    if ClaudeSDKError and isinstance(exc, ClaudeSDKError):
        return "sdk_error", f"{type(exc).__name__}: {exc}"
    return "harness_error", f"{type(exc).__name__}: {exc}"


async def run_cases(
    target: Target,
    cases: list[Case],
    *,
    reps: int = 1,
    concurrency: int = 2,
    timeout_s: float = 300,
    on_done: Callable[[Trace | RunError], None] | None = None,
) -> tuple[list[Trace], list[RunError]]:
    traces: list[Trace] = []
    errors: list[RunError] = []
    stop = asyncio.Event()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(case: Case, rep: int) -> None:
        async with sem:
            if stop.is_set():
                return
            try:
                item: Trace | RunError = await asyncio.wait_for(target.run(case, rep), timeout_s)
                traces.append(item)
            except TimeoutError:
                item = RunError(case.id, rep, "timeout", f"No result after {timeout_s:.0f} seconds.")
                errors.append(item)
            except Exception as exc:
                kind, message = _classify(exc)
                item = RunError(case.id, rep, kind, message)
                errors.append(item)
                if kind in FATAL_KINDS:
                    stop.set()
            if on_done:
                on_done(item)

    jobs = [(case, rep) for case in cases for rep in range(reps)]
    # Run the first job alone, so a setup problem (no CLI, unfaked tools) shows up once.
    if jobs:
        await one(*jobs[0])
    await asyncio.gather(*(one(case, rep) for case, rep in jobs[1:]))
    order = {case.id: i for i, case in enumerate(cases)}
    traces.sort(key=lambda t: (order.get(t.case_id, 0), t.rep))
    errors.sort(key=lambda e: (order.get(e.case_id, 0), e.rep))
    return traces, errors


def save_run(out_dir: str | Path, traces: list[Trace], errors: list[RunError], meta: dict[str, Any]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = Path(out_dir) / stamp
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "traces.jsonl", traces)
    write_jsonl(path / "errors.jsonl", errors)
    meta = {**meta, "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "traces": len(traces), "errors": len(errors),
            "cost_usd": round(sum(t.cost_usd or 0 for t in traces), 4)}
    (path / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path
