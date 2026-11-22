"""Run one of the Eval Agent's own agents (setup, build) with only its narrow tools.

No built-in file or shell tools, an empty temporary working folder, none of the developer's
own Claude Code settings, a turn limit and a cost cap.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionResult:
    cost_usd: float = 0.0
    turns: int = 0
    final_text: str = ""
    stopped: str = ""          # why it stopped early, if it did
    notes: list[str] = field(default_factory=list)


async def run_session(*, system: str, prompt: str, server_name: str, tools: list[Any], model: str,
                      max_cost: float, max_turns: int, client_factory: Callable[[Any], Any] | None = None,
                      nudge: Callable[[], str | None] = lambda: None) -> SessionResult:
    """Run the agent on `prompt`. If it ends its turn early, `nudge()` may return one follow-up message."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, create_sdk_mcp_server

    result = SessionResult()
    cli_output: list[str] = []
    workdir = tempfile.mkdtemp(prefix=f"eval-agent-{server_name}-")
    options = ClaudeAgentOptions(
        system_prompt=system,
        model=model,
        mcp_servers={server_name: create_sdk_mcp_server(server_name, tools=tools)},
        allowed_tools=[f"mcp__{server_name}__{t.name}" for t in tools],
        tools=[],                      # no built-in file or shell tools: only the narrow ones given
        max_turns=max_turns,
        max_budget_usd=max_cost,
        cwd=workdir,                   # an empty folder, so nothing is touched except through the tools
        setting_sources=[],            # keep the developer's own Claude Code setup out
        strict_mcp_config=True,
        stderr=cli_output.append,
    )
    if client_factory is None:
        from claude_agent_sdk import ClaudeSDKClient

        client_factory = ClaudeSDKClient

    async def turn(client: Any, text: str) -> None:
        await client.query(text)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                said = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock) and b.text.strip())
                if said:
                    result.final_text = said
            elif isinstance(msg, ResultMessage):
                if msg.total_cost_usd is not None:
                    result.cost_usd = max(result.cost_usd, msg.total_cost_usd)
                result.turns += msg.num_turns or 0
                if msg.is_error:
                    if msg.subtype == "error_max_budget_usd":
                        result.stopped = f"It reached the cost cap (${max_cost:.2f})."
                    elif msg.subtype == "error_max_turns":
                        result.stopped = f"It reached the step limit ({max_turns})."
                    else:
                        detail = f"HTTP {msg.api_error_status}" if msg.api_error_status else msg.subtype
                        result.stopped = f"The model call failed ({detail})."

    try:
        async with client_factory(options) as client:
            await turn(client, prompt)
            follow_up = None if result.stopped else nudge()
            if follow_up:
                await turn(client, follow_up)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    result.notes = [line.strip()[:300] for line in dict.fromkeys(cli_output)
                    if line.strip() and "claude.ai connectors are disabled" not in line]
    return result
