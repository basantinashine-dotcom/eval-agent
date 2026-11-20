import asyncio
import json

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
)

from eval_agent.dataset import Case
from eval_agent.fakes import FakeTool
from eval_agent.runner import CallableTarget, SdkTarget, looks_like_question, run_cases, save_run, split_tool_name
from eval_agent.trace import ASKED_WITHOUT_ANSWER, COMPLETED, MAX_TURNS, read_traces
from examples.support_agent import agent, fakes

STATE = {
    "customers": [{"id": "c_1", "name": "Maya", "email": "maya@example.com"}],
    "orders": [{"id": "4471", "customer_email": "maya@example.com", "total": 25.0, "status": "delivered"}],
    "refunds": [],
}


class ScriptedClient:
    """Stands in for ClaudeSDKClient. Each turn is a function(toolboxes) -> list of SDK messages."""

    def __init__(self, turns, options, toolboxes):
        self.turns, self.options, self.toolboxes = list(turns), options, toolboxes
        self.prompts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_response(self):
        for message in self.turns.pop(0)(self.toolboxes):
            yield message


def use(toolboxes, tool, args, call_id):
    text, is_error = toolboxes["shop"].call(tool, args)
    return [
        AssistantMessage(content=[ToolUseBlock(id=call_id, name=f"mcp__shop__{tool}", input=args)], model="m"),
        UserMessage(content=[ToolResultBlock(tool_use_id=call_id, content=[{"type": "text", "text": text}],
                                             is_error=is_error)]),
    ]


def say(text):
    return AssistantMessage(content=[TextBlock(text=text)], model="m")


def done(text, cost=0.01, **kw):
    return ResultMessage(subtype=kw.pop("subtype", "success"), duration_ms=1, duration_api_ms=1,
                         is_error=kw.pop("is_error", False), num_turns=1, session_id="s",
                         total_cost_usd=cost, result=text, **kw)


def target(*turns, options_factory=agent.build_options, shop_fakes=fakes.shop_fakes):
    clients = []

    def factory(options, toolboxes):
        clients.append(ScriptedClient(turns, options, toolboxes))
        return clients[-1]

    return SdkTarget("test", options_factory, {"shop": shop_fakes}, client_factory=factory), clients


def case(**kw):
    return Case(id=kw.pop("id", "c1"), task=kw.pop("task", "Refund order 4471"), starting_state=STATE, **kw)


async def test_records_tool_calls_world_changes_and_cost():
    def turn(tb):
        return [
            *use(tb, "get_order", {"order_id": "4471"}, "t1"),
            *use(tb, "issue_refund", {"order_id": "4471", "amount": 25.0, "reason": "broken"}, "t2"),
            say("Refunded $25."),
            done("Refunded $25."),
        ]

    tgt, clients = target(turn)
    trace = await tgt.run(case(), 0)
    assert trace.status == COMPLETED
    assert trace.tools_called() == ["get_order", "issue_refund"]
    assert trace.tool_calls[0].server == "shop" and not trace.tool_calls[0].is_error
    assert trace.world_before["refunds"] == []
    assert trace.world_after["refunds"][0]["amount"] == 25.0
    assert trace.final_answer == "Refunded $25." and trace.cost_usd == 0.01 and trace.model == "m"
    # The agent's own server was swapped for the fakes, keeping the same tool names.
    assert set(clients[0].options.mcp_servers) == {"shop"}
    assert clients[0].options.mcp_servers["shop"]["instance"] is not None
    # And the developer's own Claude Code setup is kept out.
    assert clients[0].options.strict_mcp_config is True
    assert clients[0].options.setting_sources == []
    assert trace.notes == []


async def test_agent_asking_for_settings_gets_a_note():
    def options_with_settings():
        options = agent.build_options()
        options.setting_sources = ["user", "project"]
        return options

    tgt, clients = target(lambda tb: [done("ok")], options_factory=options_with_settings)
    trace = await tgt.run(case(), 0)
    assert clients[0].options.setting_sources == []
    assert "without your personal settings" in trace.notes[0]


def test_auth_warning_only_for_personal_login(monkeypatch):
    import subprocess

    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    from eval_agent import runner

    monkeypatch.setattr(SubprocessCLITransport, "_find_cli", lambda self: "claude.exe")
    status = {"authMethod": "claude.ai"}
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(status), ""))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" in runner.auth_warnings()[0]
    status["authMethod"] = "api_key"
    assert runner.auth_warnings() == []
    status["authMethod"] = "claude.ai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert runner.auth_warnings() == []


async def test_scripted_reply_answers_the_agents_question():
    turns = [
        lambda tb: [say("Are you sure you want to delete your account?"), done("Are you sure?", cost=0.01)],
        lambda tb: [*use(tb, "delete_account", {"email": "maya@example.com"}, "t1"), done("Deleted.", cost=0.02)],
    ]
    tgt, clients = target(*turns)
    trace = await tgt.run(case(task="Delete my account", user_replies=["Yes, delete it."]), 0)
    assert clients[0].prompts == ["Delete my account", "Yes, delete it."]
    assert trace.status == COMPLETED and trace.replies_used == 1
    assert trace.world_after["customers"] == []
    assert [m.role for m in trace.messages] == ["user", "agent", "user"]
    assert trace.cost_usd == 0.02 and trace.usage["turn_costs_usd"] == [0.01, 0.02]


async def test_question_with_no_scripted_reply_left():
    tgt, clients = target(lambda tb: [done("Which order do you mean?")])
    trace = await tgt.run(case(task="I want my money back"), 0)
    assert trace.status == ASKED_WITHOUT_ANSWER and len(clients[0].prompts) == 1


async def test_agent_hitting_max_turns_is_a_trace_not_an_error():
    tgt, _ = target(lambda tb: [done(None, is_error=True, subtype="error_max_turns")])
    trace = await tgt.run(case(), 0)
    assert trace.status == MAX_TURNS


async def test_api_errors_go_to_errors_not_traces():
    tgt, _ = target(lambda tb: [done(None, is_error=True, api_error_status=529)])
    traces, errors = await run_cases(tgt, [case()])
    assert traces == [] and errors[0].kind == "api_error" and "529" in errors[0].message


async def test_unfaked_tool_server_stops_the_run():
    def options_with_payments():
        options = agent.build_options()
        options.mcp_servers["payments"] = create_sdk_mcp_server("payments", tools=[])
        return options

    tgt, _ = target(lambda tb: [done("x")], options_factory=options_with_payments)
    traces, errors = await run_cases(tgt, [case(id="a"), case(id="b")], concurrency=1)
    assert traces == []
    assert [e.kind for e in errors] == ["unfaked_tools"], "the second case shouldn't run"
    assert "payments" in errors[0].message


async def test_injected_failure_reaches_the_agent_and_is_noted():
    email = {"to": "m", "subject": "s", "body": "b"}
    tgt, _ = target(lambda tb: [*use(tb, "send_email", email, "t1"), *use(tb, "send_email", email, "t2"), done("ok")])
    trace = await tgt.run(case(inject_failures={"send_email": {"mode": "error", "times": "all"}}), 0)
    assert trace.tool_calls[0].is_error and trace.tool_calls[1].is_error
    assert trace.notes == ["send_email was made to fail on purpose (error, 2 times)."]
    assert "emails" not in trace.world_after


async def test_timeouts_and_crashing_fakes_are_errors():
    class SlowClient(ScriptedClient):
        async def receive_response(self):
            await asyncio.sleep(5)
            yield done("late")

    slow = SdkTarget("slow", agent.build_options, {"shop": fakes.shop_fakes},
                     client_factory=lambda o, tb: SlowClient([], o, tb))
    _, errors = await run_cases(slow, [case()], timeout_s=0.05)
    assert errors[0].kind == "timeout"

    def crashing_fakes():
        return [FakeTool("get_order", "x", {"order_id": str}, lambda args, world: world.state["nope"])]

    tgt, _ = target(lambda tb: [*use(tb, "get_order", {"order_id": "1"}, "t1"), done("ok")],
                    shop_fakes=crashing_fakes)
    _, errors = await run_cases(tgt, [case()])
    assert errors[0].kind == "fake_tool_crashed"


async def test_callable_target_for_custom_code(tmp_path):
    def my_agent(case, toolbox):
        order = json.loads(toolbox.call("get_order", {"order_id": "4471"})[0])
        toolbox.call("issue_refund", {"order_id": "4471", "amount": order["total"], "reason": "x"})
        return "Refunded."

    tgt = CallableTarget("custom", my_agent, fakes.shop_fakes)
    traces, errors = await run_cases(tgt, [case()], reps=2)
    assert errors == [] and [t.rep for t in traces] == [0, 1]
    assert traces[0].tools_called() == ["get_order", "issue_refund"]
    assert traces[0].world_after["refunds"][0]["amount"] == 25.0

    path = save_run(tmp_path, traces, errors, {"command": "test"})
    assert [t.case_id for t in read_traces(path / "traces.jsonl")] == ["c1", "c1"]
    assert json.loads((path / "run.json").read_text())["traces"] == 2


def test_question_detection_and_tool_names():
    assert looks_like_question("Sure. Can you confirm you want this deleted?")
    assert looks_like_question("Which order?\n\n**Thanks!**?")
    assert not looks_like_question("Done! Your refund is on its way.")
    # Polite closings aren't questions that need an answer (seen in real runs).
    assert not looks_like_question("I've refunded $25.\n\nAnything else I can help with?")
    assert not looks_like_question("Refunded. Is there anything else I can do for you?")
    assert not looks_like_question("Done. **Can I help with anything else?**")
    assert not looks_like_question("It has shipped. Let me know if there's anything else I can help with!")
    assert looks_like_question("Anything else first? Which order should I refund?")
    assert looks_like_question("Should I delete it, or is there anything else you need?")
    # Real questions that don't end the message (both seen in real runs).
    assert looks_like_question(
        "Deleting is permanent. Could you send me an order number from an order placed with "
        "maya@example.com? Once I've matched it to your email, I'll delete the account.")
    assert looks_like_question(
        "I can help you get a refund. Could you tell me:\n\n1. **Your order number**\n2. **The reason**")
    assert split_tool_name("mcp__shop__get_order") == ("shop", "get_order")
    assert split_tool_name("Bash") == (None, "Bash")
