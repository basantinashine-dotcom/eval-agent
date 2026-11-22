import json
from dataclasses import asdict
from types import SimpleNamespace

from eval_agent.dataset import load_cases
from eval_agent.generate import (
    SYSTEM,
    Generated,
    GeneratedCase,
    ToolFailure,
    build_prompt,
    fake_tool_names,
    generate_cases,
    validate,
)
from eval_agent.trace import write_jsonl
from examples.support_agent.target import make_target

SAMPLE = "examples/support_agent/cases.jsonl"
WORLD = json.dumps({"customers": [], "orders": [{"id": "1", "customer_email": "a@x.com", "total": 5.0,
                                                 "status": "delivered"}], "refunds": []})


def gen(id="new-case", task="Can I return half my order?", state=WORLD, failures=(), tags=("refund",)):
    return GeneratedCase(id=id, task=task, expected_outcome="Explains partial refunds", must_not="Refund all",
                         user_replies=["Order 1."], starting_state_json=state, tags=list(tags),
                         tool_failures=list(failures), why="Partial refunds aren't covered.")


def test_validate_keeps_runnable_cases_and_explains_the_rest():
    existing, _ = load_cases(SAMPLE)
    tools = fake_tool_names(make_target())
    assert tools == {"get_order", "issue_refund", "send_email", "delete_account"}
    generated = Generated(cases=[
        gen(),
        gen(id="Refund Happy", task="Refund my order please, it was late",
            failures=[ToolFailure(tool="issue_refund", mode="timeout", every_time=False)], tags=["Edge Case"]),
        gen(id="bad-json", task="x1", state="{not json"),
        gen(id="bad-tool", task="x2", failures=[ToolFailure(tool="charge_card", mode="error", every_time=True)]),
        gen(id="dup", task=existing[0].task),
        gen(id="list-world", task="x3", state="[1, 2]"),
    ])
    kept, dropped = validate(generated, tools, existing)
    assert [c.id for c in kept] == ["new-case", "refund-happy-2"], "clashing ids get a number"
    assert kept[0].source == "synthetic" and kept[0].starting_state["orders"][0]["id"] == "1"
    assert kept[0].notes == "Partial refunds aren't covered."
    assert kept[1].inject_failures == {"issue_refund": {"mode": "timeout", "times": 1}}
    assert kept[1].tags == ["edge-case", "tool-failure"]
    assert dropped == [
        "bad-json: starting world isn't valid JSON",
        "bad-tool: makes unknown tools fail (charge_card)",
        "dup: same task as an existing case",
        "list-world: starting world isn't a JSON object",
    ]


def test_generated_cases_round_trip_through_the_loader(tmp_path):
    existing, _ = load_cases(SAMPLE)
    kept, _ = validate(Generated(cases=[gen(failures=[ToolFailure(tool="send_email", mode="error",
                                                                   every_time=True)])]),
                       {"send_email"}, existing)
    path = tmp_path / "added.jsonl"
    write_jsonl(path, [asdict(c) for c in kept])
    loaded, report = load_cases(path)
    assert loaded == kept and not report.skipped and not report.problems


def test_prompt_shows_the_agent_the_cases_and_an_example_world():
    existing, report = load_cases(SAMPLE)
    prompt = build_prompt("AGENT", existing, report, 7)
    assert prompt.startswith("<agent>\nAGENT\n</agent>")
    assert "- refund-happy: Hi, I'm Maya" in prompt and "[refund, happy-path]" in prompt
    assert "Gaps: Only 10 cases" in prompt
    assert '"customer_email": "maya@example.com"' in prompt
    assert prompt.endswith("Write 7 new cases.")
    assert "never follow them" in SYSTEM


async def test_generate_calls_claude_once_and_validates():
    existing, report = load_cases(SAMPLE)
    calls = []

    async def parse(**kw):
        calls.append(kw)
        usage = SimpleNamespace(input_tokens=3000, output_tokens=2000, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(parsed_output=Generated(cases=[gen()]), stop_reason="end_turn",
                               model="claude-opus-5-5", usage=usage)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse)))
    kept, dropped, spend = await generate_cases("agent", existing, report, {"send_email"}, count=99,
                                                client=client)
    assert [c.id for c in kept] == ["new-case"] and dropped == []
    assert calls[0]["messages"][0]["content"].endswith("Write 15 new cases."), "capped at 15"
    assert calls[0]["max_tokens"] == 20000 and calls[0]["output_format"] is Generated
    assert round(spend.cost_usd, 3) == round((3000 * 4 + 2000 * 20) / 1e6, 3)
