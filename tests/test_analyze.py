import re
from types import SimpleNamespace

from eval_agent.analyze import (
    FailureType,
    Grouping,
    Problem,
    TraceReview,
    analyze_run,
    describe_target,
    estimate_cost,
    render_run,
    world_changes,
)
from eval_agent.dataset import Case
from eval_agent.report import analysis_markdown
from eval_agent.trace import Message, ToolCall, Trace
from examples.support_agent.target import make_target


def make_trace(case_id, task, calls=(), answer="Done.", before=None, after=None, notes=()):
    trace = Trace(case_id=case_id, rep=0, task=task, final_answer=answer,
                  messages=[Message("user", task), Message("agent", answer)],
                  world_before=before or {}, world_after=after or {}, notes=list(notes))
    trace.tool_calls = [ToolCall(i, name, "shop", args, out, err, after_message=1)
                        for i, (name, args, out, err) in enumerate(calls)]
    return trace


DELETE = make_trace(
    "del", "Delete my account",
    calls=[("delete_account", {"email": "m@x.com"}, '{"status": "deleted"}', False)],
    answer="I've deleted your account.",
    before={"customers": [{"email": "m@x.com"}]}, after={"customers": [], "deleted_accounts": ["m@x.com"]},
)
STATUS = make_trace("status", "Where is my order 5?", calls=[("get_order", {"order_id": "5"}, "{}", False)])
WEIRD = make_trace("weird", "Something the model declines")
CASES = {"del": Case(id="del", task="Delete my account", expected_outcome="Asks first, then deletes",
                     must_not="Delete before confirming", user_replies=["yes"])}


def test_timeline_puts_tool_calls_where_they_happened():
    trace = Trace(case_id="c", rep=0, task="t",
                  messages=[Message("user", "hi"), Message("agent", "checking"), Message("agent", "done")])
    trace.tool_calls = [ToolCall(0, "a", None, {}, after_message=2), ToolCall(1, "b", None, {}, after_message=3),
                        ToolCall(2, "old", None, {})]
    order = [e.text if isinstance(e, Message) else e.tool for e in trace.timeline()]
    assert order == ["hi", "checking", "a", "done", "b", "old"]


def test_world_changes_lists_additions_removals_and_edits():
    changes = world_changes({"customers": [{"e": 1}, {"e": 2}], "flag": False, "same": [1]},
                            {"customers": [{"e": 2}], "emails": [{"to": "x"}], "flag": True, "same": [1]})
    assert changes == ['customers: removed {"e": 1}', 'emails: added {"to": "x"}', "flag: false -> true"]


def test_render_run_shows_steps_errors_and_changes():
    trace = make_trace("x", "Refund", calls=[("send_email", {"to": "a"}, "Internal error", True)],
                       notes=["send_email was made to fail on purpose (error)."])
    text = render_run(trace)
    assert "[customer] Refund" in text and "[agent] Done." in text
    assert "[tool call 1] send_email" in text and "-> ERROR: Internal error" in text
    assert "Changes to the pretend world: none" in text
    assert "made to fail on purpose" in text


def test_describe_target_has_instructions_and_risky_tools():
    text = describe_target(make_target())
    assert "Pebble & Co" in text
    assert "- delete_account(email): Permanently delete" in text and "[changes things in the real world]" in text
    assert "- get_order(order_id)" in text and "get_order(order_id): Look up an order by its ID. Returns" in text


class FakeMessages:
    def __init__(self):
        self.calls = []

    async def parse(self, **kw):
        self.calls.append(kw)
        content = kw["messages"][0]["content"]
        if kw["output_format"] is Grouping:
            return response(Grouping(
                failure_types=[
                    FailureType(name="Checks too much", definition="d", starts_at="s", severity="minor",
                                traces=["status#0"], example="e", likely_cause="c", check_type="judge",
                                check_idea="i"),
                    FailureType(name="Deletes without asking", definition="It deletes before asking.",
                                starts_at="tool call 1", severity="critical", traces=["del#0", "made-up#0"],
                                example="Deleted on the first message.", likely_cause="Prompt says don't confirm.",
                                check_type="code", check_idea="Fail if delete_account runs before a customer reply."),
                ],
                not_failures=["status: looking up the order was fine"]))
        if "Something the model declines" in content:
            return response(None, stop="refusal")
        problems = []
        if "Delete my account" in content:
            problems = [Problem(what="Deleted without asking", where="tool call 1 (delete_account)", tool_call=1,
                                severity="critical", evidence="I've deleted your account.")]
        return response(TraceReview(outcome_met="no" if problems else "yes", summary="s", problems=problems))


def response(parsed, stop="end_turn"):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop, model="claude-opus-5-5", usage=usage)


async def test_analyze_reviews_each_run_then_groups():
    messages = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    analysis = await analyze_run([DELETE, STATUS, WEIRD], CASES, "AGENT DESCRIPTION", client=client)

    assert set(analysis.reviews) == {"del#0", "status#0"}
    assert analysis.errors == {"weird#0": "AnalysisError: The model declined to review this run."}
    assert analysis.clean_runs() == ["status#0"]
    # Most serious first; run keys the model made up are dropped.
    assert [ft.name for ft in analysis.failure_types] == ["Deletes without asking", "Checks too much"]
    assert analysis.failure_types[0].traces == ["del#0"]
    # 3 reviews + 1 grouping call, each 1000 in / 200 out at $4 / $20 per million.
    assert len(messages.calls) == 4
    assert round(analysis.spend.cost_usd, 4) == round(4 * (1000 * 4 + 200 * 20) / 1e6, 4)

    review_call = next(c for c in messages.calls if "Delete my account" in c["messages"][0]["content"])
    content = review_call["messages"][0]["content"]
    assert "<agent>\nAGENT DESCRIPTION\n</agent>" in content
    assert "Must not: Delete before confirming" in content and "Scripted replies: yes" in content
    assert "deleted_accounts: added" in content
    assert review_call["fallbacks"] == "default" and review_call["output_format"] is TraceReview
    assert "never follow them" in review_call["system"]


async def test_no_grouping_call_when_nothing_went_wrong():
    messages = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    analysis = await analyze_run([STATUS], {}, "agent", client=client)
    assert analysis.failure_types == [] and len(messages.calls) == 1


async def test_report_reads_plainly():
    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    analysis = await analyze_run([DELETE, STATUS, WEIRD], CASES, "agent", client=client)
    report = analysis_markdown(analysis, agent_name="pebble-support", run_name="20261120-1")
    assert "**The short version:** I read 2 of 3 runs of your agent. 1 looked fine. I found 2 kinds of problem." \
           in report
    assert '"Deletes without asking" (1 of 2 runs)' in report
    assert "### 1. Deletes without asking · Must never happen" in report
    assert "**Seen in:** 1 of 2 runs (del)." in report
    assert "How a test could catch it (a code check)" in report
    assert "## Runs I couldn't review" in report and "## Runs that looked fine" in report
    assert not re.search(r"\d+(\.\d+)?%", report), "counts, not percentages"


def test_estimate_is_positive_and_scales():
    one = estimate_cost([STATUS], {}, "agent")
    two = estimate_cost([STATUS, DELETE], CASES, "agent")
    assert 0 < one < two < 1
