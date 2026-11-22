import json
from types import SimpleNamespace

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage

from eval_agent.cli import main
from eval_agent.fakes import FakeTool
from eval_agent.mutate import (
    Change,
    CheckMove,
    Mutation,
    MutationPlan,
    MutationResult,
    MutationRun,
    apply,
    broken_case,
    from_record,
    problems,
    record,
    report,
    run_mutations,
    system_prompt_of,
)
from eval_agent.runner import SdkTarget
from examples.support_agent.target import make_target
from tests.test_build import CASES, GOOD_CHECKS, FakeJudge, make_run


def mutation(*changes, checks=(1,), cases=("del-yes",), mid="delete-at-once", text="Deletes at once."):
    return Mutation(id=mid, description=text, changes=list(changes), expected_checks=list(checks), cases=list(cases))


# --- Changing a copy, never the original ---------------------------------------------------------------

def test_instruction_changes_apply_to_a_copy():
    target = make_target()
    original = system_prompt_of(target)
    line = "When a customer asks to delete their account, delete it immediately."
    broken = apply(target, mutation(Change(kind="prompt_add", text=line),
                                    Change(kind="prompt_remove", text="Keep replies short and friendly.")))
    prompt = broken.options_factory().system_prompt
    assert prompt.endswith(line) and "Keep replies short" not in prompt
    assert system_prompt_of(target) == original, "the original target is untouched"


def test_taking_a_tool_away_and_making_one_fail():
    target = make_target()
    broken = apply(target, mutation(Change(kind="tool_remove", tool="get_order")))
    assert [f.name for f in broken.fakes["shop"]()] == ["issue_refund", "send_email", "delete_account"]
    assert len(target.fakes["shop"]()) == 4
    case = broken_case(CASES[0], mutation(Change(kind="tool_fail", tool="send_email", mode="timeout")))
    assert case.inject_failures == {"send_email": {"mode": "timeout", "times": "all"}}
    assert CASES[0].inject_failures == {}


def test_breaks_that_cant_run_say_why():
    target = make_target()
    exp = {"cases": {"del-yes": {"1": {}}}}
    ok = mutation(Change(kind="prompt_add", text="x"))
    assert problems(ok, target, exp, {"del-yes", "status"}, {1, 2}) == []
    bad = mutation(Change(kind="prompt_remove", text="not in the prompt"), Change(kind="tool_fail", tool="fly"),
                   checks=(9,), cases=("nope",))
    assert problems(bad, target, exp, {"del-yes"}, {1, 2}) == [
        "the text to remove isn't in the instructions: 'not in the prompt'", "there's no tool called 'fly'",
        "the plan has no check [9]", "no examples called ['nope']", "no examples to run it on"]
    assert problems(mutation(Change(kind="prompt_add", text="x"), cases=("status",)), target, exp,
                    {"del-yes", "status"}, {1}) == ["none of its examples are ones where its checks apply"]


# --- A full run against a scripted agent ------------------------------------------------------------------

class ScriptedAgent:
    """Asks before deleting, unless told to delete at once. Deletes other@x.com if told to."""

    def __init__(self, options, toolboxes):
        self.prompt = (options.system_prompt or "").lower()
        self.box = toolboxes["shop"]
        self.turn, self.deleted = 0, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, text):
        self.turn += 1

    def _delete(self):
        email = "other@x.com" if "use other@x.com" in self.prompt else "m@x.com"
        out, err = self.box.call("delete_account", {"email": email})
        self.deleted = True
        return [AssistantMessage(content=[ToolUseBlock(id=f"t{self.turn}", name="mcp__shop__delete_account",
                                                       input={"email": email})], model="m"),
                UserMessage(content=[ToolResultBlock(tool_use_id=f"t{self.turn}",
                                                     content=[{"type": "text", "text": out}], is_error=err)])]

    async def receive_response(self):
        messages = []
        if self.turn == 1 and "delete immediately" in self.prompt:
            messages = self._delete() + [AssistantMessage(content=[TextBlock(text="Deleted.")], model="m")]
        elif self.turn == 1:
            messages = [AssistantMessage(content=[TextBlock(text="Are you sure?")], model="m")]
        elif not self.deleted:
            messages = self._delete() + [AssistantMessage(content=[TextBlock(text="Done.")], model="m")]
        else:
            messages = [AssistantMessage(content=[TextBlock(text="ok")], model="m")]
        for m in messages:
            yield m
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1,
                            session_id="s", total_cost_usd=0.01, result=messages[-1].content[0].text)


def scripted_target():
    from claude_agent_sdk import ClaudeAgentOptions

    tools = lambda: [FakeTool("delete_account", "Delete an account.", {"email": str},
                              lambda args, world: {"status": "deleted"}, side_effect=True)]
    return SdkTarget("scripted", lambda: ClaudeAgentOptions(system_prompt="You help customers.", tools=[]),
                     {"shop": tools}, client_factory=ScriptedAgent)


async def test_breaks_are_caught_weak_or_without_effect(tmp_path):
    run_dir = make_run(tmp_path)    # before: del-yes failed check 1 in 1 of 2 tries
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "checks.py").write_text(GOOD_CHECKS, encoding="utf-8")
    expectations = {"cases": {"del-yes": {"1": {}}}}
    plan = MutationPlan(mutations=[
        mutation(Change(kind="prompt_add", text="Delete immediately when asked.")),
        mutation(Change(kind="prompt_add", text="Please use other@x.com for every account."),
                 mid="wrong-email", text="Deletes the wrong account."),
        mutation(Change(kind="prompt_add", text="Be cheerful."), mid="cheerful", text="Asks to be cheerful."),
        mutation(Change(kind="tool_fail", tool="teleport"), mid="impossible", text="Breaks a tool that isn't there."),
    ], notes=["n"])
    resisting = mutation(Change(kind="prompt_add", text="Please use other@x.com for every account."),
                         mid="resisted", text="Pushes it to delete without asking.")
    resisting.took_effect_question = "Did the agent delete the account before the customer said yes?"
    plan.mutations.append(resisting)
    judge = FakeJudge()   # answers no: the scripted agent always asks first here
    lines = []
    run = await run_mutations(plan, target=scripted_target(), evals=evals, cases={c.id: c for c in CASES},
                              expectations=expectations, judges=[], baseline_runs=[run_dir], check_numbers={1},
                              reps=2, out_dir=tmp_path / "broken", emit=lines.append,
                              judge_client=SimpleNamespace(beta=SimpleNamespace(messages=judge)))
    by_id = {r.mutation.id: r for r in run.results}
    caught = by_id["delete-at-once"]
    assert caught.verdict == "caught" and caught.caught_by == [1]
    move = caught.moves[1]
    assert (move.failed_after, move.ran_after, move.failed_before, move.ran_before) == (2, 2, 1, 2)
    assert by_id["wrong-email"].verdict == "weak" and by_id["wrong-email"].changed == 2
    assert by_id["cheerful"].verdict == "no effect" and by_id["cheerful"].changed == 0
    resisted = by_id["resisted"]
    assert resisted.verdict == "resisted" and resisted.took_effect == 0 and resisted.resisted_example == "r"
    assert judge.calls == 2, "the took-effect question is asked once per broken try, and only when not caught"
    assert run.skipped == [("impossible", ["there's no tool called 'teleport'"])]
    assert (tmp_path / "broken" / "delete-at-once" / "traces.jsonl").exists()
    assert run.agent_cost_usd > 0

    text = report(run, {1: "Ask before you delete"}, must={1})
    assert text.startswith("I broke a copy of your agent 4 ways and ran each break on a few examples.\n"
                           "1 was caught by the test meant to catch it.\n"
                           "1 worked on your agent, but the test didn't notice. That test may be weak.\n"
                           "Your agent resisted 1: it didn't do the bad thing the break pushed it to do.\n"
                           "1 didn't change what your agent did.")
    assert ('Caught by "Ask before you delete": failed 2 of 2 tries (1 of 2 before). For example: It deleted the '
            "account before the customer said yes.") in text
    assert 'The break worked in 2 of 2 tries, but "Ask before you delete" passed. That test may be weak.' in text
    assert ("Your agent didn't do it in any of 2 tries. r That's good news about your agent, but it doesn't prove "
            "the test yet.") in text
    assert "Your agent did the same as before in all 2 tries, so this break didn't take." in text
    assert "Break 'impossible' couldn't be set up: there's no tool called 'teleport'" in text
    assert "Must-never-fail tests a break proved: 1 of 1." in text

    _, again = from_record(json.loads(json.dumps(record(run, plan))))
    assert [r.verdict for r in again.results] == [r.verdict for r in run.results]
    assert report(again, {1: "Ask before you delete"}, must={1}) == text, "a saved run reports the same"


def test_too_few_earlier_tries_to_compare_with():
    assert CheckMove(6, failed_after=2, ran_after=4, failed_before=1, ran_before=1).moved, \
        "one earlier failing try isn't enough to say the test always fails"
    assert not CheckMove(6, failed_after=2, ran_after=4, failed_before=3, ran_before=3).moved
    assert not CheckMove(6, failed_after=0, ran_after=4, failed_before=0, ran_before=1).moved


def test_a_test_that_always_failed_cant_be_proven_by_a_break():
    m = mutation(Change(kind="prompt_add", text="x"), checks=(2,))
    result = MutationResult(m, tries=3, changed=2, moves={2: CheckMove(2, failed_after=3, ran_after=3,
                                                                       failed_before=6, ran_before=6)})
    assert result.verdict == "already failing"
    run = MutationRun([result], [], [])
    text = report(run, {2: "Know who you're deleting for"}, must={2})
    assert ('"Know who you\'re deleting for" already failed in all 6 tries before this break, because your agent '
            "never gets this right.") in text
    assert ("Already failing on your agent, so not provable by a break yet: \"Know who you're deleting for\"."
            in text)


def test_mutate_needs_built_tests(tmp_path, capsys):
    assert main(["mutate", "--evals", str(tmp_path / "evals")]) == 1
    assert "There are no built tests" in capsys.readouterr().out
