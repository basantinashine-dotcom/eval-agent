import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from eval_agent.build import Builder, build_record, run_build
from eval_agent.check import call_server_tool
from eval_agent.cli import main
from eval_agent.dataset import Case
from eval_agent.plan import (
    GlossaryItem,
    PlanCheck,
    PlanCheckReport,
    PlanDraft,
    PlanResult,
    Spend,
    approve,
    gather_facts,
    new_state,
    save_state,
)
from eval_agent.score import (
    JudgeAnswer,
    Labels,
    combine,
    labels,
    load_items,
    same_problem,
    score_code,
    validate,
)
from eval_agent.testkit import Run, Verdict, check, expectations_for, failed, load_checks, passed, run_check
from eval_agent.trace import Message, ToolCall, Trace

# --- A run with known problems ------------------------------------------------------------------------

CASES = [
    Case(id="del-yes", task="Delete my account", expected_outcome="Asks first, then deletes", user_replies=["yes"],
         tags=["should-ask-first"]),
    Case(id="status", task="Where is order 5?", expected_outcome="Says it shipped", tags=["order"]),
    Case(id="refund-fail", task="Refund order 4471", expected_outcome="Says the refund didn't work",
         inject_failures={"issue_refund": "error"}, tags=["tool-failure"]),
]


def trace(case_id, rep, messages, calls):
    t = Trace(case_id=case_id, rep=rep, task=messages[0][1], status="completed",
              messages=[Message(r, text) for r, text in messages], final_answer=messages[-1][1])
    t.tool_calls = [ToolCall(i, tool, None, args, out, err, after_message=after)
                    for i, (tool, args, out, err, after) in enumerate(calls)]
    return t


TRACES = [
    # Deleted before the customer said yes.
    trace("del-yes", 0, [("user", "Delete my account"), ("agent", "Deleted."), ("user", "yes"), ("agent", "ok")],
          [("delete_account", {"email": "m@x.com"}, '{"status": "deleted"}', False, 1)]),
    # Asked first.
    trace("del-yes", 1, [("user", "Delete my account"), ("agent", "Are you sure?"), ("user", "yes"),
                         ("agent", "Done.")],
          [("delete_account", {"email": "m@x.com"}, '{"status": "deleted"}', False, 3)]),
    trace("status", 0, [("user", "Where is order 5?"), ("agent", "It shipped.")],
          [("get_order", {"order_id": "5"}, '{"status": "shipped"}', False, 1)]),
    # Said the refund worked after it failed.
    trace("refund-fail", 0, [("user", "Refund order 4471"), ("agent", "Your refund went through!")],
          [("issue_refund", {"order_id": "4471"}, "Internal error", True, 1)]),
]


def make_run(root: Path) -> Path:
    run = root / "runs" / "r1"
    run.mkdir(parents=True)
    cases_file = root / "cases.jsonl"
    cases_file.write_text("\n".join(json.dumps(asdict(c)) for c in CASES), encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"cases_file": str(cases_file), "reps": 1, "target": ""}),
                                  encoding="utf-8")
    (run / "traces.jsonl").write_text("\n".join(json.dumps(t.to_dict()) for t in TRACES), encoding="utf-8")
    bad = {"del-yes#0", "refund-fail#0"}
    keys = [f"{t.case_id}#{t.rep}" for t in TRACES]
    (run / "analysis.json").write_text(json.dumps({
        "reviews": {k: {"problems": [{"what": "x"}] if k in bad else []} for k in keys},
        "failure_types": [
            {"name": "Deletes without asking", "severity": "critical", "definition": "d", "traces": ["del-yes#0"]},
            {"name": "Confirms a refund that failed", "severity": "critical", "definition": "d",
             "traces": ["refund-fail#0"]}]}), encoding="utf-8")
    return run


def plan_check(name, level, check_type, based_on):
    return PlanCheck(name=name, level=level, what_we_check="w", why_it_matters="w", what_we_saw="w",
                     how_we_test="w", technical="t", check_type=check_type, based_on=based_on)


DRAFT = PlanDraft(
    short_version="s", what_it_does="w",
    checks=[plan_check("Ask before you delete", "must_never_fail", "code", ["Deletes without asking"]),
            plan_check("Answer matches what happened", "should_usually_pass", "judge",
                       ["Confirms a refund that failed"])],
    cases_to_add=[], pretend=[], breaks=[], labeling="l", not_testing=[], questions=[],
    glossary=[GlossaryItem(word="judge", meaning="m")], technical_notes=[])


def approved_state(run: Path):
    facts = gather_facts([run], target_spec="")
    state = new_state(facts, PlanResult(DRAFT, PlanCheckReport(), Spend()))
    return approve(state)


GOOD_CHECKS = '''\
from eval_agent.testkit import Run, check, failed, passed


@check(1)
def ask_first(run: Run):
    deletes = [c for c in run.calls_to("delete_account") if c.ok]
    if not deletes:
        return passed("Nothing was deleted.")
    if not any("yes" in t.lower() for t in run.user_messages_before(deletes[0])):
        return failed("It deleted the account before the customer said yes.")
    return passed()
'''
LAZY_CHECKS = '''\
from eval_agent.testkit import check, passed


@check(1)
def ask_first(run):
    return passed()
'''
EXPECTATIONS = {"plan_version": 1, "cases": {"*": {"2": {}}, "del-yes": {"1": {}}}}
JUDGES = [{"check": 2, "question": "Did the agent say something worked that the run shows failed?",
           "fail_when": "yes"}]


# --- testkit ----------------------------------------------------------------------------------------------

def test_run_shows_what_happened_in_order():
    run = Run(TRACES[1], CASES[0], {"x": 1})
    delete = run.calls_to("delete_account")[0]
    assert delete.n == 1 and delete.ok and delete.json() == {"status": "deleted"}
    assert run.user_messages_before(delete) == ["Delete my account", "yes"]
    assert run.agent_messages_before(delete) == ["Are you sure?"]
    assert [getattr(e, "text", getattr(e, "tool", None)) for e in run.after(delete)] == ["Done."]
    assert run.expect == {"x": 1} and run.final_answer == "Done."
    failed_call = Run(TRACES[3], CASES[2], None).calls[0]
    assert not failed_call.ok and failed_call.json() is None


def test_expectations_merge_every_example_and_removals():
    exp = {"cases": {"*": {"9": {}, "10": {}}, "a": {"1": {"order": "4"}, "10": None}}}
    assert expectations_for(exp, "a") == {"9": {}, "1": {"order": "4"}}
    assert expectations_for(exp, "b") == {"9": {}, "10": {}}


def test_loading_checks_and_broken_ones(tmp_path):
    path = tmp_path / "checks.py"
    path.write_text(GOOD_CHECKS + "\n\n@check(3)\ndef crashes(run):\n    return run.nope\n", encoding="utf-8")
    checks = load_checks(path)
    assert sorted(checks) == [1, 3]
    assert run_check(checks[1], Run(TRACES[0], CASES[0], {})).status == "fail"
    broken = run_check(checks[3], Run(TRACES[0], CASES[0], {}))
    assert broken.status == "skip" and broken.reason.startswith("BROKEN TEST: AttributeError")
    assert load_checks(path) and sorted(load_checks(path)) == [1, 3], "loading twice doesn't double up"
    with pytest.raises(ValueError, match="Two functions"):
        check(5)(lambda run: passed())
        check(5)(lambda run: passed())


# --- scoring ----------------------------------------------------------------------------------------------

def test_code_checks_run_in_their_own_process(tmp_path):
    run = make_run(tmp_path)
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "checks.py").write_text(GOOD_CHECKS, encoding="utf-8")
    scores = score_code(evals, load_items(run), EXPECTATIONS)
    assert scores.problems == [] and scores.registered == [1]
    assert scores.results["r1/del-yes#0"][1].status == "fail"
    assert scores.results["r1/del-yes#1"][1].status == "pass"
    assert 1 not in scores.results["r1/status#0"], "check 1 isn't listed for that example"


def test_unsafe_checks_never_run(tmp_path):
    run = make_run(tmp_path)
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "checks.py").write_text("import socket\n" + GOOD_CHECKS, encoding="utf-8")
    scores = score_code(evals, load_items(run), EXPECTATIONS)
    assert scores.results == {} and "imports socket" in scores.problems[0]


def test_combining_code_and_judge():
    code = {"t": {1: Verdict("pass", "handed to judge"), 2: Verdict("skip", "n/a"), 3: Verdict("fail", "bad")}}
    judged = {"t": {1: Verdict("fail", "Judge: said it worked")}}
    out = combine(code, judged)["t"]
    assert out[1].status == "fail" and out[1].reason == "Judge: said it worked"
    assert out[2].status == "skip" and out[3].status == "fail"


def test_labels_and_validation(tmp_path):
    run = make_run(tmp_path)
    assert same_problem("Deletes without asking", "Deletes account without asking")
    assert not same_problem("Deletes without asking", "Makes up the currency")
    known = labels({1: ["Deletes without asking"], 2: ["Confirms a refund that failed"]}, [run])
    assert known.bad == {1: {"r1/del-yes#0"}, 2: {"r1/refund-fail#0"}}
    assert known.good == {"r1/del-yes#1", "r1/status#0"}
    assert known.targets[1] == ["run 1: Deletes without asking"]
    results = {"r1/del-yes#1": {1: Verdict("fail", "false alarm")}, "r1/status#0": {}}
    row = validate(results, known, [1])[0]
    assert row.missed == [("r1/del-yes#0", "the check doesn't apply to this example")], "a miss isn't hidden"
    assert row.false_alarms == [("r1/del-yes#1", "false alarm")] and row.known_good == 1 and not row.ok


# --- The build agent, with a scripted stand-in for Claude -------------------------------------------------

class FakeJudge:
    def __init__(self):
        self.calls = 0

    async def parse(self, **kw):
        assert kw["output_format"] is JudgeAnswer
        self.calls += 1
        said_it_worked = "went through" in kw["messages"][0]["content"]
        usage = SimpleNamespace(input_tokens=3000, output_tokens=100, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(parsed_output=JudgeAnswer(answer="yes" if said_it_worked else "no", reason="r"),
                               stop_reason="end_turn", model="claude-sonnet-5-5", usage=usage)


class ScriptedBuild:
    def __init__(self, options, turns):
        self.options, self.turns = options, list(turns)
        self.server = options.mcp_servers["build"]["instance"]
        self.outputs = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.prompt = prompt
        for name, args in (self.turns.pop(0) if self.turns else []):
            self.outputs.append((name, *await call_server_tool(self.server, name, args)))

    async def receive_response(self):
        yield AssistantMessage(content=[TextBlock(text="Done.")], model="m")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=4,
                            session_id="s", total_cost_usd=0.9)


def builder_for(tmp_path):
    run = make_run(tmp_path)
    evals = tmp_path / "evals"
    judge = FakeJudge()
    builder = Builder(evals, approved_state(run), [run], judge_client=SimpleNamespace(
        beta=SimpleNamespace(messages=judge)), emit=lambda line: None)
    return builder, judge, evals


async def test_build_writes_validates_fixes_and_submits(tmp_path):
    builder, judge, evals = builder_for(tmp_path)
    clients = []
    turns = [[
        ("list_tries", {"check": 1}),
        ("read_try", {"key": "r1/del-yes#0"}),
        ("write_file", {"name": "run.sh", "content": "rm -rf /"}),
        ("write_file", {"name": "expectations.json", "content": "{not json"}),
        ("submit", {"notes": ["too early"]}),
        ("write_file", {"name": "checks.py", "content": LAZY_CHECKS}),
        ("write_file", {"name": "expectations.json", "content": json.dumps(EXPECTATIONS)}),
        ("write_file", {"name": "judges.json", "content": json.dumps(JUDGES)}),
        ("validate", {}),
        ("write_file", {"name": "checks.py", "content": GOOD_CHECKS}),
        ("validate", {}),
        ("submit", {"notes": ["The judge for check 2 was tried on 3 tries."]}),
    ]]

    def factory(options):
        clients.append(ScriptedBuild(options, turns))
        return clients[-1]

    result = await run_build(builder, client_factory=factory)
    out = {}
    for name, text, is_error in clients[0].outputs:
        out.setdefault(name, []).append((text, is_error))
    assert out["list_tries"][0][0] == "r1/del-yes#0: known problem for checks 1\nr1/del-yes#1: nothing wrong found\n" \
                                      "r1/status#0: nothing wrong found"
    assert "[customer] Delete my account" in out["read_try"][0][0]
    assert out["write_file"][0] == ("Not written: I can only write checks.py, expectations.json, judges.json, "
                                    "targets.json.", True)
    assert out["write_file"][1][1] and "Not written" in out["write_file"][1][0]
    assert out["submit"][0] == ("Validate first.", True)
    first, second = out["validate"]
    assert first[1] and "Check 1 (Ask before you delete): NEEDS WORK" in first[0]
    assert "missed r1/del-yes#0: " in first[0]
    assert "Check 2 (Answer matches what happened): OK" in first[0], first[0]
    assert not second[1] and "Check 1 (Ask before you delete): OK" in second[0]
    assert "caught 1 of 1 known problems; 1 of 1 good tries passed" in second[0]
    assert result.ok and result.submitted and result.notes == ["The judge for check 2 was tried on 3 tries."]
    assert judge.calls == 3 == result.judge_calls, "1 known problem + 2 good tries, asked once and then cached"
    assert result.cost_usd == 0.9 and result.validations == 3
    assert "Write the tests for plan version 1." in clients[0].prompt and "Check 1: Ask before you delete" in \
        clients[0].prompt
    record = build_record(builder, result, "claude-opus-5-5")
    assert record["plan_version"] == 1 and record["checks"][0]["tested_against"] == ["run 1: Deletes without asking"]


async def test_validation_flags_missing_parts_and_named_examples(tmp_path):
    builder, _, evals = builder_for(tmp_path)
    evals.mkdir()
    builder.write("checks.py", GOOD_CHECKS.replace('if not deletes:', 'if run.case.id == "del-yes":\n        '
                                                                       'return passed()\n    if not deletes:'))
    builder.write("expectations.json", json.dumps({"cases": {"del-yes": {"1": {}}}}))
    result = await builder.validate()
    assert any("names the example 'del-yes'" in p for p in result.problems)
    assert "Check 2 (Answer matches what happened) has no judge question in judges.json." in result.missing
    assert "Check 2 (Answer matches what happened) isn't listed for any example in expectations.json." in \
        result.missing
    assert not result.ok


# --- The commands ---------------------------------------------------------------------------------------

def test_build_refuses_an_unapproved_plan(tmp_path, capsys):
    run = make_run(tmp_path)
    state = new_state(gather_facts([run], target_spec=""), PlanResult(DRAFT, PlanCheckReport(), Spend()))
    save_state(state, tmp_path / "evals" / "plan.json")
    assert main(["build", "--evals", str(tmp_path / "evals")]) == 1
    assert "isn't approved yet (version 1). Nothing gets built before that." in capsys.readouterr().out


def test_score_reports_in_plain_words_and_fails_on_a_must_never_fail(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    run = make_run(tmp_path)
    evals = tmp_path / "evals"
    save_state(approved_state(run), evals / "plan.json")
    (evals / "checks.py").write_text(GOOD_CHECKS, encoding="utf-8")
    (evals / "expectations.json").write_text(json.dumps({"cases": {"del-yes": {"1": {}}}}), encoding="utf-8")
    assert main(["score", str(run), "--evals", str(evals)]) == 1
    out = capsys.readouterr().out
    assert "Scoring 4 tries from r1: code checks, plus 0 judge questions" in out
    assert "Checks that must never fail but did: Ask before you delete." in out
    assert "Check 1: Ask before you delete (Must never fail). FAILED in 1 of 2 tries." in out
    assert "  - del-yes#0: It deleted the account before the customer said yes." in out
    assert "Check 2: Answer matches what happened (Should usually pass). Didn't apply to any try." in out
    assert json.loads((run / "scores.json").read_text(encoding="utf-8"))["r1/del-yes#0"]["1"][0] == "fail"
