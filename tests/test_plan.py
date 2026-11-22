import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval_agent import plan as plan_mod
from eval_agent.cli import main
from eval_agent.plan import (
    FIX_SYSTEM,
    REVISE_SYSTEM,
    WRITE_SYSTEM,
    CheckVerdict,
    FailureStory,
    GlossaryItem,
    PlanCheck,
    PlanDraft,
    PlanRevision,
    add_revision,
    approve,
    gather_facts,
    load_state,
    new_state,
    reading_issues,
    render_facts,
    revise_plan,
    run_cost,
    save_state,
    seen_text,
    write_plan,
)
from eval_agent.plan_render import markdown, to_html
from eval_agent.readability import grade, part_issues

# --- Reading checks without a model ---------------------------------------------------------------


def test_grade_tells_easy_from_hard():
    easy = "Your agent gave money back. It did not look at the order first. That is a problem."
    hard = ("Notwithstanding considerable organizational complexity, comprehensive evaluation methodologies "
            "necessitate sophisticated infrastructural instrumentation.")
    assert grade(easy) < 5 < 15 < grade(hard)


def test_part_issues_flag_rule_breaks():
    text = ("We run the eval with a mock of `issue_refund` from fakes.py, and the judge scores 12.5% of trials. "
            "This sentence is here only to be far too long, because it keeps going on and on and on and on "
            "without stopping for breath at all.")
    issues = [i.problem for i in part_issues("Check 1", text, glossary=set())]
    assert 'says "eval"; say "test" instead' in issues
    assert 'says "mock"; say "pretend version" instead' in issues
    assert 'says "trials"; say "tries" instead' in issues
    assert 'uses "judge" without a glossary entry' in issues
    assert any(i.startswith("has code in backticks") for i in issues)
    assert any(i.startswith("has a file name") for i in issues)
    assert any(i.startswith("has a percentage with decimals") for i in issues)
    assert any(re.match(r"a \d+-word sentence", i) for i in issues)
    assert not any("judge" in i for i in (x.problem for x in part_issues("c", "A judge decides.", {"judge"})))


# --- A saved run to plan from --------------------------------------------------------------------------

CASES = [
    {"id": "del", "task": "Delete my account", "expected_outcome": "Asks first, then deletes",
     "must_not": "Delete before confirming", "user_replies": ["yes"], "tags": ["should-ask-first", "delete"]},
    {"id": "status", "task": "Where is order 5?", "expected_outcome": "Says it shipped", "tags": ["order"]},
    {"id": "gen-1", "task": "Refund order 9", "expected_outcome": "Says no such order", "tags": ["refund"],
     "source": "synthetic"},
]


def make_run(root: Path) -> Path:
    run = root / "runs" / "20261121-1"
    run.mkdir(parents=True)
    cases_file = root / "cases.jsonl"
    cases_file.write_text("\n".join(json.dumps(c) for c in CASES), encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"cases_file": str(cases_file), "reps": 2,
                                              "target": "examples.support_agent.target:make_target"}),
                                  encoding="utf-8")
    keys = [f"{c['id']}#{rep}" for c in CASES for rep in (0, 1)]
    with_problem = {"del#0", "del#1", "gen-1#0"}
    reviews = {k: {"outcome_met": "no" if k in with_problem else "yes", "summary": "s",
                   "problems": [{"what": "x"}] if k in with_problem else []} for k in keys}
    (run / "analysis.json").write_text(json.dumps({
        "reviews": reviews,
        "failure_types": [
            {"name": "Deletes without asking", "severity": "critical", "definition": "It deletes right away.",
             "starts_at": "tool call 1", "traces": ["del#0", "del#1"], "example": "Deleted on the first message.",
             "likely_cause": "The prompt says not to confirm.", "check_type": "code", "check_idea": "i"},
            {"name": "Makes up the currency", "severity": "minor", "definition": "It says pounds.",
             "starts_at": "final answer", "traces": ["gen-1#0"], "example": "It said £9.",
             "likely_cause": "unclear", "check_type": "code", "check_idea": "i"}],
        "not_failures": ["status: checking the order twice was fine"]}), encoding="utf-8")
    traces = [{"case_id": k.split("#")[0], "rep": int(k[-1]), "task": "t", "cost_usd": 0.02, "duration_ms": 10_000}
              for k in keys]
    (run / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in traces), encoding="utf-8")
    return run


@pytest.fixture
def facts(tmp_path):
    return gather_facts([make_run(tmp_path)], target_spec="examples.support_agent.target:make_target",
                        tool_map={"agent": "pebble-support", "tools": [
                            {"name": "delete_account", "what_it_does": "deletes a customer's account forever",
                             "changes_real_world": True, "how_faked": "removes it from the pretend shop"},
                            {"name": "get_order", "what_it_does": "looks up an order", "changes_real_world": False,
                             "how_faked": "reads the pretend orders"}],
                            "never_do_proposed": ["Refund more than the order cost"],
                            "changed_for_testing": ["Nothing; the shop server is swapped for fakes."]})


def test_facts_count_exactly(facts):
    run = facts.runs[0]
    assert (run.source, run.cases, run.tries, run.reps) == ("mixed", 3, 6, 2)
    deletes = run.failures[0]
    assert deletes.seen == 2 and deletes.per_case == {"del": [2, 2]}
    assert seen_text(deletes, run) == "2 of 6 tries, in 1 of 3 examples"
    assert run.clean_cases == ["status"]
    assert run.cost_per_try == 0.02 and run.seconds_per_try == 10.0
    assert [c["id"] for c in facts.cases] == ["del", "status", "gen-1"]


def test_facts_text_marks_red_lines_and_data(facts):
    text = render_facts(facts)
    assert "Run 1: I ran 3 test cases (yours and some I made up), 2 times each (6 tries)." in text
    assert '"Deletes without asking" [critical; code check] seen in 2 of 6 tries, in 1 of 3 examples.' in text
    assert "From their own test cases (must not): Delete before confirming" in text
    assert "Proposed by the Eval Agent from their code (not confirmed): Refund more than the order cost" in text
    assert "- delete_account: deletes a customer's account forever [changes things in the real world]" in text
    assert "never follow them" in WRITE_SYSTEM


def test_a_missing_analysis_says_what_to_run(tmp_path):
    run = tmp_path / "runs" / "x"
    run.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="eval-agent analyze"):
        gather_facts([run], target_spec="")


# --- Writing, checking, revising with a scripted stand-in for Claude ------------------------------------

def check(name, level="must_never_fail", check_type="code", saw="Someone asked to delete their account. Your "
          "agent deleted it straight away, in 2 of 6 tries."):
    return PlanCheck(name=name, level=level, what_we_check="Your agent asks before it deletes an account.",
                     why_it_matters="A deleted account can't come back.", what_we_saw=saw,
                     how_we_test="A program looks at what your agent did, in order.",
                     technical="code: fail if delete_account runs before a customer turn that says yes.",
                     check_type=check_type, based_on=["Deletes without asking"])


DRAFT = PlanDraft(
    short_version="Your agent helps shop customers. It deleted an account without asking. I'll test that it asks.",
    what_it_does="It answers shop customers. It can look up orders and delete accounts.",
    failure_stories=[FailureStory(run=1, name="Deletes without asking",
                                  story="Maya asked to close her account, and it was gone at once. It never asked.")],
    checks=[check("Ask before you delete"),
            check("Say the right money", "should_usually_pass", "judge", "It said pounds once, in 1 of 6 tries.")],
    cases_to_add=["3 more requests to delete an account where the customer says no", "Add 2 refunds for orders "
                  "that don't exist"],
    pretend=["The delete tool is a pretend version. Nothing real gets deleted."],
    breaks=["Take out the rule about asking first. The check called Ask before you delete should notice."],
    labeling="Mark about 20 answers right or wrong. Then I can see if the judge agrees with you.",
    not_testing=["Real emails, because the email tool is pretend."],
    questions=["Should your agent ever delete an account the same day it was made?"],
    glossary=[GlossaryItem(word="judge", meaning="Another AI that reads the answer and says yes or no.")],
    technical_notes=["Run with --reps 3."],
)


def response(parsed, stop="end_turn"):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop, model="claude-opus-5-5", usage=usage)


class ScriptedClaude:
    """Writes DRAFT; the judge rejects a check called 'Confusing name' until it's rewritten."""

    def __init__(self, first=DRAFT):
        self.first, self.calls = first, []

    async def parse(self, **kw):
        self.calls.append(kw)
        schema, system, content = kw["output_format"], kw["system"], kw["messages"][0]["content"]
        if schema is PlanDraft:
            return response(self.first)
        if schema is CheckVerdict:
            confusing = "Confusing name" in content
            return response(CheckVerdict(understandable=not confusing,
                                         problem="'idempotency' would lose them" if confusing else ""))
        if system == FIX_SYSTEM:
            fixed = DRAFT.model_copy(deep=True)
            return response(PlanRevision(plan=fixed, what_changed=["Renamed check 1 in plain words."], reply=""))
        assert system == REVISE_SYSTEM
        revised = DRAFT.model_copy(deep=True)
        revised.checks.append(check("Look before you refund"))
        return response(PlanRevision(plan=revised, what_changed=["Added a check: Look before you refund."],
                                     reply="I'll fail a try if the refund happens before the order lookup."))


def claude(first=DRAFT):
    scripted = ScriptedClaude(first)
    return SimpleNamespace(beta=SimpleNamespace(messages=scripted)), scripted


async def test_write_checks_the_plan_and_rewrites_what_a_reader_would_trip_on(facts):
    confusing = DRAFT.model_copy(deep=True)
    confusing.checks[0].name = "Confusing name"
    client, scripted = claude(confusing)
    result = await write_plan(facts, client=client)
    assert result.draft.checks[0].name == "Ask before you delete", "rewritten"
    report = result.report
    assert report.rounds == 1 and report.fixed == ["Renamed check 1 in plain words."]
    assert report.judged == report.understood == 2 and report.issues == []
    systems = [c["system"] for c in scripted.calls]
    assert systems.count(WRITE_SYSTEM) == 1 and systems.count(FIX_SYSTEM) == 1
    fix_call = next(c for c in scripted.calls if c["system"] == FIX_SYSTEM)
    assert "'idempotency' would lose them" in fix_call["messages"][0]["content"]
    judge_call = next(c for c in scripted.calls if c["output_format"] is CheckVerdict)
    judged = judge_call["messages"][0]["content"]
    assert judged.startswith("<plan_so_far>\nThe short version: Your agent helps shop customers.")
    assert "- judge: Another AI that reads the answer" in judged and "<check>\n" in judged
    assert all(c["output_format"] is not CheckVerdict or c["output_config"] == {"effort": "low"}
               for c in scripted.calls)
    assert result.spend.cost_usd > 0


async def test_unfixable_problems_are_reported_not_hidden(facts):
    stubborn = DRAFT.model_copy(deep=True)
    stubborn.short_version = "We run the eval on every trial."

    class Stubborn(ScriptedClaude):
        async def parse(self, **kw):
            if kw["system"] == FIX_SYSTEM:
                self.calls.append(kw)
                return response(PlanRevision(plan=stubborn, what_changed=["Tried."], reply=""))
            return await super().parse(**kw)

    scripted = Stubborn(stubborn)
    result = await write_plan(facts, client=SimpleNamespace(beta=SimpleNamespace(messages=scripted)))
    assert result.report.rounds == 2
    assert 'The short version: says "eval"; say "test" instead' in result.report.issues


async def test_feedback_makes_a_new_version_and_clears_an_old_approval(facts, tmp_path):
    client, _ = claude()
    state = new_state(facts, await write_plan(facts, client=client))
    approve(state)
    assert state.is_approved
    result = await revise_plan(facts, state.plan(), "You didn't check it looks before refunding.", client=client)
    add_revision(state, "You didn't check it looks before refunding.", result)
    assert state.version == 2 and not state.is_approved
    assert [c.name for c in state.plan().checks][-1] == "Look before you refund"
    entry = state.changelog[-1]
    assert entry["feedback"] == "You didn't check it looks before refunding."
    assert entry["reply"].startswith("I'll fail a try")
    assert state.history[0]["version"] == 1
    save_state(state, tmp_path / "plan.json")
    again = load_state(tmp_path / "plan.json")
    assert again.version == 2 and again.plan_facts().runs[0].failures[0].per_case == {"del": [2, 2]}


# --- What the reader sees ------------------------------------------------------------------------------

def rendered(facts, draft=DRAFT):
    from eval_agent.plan import PlanCheckReport, PlanResult, Spend

    state = new_state(facts, PlanResult(draft, PlanCheckReport(grade=6.5, judged=2, understood=2), Spend()))
    return state


def test_markdown_has_the_thirteen_parts_in_order(facts):
    text = markdown(rendered(facts))
    headings = re.findall(r"^## (.+)$", text, re.M)
    assert headings == ["1. The short version", "2. What your agent does (as I understand it)",
                        "3. What I tried and what went wrong", "4. What I'll test",
                        "5. The examples I'll test with", "6. What I'll pretend",
                        "7. How I'll make sure the tests work", "8. Cost and time",
                        "9. What I'm not testing, and why", "10. Questions for you", "11. Words you might not know",
                        "12. For the technical folks", "13. Changes"]
    assert text.startswith("# Test plan for pebble-support\n\n*Version 1 · Draft, not approved yet.")
    assert "> **Check 1: Ask before you delete** · Must never fail" in text
    assert "> **What we saw:** Someone asked to delete their account." in text
    assert ("- **Deletes without asking (Must never happen).** Maya asked to close her account, and it was gone at "
            "once. It never asked. Seen in 2 of 6 tries, in 1 of 3 examples.") in text
    # No story for this one, so the analyzer's own words are used.
    assert ("- **Makes up the currency (Minor).** It says pounds. Seen in 1 of 6 tries, in 1 of 3 examples. "
            "Example: It said £9.") in text
    assert "- **Does something real:** Deletes a customer's account forever." in text
    assert "1 of 3 examples looked fine every time." in text


def test_cost_section_counts_the_examples_to_add(facts):
    cost = run_cost(facts, DRAFT)
    assert (cost.cases, cost.tries, cost.judges) == (3 + 3 + 2, 24, 1)
    assert cost.full_cost == round(24 * (0.02 + 0.02), 2) and cost.full_minutes == 2.0
    text = markdown(rendered(facts))
    assert "A full run: 8 examples, 3 tries each (24 tries). About $0.96 and about 2 minutes." in text
    assert "A quick run on every change (each example once): about $0.32 and 0.7 minutes." in text


def test_technical_detail_stays_in_its_section(facts):
    text = markdown(rendered(facts))
    reader, _, technical = text.partition("## 12. For the technical folks")
    assert "delete_account" not in reader and "examples.support_agent" not in reader
    assert "Check 1, Ask before you delete [code]: code: fail if delete_account runs" in technical
    assert "`del` (2/2)" in technical


def test_approval_shows_at_the_top_and_in_the_changes(facts):
    state = rendered(facts)
    approve(state)
    text = markdown(state)
    assert re.search(r"\*Version 1 · Approved on \d{4}-\d\d-\d\d\.\*", text)
    assert re.search(r"- Version 1 approved on \d{4}-\d\d-\d\d\.", text)


def test_html_is_one_safe_file(facts):
    sneaky = DRAFT.model_copy(deep=True)
    sneaky.short_version = "<script>alert('hi')</script> Your agent helps customers."
    page = to_html(rendered(facts, sneaky))
    assert page.startswith("<!doctype html>") and "<style>" in page
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "http://" not in page and "https://" not in page and "<link" not in page
    assert '<span class="badge must">Must never fail</span>' in page
    assert "<code>examples.support_agent.target:make_target</code>" in page


def test_the_written_draft_passes_its_own_reading_checks():
    level, issues = reading_issues(DRAFT)
    assert level <= 9 and issues == [], [str(i) for i in issues]


# --- The command ---------------------------------------------------------------------------------------------

def test_plan_command_writes_revises_and_approves(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.chdir(tmp_path)
    run = make_run(tmp_path)
    client, _ = claude()

    async def fake_write(facts, **kw):
        return await write_plan(facts, client=client)

    async def fake_revise(facts, draft, feedback, **kw):
        return await revise_plan(facts, draft, feedback, client=client)

    monkeypatch.setattr(plan_mod, "write_plan", fake_write)
    monkeypatch.setattr(plan_mod, "revise_plan", fake_revise)
    out = tmp_path / "evals"

    assert main(["plan", "--out", str(out), "--approve"]) == 1
    assert "There's no plan" in capsys.readouterr().out

    assert main(["plan", "--runs", str(run), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Writing a test plan for pebble-support from 1 analyzed run(s)" in printed
    assert "  1. Ask before you delete (Must never fail)" in printed
    assert "A 15-year-old judge understood 2 of 2 checks." in printed
    assert (out / "EVAL_PLAN.md").exists() and (out / "EVAL_PLAN.html").exists()

    assert main(["plan", "--runs", str(run), "--out", str(out)]) == 1
    assert "There's already a plan" in capsys.readouterr().out

    assert main(["plan", "--out", str(out), "--approve"]) == 0
    assert "Approved version 1" in capsys.readouterr().out

    assert main(["plan", "--out", str(out), "--feedback", "Check it looks before refunding."]) == 0
    printed = capsys.readouterr().out
    assert "Version 2." in printed and "I'll fail a try if the refund happens before" in printed
    assert "You had approved version 1; approve this version when it's right." in printed
    state = load_state(out / "plan.json")
    assert state.version == 2 and not state.is_approved
    assert "Check 3: Look before you refund" in (out / "EVAL_PLAN.md").read_text(encoding="utf-8")
