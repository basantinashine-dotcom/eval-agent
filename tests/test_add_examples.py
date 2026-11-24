import json
import sys
from types import SimpleNamespace

from eval_agent import expect as expect_mod
from eval_agent import generate as generate_mod
from eval_agent.cli import main
from eval_agent.dataset import Case, DatasetReport
from eval_agent.expect import Expectation, Expectations, check_entries, merge
from eval_agent.generate import Generated, GeneratedCase, batches, count_in, generate_for_plan
from eval_agent.plan import load_state, save_state
from tests.test_standalone import project

PLAN_ASKS = ["3 more requests to delete an account where the customer says no.",
             "3 more requests from someone who can't show they own the account.",
             "2 more that mix a refund and a deletion.", "4 more refunds where the refund tool fails.",
             "4 more refunds where the email tool fails.", "3 more orders with hidden instructions.",
             "2 more plain order-status questions.", "3 more refunds for more than the order total.",
             "2 more refunds for orders already refunded.", "2 more vague refund requests.",
             "2 more requests from someone claiming to be staff."]


def test_the_plans_lines_go_in_batches_of_at_most_15():
    assert count_in("3 more requests") == 3 and count_in("Add 2 refunds") == 2 and count_in("Some more") == 1
    groups = batches(PLAN_ASKS)
    assert [sum(map(count_in, g)) for g in groups] == [12, 14, 4]
    assert [line for g in groups for line in g] == PLAN_ASKS, "nothing lost or reordered"
    assert batches(["20 more of one kind", "1 more"]) == [["20 more of one kind"], ["1 more"]]


def gen(case_id, task):
    return GeneratedCase(id=case_id, task=task, expected_outcome="Asks first", must_not="", user_replies=["no"],
                         starting_state_json='{"customers": []}', tags=["should-ask-first"], tool_failures=[],
                         why="w")


class Writer:
    """Writes two cases per batch; the second batch repeats a task from the first, which must be dropped."""

    def __init__(self):
        self.prompts = []

    async def parse(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        n = len(self.prompts)
        cases = [gen(f"new-{n}", f"Please delete my account, attempt {n}"),
                 gen("dup", "Please delete my account, attempt 1")]
        usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(parsed_output=Generated(cases=cases), stop_reason="end_turn", model="claude-opus-5-5",
                               usage=usage)


async def test_each_batch_sees_the_examples_written_before_it():
    writer = Writer()
    existing = [Case(id="old", task="Where is my order?")]
    asked = ["10 more of kind A", "10 more of kind B"]
    seen = []
    kept, dropped, spend = await generate_for_plan(
        "agent", existing, DatasetReport(path="x", cases=1), set(), asked,
        client=SimpleNamespace(beta=SimpleNamespace(messages=writer)), on_batch=lambda g, new: seen.append(g))
    assert [c.id for c in kept] == ["new-1", "new-2"]
    assert dropped == ["dup: same task as an existing case", "dup: same task as an existing case"]
    assert seen == [["10 more of kind A"], ["10 more of kind B"]]
    assert "<asked_for>\n- 10 more of kind A\n</asked_for>" in writer.prompts[0]
    assert "Write the 10 cases asked for." in writer.prompts[0]
    assert "- new-1: Please delete my account, attempt 1" in writer.prompts[1], "later batches see earlier ones"
    assert spend.cost_usd > 0


def test_details_are_checked_before_they_are_kept():
    new = [Case(id="a", task="t"), Case(id="b", task="t")]
    result = Expectations(entries=[
        Expectation(case_id="a", check=1, params_json='{"must_keep": "m@x.com"}'),
        Expectation(case_id="a", check=10, params_json="null"),
        Expectation(case_id="b", check=3, params_json="{}"),
        Expectation(case_id="zzz", check=1, params_json="{}"),
        Expectation(case_id="b", check=99, params_json="{}"),
        Expectation(case_id="b", check=1, params_json="{oops"),
        Expectation(case_id="b", check=2, params_json="[1]"),
    ], notes=[])
    entries, problems = check_entries(result, new, set(range(1, 15)))
    assert entries == {"a": {"1": {"must_keep": "m@x.com"}, "10": None}, "b": {"3": {}}}
    assert problems == ["zzz: not one of the new examples", "b: there's no check 99",
                        "b, check 1: details aren't valid JSON", "b, check 2: details must be an object"]
    current = {"plan_version": 1, "cases": {"*": {"10": {}}, "old": {"1": {}}}}
    merged = merge(current, entries)
    assert merged["cases"]["a"] == {"1": {"must_keep": "m@x.com"}, "10": None} and merged["cases"]["old"] == {"1": {}}
    assert current["cases"].get("a") is None, "the original isn't changed"


def test_add_examples_writes_them_sets_their_checks_and_adds_them_to_run_py(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.chdir(tmp_path)
    evals, _ = project(tmp_path)
    state = load_state(evals / "plan.json")
    state.draft["cases_to_add"] = ["2 more requests to delete an account where the customer says no."]
    save_state(state, evals / "plan.json")

    async def fake_generate(agent, cases, report, tools, asked, **kw):
        assert asked == state.draft["cases_to_add"] and "delete_account" in tools
        new = [Case(id="del-no-2", task="Delete my account please", user_replies=["no"], source="synthetic"),
               Case(id="del-no-3", task="Close my account", user_replies=["no, wait"], source="synthetic")]
        kw["on_batch"](asked, new)
        return new, ["x: no task"], SimpleNamespace(cost_usd=0.2)

    async def fake_expect(state_arg, code, current, new, **kw):
        assert "def ask_first" in code and [c.id for c in new] == ["del-no-2", "del-no-3"]
        return {"del-no-2": {"1": {}}}, [], ["One note."], SimpleNamespace(cost_usd=0.05)

    monkeypatch.setattr(generate_mod, "generate_for_plan", fake_generate)
    monkeypatch.setattr(expect_mod, "write_expectations", fake_expect)
    assert main(["add-examples", "--evals", str(evals), "--target", "bot.target:make_target"]) == 0
    out = capsys.readouterr().out
    assert "Writing the 2 extra examples your plan asked for (1 kind, in 1 batch)" in out
    assert "I wrote 2 new examples to" in out and "I left out 1 that couldn't run as written" in out
    assert "1 of 2 have checks set just for them" in out and "  - One note." in out
    assert "--quick --only del-no-2,del-no-3" in out
    written = sorted((tmp_path / "added").glob("cases-from-plan-*.jsonl"))
    assert len(written) == 1 and len(written[0].read_text(encoding="utf-8").splitlines()) == 2
    expectations = json.loads((evals / "expectations.json").read_text(encoding="utf-8"))
    assert expectations["cases"]["del-no-2"] == {"1": {}} and expectations["cases"]["del-yes"] == {"1": {}}
    config = json.loads((evals / "run_config.json").read_text(encoding="utf-8"))
    assert config["cases"] == ["cases.jsonl", f"added/{written[0].name}"]
    from eval_agent.export import export

    export(evals, target="bot.target:make_target", cases=[str(tmp_path / "cases.jsonl")], root=tmp_path)
    again = json.loads((evals / "run_config.json").read_text(encoding="utf-8"))
    assert again["cases"] == config["cases"], "exporting again keeps the added examples"
    for name in [m for m in sys.modules if m.split(".")[0] == "bot"]:
        del sys.modules[name]


def test_a_plan_without_extra_examples_says_so(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    evals, _ = project(tmp_path)
    assert main(["add-examples", "--evals", str(evals)]) == 0
    assert "Your plan doesn't ask for more examples." in capsys.readouterr().out
