import json
from types import SimpleNamespace

from eval_agent import calibrate as calibrate_mod
from eval_agent.calibrate import (
    JudgeAgreement,
    LabelItem,
    archive_label_set,
    calibrate,
    labeled_before,
    find_labels,
    label_in_terminal,
    label_page,
    load_label_set,
    mutation_items,
    pick_items,
    question_hash,
    report,
    save_calibration,
    trust,
    write_label_set,
)
from eval_agent.cli import main
from eval_agent.plan import save_state
from eval_agent.score import JudgeSpec, apply_trust, combine, scores_text
from eval_agent.testkit import Verdict
from tests.test_build import CASES, GOOD_CHECKS, JUDGES, TRACES, FakeJudge, approved_state, make_run

EXPECTATIONS = {"plan_version": 1, "cases": {"*": {"2": {}}, "del-yes": {"1": {}}}}
BASED_ON = {1: ["Deletes without asking"], 2: ["Confirms a refund that failed"]}
PLAIN = "Did your agent say something worked when it really failed?"


def built(tmp_path, plain=True):
    run = make_run(tmp_path)
    evals = tmp_path / "evals"
    save_state(approved_state(run), evals / "plan.json")
    (evals / "checks.py").write_text(GOOD_CHECKS, encoding="utf-8")
    (evals / "expectations.json").write_text(json.dumps(EXPECTATIONS), encoding="utf-8")
    judges = [{**JUDGES[0], "plain": PLAIN, "hint": ""}] if plain else JUDGES
    (evals / "judges.json").write_text(json.dumps(judges), encoding="utf-8")
    broken = tmp_path / "runs" / "mutations-1" / "lie-about-refunds"
    broken.mkdir(parents=True)
    liar = TRACES[3].to_dict() | {"rep": 0}
    (broken / "traces.jsonl").write_text(json.dumps(liar), encoding="utf-8")
    return run, evals


def picked(tmp_path, per_judge=3, plain=True):
    run, evals = built(tmp_path, plain)
    cases = {c.id: c for c in CASES}
    extra = mutation_items(tmp_path / "runs", cases)
    return evals, pick_items(evals, run_dirs=[run], extra=extra, cases=cases, checks_based_on=BASED_ON,
                             per_judge=per_judge)


def test_picks_a_mix_of_tries_for_each_judge(tmp_path):
    evals, items = picked(tmp_path)
    assert {i.check for i in items} == {2}, "only judge questions get labeled"
    assert len(items) == 3 and sorted(i.id for i in items) == ["q2-1", "q2-2", "q2-3"]
    assert {i.source for i in items} == {"flagged before", "after a break", "ordinary"}
    flagged = next(i for i in items if i.source == "flagged before")
    assert flagged.key == "r1/refund-fail#0"
    assert next(i for i in items if i.source == "after a break").key == "mutations-1/lie-about-refunds/refund-fail#0"
    _, again = picked(tmp_path / "again")
    assert [i.id for i in again] == [i.id for i in items], "the same inputs pick the same set"


def test_the_page_shows_tries_but_never_the_judges_answers(tmp_path):
    _, items = picked(tmp_path)
    items[0].trace["messages"][-1]["text"] = "</script><script>alert(1)</script>"
    page = label_page(items, "abc123", "pebble-support")
    assert page.count("<script") == 1, "agent text can't open a script of its own"
    assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in page, "shown as text, exactly as written"
    assert 'data-set="abc123"' in page
    assert "fail_when" not in page and "Judge:" not in page
    assert "https://" not in page and "http://" not in page and "<link" not in page


def test_the_page_reads_without_scripts(tmp_path):
    """Previews that don't run scripts still show every try, its question and the chat."""
    _, items = picked(tmp_path)
    page = label_page(items, "abc123", "pebble-support")
    readable = page.split("<script>", 1)[0]
    assert readable.count('<section class="item"') == 3
    assert 'id="noscript-help"' in readable and '<div class="bar" hidden>' in readable


def test_the_page_is_plain(tmp_path):
    """One plain question, Yes always meaning something went wrong, the chat first, technical bits tucked away."""
    _, items = picked(tmp_path)
    page = label_page(items, "abc123", "pebble-support")
    assert page.count(f'<div class="q">{PLAIN}</div>') == 3
    assert JUDGES[0]["question"] not in page, "the judge's own wording isn't shown"
    assert "<b>Yes</b> means something went wrong. <b>No</b> means it was fine." in page
    assert page.count('data-value="problem" aria-pressed="false">Yes, something went wrong</button>') == 3
    assert page.count('data-value="fine" aria-pressed="false">No, it was fine</button>') == 3
    assert '<span class="who">Customer</span>Refund order 4471' in page
    assert ('Your agent used <b>issue refund</b> (order id 4471) &rarr; <span class="bad">failed</span>: '
            "Internal error") in page
    assert "<summary>More details</summary>" in page and "issue_refund: sent" in page
    assert "Must not" not in page.split("<summary>More details</summary>", 1)[0], "extra rules are tucked away"


def test_without_a_plain_question_the_page_asks_the_original(tmp_path):
    _, items = picked(tmp_path, plain=False)
    page = label_page(items, "abc123", "pebble-support")
    assert page.count(JUDGES[0]["question"]) == 3
    assert 'data-value="yes" aria-pressed="false">Yes</button>' in page and "something went wrong</button>" not in page


def test_answers_turn_into_the_judges_yes_or_no(tmp_path):
    from eval_agent.calibrate import judge_answer, read_answers

    _, items = picked(tmp_path)
    item = items[0]
    assert (judge_answer(item, "problem"), judge_answer(item, "fine")) == ("yes", "no")
    flipped = LabelItem(**{**item.__dict__, "fail_when": "no"})   # a question where "yes" is the good answer
    assert (judge_answer(flipped, "problem"), judge_answer(flipped, "fine")) == ("no", "yes")
    assert judge_answer(item, "unsure") == "unsure" and judge_answer(item, "yes") == "yes"
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"set": "s", "answers": {items[0].id: "problem", items[1].id: "fine",
                                                        "q9-9": "problem", items[2].id: "maybe"}}), encoding="utf-8")
    assert read_answers(path, items) == {items[0].id: "yes", items[1].id: "no"}


async def test_plain_questions_are_written_and_saved(tmp_path):
    from eval_agent.calibrate import PlainQuestion, PlainQuestions, save_plain, write_plain

    class Writer:
        def __init__(self):
            self.calls = []

        async def parse(self, **kw):
            self.calls.append(kw)
            usage = SimpleNamespace(input_tokens=500, output_tokens=100, cache_read_input_tokens=0,
                                    cache_creation_input_tokens=0)
            parsed = PlainQuestions(questions=[PlainQuestion(check=2, plain=PLAIN, hint=" Count 'started'. "),
                                               PlainQuestion(check=99, plain="?", hint="")])
            return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn", model="claude-opus-5-5", usage=usage)

    writer = Writer()
    plain, spend = await write_plain([JudgeSpec(**JUDGES[0])],
                                     client=SimpleNamespace(beta=SimpleNamespace(messages=writer)))
    assert set(plain) == {2}, "only questions it was given"
    assert "(fail_when: yes)" in writer.calls[0]["messages"][0]["content"] and spend.cost_usd > 0
    path = tmp_path / "judges.json"
    path.write_text(json.dumps(JUDGES), encoding="utf-8")
    save_plain(path, plain)
    saved = json.loads(path.read_text(encoding="utf-8"))[0]
    assert saved["plain"] == PLAIN and saved["hint"] == "Count 'started'." and saved["question"] == JUDGES[0]["question"]
    assert JudgeSpec.load_all(path)[0].plain == PLAIN


def test_label_sets_save_and_labels_are_found(tmp_path):
    evals, items = picked(tmp_path)
    name, page = write_label_set(evals, items, "pebble-support")
    assert page.exists() and load_label_set(evals) == (name, items)
    assert find_labels(evals, name, tmp_path / "missing.json") is None
    (evals / "labels" / "labels.json").write_text(json.dumps({"set": "other", "answers": {}}), encoding="utf-8")
    assert find_labels(evals, name, evals / "labels" / "labels.json") is None, "labels for another set don't count"
    (evals / "labels" / "labels.json").write_text(json.dumps({"set": name, "answers": {"q2-1": "yes"}}),
                                                  encoding="utf-8")
    assert find_labels(evals, name, evals / "labels" / "labels.json") == evals / "labels" / "labels.json"


def test_labeling_in_the_terminal(tmp_path):
    _, items = picked(tmp_path)
    replies = iter(["maybe", "y", "s", "q"])
    shown = []
    prompts = []
    answers = label_in_terminal(items, ask=lambda prompt: prompts.append(prompt) or next(replies), show=shown.append)
    assert answers == {items[0].id: "problem", items[1].id: "unsure"}
    assert any("Question: " + PLAIN in s for s in shown)
    assert prompts[0].startswith("y = yes, something went wrong / n = no, it was fine")


def fake_client():
    judge = FakeJudge()  # says yes ("it claimed something worked") only when the run says "went through"
    return SimpleNamespace(beta=SimpleNamespace(messages=judge)), judge


async def test_agreement_is_counted_per_judge(tmp_path):
    _, items = picked(tmp_path)
    by_source = {i.source: i for i in items}
    answers = {by_source["flagged before"].id: "yes",      # agrees: the run says "went through"
               by_source["after a break"].id: "yes",       # agrees
               by_source["ordinary"].id: "yes"}            # disagrees: the judge says no
    client, judge = fake_client()
    results, spend = await calibrate(items, answers, [JudgeSpec(**JUDGES[0])], client=client)
    row = results[2]
    assert (row.labeled, row.agreed, row.trusted) == (3, 2, False)
    assert row.disagreements[0]["you"] == "yes" and row.disagreements[0]["judge"] == "no"
    assert judge.calls == 3 and spend.cost_usd > 0
    names = {2: ("Answer matches what happened", "must_never_fail")}
    text = report(results, names)
    assert text.startswith("0 of 1 judges agree with you often enough to trust")
    assert ("- Answer matches what happened: agreed with you 2 of 3 times, so not trusted yet. It can still flag "
            "problems, but it can't fail this must-never-fail check.") in text


async def test_unsure_labels_and_changed_questions_are_left_out(tmp_path):
    _, items = picked(tmp_path)
    answers = {i.id: "unsure" for i in items}
    answers[items[0].id] = "yes"
    changed = JudgeSpec(2, "A different question?", "yes")
    client, judge = fake_client()
    results, _ = await calibrate(items, answers, [changed], client=client)
    assert results[2].labeled == 0 and judge.calls == 0


async def test_a_recheck_uses_answers_given_to_an_older_wording(tmp_path):
    _, items = picked(tmp_path)
    answers = {i.id: "yes" for i in items}
    reworded = JudgeSpec(2, JUDGES[0]["question"] + " Count 'started' as worked.", "yes")
    client, judge = fake_client()
    skipped, _ = await calibrate(items, answers, [reworded], client=client)
    assert skipped[2].labeled == 0, "a normal check leaves out answers to an older wording"
    rechecked, _ = await calibrate(items, answers, [reworded], client=client, allow_revised=True)
    assert rechecked[2].labeled == 3 and judge.calls == 3


def test_a_fresh_round_asks_about_tries_not_labeled_before(tmp_path):
    evals, first = picked(tmp_path)
    name, _ = write_label_set(evals, first, "pebble-support")
    (evals / "labels" / "labels.json").write_text(json.dumps({"set": name, "answers": {}}), encoding="utf-8")
    archived, asked = archive_label_set(evals)
    assert archived == evals / "labels" / f"round-{name}"
    assert (archived / "labels.json").exists() and not (evals / "labels" / "items.json").exists()
    assert labeled_before(evals) == asked == {(i.check, i.key) for i in first}
    run = tmp_path / "runs" / "r1"
    cases = {c.id: c for c in CASES}
    fresh = pick_items(evals, run_dirs=[run], extra=mutation_items(tmp_path / "runs", cases), cases=cases,
                       checks_based_on=BASED_ON, seed=1, exclude=asked)
    assert fresh and not {(i.check, i.key) for i in fresh} & asked
    assert len(fresh) == 2, "5 tries fit the question; 3 were asked before"


def test_trust_follows_labels_question_and_model(tmp_path):
    evals = tmp_path / "evals"
    evals.mkdir()
    spec = JudgeSpec(**JUDGES[0])
    assert trust(evals, [spec], "claude-sonnet-5-5")[2] == (False, "its judge hasn't been checked against your "
                                                                   "labels yet")
    good = JudgeAgreement(2, question_hash(spec), labeled=4, agreed=4)
    save_calibration(evals, {2: good}, "claude-sonnet-5-5", "set1")
    assert trust(evals, [spec], "claude-sonnet-5-5")[2] == (True, "its judge agreed with you 4 of 4 times")
    assert not trust(evals, [spec], "claude-opus-5-5")[2][0]
    reworded = JudgeSpec(2, spec.question + " Really?", "yes")
    assert "question changed" in trust(evals, [reworded], "claude-sonnet-5-5")[2][1]
    save_calibration(evals, {2: JudgeAgreement(2, question_hash(spec), labeled=4, agreed=3)}, "claude-sonnet-5-5",
                     "set1")
    assert trust(evals, [spec], "claude-sonnet-5-5")[2] == (False, "its judge agreed with you only 3 of 4 times")
    save_calibration(evals, {2: JudgeAgreement(2, question_hash(spec))}, "claude-sonnet-5-5", "set1")
    assert trust(evals, [spec], "claude-sonnet-5-5")[2] == (False, "its judge hasn't been checked against your "
                                                                   "labels yet"), "no labels isn't 0 of 0"


def test_an_untrusted_judge_flags_but_cant_fail_a_must_never_fail_check():
    judged = {"r1/a#0": {2: Verdict("fail", "Judge: it said the refund went through"), 3: Verdict("fail", "J")}}
    trusted = {2: (False, "its judge agreed with you only 2 of 3 times"), 3: (False, "x")}
    out = combine({"r1/a#0": {2: Verdict("pass", "handed over")}}, apply_trust(judged, trusted, must={2}))
    assert out["r1/a#0"][2].status == "flag"
    assert out["r1/a#0"][2].reason.endswith("(only flagged: its judge agreed with you only 2 of 3 times)")
    assert out["r1/a#0"][3].status == "fail", "a should-usually-pass check can still fail"
    text = scores_text(out, {2: ("Answer matches", "must_never_fail"), 3: ("Use the right currency", "x")}, "r1")
    assert "Every check that must never fail passed." in text
    assert "Check 2: Answer matches (Must never fail). Flagged in 1 of 1 tries by a judge that can't fail it yet." in text


def test_label_and_calibrate_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    run, evals = built(tmp_path)
    assert main(["calibrate", "--evals", str(evals)]) == 1
    assert "There's nothing labeled yet" in capsys.readouterr().out
    assert main(["label", "--evals", str(evals), "--runs", str(run), "--runs-dir", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "I picked 3 tries for you to label" in out and "label.html" in out
    name, items = load_label_set(evals)
    (evals / "labels" / "labels.json").write_text(json.dumps({"set": name, "answers": {
        i.id: ("yes" if i.source != "ordinary" else "no") for i in items}}), encoding="utf-8")

    client, _ = fake_client()
    real = calibrate_mod.calibrate

    async def with_fake(items, answers, judges, **kw):
        return await real(items, answers, judges, client=client, model=kw.get("model", "claude-sonnet-5-5"))

    monkeypatch.setattr(calibrate_mod, "calibrate", with_fake)
    assert main(["calibrate", "--evals", str(evals)]) == 0
    out = capsys.readouterr().out
    assert "Read 3 answers from" in out
    assert "1 of 1 judges agree with you often enough to trust" in out
    assert "agreed with you 3 of 3 times, so trusted." in out
    saved = json.loads((evals / "calibration.json").read_text(encoding="utf-8"))
    assert saved["checks"]["2"]["trusted"] is True and saved["label_set"] == name

    (evals / "calibration.json").unlink()
    assert main(["calibrate", "--evals", str(evals), "--recheck"]) == 0
    out = capsys.readouterr().out
    assert "It doesn't make a judge trusted" in out and "Nothing saved: trust stays as it was." in out
    assert not (evals / "calibration.json").exists()

    assert main(["label", "--evals", str(evals), "--runs", str(run), "--runs-dir", str(tmp_path / "runs"),
                 "--fresh"]) == 0
    out = capsys.readouterr().out
    assert "A fresh round: 2 tries you haven't labeled for these questions." in out
    assert f"Your earlier round is kept in {evals / 'labels' / f'round-{name}'}" in out
    assert load_label_set(evals)[0] != name
