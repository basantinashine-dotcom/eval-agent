import pytest

from eval_agent.dataset import load_cases

SAMPLE = "examples/support_agent/cases.jsonl"


def test_sample_cases_read_cleanly():
    cases, report = load_cases(SAMPLE)
    assert report.cases == len(cases) == 10
    assert not report.skipped and not report.problems and not report.missing_outcome
    assert not report.duplicate_tasks
    assert report.should_not_act == 7
    assert report.with_replies == 3
    by_id = {c.id: c for c in cases}
    assert by_id["refund-email-fails"].inject_failures == {"send_email": {"mode": "error", "times": "all"}}
    assert by_id["delete-confirm-yes"].user_replies == ["Yes, delete it."]
    assert "Gaps: Only 10 cases" in report.summary()


def test_csv_with_other_column_names(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text(
        "Prompt,Expected,Tags,Replies,Extra\n"
        "Delete my account,Asks first,Should Ask First,yes | no,x\n"
        ",No task here,,,\n"
        "Where is order 5?,Says it shipped,status,,\n",
        encoding="utf-8",
    )
    cases, report = load_cases(path)
    assert [c.task for c in cases] == ["Delete my account", "Where is order 5?"]
    assert cases[0].expected_outcome == "Asks first"
    assert cases[0].tags == ["should-ask-first"]
    assert cases[0].user_replies == ["yes", "no"]
    assert cases[0].id == "case-001"
    assert report.skipped == ["row 2: no task"]
    assert report.ignored_columns == {"Extra"}
    assert report.should_not_act == 1


def test_bad_jsonl_lines_are_skipped_with_a_reason(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"task": "a", "expected_outcome": "b"}\nnot json\n{"task": "c"}\n', encoding="utf-8")
    cases, report = load_cases(path)
    assert [c.task for c in cases] == ["a", "c"]
    assert report.skipped[0].startswith("line 2: not valid JSON")
    assert report.missing_outcome == ["case-003"]
    assert any("can't be graded" in gap for gap in report.gaps())
    assert any("refuse or ask first" in gap for gap in report.gaps())


def test_duplicates_and_reused_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "x", "task": "Refund order 1"}\n{"id": "x", "task": "refund order 1!"}\n',
                    encoding="utf-8")
    cases, report = load_cases(path)
    assert [c.id for c in cases] == ["x", "x-2"]
    assert report.duplicate_tasks == [("x", "x-2")]


def test_spreadsheet_files_get_a_helpful_message(tmp_path):
    path = tmp_path / "cases.xlsx"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Save it as CSV"):
        load_cases(path)
