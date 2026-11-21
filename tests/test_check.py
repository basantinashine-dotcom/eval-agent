from pathlib import Path

from eval_agent.check import changed_files, check_target, sample_args, snapshot, static_problems
from tests.minirepo import FAKES, TARGET, make_repo

SPEC = "evals.target:make_target"


def written(root: Path) -> list[Path]:
    return [root / "evals" / "fakes.py", root / "evals" / "target.py"]


def test_static_check_flags_commands_network_and_file_writes(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import subprocess\nfrom urllib.request import urlopen\nimport os\n"
        "os.system('rm -rf /')\nopen('x.txt', 'w')\nfrom pathlib import Path\nPath('y').write_text('hi')\n"
        "eval('1')\n", encoding="utf-8")
    problems = static_problems([bad], tmp_path)
    joined = "\n".join(problems)
    for expected in ("line 1: imports subprocess", "line 2: imports from urllib.request", "line 4: calls .system()",
                     "line 5: opens a file for writing", "line 7: calls .write_text()", "line 8: calls eval()"):
        assert f"bad.py {expected}" in joined


def test_static_check_allows_ordinary_fake_code(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("import json, os\nkey = os.environ.get('X')\nitems = [1, 2]\nitems.remove(1)\n"
                  "name = 'a-b'.replace('-', '_')\nwith open('data.json') as fh:\n    json.load(fh)\n",
                  encoding="utf-8")
    assert static_problems([ok], tmp_path) == []


def test_sample_args_fit_both_schema_forms():
    assert sample_args({"order_id": str, "amount": float, "email": str}) == \
        {"order_id": "test-1", "amount": 1.0, "email": "test@example.com"}
    assert sample_args({"type": "object", "properties": {"n": {"type": "integer"}, "mode": {"enum": ["a", "b"]},
                                                         "tags": {"type": "array"}}}) == \
        {"n": 1, "mode": "a", "tags": []}


def test_a_good_target_passes(tmp_path):
    root = make_repo(tmp_path)
    result = check_target(SPEC, root, written(root))
    assert result.ok, result.text()
    assert result.kind == "sdk" and result.name == "notes-bot"
    assert [(t.name, t.side_effect) for t in result.tools] == [("find_note", False), ("delete_note", True)]


def test_the_sample_agents_hand_written_target_passes():
    result = check_target("examples.support_agent.target:make_target", Path("."))
    assert result.ok, result.text()
    assert {t.name for t in result.tools} == {"get_order", "issue_refund", "send_email", "delete_account"}


def test_missing_tools_and_mismatched_schemas_are_problems(tmp_path):
    fakes = FAKES.replace(
        "    return [fake_like(agent.find_note, find_note), fake_like(agent.delete_note, delete_note, side_effect=True)]",
        "    from eval_agent.fakes import FakeTool\n"
        "    return [FakeTool('find_note', 'Find a note.', {'title': str, 'folder': str}, find_note)]")
    root = make_repo(tmp_path, fakes=fakes)
    result = check_target(SPEC, root, written(root))
    assert not result.ok
    text = result.text()
    assert "notes/delete_note is a real tool with no pretend version." in text
    assert "notes/find_note: the pretend input schema differs" in text
    assert "notes/find_note: the pretend description differs" in text


def test_unfaked_server_and_crashing_fake_are_problems(tmp_path):
    fakes = FAKES.replace('    notes = world.state.get("notes", [])\n    world.state["notes"]',
                          '    notes = world.state["notes"]\n    world.state["notes"]')
    target = TARGET.replace('{"notes": fakes.notes_fakes}', '{"notez": fakes.notes_fakes}')
    root = make_repo(tmp_path, fakes=fakes, target=target)
    text = check_target(SPEC, root, written(root)).text()
    assert "Tool server 'notes' has no pretend version." in text
    assert "There's a pretend server 'notez'" in text
    assert "Pretend tool notez/delete_note: KeyError" in text and "on an empty world" in text


def test_network_is_blocked_while_checking(tmp_path):
    target = "import socket\nsocket.create_connection(('example.com', 80), timeout=5)\n" + TARGET
    root = make_repo(tmp_path, target=target)
    result = check_target(SPEC, root)  # skip the static check to prove the process itself blocks it
    assert not result.ok
    assert "network is blocked" in result.text()


def test_changing_repo_files_is_a_problem(tmp_path):
    fakes = FAKES.replace('def find_note(args, world):\n',
                          'def find_note(args, world):\n    open("touched.txt", "w").write("x")\n')
    root = make_repo(tmp_path, fakes=fakes)
    assert any("opens a file for writing" in p for p in check_target(SPEC, root, written(root)).problems)
    result = check_target(SPEC, root)  # without the static check, the snapshot still catches it
    assert any("changed files in the repo" in p and "touched.txt (new)" in p for p in result.problems)


def test_a_broken_import_says_what_is_missing(tmp_path):
    root = make_repo(tmp_path, target="import not_a_real_package\n" + TARGET)
    result = check_target(SPEC, root, written(root))
    assert "No module named 'not_a_real_package'" in result.text() and "pip install" in result.text()


def test_snapshot_spots_new_changed_and_deleted_files(tmp_path):
    (tmp_path / "a.txt").write_text("1", encoding="utf-8")
    (tmp_path / "b.txt").write_text("1", encoding="utf-8")
    before = snapshot(tmp_path)
    (tmp_path / "a.txt").write_text("22", encoding="utf-8")
    (tmp_path / "b.txt").unlink()
    (tmp_path / "c.txt").write_text("1", encoding="utf-8")
    assert changed_files(before, snapshot(tmp_path)) == ["a.txt (changed)", "b.txt (deleted)", "c.txt (new)"]
