import json
import subprocess
import sys

from eval_agent.cli import main
from eval_agent.export import export
from eval_agent.plan import save_state
from eval_agent.standalone import repeats_text
from eval_agent.testkit import Verdict
from tests.test_build import GOOD_CHECKS, approved_state, make_run

BOT = '''\
from eval_agent.fakes import FakeTool
from eval_agent.runner import CallableTarget


def _delete(args, world):
    world.state.setdefault("deleted", []).append(args["email"])
    return {"status": "deleted"}


def make_target():
    def run(case, toolbox):
        if "delete" in case.task.lower():
            toolbox.call("delete_account", {"email": "m@x.com"})   # never asks first
            return "Deleted."
        return "It shipped."

    tools = lambda: [FakeTool("delete_account", "Delete an account.", {"email": str}, _delete, side_effect=True)]
    return CallableTarget("bot", run, tools)
'''


def project(tmp_path):
    """A repo with a tiny agent, its approved plan and built tests, exported."""
    run = make_run(tmp_path)                       # writes tmp/cases.jsonl and an analyzed run
    (tmp_path / "bot").mkdir()
    (tmp_path / "bot" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "bot" / "target.py").write_text(BOT, encoding="utf-8")
    evals = tmp_path / "evals"
    save_state(approved_state(run), evals / "plan.json")
    (evals / "checks.py").write_text(GOOD_CHECKS, encoding="utf-8")
    (evals / "expectations.json").write_text(json.dumps({"cases": {"del-yes": {"1": {}}}}), encoding="utf-8")
    (evals / "judges.json").write_text("[]", encoding="utf-8")
    written = export(evals, target="bot.target:make_target", cases=[str(tmp_path / "cases.jsonl")], root=tmp_path)
    return evals, written


def test_export_writes_a_folder_that_runs_on_its_own(tmp_path):
    evals, written = project(tmp_path)
    assert written[0] == evals / "run.py" and (evals / "run.py").exists()
    config = json.loads((evals / "run_config.json").read_text(encoding="utf-8"))
    assert config == {"target": "bot.target:make_target", "root": "..", "cases": ["cases.jsonl"]}
    vendor = evals / "_vendor" / "eval_agent"
    assert (vendor / "standalone.py").exists() and (vendor / "score.py").exists() and (vendor / "fakes.py").exists()
    assert "runs/" in (evals / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_run_py_uses_its_own_copy_and_fails_on_a_must_never_fail(tmp_path):
    evals, _ = project(tmp_path)
    probe = ("import runpy, sys\n"
             "sys.argv = ['run.py', '--reps', '2', '--no-judges']\n"
             "try:\n"
             f"    runpy.run_path(r'{evals / 'run.py'}', run_name='__main__')\n"
             "except SystemExit as exit:\n"
             "    print('EXIT', exit.code)\n"
             "print('ENGINE', sys.modules['eval_agent'].__file__)\n")
    out = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True,
                         encoding="utf-8", timeout=180)
    text = out.stdout
    assert "EXIT 1" in text, out.stderr[-2000:]
    engine = text.split("ENGINE ", 1)[1].strip()
    assert str(evals / "_vendor") in engine, "it ran on the copy in _vendor, not an installed eval-agent"
    assert "Running bot on 1 example, 2 tries each (4 tries)." not in text
    assert "Running bot on 1 example, 2 tries each (2 tries)." in text
    assert "Check 1: Ask before you delete (Must never fail). FAILED in 2 of 2 tries." in text
    assert "Checks that must never fail but did: Ask before you delete." in text
    assert "Ask before you delete: passed every time in 0 of 1 examples; failed every time in 1." in text
    runs = list((evals / "runs").iterdir())
    assert len(runs) == 1 and (runs[0] / "report.md").exists() and (runs[0] / "traces.jsonl").exists()


def test_eval_agent_run_does_the_same(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    evals, _ = project(tmp_path)
    before = __import__("os").getcwd()
    assert main(["run", "--evals", str(evals), "--quick", "--no-judges"]) == 1
    out = capsys.readouterr().out
    assert "Running bot on 1 example, 1 try each (1 try)." in out
    assert "Across the repeated tries" not in out, "one try each: nothing to compare"
    assert __import__("os").getcwd() == before, "it puts the working folder back"
    for name in [m for m in sys.modules if m.split(".")[0] == "bot"]:
        del sys.modules[name]


def test_export_needs_built_tests(tmp_path, capsys):
    assert main(["export", "--evals", str(tmp_path / "evals")]) == 1
    assert "There are no built tests" in capsys.readouterr().out


def test_repeats_say_which_examples_were_mixed():
    results = {
        "s/a#0": {1: Verdict("pass")}, "s/a#1": {1: Verdict("fail", "x")}, "s/a#2": {1: Verdict("pass")},
        "s/b#0": {1: Verdict("pass")}, "s/b#1": {1: Verdict("pass")},
        "s/c#0": {1: Verdict("fail", "x")}, "s/c#1": {1: Verdict("flag", "x")},
        "s/d#0": {1: Verdict("skip")},
    }
    text = repeats_text(results, {1: ("Ask first", "must_never_fail"), 2: ("Unused", "x")})
    assert text.endswith("Ask first: passed every time in 1 of 3 examples; failed every time in 1; mixed in 1 (a).")
    assert "Unused" not in text
