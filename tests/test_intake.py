import shutil
import subprocess
import sys

import pytest
import yaml
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from eval_agent.check import CheckResult, ToolInfo, call_server_tool
from eval_agent.cli import main, smoke_run
from eval_agent.intake import RepoFiles, estimate_cost, run_intake, secret_reason
from eval_agent.toolmap import SmokeResult, map_problems, summary, tool_map_data, tool_map_yaml
from tests.minirepo import FAKES, TARGET, make_repo

MAP = {
    "agent_name": "notes-bot", "framework": "claude_agent_sdk", "summary": "Finds and deletes notes.",
    "entry_point": "notes_bot/agent.py: build_options()", "model": "default",
    "tools": [
        {"name": "find_note", "server": "notes", "what_it_does": "finds a note", "changes_real_world": False,
         "how_faked": "looks in the pretend notes"},
        {"name": "delete_note", "server": "notes", "what_it_does": "deletes a note forever",
         "changes_real_world": True, "how_faked": "removes it from the pretend notes"},
    ],
    "never_do": ["Delete a note without asking"], "not_covered": [], "questions": ["Is delete undoable?"],
    "smoke_task": "Find my Groceries note",
}


@pytest.fixture(autouse=True)
def forget_repo_modules():
    yield
    for name in [m for m in sys.modules if m.split(".")[0] in {"evals", "notes_bot"}]:
        del sys.modules[name]


# --- What the setup agent may read and write -------------------------------------------------------------

def test_lists_code_but_not_secrets_caches_or_excluded_files(tmp_path):
    root = make_repo(tmp_path, with_evals=False)
    for rel in (".venv/lib/x.py", "notes_bot/__pycache__/agent.pyc", "certs/server.pem", "PLANTED.md",
                ".env.local", ".env.example", "logo.png"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x", encoding="utf-8")
    files = RepoFiles(root, exclude=("PLANTED*",)).files()
    assert files == [".env.example", "README.md", "notes_bot/__init__.py", "notes_bot/agent.py"]


def test_respects_gitignore_in_a_git_repo(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git isn't installed")
    root = make_repo(tmp_path, with_evals=False)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("config.toml\n", encoding="utf-8")
    (root / "config.toml").write_text("handle = 'me'\n", encoding="utf-8")
    files = RepoFiles(root).files()
    assert "config.toml" not in files and "notes_bot/agent.py" in files and ".gitignore" in files


def test_secret_names():
    assert secret_reason(".env") and secret_reason("app/.env.production") and secret_reason("id_rsa")
    assert secret_reason("client_secret.json") and secret_reason("data/app.sqlite")
    assert not secret_reason(".env.example") and not secret_reason("agent.py")


def test_reading_is_limited_to_the_repos_readable_files(tmp_path):
    root = make_repo(tmp_path / "repo", with_evals=False)
    (tmp_path / "outside.txt").write_text("private", encoding="utf-8")
    files = RepoFiles(root)
    with pytest.raises(PermissionError, match="may hold keys"):
        files.read(".env")
    with pytest.raises(PermissionError, match="outside the repo"):
        files.read("../outside.txt")
    text = files.read("notes_bot/agent.py", start=4, count=2)
    assert text.splitlines()[0] == "notes_bot/agent.py"
    assert '    4  @tool("find_note"' in text and "read on with start_line=6" in text


def test_search_finds_tool_definitions(tmp_path):
    files = RepoFiles(make_repo(tmp_path, with_evals=False))
    assert files.search(r"@tool\(") == ('notes_bot/agent.py:4: @tool("find_note", "Find a note by its title.", '
                                        '{"title": str})\nnotes_bot/agent.py:9: @tool("delete_note", "Delete a note '
                                        'forever.", {"title": str})')
    assert files.search("nothing-like-this") == "No matches."
    assert "valid regular expression" in files.search("(")


def test_writes_only_inside_the_output_folder(tmp_path):
    root = make_repo(tmp_path, with_evals=False)
    files = RepoFiles(root)
    with pytest.raises(PermissionError, match="only write inside evals/"):
        files.write("notes_bot/agent.py", "oops")
    with pytest.raises(PermissionError, match="only write"):
        files.write("evals/run.sh", "rm -rf /")
    assert files.write("evals/fakes.py", FAKES) == f"Wrote evals/fakes.py ({len(FAKES.splitlines())} lines)."
    assert (root / "evals" / "__init__.py").exists(), "made a package so `evals.target` resolves to this folder"
    assert files.write("evals/fakes.py", FAKES + "\n").startswith("Wrote"), "it may replace its own files"
    assert "evals/fakes.py" in files.files() and "evals/__init__.py" in files.files()


def test_never_replaces_files_it_did_not_write(tmp_path):
    root = make_repo(tmp_path)
    with pytest.raises(PermissionError, match="already exists"):
        RepoFiles(root).write("evals/target.py", "x")
    assert RepoFiles(root, overwrite=True).write("evals/target.py", TARGET).startswith("Wrote")


def test_estimate_grows_with_the_code(tmp_path):
    small = estimate_cost(RepoFiles(make_repo(tmp_path / "a", with_evals=False)))
    big_root = make_repo(tmp_path / "b", with_evals=False)
    (big_root / "notes_bot" / "big.py").write_text("x = 1\n" * 50_000, encoding="utf-8")
    assert 0 < small < estimate_cost(RepoFiles(big_root)) < 5


# --- The setup agent's session, with a scripted stand-in for Claude --------------------------------------

class ScriptedIntake:
    """Stands in for ClaudeSDKClient: each query runs the next list of (tool, args) through the real tools."""

    def __init__(self, options, turns, result=None):
        self.options, self.turns = options, list(turns)
        self.server = options.mcp_servers["intake"]["instance"]
        self.prompts, self.outputs = [], []
        self.result = result or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.prompts.append(prompt)
        for name, args in (self.turns.pop(0) if self.turns else []):
            self.outputs.append((name, *await call_server_tool(self.server, name, args)))

    async def receive_response(self):
        yield AssistantMessage(content=[TextBlock(text="Done: the notes agent is set up.")], model="m")
        yield ResultMessage(**{"subtype": "success", "duration_ms": 1, "duration_api_ms": 1, "is_error": False,
                               "num_turns": 5, "session_id": "s", "total_cost_usd": 0.42, **self.result})


def scripted(*turns, result=None):
    clients = []

    def factory(options):
        clients.append(ScriptedIntake(options, turns, result))
        return clients[-1]

    return factory, clients


async def test_setup_session_reads_writes_checks_and_hands_back_a_map(tmp_path):
    root = make_repo(tmp_path, with_evals=False)
    wrong_map = {**MAP, "tools": [{**MAP["tools"][1], "changes_real_world": False}, MAP["tools"][0]]}
    factory, clients = scripted([
        ("list_files", {}),
        ("read_file", {"path": ".env"}),
        ("write_file", {"path": "notes_bot/agent.py", "content": "hacked"}),
        ("submit_map", MAP),                       # too early: nothing checked yet
        ("write_file", {"path": "evals/fakes.py", "content": FAKES}),
        ("write_file", {"path": "evals/target.py", "content": TARGET}),
        ("check_target", {}),
        ("submit_map", wrong_map),                 # says delete_note doesn't change anything
        ("submit_map", MAP),
    ])
    lines = []
    result, files = await run_intake(root, emit=lines.append, client_factory=factory)

    outputs = {name: [] for name, _, _ in clients[0].outputs}
    for name, text, is_error in clients[0].outputs:
        outputs[name].append((text, is_error))
    assert ".env" not in outputs["list_files"][0][0].splitlines()
    assert outputs["read_file"][0][1] and "won't read .env" in outputs["read_file"][0][0]
    assert outputs["write_file"][0][1], "can't write outside evals/"
    assert (root / "notes_bot" / "agent.py").read_text(encoding="utf-8").startswith("from claude_agent_sdk")
    assert outputs["submit_map"][0] == ("check_target hasn't passed yet. Fix the problems and check again first.",
                                        True)
    assert outputs["check_target"][0] == ("PASSED: notes-bot (sdk)\nPretend tools: notes/find_note, "
                                          "notes/delete_note (changes things)", False)
    assert outputs["submit_map"][1][1] and "changes_real_world=False" in outputs["submit_map"][1][0]
    assert outputs["submit_map"][2] == ("Saved. Now reply with one short sentence and stop.", False)

    assert result.ok and result.map["agent_name"] == "notes-bot"
    assert result.written == ["evals/__init__.py", "evals/fakes.py", "evals/target.py"]
    assert result.cost_usd == 0.42 and result.checks_run == 1
    assert len(clients[0].prompts) == 1, "no nudge needed"
    assert "The repo is" in clients[0].prompts[0] and "evals.target:make_target" in clients[0].prompts[0]
    options = clients[0].options
    assert options.tools == [] and options.setting_sources == [] and options.strict_mcp_config
    assert options.max_budget_usd == 3.0 and str(root) not in str(options.cwd)
    assert "checking what I wrote" in lines and f"wrote evals/fakes.py ({len(FAKES.splitlines())} lines)." in lines


async def test_it_gets_one_nudge_when_it_stops_early(tmp_path):
    root = make_repo(tmp_path, with_evals=False)
    factory, clients = scripted([("list_files", {})], [])
    result, _ = await run_intake(root, emit=lambda line: None, client_factory=factory, note="It starts in agent.py")
    assert not result.ok and result.map is None
    assert len(clients[0].prompts) == 2 and "call submit_map" in clients[0].prompts[1]
    assert "A note from the user" in clients[0].prompts[0] and "It starts in agent.py" in clients[0].prompts[0]


async def test_reaching_the_cost_cap_stops_it(tmp_path):
    root = make_repo(tmp_path, with_evals=False)
    factory, clients = scripted([], result={"subtype": "error_max_budget_usd", "is_error": True})
    result, _ = await run_intake(root, emit=lambda line: None, client_factory=factory, max_cost=0.5)
    assert result.stopped == "It reached the cost cap ($0.50)." and len(clients[0].prompts) == 1


# --- The tool map and the summary -------------------------------------------------------------------------

CHECK = CheckResult(ok=True, kind="sdk", name="notes-bot", tools=[
    ToolInfo("notes", "find_note", "Find", False), ToolInfo("notes", "delete_note", "Delete", True)])


def test_map_problems_spot_disagreements():
    assert map_problems(MAP, CHECK) == []
    extra = {**MAP, "tools": MAP["tools"][:1] + [{"name": "send", "server": "mail", "changes_real_world": True}]}
    assert map_problems(extra, CHECK) == [
        "notes/delete_note has a pretend version but isn't in the map.",
        "mail/send is in the map but has no pretend version (check the server name).",
    ]


def test_tool_map_yaml_reads_back():
    text = tool_map_yaml(tool_map_data(MAP, CHECK, "evals.target:make_target"))
    assert text.startswith("# Written by `eval-agent setup`")
    data = yaml.safe_load(text)
    assert data["framework"] == "Claude Agent SDK" and data["check"] == "passed"
    assert data["tools"][1] == {"name": "delete_note", "server": "notes", "what_it_does": "deletes a note forever",
                                "changes_real_world": True, "faked": True,
                                "how_faked": "removes it from the pretend notes"}
    assert data["never_do_proposed"] == ["Delete a note without asking"]


def test_summary_reads_plainly():
    data = tool_map_data(MAP, CHECK, "evals.target:make_target")
    smoke = SmokeResult(ran=True, status="completed", task="Find my Groceries note", tools=["find_note"],
                        answer="Your Groceries note says: eggs.", cost_usd=0.02)
    text = summary(data, ["evals/fakes.py", "evals/target.py"], 0.84, smoke, "evals/runs/1")
    assert "Your agent: notes-bot (Claude Agent SDK). Finds and deletes notes." in text
    assert "Its tools (2; 1 change things in the real world):" in text
    assert "delete_note  CHANGES THINGS  pretend version: yes  - deletes a note forever" in text
    assert "Things it should probably never do (from your code; tell me if I'm wrong):\n  - Delete a note" in text
    assert ('First run ($0.020): asked "Find my Groceries note", it used find_note and answered: '
            '"Your Groceries note says: eggs."') in text
    assert "Nothing in your repo changed." in text
    assert "Setup cost: $0.84 (reading your code) + $0.02 (your agent's first run)." in text


# --- The setup command ------------------------------------------------------------------------------------

CUSTOM_TARGET = '''\
from eval_agent.fakes import FakeTool
from eval_agent.runner import CallableTarget

from . import fakes


def make_target():
    def run(case, toolbox):
        text, _ = toolbox.call("find_note", {"title": "Groceries"})
        WRITE
        return "Found it: " + text

    tools = lambda: [FakeTool("find_note", "Find a note.", {"title": str}, fakes.find_note)]
    return CallableTarget("notes-custom", run, tools, example_world=fakes.EXAMPLE_WORLD)
'''


def test_smoke_run_proves_the_target_works(tmp_path):
    root = make_repo(tmp_path, target=CUSTOM_TARGET.replace("WRITE", "pass"))
    smoke, run_dir = smoke_run(root, "evals.target:make_target", MAP, "evals", 60)
    assert smoke.ok and smoke.tools == ["find_note"]
    assert smoke.answer.startswith('Found it: {"text": "eggs", "title": "Groceries"}')
    assert (root / "evals" / "runs").is_dir() and run_dir.startswith(str(root / "evals" / "runs"))


def test_smoke_run_notices_real_side_effects(tmp_path):
    root = make_repo(tmp_path, target=CUSTOM_TARGET.replace("WRITE", "open('sent.txt', 'w').close()"))
    smoke, _ = smoke_run(root, "evals.target:make_target", MAP, "evals", 60)
    assert smoke.ran and not smoke.ok and smoke.changed_files == ["sent.txt (new)"]


def test_setup_wont_replace_an_existing_target(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    root = make_repo(tmp_path)
    assert main(["setup", str(root)]) == 1
    assert "already has target.py, fakes.py. Use --force" in capsys.readouterr().out


def test_setup_writes_the_map_and_explains_it(tmp_path, monkeypatch, capsys):
    from eval_agent import intake
    from eval_agent.intake import IntakeResult

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    root = make_repo(tmp_path, with_evals=False)

    async def fake_run_intake(root_arg, *, files, **kw):
        files.write("evals/fakes.py", FAKES)
        files.write("evals/target.py", TARGET)
        return IntakeResult(map=MAP, check=CHECK, checks_run=1, written=sorted(files.written), cost_usd=0.5), files

    monkeypatch.setattr(intake, "run_intake", fake_run_intake)
    assert main(["setup", str(root), "--no-smoke"]) == 0
    out = capsys.readouterr().out
    assert "I won't change your code." in out and "Your agent: notes-bot" in out
    assert "I didn't run your agent yet (--no-smoke)." in out
    assert "eval-agent explore --target evals.target:make_target --cases YOUR_CASES.jsonl" in out
    assert yaml.safe_load((root / "evals" / "tool_map.yaml").read_text(encoding="utf-8"))["agent"] == "notes-bot"
    assert (root / "evals" / ".gitignore").read_text(encoding="utf-8") == "runs/\n"
