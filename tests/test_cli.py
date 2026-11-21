import os

from eval_agent.cli import load_dotenv


def test_dotenv_sets_missing_vars_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "EVAL_TEST_A=from-file\n"
        "export EVAL_TEST_B='quoted'\n"
        "EVAL_TEST_EMPTY=\n"
        "EVAL_TEST_KEEP=from-file\n"
        "not a setting\n",
        encoding="utf-8",
    )
    for name in ("EVAL_TEST_A", "EVAL_TEST_B", "EVAL_TEST_EMPTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EVAL_TEST_KEEP", "already-set")

    assert load_dotenv(env) == ["EVAL_TEST_A", "EVAL_TEST_B"]
    assert os.environ["EVAL_TEST_A"] == "from-file"
    assert os.environ["EVAL_TEST_B"] == "quoted"
    assert "EVAL_TEST_EMPTY" not in os.environ, "a blank template line must not set an empty key"
    assert os.environ["EVAL_TEST_KEEP"] == "already-set"
    for name in ("EVAL_TEST_A", "EVAL_TEST_B"):
        monkeypatch.delenv(name)


def test_no_dotenv_is_fine(tmp_path):
    assert load_dotenv(tmp_path / ".env") == []
