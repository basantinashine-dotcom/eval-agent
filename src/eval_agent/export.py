"""Make the tests folder work on its own: evals/run.py (docs/prd.md FR-34).

Copies the eval_agent engine into evals/_vendor/ and writes run.py next to it, so the tests run
with only Python, the agent's own packages and keys, even where the Eval Agent isn't installed
(in CI, or after you stop using it). run_config.json says how to start the agent and which
example files to use, relative to the repo's folder.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

RUN_PY = '''"""Run your agent's tests and get a plain report.

    python run.py            every example, 3 tries each
    python run.py --quick    every example once: a fast check for each change
    python run.py --help     all the options

Written by `eval-agent export`. It works without the Eval Agent installed: it uses the copy of
its engine in _vendor/. It needs Python 3.11 or newer, your agent's own packages and keys, and
the anthropic and pydantic packages for the AI graders (plus claude-agent-sdk if your agent is
built on it). It exits with 1 if a "must never fail" check failed, so CI can block the change.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_vendor"))   # this folder's own copy comes first

from eval_agent.standalone import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(evals_dir=HERE))
'''


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("eval-agent")
    except Exception:
        return "unknown"


def export(evals: Path, *, target: str, cases: list[str], root: Path) -> list[Path]:
    """Write run.py, run_config.json and _vendor/eval_agent into evals. Returns what it wrote."""
    evals, root = evals.resolve(), root.resolve()
    package = Path(__file__).resolve().parent
    vendor = evals / "_vendor" / "eval_agent"
    if vendor.exists():
        shutil.rmtree(vendor)
    vendor.mkdir(parents=True)
    for source in sorted(package.glob("*.py")):
        shutil.copy2(source, vendor / source.name)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    (evals / "_vendor" / "README.md").write_text(
        f"A copy of the eval-agent engine (version {_version()}, copied {stamp}) so `run.py` works on its own.\n"
        "Don't edit it; run `eval-agent export` again to update it.\n", encoding="utf-8")

    def from_root(name: str) -> str:
        path = Path(name).resolve()
        return (path.relative_to(root) if path.is_relative_to(root) else path).as_posix()

    # Keep example files added since the last export (e.g. by add-examples).
    old = evals / "run_config.json"
    kept = json.loads(old.read_text(encoding="utf-8")).get("cases", []) if old.exists() else []
    config = {
        "target": target,
        "root": Path(os.path.relpath(root, evals)).as_posix(),
        "cases": list(dict.fromkeys([from_root(c) for c in cases] + kept)),
    }
    (evals / "run_config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    (evals / "run.py").write_text(RUN_PY, encoding="utf-8")
    gitignore = evals / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if "runs/" not in lines:
        gitignore.write_text("\n".join(lines + ["runs/"]) + "\n", encoding="utf-8")
    return [evals / "run.py", evals / "run_config.json", vendor]
