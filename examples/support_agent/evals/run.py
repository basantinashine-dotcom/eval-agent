"""Run your agent's tests and get a plain report.

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
