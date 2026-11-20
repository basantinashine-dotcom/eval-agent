# Eval Agent

Give it your agent's repo. It runs your agent, writes an eval plan anyone can understand, gets
your approval, builds the eval, proves the eval works, and runs it.

**Status:** early. Milestone M0 (run an agent safely and record what it does) is in progress.

- [Product requirements (PRD)](docs/prd.md)

v1 focuses on tool-using agents built on the Claude Agent SDK or in custom Python code.

## What works so far

```
eval-agent cases PATH          read a golden dataset (JSONL, JSON or CSV) and say how it was read
eval-agent explore --target module:function --cases PATH [--reps 3]
                               run the agent on each case with pretend tools; save traces
eval-agent analyze [RUN]       find what went wrong and group it into failure types, in plain words
eval-agent generate --target module:function --cases PATH
                               write new test cases that cover what yours miss (into added/)
```

`analyze` and `generate` call Claude and show a cost estimate first (`--max-cost` stops them).

- **Pretend tools** (`eval_agent/fakes.py`): same names and schemas as the real ones, acting
  on a fresh world per run. They record every call and can be made to fail.
- **Runner** (`eval_agent/runner.py`):
  - swaps a Claude Agent SDK agent's tool servers for fakes, and refuses to run tools that have no
    fake;
  - answers the agent's questions with scripted replies;
  - saves `traces.jsonl`, plus `errors.jsonl` for runs that couldn't finish, which are never
    counted as agent failures.
- **Analyzer** (`eval_agent/analyze.py`, `report.py`): reviews each run against its case, then
  groups problems into failure types with counts per case, examples, likely causes and a test idea.
  The report is written for a 15-year-old reader.
- **Case generator** (`eval_agent/generate.py`): proposes new cases for coverage gaps and drops ones
  that can't run, with a reason. Your own dataset is never changed.
- **Sample agent** (`examples/support_agent/`): a small Pebble & Co support agent with refund,
  email and delete-account tools, 10 test cases plus 10 generated ones, and planted bugs for the
  eval to find. Without seeing the answer key, the analyzer found the delete-without-asking bug
  (5 of 6 delete runs) and, on generated cases, a refund confirmed after the refund tool failed.
- **Custom-code example:** the Threads reply agent's `evals/target.py` (in the
  `marketingagentthreads` repo) runs its real drafter and policy with posting faked.

## Try it

Running the sample agent needs Claude Code (`claude.exe`) installed and an Anthropic API key. It
makes real model calls, which cost money.

Copy `.env.example` to `.env` and put your key in it; `eval-agent` reads it automatically and git
ignores it.

Use an API key (`ANTHROPIC_API_KEY`), not your personal Claude login. On a claude.ai login, Claude
Code adds your account email and your computer's details to the agent's context, which changes how
it behaves; `explore` warns when that's the case. The runner also keeps your own Claude Code
plugins, skills and claude.ai connectors away from the agent under test.

```
pip install -e ".[dev]"
pytest
eval-agent cases examples/support_agent/cases.jsonl
eval-agent explore --target examples.support_agent.target:make_target --cases examples/support_agent/cases.jsonl
```
