# Eval Agent

Give it your agent's repo. It runs your agent, writes an eval plan anyone can understand, gets
your approval, builds the eval, proves the eval works, and runs it.

**Status:** early. Milestone M0 (run an agent safely and record what it does) is in progress.

- [Product requirements (PRD)](docs/prd.md)

v1 focuses on tool-using agents built on the Claude Agent SDK or in custom Python code.

## What works so far

```
eval-agent cases PATH          read a golden dataset (JSONL, JSON or CSV) and say how it was read
eval-agent explore --target module:function --cases PATH
                               run the agent on each case with pretend tools; save traces
```

- **Pretend tools** (`eval_agent/fakes.py`): same names and schemas as the real ones, acting
  on a fresh world per run. They record every call and can be made to fail.
- **Runner** (`eval_agent/runner.py`):
  - swaps a Claude Agent SDK agent's tool servers for fakes, and refuses to run tools that have no
    fake;
  - answers the agent's questions with scripted replies;
  - saves `traces.jsonl`, plus `errors.jsonl` for runs that couldn't finish, which are never
    counted as agent failures.
- **Sample agent** (`examples/support_agent/`): a small Pebble & Co support agent with refund,
  email and delete-account tools, 10 test cases, and planted bugs for the eval to find.

## Try it

Running the sample agent needs Claude Code (`claude.exe`) installed and an Anthropic API key or
Claude login. It makes real model calls, which cost money.

```
pip install -e ".[dev]"
pytest
eval-agent cases examples/support_agent/cases.jsonl
eval-agent explore --target examples.support_agent.target:make_target --cases examples/support_agent/cases.jsonl
```
