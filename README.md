# Eval Agent

Give it your agent's repo. It runs your agent, writes an eval plan anyone can understand, gets
your approval, builds the eval, proves the eval works, and runs it.

**Status:** early. Milestone M0 (run an agent safely and record what it does) is in progress.

- [Product requirements (PRD)](docs/prd.md)

v1 focuses on tool-using agents built on the Claude Agent SDK or in custom Python code.

## What works so far

```
eval-agent setup [REPO]        read your agent's repo, find its tools, and write pretend versions
                               of them into evals/ (then run the agent once to prove it works)
eval-agent cases PATH          read a golden dataset (JSONL, JSON or CSV) and say how it was read
eval-agent explore --target module:function --cases PATH [--reps 3]
                               run the agent on each case with pretend tools; save traces
eval-agent analyze [RUN]       find what went wrong and group it into failure types, in plain words
eval-agent generate --target module:function --cases PATH
                               write new test cases that cover what yours miss (into added/)
eval-agent plan [--runs RUN ...]
                               write the eval plan in plain words, from your analyzed runs
eval-agent plan --feedback "..."
                               revise it: a new version, with what changed
eval-agent plan --approve      approve the current version (nothing gets built before this)
```

`setup`, `analyze`, `generate` and `plan` call Claude and show a cost estimate first (`--max-cost` stops
them).

- **Setup** (`eval_agent/intake.py`, `check.py`, `toolmap.py`): an agent built on the Claude Agent SDK
  reads your repo and writes `evals/target.py` and `evals/fakes.py` itself, for Claude Agent SDK agents
  and custom code. It:
  - can read only files git doesn't ignore, never `.env` or other files that look like they hold
    secrets, and can write only inside `evals/`. It runs no commands and never edits your code;
  - checks what it wrote before trusting it: the file can't run commands or touch the network, and it
    loads in a separate process with the network blocked. Every real tool has a pretend version with
    the same name, description and schema, and each pretend tool runs once without crashing;
  - writes `evals/tool_map.yaml` (which tools change things in the real world, what was patched for
    testing, what it thinks your agent must never do) for you to check;
  - runs your agent once on a simple task and says if any file in your repo changed.

  Tried on a copy of the sample agent (without its answer key) and on the Threads reply agent (custom
  code): both passed their checks on the first try, for $0.13 and $0.59.
- **Eval plan** (`eval_agent/plan.py`, `plan_render.py`, `readability.py`): turns what the runs showed
  into a plan in 13 parts, from "the short version" to "words you might not know". Each check says
  what it tests, why it matters, what happened with your agent, and how it's tested. Claude writes the
  words; code fills in the counts, tools, examples, cost and time, so those are exact. Before you see
  it, the plan is checked: reading level (grade 9 or below), long sentences, jargon and code words, and
  a yes/no judge asking "would a 15-year-old understand this check?" Anything that fails gets
  rewritten. Saved as `evals/EVAL_PLAN.md`, one self-contained `EVAL_PLAN.html` you can send to anyone,
  and `plan.json` with every version, the feedback, the changelog and your approval.
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
  Its plan, [`examples/support_agent/evals/EVAL_PLAN.md`](examples/support_agent/evals/EVAL_PLAN.md),
  has 14 checks covering all five planted bugs, including two that never happened in testing. It
  reads at grade 6.6, and the 15-year-old judge understood every check. Writing it cost $0.72.
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

For your own agent, start with `eval-agent setup path/to/your/repo`, then run `explore` from your
repo's folder with `--target evals.target:make_target`. Your agent's packages need to be installed
where eval-agent runs.
