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
eval-agent build               turn the approved plan into tests, proven on the runs you already have
eval-agent score [RUN]         score a run with those tests
eval-agent mutate              break a copy of your agent on purpose, and see if the right test notices
eval-agent label               label about 20 of your agent's answers yourself (a page in your browser)
eval-agent calibrate           see how often each judge agrees with your labels
eval-agent add-examples        write the extra examples your approved plan asked for, and set their checks
eval-agent export              make the tests run on their own: writes evals/run.py
eval-agent run                 run the tests and report (same as python evals/run.py)
```

**Run your tests anywhere.** After `export`, `python evals/run.py` runs every example 3 times. It
scores each try with your checks and trusted graders, and reports in plain words, including which
checks gave mixed results across tries. It uses its own copy of the engine in `evals/_vendor/`, so
the Eval Agent doesn't need to be installed: only Python, your agent's packages and keys, plus
`anthropic` and `pydantic` for the graders. `--quick` runs each example once, for a fast check on
every change. It exits with 1 if a "must never fail" check failed, so CI can block the change.
Commit `evals/` (including `_vendor/`) to your repo.

`setup`, `analyze`, `generate`, `plan`, `build` and `score` call Claude and show a cost estimate or cap
first (`--max-cost` stops them).

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
- **Build** (`eval_agent/build.py`, `testkit.py`, `score.py`): only from an approved plan. An agent with
  narrow tools writes the tests into `evals/`:
  - `checks.py`: a small function for each check that a program can score;
  - `judges.json`: a yes/no question for each check that needs judgment, asked by a different model
    from your agent's;
  - `expectations.json`: which checks apply to which example, taken from the example's own text,
    never from what your agent did.

  It proves the tests before handing them over. Every try the analysis found a check's problem in
  must fail that check, and tries with nothing wrong must pass. Code checks run in a separate
  process with the network blocked. `score` then uses the tests on any run, and fails if a "must
  never fail" check failed.
- **Breaks** (`eval_agent/mutate.py`): proves the tests that never saw a failure. Claude turns the
  plan's breaks into exact changes to a copy of your agent: add or remove a line in its instructions,
  make a tool fail every time, or take a tool away. Your code is never edited. It also adds a break for
  every "must never fail" test that has nothing to catch yet. Each break runs on a few examples and is
  scored with the built tests. It counts as **caught** when the test it was meant for fails more often
  than before the break, **weak** when your agent changed but no test noticed, and **no effect** when
  your agent did the same as before.
- **Checking the judges** (`eval_agent/calibrate.py`): `label` picks about 3 tries per judge question.
  It mixes tries that went wrong before, tries from the breaks, and ordinary ones, and writes
  `evals/labels/label.html`, one page you open in any browser. You answer each question yourself,
  without seeing the judge's answers, and save `labels.json`. `calibrate` asks the judges the same
  questions and counts how often each agrees with you. A judge that agrees at least 9 times in 10 is
  trusted. One that isn't (or was never checked, or whose question changed) can still flag a problem,
  but `score` won't let it fail a "must never fail" check. After rewriting a judge's question to match
  how you answered, `calibrate --recheck` shows whether it now says what you meant. It never grants
  trust, because those are the answers the question was rewritten from. `label --fresh` starts a new
  round with tries you haven't labeled, and keeps the old round.
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
