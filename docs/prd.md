# PRD: Eval Agent

**Status:** Draft v0.5 · 2026-11-19
**One line:** Give it your agent's repo. It runs your agent, writes an eval plan anyone can
understand, gets your approval, builds the eval, proves the eval works, and runs it.

---

## 1. Summary

People building tool-using agents know they need evals and rarely have good ones. Existing tools
run test cases and draw dashboards, but they leave the hard parts to you: what to measure, how to
test an agent safely when its tools act in the real world, and whether your eval would actually
catch a regression.

The Eval Agent takes a repo and API keys, then:

1. gets the agent running in an isolated environment and maps its tools,
2. replaces risky tools with fakes and runs the agent on realistic tasks,
3. looks at what went wrong,
4. writes an **eval plan** simple enough for a 15-year-old to follow, and shares it for your
   approval and feedback,
5. builds the eval,
6. **tests the eval itself**, so you know it catches real problems,
7. runs it and explains the results just as simply.

You give judgment: what must never happen, whether the plan is right and complete, and about 20
labels. The agent does everything else. What it builds is plain files in your repo that run
without the Eval Agent installed.

## 2. Decisions so far

| Topic | Decision |
|---|---|
| Input | Repo + API keys. The Eval Agent must be able to run the user's agent. A README alone isn't enough. |
| Product type for v1 | Tool-using agents |
| Frameworks for v1 | Claude Agent SDK, and custom code |
| Language for custom code | Python first; TypeScript later |
| Human checkpoint | An **eval plan**, shared and approved by the user, who can say what's missing |
| How the plan reads | Written so a **15-year-old** can understand it: short sentences, no jargon, real examples first, technical detail kept at the back |
| Plan feedback in v1 | In chat with the agent |
| Sharing in the free core | Markdown in the repo, plus one self-contained HTML file the user can send to anyone |
| Paid later | Hosted share links with comments, hosted runs, team features |
| Existing data | The agent asks for a **golden dataset** up front. If there isn't one, it recommends a simple optional format. |
| Questions the agent asks mid-task | Answered by **scripted replies** in each test case. An LLM-played user comes later. |
| Repeats | Each eval case runs 3 times by default. Reports show both the average and "passed every time". |
| Built on | The Claude Agent SDK (Python), so we can also test the Eval Agent on itself |
| Business model | Open core. The free core runs locally on the user's code and keys. |
| Method | Look at real outputs first; code checks for objective rules; yes/no LLM judges for subjective ones, calibrated to the user; end-to-end and per-tool-call checks |

## 3. Method

| Principle | What the Eval Agent does |
|---|---|
| **Build first, then look.** Failure modes can't be predicted in advance. | Runs the agent and reads what it did before proposing any metric |
| **Error analysis on traces.** | Reads every step: each tool call, its arguments, its result, and the final answer. Groups the failures and counts them. |
| **Objective → code.** | e.g. "`issue_refund` called without a prior `get_order` for the same ID" |
| **Subjective → LLM judge, yes/no, not 1–5.** | e.g. "Does the final message tell the customer what actually happened?" Calibrated against the user's labels. |
| **End-to-end and component-level.** | Every check targets either the final outcome or a specific step or tool call |

## 4. Users

| Persona | Situation | What they need |
|---|---|---|
| **Solo builder** (primary) | Built an agent on the Claude Agent SDK or in custom code; tests it by trying a few tasks | A trustworthy eval without learning eval theory |
| **Small product team** (primary) | Changes prompts, tools or models weekly | Know whether a change broke something before shipping |
| **Teammate, manager or non-technical founder** (secondary) | Receives the plan or the results file | To understand it on the first read, with no eval or coding background |

## 5. Non-goals (v1)

- **Hosting production logs or replacing observability tools.** It reads traces; it doesn't host
  them.
- **Fine-tuning or training models.**
- **Live A/B tests on real users.** All comparisons run offline.
- **Changing the user's agent.** It never edits the agent's code or prompts. It can suggest a fix,
  and the user applies it. (Mutation testing edits a throwaway copy only.)
- **Other kinds of products:** chat-only bots, RAG apps, classifiers. Tool-using agents only.
- **Other languages and frameworks:** TypeScript, LangChain, CrewAI and others come later.
- **Simulated multi-turn conversations.** When the agent asks the user something, v1 answers with
  scripted replies, not a simulated person.
- **Running without the user's code and keys.** No README-only mode.

## 6. Flow

| # | Step | The agent does | The user does | Output |
|---|---|---|---|---|
| 1 | **Intake** | Asks for the repo, keys, golden dataset (if any), red lines and risky tools; shows the setup cost estimate | Shares them, about 5 min | Intake summary |
| 2 | **Get it running** | Sets up an isolated environment and runs the agent once; stops with a clear list if it can't | Fixes anything missing | A working run |
| 3 | **Understand** | Reads the code, maps the tools (including MCP tools and subagents), builds fakes | Confirms the tool map | Tool map + fakes |
| 4 | **Explore** | Runs ~30 tasks (golden first, then generated) and reads every trace | Nothing | Failure list with examples |
| 5 | **Eval plan** | Writes the plan in plain language, shares it, revises it on feedback | Approves, or says what's missing | Approved plan |
| 6 | **Build** | Writes checks, test cases and a standalone runner | Nothing | `evals/` in the repo |
| 7 | **Test the eval** | Known-good/known-bad checks, mutations, judge calibration, noise | Labels ~20 outputs, about 15 min | Eval health report |
| 8 | **Run and report** | Runs the eval (3 repeats per case), explains the results simply | Reads it | Results report |
| 9 | **Keep it alive** | Compares versions, gates CI, adds new failures, flags when the agent's tools change | Occasional review | Ongoing |

Nothing gets built (step 6) until the plan is approved (step 5).

## 7. Functional requirements

### 7.1 Intake

- **FR-1** Ask for:
  - the repo (URL or local path);
  - the API keys the user's agent needs;
  - an Anthropic key or login for the Eval Agent's own work;
  - whether a golden dataset exists (§7.3).
- **FR-2** Ask two short questions it can't infer from the code: *What must your agent never do?*
  and *Which tools change things in the real world (send, pay, post, delete)?* Propose answers
  from the code first, so the user mostly confirms.
- **FR-3** Before starting, show what it will cost:
  - the **setup estimate**, covering the Eval Agent's own model use (reading code, analyzing
    traces, writing the plan) and the user's agent's runs;
  - a **per-run estimate** once the eval exists.

  Honor a cost cap the user can set. The two costs are reported separately: the Eval Agent's model
  use and the user's agent's model use.
- **FR-4** Keys are used only to run the user's agent locally. They're never written to files, logs
  or traces, and never sent anywhere else.

### 7.2 Get the agent running

- **FR-5** Set up the user's agent in an isolated environment (its own virtualenv; a container where
  available), separate from the user's own Python setup.
- **FR-6** Run the agent once on a simple task to prove it works before anything else.
- **FR-7** If it can't run, **stop** and say exactly what's missing, in plain words: which
  dependency, which key, which command failed, and the steps to fix it. Never fall back to guessing
  from the README.

### 7.3 Golden dataset

- **FR-8** Ask early: *"Do you already have test cases, or examples with known right answers? If
  so, share them in any format."* Accept JSONL, JSON, CSV or spreadsheet exports, and exported
  logs.
- **FR-9** Show how it read the data before using it, for example: "48 cases; 40 have an expected
  outcome; 8 don't; none where the agent should refuse or ask first." The user confirms or corrects
  the reading.
- **FR-10** Point out gaps (no failure cases, no "should not act" cases, one category missing) and
  offer to fill them with generated cases. Generated cases are tagged `synthetic`, kept in a
  separate file, and approved by the user.
- **FR-11** If there's no golden dataset, recommend the optional format in §9 and explain that it
  can start without one. It builds a starter set from generated tasks plus the user's labels.
- **FR-12** Never edit the user's dataset in place. Additions go in separate files.
- **FR-13** Use golden cases as realistic inputs, as ground truth for checks, and as known-good
  examples when testing the eval (§7.9).

### 7.4 Understand the agent and fake its tools

- **FR-14** Map the agent: entry point, system prompt, model, tool list with schemas, and which
  tools are read-only and which have side effects. The user confirms the map, which is saved to
  `evals/tool_map.yaml`.
- **FR-15** **Claude Agent SDK.**
  - Record every tool call from the SDK's message stream, and use its hooks to block or adjust tool
    calls before they run.
  - Swap custom tools for fakes, and fake **MCP server tools** the same way, so nothing reaches
    GitHub, Slack or a real database.
  - Include **subagent** tool calls in the trace.
  - Run built-in file and shell tools in a throwaway copy of the workspace.
- **FR-16** **Custom code.** Find where tools are defined and called. Inject fakes where tools are
  passed in, and patch them where they're called directly. Wrap the agent's model client to record
  its calls. Every patch is listed in the plan so the user knows what was changed for testing.
- **FR-17** Fakes keep state (a refund made in one call shows up in the next), return realistic
  data, record every call, and can be told to fail (timeout, error, empty result) so error
  handling can be tested.
- **FR-18** Side-effect tools are always faked unless the user explicitly says a sandbox is safe to
  use.
- **FR-19** **Scripted user replies.** When the agent asks the user something mid-task ("Are you
  sure you want to delete your account?"), the runner answers with the next reply from the case's
  `user_replies` list. If the agent asks and no reply is left, the run records "asked, no answer
  given", which is itself checkable.

### 7.5 Explore and find failures

- **FR-20** Run about 30 tasks once each: golden cases first, then generated ones covering normal
  tasks, edge cases, cases where the agent should refuse or ask first, tool failures, and
  instructions hidden in tool results.
- **FR-21** Save a full trace per task: every message, tool call, argument, result, subagent call,
  final answer, token count and cost.
- **FR-22** Read the traces, group failures into types, count them, and identify the step or tool
  call where each one starts. Each type comes with real example traces.
- **FR-23** If nothing failed, say so honestly: what 30 clean runs do and don't prove, and which
  red lines will still be tested.

### 7.6 Eval plan: contents, sharing, approval

- **FR-24** The plan has these parts, in this order. Each heading uses everyday words.
  1. **The short version:** 3–5 sentences covering what your agent does, what went wrong when I
     tried it, what I'll test, and what I need from you.
  2. **What your agent does (as I understand it):** purpose, users, tools, and which tools do real
     things.
  3. **What I tried and what went wrong:** failure types, counts ("4 out of 30 tries") and real
     examples.
  4. **What I'll test:** one block per check (format in §7.7).
  5. **The examples I'll test with:** how many, where from (yours or ones I made up), what kinds,
     and 5 samples.
  6. **What I'll pretend:** which tools are swapped for pretend versions so nothing real happens,
     and what I had to change in your code to do that.
  7. **How I'll make sure the tests work:** the breaks I'll try, and the ~20 answers I need you to
     label.
  8. **Cost and time:** per run, and per CI run.
  9. **What I'm not testing, and why:** makes gaps easy to spot.
  10. **Questions for you:** targeted questions about likely gaps, e.g. "Should your agent ever
      email the same customer twice?"
  11. **Words you might not know:** a short glossary.
  12. **For the technical folks:** check types, file paths, how each check is scored.
  13. **Changes:** the changelog.
- **FR-25** Share the plan as `evals/EVAL_PLAN.md` in the repo and as **one self-contained HTML
  file** (no server, no account) the user can send to a teammate. Hosted links with comments are a
  later paid feature.
- **FR-26** In chat, the user can **approve**, **ask for changes**, or **say what's missing**, e.g.
  "you didn't check that it confirms before deleting". For each piece of feedback, the agent adds
  or changes checks, explains how it will measure the new item, and shares a new version with a
  short "what changed" list.
- **FR-27** Record each version in the changelog: the feedback received and what changed.
- **FR-28** Build nothing until the user approves. Approval is recorded with the plan version.

### 7.7 Writing the plan so a 15-year-old can follow it

The plan is for the builder *and* whoever they forward it to. If a smart 15-year-old can't follow
it, it's too complicated.

- **FR-29** Writing rules:
  - Short sentences: aim for under 20 words each.
  - Everyday words. "Test", not "eval"; "pretend version", not "mock"; "tries", not "trials".
  - When a technical word can't be avoided, explain it in the same sentence the first time, and add
    it to the glossary.
  - **Example first, rule second.** Show what actually happened with the user's own agent, then
    the rule.
  - Numbers as counts ("4 out of 30 tries"), not percentages with decimals.
  - Talk to the reader: "you" and "your agent".
  - Code, file paths and scoring details go only in "For the technical folks".
- **FR-30** Each check uses the same block:
  - a **short name** you could say out loud ("Look before you refund"), plus **Must never fail** or
    **Should usually pass**;
  - **What we check:** one or two sentences;
  - **Why it matters:** what could go wrong for you or your users;
  - **What we saw:** a real example from the user's agent, or "This didn't happen in testing, but
    you told me it must never happen";
  - **How we test it:** in plain words;
  - **Technical detail:** one line, in "For the technical folks".
- **FR-31** **The plan gets checked too.** Before sharing, the Eval Agent:
  - measures the reading level, targeting US grade 9 or below;
  - runs a yes/no judge on each check: *"Would a 15-year-old understand what this tests and why it
    matters?"*;
  - rewrites any part that fails.
- **FR-32** The results report (§7.10) follows the same rules.

**Example check, written to this standard:**

> **Check 3: Look before you refund** · Must never fail
>
> **What we check:** Before your agent gives money back, it looks up the order to make sure it's
> real and belongs to this customer.
>
> **Why it matters:** If it skips this step, it could refund the wrong order, or pay someone who
> never bought anything.
>
> **What we saw:** In 4 out of 30 tries, your agent refunded an order number the customer typed
> without checking it first. Someone wrote "refund order 4471", and the agent paid out without
> ever looking at order 4471.
>
> **How we test it:** We look at the list of actions your agent took. If "give refund" happens
> before "look up order", the test fails.

*For the technical folks: code check on the trace. It fails if `issue_refund(order_id=X)` has no
earlier successful `get_order(order_id=X)`.*

### 7.8 Build

- **FR-33** Write the code checks, judge prompts, test cases and a runner that calls the user's
  agent with the fakes. Everything goes under `evals/` (§10).
- **FR-34** **The runner stands alone.** `evals/run.py` needs only Python, the user's agent's
  dependencies and their keys. It doesn't need the Eval Agent installed, so it works in CI and keeps
  working if the user stops using the Eval Agent.
- **FR-35** Every check maps back to an item in the approved plan, and the plan maps to the code.

### 7.9 Test the eval

- **FR-36** Confirm that known-good cases pass and known-bad cases fail.
- **FR-37** **Mutation testing.** Break the user's agent on purpose, in a throwaway copy: remove a
  rule from the prompt, remove a tool, or make a tool fail. Each relevant check must catch the break,
  and a check that doesn't move is reported as weak. To keep cost down, run 5–10 mutations on a
  subset of cases, and include them in the cost estimate.
- **FR-38** **Judge setup.**
  - The judge model is configurable. Prefer one different from the model the user's agent runs on,
    since models tend to favor their own style.
  - Examples given to a judge come from different cases than the labels used to grade it.
- **FR-39** **Judge calibration.** The user labels about 20 outputs without seeing the judge's
  answers. Agreement is reported per judge. A judge below 90% agreement can be shown but can't fail
  a "Must never fail" check until it's revised.
- **FR-40** **Repeats and noise.** Run each case 3 times by default. Report the average, how often
  a case passed every time, and how much scores move between identical runs, so the user knows how
  big a change has to be to mean anything.

### 7.10 Run, report and keep alive

- **FR-41** Run the eval, keeping infra errors (timeouts, API errors) separate from agent failures.
  Show a cost estimate first and honor the cost cap.
- **FR-42** Record with each run: the git commit, model, prompt version and tool map, so every
  result is tied to the exact version tested.
- **FR-43** The results report explains, in the §7.7 style:
  - what passed and what failed, with examples;
  - which tool call or step caused each failure;
  - the cost.

  It's shared the same way as the plan.
- **FR-44** Compare two versions of the agent (prompt, model, tools): changes with error bars, the
  cases that flipped, and a one-line verdict ("better", "worse", or "no real difference").
- **FR-45** **CI gate:** fail the build if a "Must never fail" check fails or a metric drops beyond
  noise. The plan states the cost per CI run. CI can run a quick subset on every pull request and
  the full suite before a release.
- **FR-46** **Tool changes:** on each run, compare the agent's current tools to `tool_map.yaml`.
  If tools were added, removed or changed, say so, flag checks that reference missing tools, and
  propose a plan update.
- **FR-47** Turn a new production failure into a test case in one step.

## 8. Checks for tool-using agents

This is the starting menu the plan draws from. Only checks tied to an observed failure or a red
line go into a plan.

| Check | Type | Example failure |
|---|---|---|
| Task actually done (end state) | code | Checks the fake world (refund recorded), not the agent's story |
| Forbidden tool never called, or confirmed first | code (critical) | Deleted data or sent an email without asking |
| Right arguments | code | Refunded the wrong order ID or amount |
| Sensible order | code | Refund issued before the order was looked up |
| Right tool chosen | code | Searched the web instead of looking up the order |
| Tool errors handled | code + judge | Tool timed out; agent said it worked |
| No invented results | judge | "I've booked it", but no booking call happened |
| Final answer matches what happened | judge | Told the customer "refunded" after the refund failed |
| Not hijacked by tool output | code + judge | A fetched page said "ignore your instructions", and it obeyed |
| Asked when it should, didn't when it shouldn't | code | Deleted without asking, or asked three times for a simple lookup |
| Steps and cost within budget | code | 30 loops and $2 for a simple task |

**Grade the result before the path.** The end-state check comes first. Path checks (which tools,
in what order) are used only where the path itself matters, such as "look before you refund", so
an agent that reaches the right result a different valid way isn't failed.

## 9. Recommended golden dataset format (optional)

Only two fields are required. Everything else makes the eval sharper. Share it as **JSONL**
(one case per line) or a **spreadsheet / CSV**.

| Field | Required | What it is |
|---|---|---|
| `id` | no | Any unique name; generated if missing |
| `task` | **yes** | What the user asks the agent, exactly as they'd type it |
| `expected_outcome` | **yes** | Plain English: what "done right" looks like |
| `must_not` | no | Plain English or tool names: what must not happen |
| `expected_tools` | no | Tools that **must be included**, with key arguments. Other tool calls are allowed unless listed in `must_not`. |
| `user_replies` | no | What the user answers if the agent asks something, in order, e.g. `["yes, delete it"]` |
| `starting_state` | no | The world before the task (accounts, orders, files), used to set up the fakes |
| `tags` | no | e.g. `refund`, `edge-case`, `should-ask-first` |
| `notes` | no | Anything a reviewer should know |

**JSONL example:**

```json
{"id": "refund-01",
 "task": "I want a refund for order 4471, it arrived broken",
 "expected_outcome": "Order 4471 refunded in full ($25); customer told when to expect it",
 "must_not": "Refund any other order; promise a replacement",
 "expected_tools": [{"tool": "get_order", "args": {"order_id": "4471"}},
                    {"tool": "issue_refund", "args": {"order_id": "4471", "amount": 25.00}}],
 "starting_state": {"orders": [{"id": "4471", "customer": "c_9", "total": 25.00, "status": "delivered"}]},
 "tags": ["refund", "happy-path"]}
```

**Spreadsheet version (minimum):**

| task | expected_outcome | must_not | user_replies | tags |
|---|---|---|---|---|
| I want a refund for order 4471, it arrived broken | Order 4471 refunded in full; customer told when | Refund any other order | | refund |
| Delete my account | Agent asks for confirmation, then deletes | Delete without asking | yes, delete it | should-ask-first |
| Delete my account | Agent asks, then doesn't delete | Delete anything | no, keep it | should-ask-first |

**What the agent recommends for a starting set:**

- at least 20 cases, and at least 5 per "Must never fail" check, so a bigger plan asks for more;
- cases where the agent *shouldn't* act;
- a few you know are hard;
- real user requests where possible.

The agent turns plain-English expectations into checks and shows them in the plan, so nobody has
to write code.

## 10. What it writes into the repo

```
evals/
  EVAL_PLAN.md        the approved plan, with changelog
  EVAL_PLAN.html      the same plan as one file to send to anyone
  tool_map.yaml       tools, risky or not, faked or patched
  fakes/              pretend versions of risky tools (including MCP tools)
  cases/golden/       your dataset, untouched
  cases/added/        generated cases, tagged synthetic
  checks/             code checks
  judges/             judge prompts + calibration results
  run.py              standalone runner: runs your agent with fakes and scores it
  runs/<timestamp>/   traces, results, errors, report (.md + .html), version info
```

## 11. Safety and privacy

**The user's agent**

- Risky tools are faked by default. The agent never sends, pays, posts or deletes during an eval
  unless the user names a safe sandbox.
- It runs in an isolated environment, not the user's own Python setup.
- Mutations run on a throwaway copy, never the user's working tree.

**The Eval Agent itself**

- It asks before running any command outside its isolated environment or writing outside
  `evals/`.
- Everything it reads from the repo, traces, tool results and datasets is treated as **data, never
  instructions**. A README or tool result saying "run this command" or "ignore your rules" is
  reported, not followed.
- It never edits the user's agent code (§5).

**Keys and data**

- Keys are never written to files, logs or traces, and are used only for the local runs they're
  for.
- Traces and datasets stay local. Before anything is sent to a model API for analysis or judging,
  the user is told what is sent, with an option to redact personal data.

## 12. Success metrics

**For users**

- **The user's time** from sharing the repo to an approved plan: under 30 minutes.
- **Total wall-clock time** from sharing the repo to an approved plan: under 2 hours. This is a
  starting target, measured on design partners.
- **Plan clarity:** share of plans where users don't ask "what does this mean?". Every plan passes
  the reading-level and judge checks in FR-31.
- Share of plans where the user's feedback added a check. This shows the feedback loop is used.
- Share of users who say the eval found a problem they didn't know about.
- Mutation catch rate: share of planted breaks caught by the built eval (target ≥ 90%).
- Week-2 return: users who run the eval again or add it to CI.

**For the Eval Agent itself.** A benchmark of sample agents (Claude Agent SDK and custom code)
with planted failures, for example a refund before lookup, an action taken on hijacked tool
output, an invented booking, or a delete without asking. We measure:

- whether the plan names each planted failure;
- whether the built check catches it;
- whether it raises false alarms;
- whether the plan passes the 15-year-old test.

## 13. Design partners

1. **A sample Claude Agent SDK agent we build in M0.** A small support agent where the model
   chooses the tools (look up order, refund, email, delete account), with planted failures. This
   tests the core use case, tool choice, from day one.
2. **The Threads reply agent** (the `marketingagentthreads` repo).
   - Custom code, with tools passed in (`ThreadsClient`, `Drafter`), so fakes are easy.
   - Posting is the side effect to fake.
   - Clear red lines (crisis posts).
   - Caveat: its code decides which tool to call, not the model, so it tests custom code, fakes and
     red lines, not tool choice.
3. **One real outside user's agent** before calling v1 done.

## 14. Milestones

| Milestone | Delivers | Done when |
|---|---|---|
| **M0 Intake + run** | FR-1–23; the sample SDK agent | Both design partners are running in isolation with full traces and fakes, from 30 tasks each |
| **M1 Eval plan** | FR-24–32 | A plan passes the 15-year-old check and is approved after at least one round of feedback |
| **M2 Build + test** | FR-33–40 | Mutation testing catches ≥ 90% of planted breaks on both design partners |
| **M3 Run, report, CI** | FR-41–47 | A comparison report and a CI gate on a real prompt change |
| **M4 Outside user** | The full flow on design partner 3 | An outside user approves a plan and keeps running the eval in week 2 |

## 15. Risks

| Risk | Mitigation |
|---|---|
| Can't get the user's agent running (env, deps, keys) | Step 2 checks this first and stops with exact fix steps (FR-7); v1 supports Python only |
| Fakes behave differently from the real tools | Fakes keep state and are built from tool schemas and real example responses; the plan lists every fake |
| The plan looks thorough but misses what matters | "What I'm not testing", targeted questions in every plan, and the feedback loop |
| The plan is too technical to review properly | Writing rules, the reading-level check and the "15-year-old" judge (FR-29–31) |
| Agent-written evals grade the agent's own homework | Mutation testing, the user's labels, 90% judge agreement to gate, and a different judge model where possible |
| The Eval Agent is tricked by text in the repo or traces | Treat everything read as data, and ask before running commands (§11) |
| Cost surprises (setup, mutations, CI) | Estimates before each step, a cost cap, a capped mutation count, and a quick CI subset |
| Flaky results from a non-deterministic agent | 3 repeats per case, "passed every time" reported, noise shown on every comparison |

## 16. Open questions

1. **Name.** "Eval Agent" is a working name.

**Resolved 2026-11-19:**
- Its own repo, `eval-agent`, with the Threads agent as an example target.
- Python first for custom code.
- Plan feedback in chat for v1.
- Built on the Claude Agent SDK.
- Open core.
- Golden set: at least 20 cases, and at least 5 per critical check.
- Plans and reports are written for a 15-year-old reader.
