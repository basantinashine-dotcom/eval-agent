"""Check the judges against the owner's own labels (docs/prd.md FR-38, FR-39).

1. `label` picks about 3 tries per judge question: some the earlier review flagged, some from
   breaking the agent on purpose, some ordinary. It writes a labeling page (one file, opens
   in any browser) that shows each try and the judge's question, never the judge's answer.
2. The owner answers yes / no / not sure; the page saves labels.json.
3. `calibrate` asks each judge the same questions and counts how often it agrees with the owner.
   A judge that agrees at least 9 times in 10 (on at least 3 labels) is trusted. One that isn't
   can still flag problems, but it can't fail a "must never fail" check (score.apply_trust).
   Changing a judge's question makes its old agreement count no longer apply.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .analyze import Spend, render_case, render_run, world_changes
from .dataset import Case
from .score import JUDGE_MODEL, Item, JudgeSpec, judge_one, labels, load_items, score_code
from .testkit import expectations_for
from .trace import Trace, read_traces

PER_JUDGE = 3
TRUST_AT = 0.9
MIN_LABELS = 3
ANSWERS = ("yes", "no", "unsure")


def question_hash(spec: JudgeSpec) -> str:
    return hashlib.sha1(f"{spec.question}|{spec.fail_when}".encode()).hexdigest()[:10]


@dataclass
class LabelItem:
    id: str
    check: int
    question: str
    key: str                     # the try, e.g. "20261120-212713/refund-happy#0"
    source: str                  # "flagged before", "after a break" or "ordinary"
    trace: dict[str, Any]
    case: dict[str, Any] | None
    notes: dict[str, Any] = field(default_factory=dict)


def mutation_items(runs_dir: Path, cases: dict[str, Case]) -> list[Item]:
    """Tries from earlier break runs (runs/mutations-*/<break>/traces.jsonl)."""
    items = []
    for path in sorted(runs_dir.glob("mutations-*/*/traces.jsonl")):
        for trace in read_traces(path):
            key = f"{path.parent.parent.name}/{path.parent.name}/{trace.case_id}#{trace.rep}"
            items.append(Item(key, trace, cases.get(trace.case_id)))
    return items


def pick_items(evals: Path, *, run_dirs: list[Path], extra: list[Item], cases: dict[str, Case],
               checks_based_on: dict[int, list[str]], per_judge: int = PER_JUDGE, seed: int = 0) -> list[LabelItem]:
    """About `per_judge` tries for each judge question, from different kinds of tries, in shuffled order."""
    expectations = json.loads((evals / "expectations.json").read_text(encoding="utf-8"))
    judges = JudgeSpec.load_all(evals / "judges.json")
    items = [i for run_dir in run_dirs for i in load_items(run_dir)] + extra
    code = score_code(evals, items, expectations)
    if code.problems:
        raise RuntimeError("The tests can't run: " + "; ".join(code.problems))
    known = labels(checks_based_on, run_dirs)
    extra_keys = {i.key for i in extra}
    rng = random.Random(seed)
    picked: list[LabelItem] = []
    for spec in judges:
        groups: dict[str, list[tuple[Item, dict[str, Any]]]] = {"flagged before": [], "after a break": [],
                                                                 "ordinary": []}
        for item in items:
            applies = expectations_for(expectations, item.trace.case_id)
            if str(spec.check) not in applies:
                continue
            if spec.check in code.registered:
                verdict = code.results.get(item.key, {}).get(spec.check)
                if verdict is None or verdict.status != "pass":
                    continue   # the judge only sees tries its code part hands over
            source = ("flagged before" if item.key in known.bad.get(spec.check, set())
                      else "after a break" if item.key in extra_keys else "ordinary")
            groups[source].append((item, applies[str(spec.check)]))
        for group in groups.values():
            rng.shuffle(group)
        chosen: list[tuple[str, Item, dict[str, Any]]] = []
        while len(chosen) < per_judge and any(groups.values()):
            for source, group in groups.items():
                if group and len(chosen) < per_judge:
                    item, notes = group.pop()
                    chosen.append((source, item, notes))
        for n, (source, item, notes) in enumerate(chosen, start=1):
            picked.append(LabelItem(f"q{spec.check}-{n}", spec.check, spec.question, item.key, source,
                                    item.trace.to_dict(), asdict(item.case) if item.case else None, notes))
    rng.shuffle(picked)
    return picked


def set_id(items: list[LabelItem]) -> str:
    raw = json.dumps([(i.id, i.key, i.question) for i in items])
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


# -- the labeling page ------------------------------------------------------------------------------

def _events(trace: Trace) -> list[dict[str, Any]]:
    events = []
    for event in trace.timeline():
        if hasattr(event, "role"):
            events.append({"type": "customer" if event.role == "user" else "agent", "text": event.text})
        else:
            # Text as people read it (real line breaks), anything else as JSON.
            args = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={json.dumps(v, ensure_ascii=False)}"
                             for k, v in event.input.items())
            events.append({"type": "tool", "name": event.tool, "args": args, "output": event.output or "",
                           "error": bool(event.is_error)})
    return events


def page_data(items: list[LabelItem], set_name: str, agent_name: str) -> dict[str, Any]:
    out = []
    for n, item in enumerate(items, start=1):
        trace = Trace.from_dict(item.trace)
        case = item.case or {}
        out.append({
            "id": item.id, "n": n, "question": item.question,
            "task": case.get("task") or trace.task, "expected": case.get("expected_outcome", ""),
            "must_not": case.get("must_not", ""), "notes": item.notes,
            "failures": sorted((case.get("inject_failures") or {}).keys()),
            "events": _events(trace), "changes": world_changes(trace.world_before, trace.world_after),
        })
    return {"set": set_name, "agent": agent_name, "items": out}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label your agent's answers</title>
<style>
:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#64625b;--line:#e2dfd6;--card:#fff;--accent:#2b5d8a;--cust:#eef3f8;
--agent:#f3f1ea;--tool:#faf8f2;--err:#a33;--yes:#2e6b3a;--no:#9b2c2c}
@media (prefers-color-scheme:dark){:root{--bg:#151513;--fg:#ecebe6;--muted:#a3a198;--line:#34332f;--card:#1e1e1b;
--accent:#8db8de;--cust:#1d2731;--agent:#24231f;--tool:#1b1b18;--err:#f09a9a;--yes:#9fd3a8;--no:#f0a3a3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif}
main{max-width:46rem;margin:0 auto;padding:2rem 1rem 6rem}
h1{font-size:1.6rem;margin:0 0 .3rem}p.lead{color:var(--muted);margin:.2rem 0 1.5rem}
.item{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.2rem;margin:1.4rem 0}
.num{color:var(--muted);font-size:.85rem}.q{font-weight:600;font-size:1.05rem;margin:.3rem 0 .8rem}
.ex{font-size:.9rem;color:var(--muted);margin:.2rem 0}.ex b{color:var(--fg)}
.chat{margin:.8rem 0;display:flex;flex-direction:column;gap:.45rem}
.msg{padding:.55rem .75rem;border-radius:10px;white-space:pre-wrap;word-wrap:break-word;max-width:92%}
.customer{background:var(--cust);align-self:flex-end}.agent{background:var(--agent)}
.who{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);display:block}
.tool{background:var(--tool);border:1px dashed var(--line);font:13px/1.45 ui-monospace,Consolas,monospace;
white-space:pre-wrap;word-wrap:break-word;padding:.45rem .6rem;border-radius:8px}.tool .err{color:var(--err)}
.changes{font-size:.88rem;color:var(--muted)}.answers{display:flex;gap:.5rem;margin-top:.9rem;flex-wrap:wrap}
.answers button{font:inherit;padding:.45rem 1.1rem;border-radius:99px;border:1px solid var(--line);
background:transparent;color:var(--fg);cursor:pointer}
.answers button[aria-pressed=true].yes{background:var(--yes);color:var(--card);border-color:var(--yes)}
.answers button[aria-pressed=true].no{background:var(--no);color:var(--card);border-color:var(--no)}
.answers button[aria-pressed=true].unsure{background:var(--muted);color:var(--card);border-color:var(--muted)}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);
padding:.7rem 1rem;display:flex;justify-content:center;gap:1rem;align-items:center}
.bar button{font:inherit;font-weight:600;padding:.55rem 1.3rem;border-radius:99px;border:0;background:var(--accent);
color:var(--card);cursor:pointer}
</style></head><body><main>
<h1>Label your agent's answers</h1>
<p class="lead">For each try, read what happened and answer the question yourself, the way you would if you were
reviewing the chat. The AI judges answer the same questions; your answers show how far to trust them. You won't see
their answers here. When you're done, click <b>Save labels</b> and put the file it saves (labels.json) in the
<code>labels</code> folder next to this page. If you're unsure, pick "Not sure".</p>
<div id="items"></div></main>
<div class="bar"><span id="progress"></span><button id="save">Save labels</button></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const storeKey = 'eval-agent-labels-' + data.set;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(storeKey) || '{}'); } catch (e) { answers = {}; }
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text; return n; };
function render() {
  const root = document.getElementById('items');
  root.innerHTML = '';
  for (const item of data.items) {
    const card = el('section', 'item');
    card.appendChild(el('div', 'num', 'Try ' + item.n + ' of ' + data.items.length));
    card.appendChild(el('div', 'q', item.question));
    const ex = el('div', 'ex'); ex.innerHTML = '<b>Done right:</b> '; ex.appendChild(document.createTextNode(item.expected || '(not written)'));
    card.appendChild(ex);
    if (item.must_not) { const m = el('div', 'ex'); m.innerHTML = '<b>Must not:</b> '; m.appendChild(document.createTextNode(item.must_not)); card.appendChild(m); }
    if (item.failures.length) card.appendChild(el('div', 'ex', 'In this test, these tools were made to fail on purpose: ' + item.failures.join(', ')));
    if (Object.keys(item.notes).length) card.appendChild(el('div', 'ex', 'Details for this example: ' + JSON.stringify(item.notes)));
    const chat = el('div', 'chat');
    for (const e of item.events) {
      if (e.type === 'tool') {
        const t = el('div', 'tool');
        t.appendChild(document.createTextNode('Tool: ' + e.name + '(' + e.args + ')\\n'));
        const out = el('span', e.error ? 'err' : '', (e.error ? 'ERROR: ' : '→ ') + e.output);
        t.appendChild(out); chat.appendChild(t);
      } else {
        const m = el('div', 'msg ' + e.type);
        m.appendChild(el('span', 'who', e.type === 'customer' ? 'Customer' : 'Your agent'));
        m.appendChild(document.createTextNode(e.text)); chat.appendChild(m);
      }
    }
    card.appendChild(chat);
    card.appendChild(el('div', 'changes', 'Changes to the pretend world: ' + (item.changes.length ? item.changes.join('; ') : 'none')));
    const row = el('div', 'answers');
    for (const [value, label] of [['yes', 'Yes'], ['no', 'No'], ['unsure', 'Not sure']]) {
      const b = el('button', value, label);
      b.setAttribute('aria-pressed', answers[item.id] === value ? 'true' : 'false');
      b.onclick = () => { answers[item.id] = value;
        try { localStorage.setItem(storeKey, JSON.stringify(answers)); } catch (e) {}
        render(); };
      row.appendChild(b);
    }
    card.appendChild(row);
    root.appendChild(card);
  }
  const done = data.items.filter(i => answers[i.id]).length;
  document.getElementById('progress').textContent = done + ' of ' + data.items.length + ' answered';
}
document.getElementById('save').onclick = () => {
  const blob = new Blob([JSON.stringify({set: data.set, answers: answers}, null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'labels.json';
  document.body.appendChild(a); a.click(); a.remove();
};
render();
</script></body></html>
"""


def label_page(items: list[LabelItem], set_name: str, agent_name: str) -> str:
    raw = json.dumps(page_data(items, set_name, agent_name), ensure_ascii=False)
    # Whatever the agent or its tools wrote, it can't end the data's <script> or start markup of its own.
    safe = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return PAGE.replace("__DATA__", safe)


def write_label_set(evals: Path, items: list[LabelItem], agent_name: str) -> tuple[str, Path]:
    folder = evals / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    name = set_id(items)
    (folder / "items.json").write_text(json.dumps(
        {"set": name, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
         "items": [asdict(i) for i in items]}, indent=1, ensure_ascii=False), encoding="utf-8")
    page = folder / "label.html"
    page.write_text(label_page(items, name, agent_name), encoding="utf-8")
    return name, page


def load_label_set(evals: Path) -> tuple[str, list[LabelItem]]:
    data = json.loads((evals / "labels" / "items.json").read_text(encoding="utf-8"))
    return data["set"], [LabelItem(**i) for i in data["items"]]


def find_labels(evals: Path, set_name: str, given: Path | None = None) -> Path | None:
    """labels.json for this set: the given file, the labels folder, or the newest one in Downloads."""
    candidates = [given] if given else [evals / "labels" / "labels.json"]
    if not given:
        downloads = Path.home() / "Downloads"
        if downloads.is_dir():
            candidates += sorted(downloads.glob("labels*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("set") == set_name:
            return path
    return None


def read_answers(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in (data.get("answers") or {}).items() if v in ANSWERS}


def label_in_terminal(items: list[LabelItem], ask: Callable[[str], str] = input,
                      show: Callable[[str], None] = print) -> dict[str, str]:
    answers: dict[str, str] = {}
    for n, item in enumerate(items, start=1):
        trace = Trace.from_dict(item.trace)
        case = Case(**item.case) if item.case else None
        show(f"\n--- Try {n} of {len(items)} ---\n{render_case(case)}\n{render_run(trace)}")
        if item.notes:
            show(f"Details for this example: {json.dumps(item.notes, ensure_ascii=False)}")
        show(f"\nQuestion: {item.question}")
        while True:
            reply = ask("Your answer (y = yes, n = no, s = not sure, q = stop here): ").strip().lower()
            if reply in ("y", "n", "s", "q"):
                break
        if reply == "q":
            break
        answers[item.id] = {"y": "yes", "n": "no", "s": "unsure"}[reply]
    return answers


# -- comparing the judges with the labels ----------------------------------------------------------------

@dataclass
class JudgeAgreement:
    check: int
    question_hash: str
    labeled: int = 0
    agreed: int = 0
    disagreements: list[dict[str, str]] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        return self.labeled >= MIN_LABELS and self.agreed / self.labeled >= TRUST_AT


async def calibrate(items: list[LabelItem], answers: dict[str, str], judges: list[JudgeSpec], *,
                    model: str = JUDGE_MODEL, client: Any = None) -> tuple[dict[int, JudgeAgreement], Spend]:
    """Ask each judge its question on the labeled tries and count agreement with the owner."""
    own = client is None
    if own:
        import anthropic

        client = anthropic.AsyncAnthropic()
    spend = Spend()
    specs = {j.check: j for j in judges}
    results = {n: JudgeAgreement(n, question_hash(spec)) for n, spec in specs.items()}
    sem = asyncio.Semaphore(6)

    async def one(item: LabelItem) -> None:
        spec = specs.get(item.check)
        answer = answers.get(item.id)
        if spec is None or answer not in ("yes", "no") or spec.question != item.question:
            return
        trace = Trace.from_dict(item.trace)
        case = Case(**item.case) if item.case else None
        async with sem:
            try:
                verdict = await judge_one(client, model, spec, Item(item.key, trace, case), item.notes, spend)
            except Exception:
                return
        judged = spec.fail_when if verdict.status == "fail" else ("no" if spec.fail_when == "yes" else "yes")
        row = results[item.check]
        row.labeled += 1
        if judged == answer:
            row.agreed += 1
        else:
            row.disagreements.append({"item": item.id, "try": item.key, "you": answer, "judge": judged,
                                      "judge_reason": verdict.reason.removeprefix("Judge: ")})

    try:
        await asyncio.gather(*(one(i) for i in items))
    finally:
        if own:
            await client.close()
    return results, spend


def save_calibration(evals: Path, results: dict[int, JudgeAgreement], model: str, set_name: str) -> None:
    (evals / "calibration.json").write_text(json.dumps({
        "judge_model": model, "label_set": set_name, "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checks": {str(n): {**asdict(r), "trusted": r.trusted} for n, r in results.items()},
    }, indent=1, ensure_ascii=False), encoding="utf-8")


def trust(evals: Path, judges: list[JudgeSpec], judge_model: str) -> dict[int, tuple[bool, str]]:
    """Per judge check: may it fail a must-never-fail check, and why (plain words)."""
    path = evals / "calibration.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out = {}
    for spec in judges:
        row = (saved.get("checks") or {}).get(str(spec.check))
        if not row:
            out[spec.check] = (False, "its judge hasn't been checked against your labels yet")
        elif row.get("question_hash") != question_hash(spec):
            out[spec.check] = (False, "its judge's question changed since you labeled, so it needs checking again")
        elif saved.get("judge_model") != judge_model:
            out[spec.check] = (False, f"its judge was checked with {saved.get('judge_model')}, not {judge_model}")
        elif not row.get("trusted"):
            out[spec.check] = (False, f"its judge agreed with you only {row['agreed']} of {row['labeled']} times")
        else:
            out[spec.check] = (True, f"its judge agreed with you {row['agreed']} of {row['labeled']} times")
    return out


def report(results: dict[int, JudgeAgreement], names: dict[int, tuple[str, str]]) -> str:
    """names: check -> (name, level)."""
    lines = []
    trusted = sum(1 for r in results.values() if r.trusted)
    lines.append(f"{trusted} of {len(results)} judges agree with you often enough to trust (at least 9 times in 10, "
                 f"on at least {MIN_LABELS} labels).")
    for n, r in sorted(results.items()):
        name, level = names.get(n, (f"check {n}", ""))
        if not r.labeled:
            lines.append(f"- {name}: no labels to compare yet.")
            continue
        status = "trusted" if r.trusted else "not trusted yet"
        extra = (" It can still flag problems, but it can't fail this must-never-fail check."
                 if not r.trusted and level == "must_never_fail" else "")
        lines.append(f"- {name}: agreed with you {r.agreed} of {r.labeled} times, so {status}.{extra}")
        for d in r.disagreements[:3]:
            lines.append(f"    {d['try']}: you said {d['you']}, the judge said {d['judge']} ({d['judge_reason']})")
    return "\n".join(lines)
