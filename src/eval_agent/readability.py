"""Is this easy to read? Checks that need no model (docs/prd.md FR-29 and FR-31).

- Reading level: the Flesch-Kincaid grade (US school grade), targeting 9 or below.
- Long sentences: the rules aim for under 20 words; over 25 gets flagged.
- Words to avoid: "eval", "mock", "trials" have everyday replacements.
- Technical words must be explained in the glossary.
- Code, file names and ids belong only in the technical section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_GRADE = 9.0
PART_GRADE_LIMIT = 10.0   # one part may run a little above the target; the whole plan may not
PART_MIN_WORDS = 30       # grades of very short texts swing too much to judge
LONG_SENTENCE = 25

# Never use these: there's an everyday word that says the same thing.
REPLACE = {
    "eval": "test", "evals": "tests", "evaluation": "test", "evaluations": "tests",
    "mock": "pretend version", "mocks": "pretend versions", "mocked": "pretend",
    "trial": "try", "trials": "tries", "trajectory": "steps", "trajectories": "steps",
}
# Fine when explained: they must then be in the glossary.
JARGON = {
    "llm", "api", "apis", "mcp", "sdk", "json", "schema", "schemas", "harness", "assertion", "assertions",
    "regression", "regressions", "deterministic", "nondeterministic", "stochastic", "rubric", "metric",
    "metrics", "mutation", "mutations", "ci", "pipeline", "endpoint", "token", "tokens", "prompt",
    "prompts", "judge", "calibration", "trace", "traces", "repo", "repository",
}
_CODE = [
    (re.compile(r"`[^`]+`"), "code in backticks"),
    (re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b"), "a code name"),
    (re.compile(r"\b[\w-]+\.(?:py|yaml|yml|json|jsonl|md|toml|csv|html)\b"), "a file name"),
    (re.compile(r"\b\w+\(\)"), "a function call"),
    (re.compile(r"\d+\.\d+\s*%"), "a percentage with decimals (use counts)"),
]
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d[\d,.]*")


def sentences(text: str) -> list[str]:
    parts = []
    for line in re.split(r"\n+", text):
        parts += [s.strip() for s in re.split(r"(?<=[.!?])[\"')\]]*\s+", line) if s.strip()]
    return parts


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def syllables(word: str) -> int:
    w = word.lower().strip("'’-")
    if not w or w[0].isdigit():
        return 1
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def grade(text: str) -> float:
    """Flesch-Kincaid grade level. 0 for empty text."""
    ws = words(text)
    if not ws:
        return 0.0
    count = max(1, len(sentences(text)))
    syl = sum(syllables(w) for w in ws)
    return round(0.39 * (len(ws) / count) + 11.8 * (syl / len(ws)) - 15.59, 1)


@dataclass
class ReadIssue:
    part: str
    problem: str

    def __str__(self) -> str:
        return f"{self.part}: {self.problem}"


def part_issues(part: str, text: str, glossary: set[str]) -> list[ReadIssue]:
    """Problems in one reader-facing part of a plan."""
    issues = []
    n = len(words(text))
    g = grade(text)
    if n >= PART_MIN_WORDS and g > PART_GRADE_LIMIT:
        issues.append(ReadIssue(part, f"reads at grade {g:g}; aim for {TARGET_GRADE:g} or below"))
    for sentence in sentences(text):
        length = len(words(sentence))
        if length > LONG_SENTENCE:
            issues.append(ReadIssue(part, f"a {length}-word sentence: \"{sentence[:80]}...\""))
    lowered = [w.lower().strip("'’-") for w in words(text)]
    for word in dict.fromkeys(lowered):
        if word in REPLACE:
            issues.append(ReadIssue(part, f"says \"{word}\"; say \"{REPLACE[word]}\" instead"))
        elif word in JARGON and word not in glossary and word.rstrip("s") not in glossary:
            issues.append(ReadIssue(part, f"uses \"{word}\" without a glossary entry"))
    for pattern, label in _CODE:
        match = pattern.search(text)
        if match:
            issues.append(ReadIssue(part, f"has {label} (\"{match.group(0)}\"); keep that for the technical "
                                          "section"))
    return issues
