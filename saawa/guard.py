"""Fact-Consistency Guard and the numeric fact consistency (NFC) score.

NFC = |F_source ∩ F_advisory| / |F_source ∪ F_advisory|

F_source is the set of forecast facts for the season (onset window, early and late dry-spell
lengths, cessation window). F_advisory is every such fact found in the advisory's text. A score
of 1 means nothing was lost and nothing was invented; anything less sends the advisory to a person.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .forecast import WEEK_RE, Forecast, Week, Window

Fact = tuple[str, str]

DAYS_RE = re.compile(r"\b(\d{1,3})\s*(?:-|–|to)\s*(\d{1,3})\s*days\b|\b(\d{1,3})\s*days\b", re.IGNORECASE)
START_WORDS = re.compile(r"\b(start|starts|begin|begins|onset)\b", re.IGNORECASE)
END_WORDS = re.compile(r"\b(end|ends|stop|stops|cessation)\b", re.IGNORECASE)
LATE_SPELL_WORDS = re.compile(r"\b(late|second|2nd)\b", re.IGNORECASE)


def _window_key(w: Window) -> str:
    return f"{w.start}|{w.end}"


def source_facts(fc: Forecast) -> set[Fact]:
    return {
        ("onset", _window_key(fc.onset)),
        ("cessation", _window_key(fc.cessation)),
        ("early_dry_spell", f"{fc.early_dry[0]}-{fc.early_dry[1]}"),
        ("late_dry_spell", f"{fc.late_dry[0]}-{fc.late_dry[1]}"),
    }


def extract_facts(text: str) -> set[Fact]:
    """Find every onset/cessation window and dry-spell length stated in free text."""
    facts: set[Fact] = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if not sentence.strip():
            continue
        weeks = [Week.parse(m.group(0)) for m in WEEK_RE.finditer(sentence)]
        kind = "onset" if START_WORDS.search(sentence) else "cessation" if END_WORDS.search(sentence) else "window"
        for i in range(0, len(weeks) - 1, 2):
            facts.add((kind, _window_key(Window(weeks[i], weeks[i + 1]))))
        if len(weeks) % 2:
            facts.add(("week", str(weeks[-1])))
        for m in DAYS_RE.finditer(sentence):
            lo, hi = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(3))
            if "dry" in sentence.lower():
                kind_days = "late_dry_spell" if LATE_SPELL_WORDS.search(sentence) else "early_dry_spell"
            else:
                kind_days = "days"
            facts.add((kind_days, f"{int(lo)}-{int(hi)}"))
    return facts


def nfc(source: set[Fact], advisory: set[Fact]) -> float:
    union = source | advisory
    return len(source & advisory) / len(union) if union else 1.0


@dataclass
class GuardResult:
    passed: bool
    nfc: float
    missing: list[Fact] = field(default_factory=list)
    invented: list[Fact] = field(default_factory=list)

    def explain(self) -> str:
        if self.passed:
            return f"PASS (NFC = {self.nfc:.2f}). Every forecast fact is present and nothing was added."
        lines = [f"SENT TO A PERSON (NFC = {self.nfc:.2f})."]
        lines += [f"Missing from advisory: {k} = {v}" for k, v in self.missing]
        lines += [f"Not in the GMet forecast: {k} = {v}" for k, v in self.invented]
        return "\n".join(lines)


def check(fc: Forecast, advisory_text: str) -> GuardResult:
    src, adv = source_facts(fc), extract_facts(advisory_text)
    score = nfc(src, adv)
    return GuardResult(passed=score == 1.0, nfc=score,
                       missing=sorted(src - adv), invented=sorted(adv - src))


def numbers_preserved(english: str, translated: str) -> bool:
    """A translation must carry the same digits as the English advisory (dates and day counts)."""
    return Counter(re.findall(r"\d+", english)) == Counter(re.findall(r"\d+", translated))
