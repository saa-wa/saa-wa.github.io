"""The full Saa Wa? pipeline: forecast -> advisory -> guard -> Dagaare -> human approval.

An advisory is released only when it has passed the Fact-Consistency Guard, a MoFA extension
officer has approved the English text, and a Dagaare reviewer has approved the translation.
Every approval or rejection is appended to data/review_log.jsonl.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import advisory as advisory_mod
from . import guard, translate
from .advisory import Advisory
from .forecast import Forecast

ROOT = Path(__file__).resolve().parent.parent
REVIEW_LOG = ROOT / "data" / "review_log.jsonl"


@dataclass
class Result:
    forecast: Forecast
    advisory: Advisory | None
    generator: str
    raw_generation: str = ""
    guard: guard.GuardResult | None = None
    dagaare: translate.Translation | None = None
    dagaare_missing: list[str] = field(default_factory=list)
    numbers_ok: bool | None = None

    @property
    def key(self) -> str:
        """Identifies this exact advisory text, so approvals cannot carry over to a changed one."""
        payload = (self.advisory.to_json() if self.advisory else "") + (self.dagaare.text or "" if self.dagaare else "")
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def is_draft(self) -> bool:
        return bool(self.dagaare and self.dagaare.text and self.dagaare.status == translate.DRAFT_STATUS)

    @property
    def status(self) -> str:
        if self.advisory is None:
            return "Blocked. The model did not return a valid advisory; a person must write it."
        if not self.guard.passed:
            return "Blocked. The Fact-Consistency Guard found a mismatch; sent to a person."
        approvals = approvals_for(self.key)
        if self.dagaare is None or self.dagaare.text is None:
            waiting = "Dagaare translation"
        elif not self.numbers_ok:
            return "Blocked. The Dagaare text changes a number from the English advisory."
        elif self.is_draft:
            waiting = "approval of the Dagaare templates and glossary"
        else:
            waiting = ""
        needed = [r for r in ("mofa", "dagaare") if r not in approvals]
        if waiting or needed:
            items = ([waiting] if waiting else []) + [{"mofa": "MoFA approval", "dagaare": "Dagaare reviewer approval"}[r]
                                                      for r in needed]
            return "Awaiting " + ", ".join(items) + "."
        return "Released. Both reviewers approved; ready for SMS and radio."


def run(fc: Forecast, generator: str = "rule") -> Result:
    if generator == "model":
        from . import llm
        loaded = llm.load_finetuned()
        if loaded is None:
            raise RuntimeError("No fine-tuned Stage 1 model found under models/stage1-*")
        tok, model, name = loaded
        raw = llm.generate(tok, model, fc.source_text())
        adv = llm.parse_advisory(raw)
        result = Result(fc, adv, f"fine-tuned model ({name})", raw_generation=raw)
    else:
        adv = advisory_mod.generate(fc)
        result = Result(fc, adv, "rule-based baseline")

    text = adv.text_for_checking() if adv else result.raw_generation
    result.guard = guard.check(fc, text)
    if adv and result.guard.passed:
        if translate.has_finetuned_model():
            result.dagaare = translate.translate(adv.message_en)
        else:
            result.dagaare, result.dagaare_missing = translate.render_dagaare(fc, include_drafts=True)
        if result.dagaare.text:
            result.numbers_ok = guard.numbers_preserved(adv.message_en, result.dagaare.text)
    return result


def record_review(result: Result, role: str, reviewer: str, decision: str, note: str = "") -> None:
    REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "key": result.key, "season": result.forecast.season, "role": role,
        "reviewer": reviewer, "decision": decision, "note": note,
        "advisory": json.loads(result.advisory.to_json()) if result.advisory else None,
        "dagaare": result.dagaare.text if result.dagaare else None,
    }
    with open(REVIEW_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def approvals_for(key: str) -> set[str]:
    if not REVIEW_LOG.exists():
        return set()
    latest: dict[str, str] = {}
    for line in open(REVIEW_LOG, encoding="utf-8"):
        e = json.loads(line)
        if e["key"] == key:
            latest[e["role"]] = e["decision"]
    return {role for role, decision in latest.items() if decision == "approve"}
