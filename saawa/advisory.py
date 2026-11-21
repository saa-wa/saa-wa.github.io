"""Stage 1 baseline: a rule-based advisory generator.

It turns a structured Forecast into the advisory JSON schema used throughout the project,
including a plain-English message of at most 320 characters (two SMS messages).

The rules follow GMet's own seasonal recommendations (early-maturing and drought-tolerant
varieties where dry spells are long or the season is short; follow forecast updates). They are
the baseline the fine-tuned model is compared against, and they draft the target advisories that
MoFA extension agents then correct. They are not agronomic advice until a person approves them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .forecast import Forecast

SMS_LIMIT = 320
CROPS = ("maize", "millet", "sorghum", "groundnut", "cowpea")


@dataclass
class Advisory:
    season: int
    onset_window: str
    onset_category: str
    early_dry_spell_days: str
    early_dry_spell_category: str
    late_dry_spell_days: str
    late_dry_spell_category: str
    cessation_window: str
    cessation_category: str
    crop_actions: dict[str, str] = field(default_factory=dict)
    message_en: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Advisory":
        data = json.loads(text)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def text_for_checking(self) -> str:
        """All the advisory's wording in one string, which the Fact-Consistency Guard reads."""
        return "\n".join([
            f"Onset expected {self.onset_window}.",
            f"Early dry spell of {self.early_dry_spell_days} days.",
            f"Late dry spell of {self.late_dry_spell_days} days.",
            f"Rains expected to end {self.cessation_window}.",
            self.message_en,
        ])


def _days(r: tuple[int, int]) -> str:
    return str(r[0]) if r[0] == r[1] else f"{r[0]} to {r[1]}"


def crop_actions(fc: Forecast) -> dict[str, str]:
    long_early_dry = fc.early_dry_category == "long" or fc.early_dry[1] >= 12
    early_end = fc.cessation_category == "early"
    actions = {
        "maize": "Maize suffers most in a dry spell, so sow it only after the rains are steady"
                 + (" and choose an early-maturing variety." if long_early_dry or early_end else "."),
        "millet": "Millet copes with dry spells better than maize"
                  + (", so it is a good choice this season." if long_early_dry else "."),
        "sorghum": "Sorghum copes with dry spells better than maize"
                   + (", so it is a good choice this season." if long_early_dry else "."),
        "groundnut": "Use an early-maturing groundnut variety"
                     + (" so it is ready before the rains end." if early_end else "."),
        "cowpea": "Use an early-maturing cowpea variety"
                  + (" so it is ready before the rains end." if early_end else "."),
    }
    return actions


# Every SMS is assembled from these sentence templates, so the same message can be rendered in
# Dagaare from a translator-approved copy of the templates (see translate.render_dagaare).
# Slots: week phrases use "week", day ranges use "range"; both are templates too.
TEMPLATES_EN = {
    "header": "[Saa Wa? {season}]",
    "onset": "GMet expects the rains to start from the {onset_start} to the {onset_end}.",
    "onset_late": "GMet expects the rains to start from the {onset_start} to the {onset_end}, later than usual.",
    "onset_early": "GMet expects the rains to start from the {onset_start} to the {onset_end}, earlier than usual.",
    "dry": "A dry spell of {dry_days} days may follow, so do not plant on the first rains.",
    "end": "Rains should end from the {cess_start} to the {cess_end}.",
    "seed": "Use early-maturing seed.",
    "wait": "Wait for steady rains before sowing.",
    "ask": "Ask your extension officer.",
    "week": "{ordinal} week of {month}",
    "range": "{lo} to {hi}",
}


def message_parts(fc: Forecast) -> list[tuple[str, dict]]:
    """The template keys and slot values of the SMS, before rendering into a language."""
    onset_key = {"late": "onset_late", "early": "onset_early"}.get(fc.onset_category, "onset")
    return [
        ("header", {"season": fc.season}),
        (onset_key, {"onset_start": fc.onset.start, "onset_end": fc.onset.end}),
        ("dry", {"dry_days": fc.early_dry}),
        ("end", {"cess_start": fc.cessation.start, "cess_end": fc.cessation.end}),
        ("seed" if fc.early_dry_category == "long" or fc.cessation_category == "early" else "wait", {}),
        ("ask", {}),
    ]


def render_parts(parts: list[tuple[str, dict]], templates: dict[str, str],
                 week_words: dict[str, str] | None = None) -> str:
    """Fill templates. week_words maps "1st".."4th" and month names to the target language."""
    words = week_words or {}

    def slot(value) -> str:
        if hasattr(value, "month"):  # a Week
            ordinal, month = str(value).split(" week of ")
            return templates["week"].format(ordinal=words.get(ordinal, ordinal), month=words.get(month, month))
        if isinstance(value, tuple):  # a day range
            return str(value[0]) if value[0] == value[1] else templates["range"].format(lo=value[0], hi=value[1])
        return str(value)

    return " ".join(templates[key].format(**{k: slot(v) for k, v in slots.items()}) for key, slots in parts)


def fit_parts(parts: list[tuple[str, dict]], limit: int = SMS_LIMIT) -> list[tuple[str, dict]]:
    """Drop the least essential sentences (advice, then end of season) until the SMS fits."""
    parts = list(parts)
    for key in ("ask", "seed", "wait", "end"):
        if len(render_parts(parts, TEMPLATES_EN)) <= limit:
            break
        parts = [p for p in parts if p[0] != key]
    return parts


def build_message(fc: Forecast) -> str:
    return render_parts(fit_parts(message_parts(fc)), TEMPLATES_EN)


def generate(fc: Forecast) -> Advisory:
    return Advisory(
        season=fc.season,
        onset_window=f"{fc.onset.start} to {fc.onset.end}",
        onset_category=fc.onset_category,
        early_dry_spell_days=_days(fc.early_dry),
        early_dry_spell_category=fc.early_dry_category,
        late_dry_spell_days=_days(fc.late_dry),
        late_dry_spell_category=fc.late_dry_category,
        cessation_window=f"{fc.cessation.start} to {fc.cessation.end}",
        cessation_category=fc.cessation_category,
        crop_actions=crop_actions(fc),
        message_en=build_message(fc),
    )
