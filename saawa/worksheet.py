"""Translator worksheets for the Dagaare side of Saa Wa?

message_templates.csv  the sentence templates every SMS is built from. Translators keep the {slots}
                       exactly as written; once approved, any advisory can be rendered in Dagaare.
glossary.csv           weather and farming terms, week ordinals and month names, agreed before any
                       sentence is translated.
parallel_worksheet.csv free sentences (crop advice and the real seasons' messages) for the
                       English-Dagaare parallel corpus that fine-tunes NLLB.

Dagaare columns are left blank for native speakers; a second speaker reviews each row and sets
status to "approved". Existing entries are kept when the worksheets are regenerated.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import advisory
from .forecast import load_curated
from .translate import sentences

ROOT = Path(__file__).resolve().parent.parent
DAGAARE_DIR = ROOT / "data" / "dagaare"

GLOSSARY_TERMS = [
    ("rainfall onset", "The date the rainy season truly starts"),
    ("false onset", "Early rains followed by a long dry break"),
    ("dry spell", "A run of days with no useful rain during the season"),
    ("early dry spell", "Longest dry run in the first 50 days of the season"),
    ("late dry spell", "Longest dry run from day 51 to the end of the season"),
    ("cessation", "The end of the rainy season"),
    ("below normal", "Less than the 30-year average"),
    ("near normal", "Close to the 30-year average"),
    ("above normal", "More than the 30-year average"),
    ("likely", "Fixed phrase for forecast probability"),
    ("possible", "Fixed phrase for lower forecast probability"),
    ("steady rains", "Rains that continue without a long break"),
    ("first rains", "The first rains of the season"),
    ("early-maturing variety", "Seed that ripens quickly"),
    ("drought-tolerant variety", "Seed that survives dry spells"),
    ("extension officer", "MoFA agricultural extension agent"),
    ("1st", "As in '1st week of May'"), ("2nd", "As in '2nd week of May'"),
    ("3rd", "As in '3rd week of May'"), ("4th", "As in '4th week of May'"),
    ("March", ""), ("April", ""), ("May", ""), ("June", ""), ("July", ""), ("August", ""),
    ("September", ""), ("October", ""), ("November", ""),
    ("maize", ""), ("millet", ""), ("sorghum", ""), ("groundnut", ""), ("cowpea", ""),
]

TEMPLATE_FIELDS = ["key", "english", "dagaare", "translator", "reviewer", "status", "notes"]
GLOSSARY_FIELDS = ["term_en", "meaning", "dagaare", "translator", "reviewer", "status", "notes"]
PARALLEL_FIELDS = ["id", "english", "dagaare", "translator", "reviewer", "status", "notes"]


def _existing(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def _write(path: Path, fields: list[str], key: str, rows: list[dict]) -> int:
    old = _existing(path, key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for row in rows:
            kept = old.get(row[key])
            w.writerow({**row, **{k: v for k, v in (kept or {}).items() if k in fields and v}})
    return len(rows)


def parallel_sentences() -> list[str]:
    found: dict[str, None] = {}
    for fc in load_curated():
        for s in sentences(advisory.build_message(fc)):
            found.setdefault(s, None)
        for action in advisory.crop_actions(fc).values():
            found.setdefault(action, None)
    return list(found)


def write() -> dict[str, int]:
    DAGAARE_DIR.mkdir(parents=True, exist_ok=True)
    todo = {"status": "to translate"}
    return {
        "templates": _write(DAGAARE_DIR / "message_templates.csv", TEMPLATE_FIELDS, "key",
                            [{"key": k, "english": v, **todo} for k, v in advisory.TEMPLATES_EN.items()]),
        "glossary_terms": _write(DAGAARE_DIR / "glossary.csv", GLOSSARY_FIELDS, "term_en",
                                 [{"term_en": t, "meaning": m, **todo} for t, m in GLOSSARY_TERMS]),
        "parallel_sentences": _write(DAGAARE_DIR / "parallel_worksheet.csv", PARALLEL_FIELDS, "english",
                                     [{"id": f"S{i:04d}", "english": s, **todo}
                                      for i, s in enumerate(parallel_sentences(), 1)]),
    }


def drafted(path: Path, key: str) -> dict[str, str]:
    """Every row with Dagaare text, approved or not, for an unverified preview only."""
    return {k: r["dagaare"].strip() for k, r in _existing(path, key).items() if r.get("dagaare", "").strip()}


def approved(path: Path, key: str) -> dict[str, str]:
    """Rows a reviewer has approved, as {key: dagaare}."""
    return {k: r["dagaare"].strip() for k, r in _existing(path, key).items()
            if r.get("status", "").strip().lower() == "approved" and r.get("dagaare", "").strip()}


if __name__ == "__main__":
    print(write())
