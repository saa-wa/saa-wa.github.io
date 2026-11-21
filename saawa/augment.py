"""Counterfactual augmentation: build the Forecast-Advisory corpus from the real GMet seasons.

Each real Upper West forecast is varied systematically (onset and cessation shifted by whole weeks,
dry-spell ranges moved and widened) and every variant is written out in one of three GMet styles:
the table layout, the narrative summary, and a "raw PDF" layout with the LTM and forecast columns
run together. Targets come from the rule-based generator and are drafts until MoFA agents correct them.

Splits follow the proposal: variants of the 2024 season train, 2025 validates, 2026 tests, so the
model is always tested on a season it has never seen.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

from . import advisory
from .forecast import Forecast, Range, Window, load_curated

SPLIT_BY_SEASON = {2024: "train", 2025: "validation", 2026: "test"}
VARIANTS_PER_SEASON = {2024: 130, 2025: 35, 2026: 35}
STYLES = ("table", "narrative", "raw_pdf")

INSTRUCTION = (
    "You are Saa Wa?, an assistant for MoFA extension officers in Ghana's Upper West Region. "
    "Read the GMet seasonal forecast below and return one JSON object for Upper West farmers with "
    "exactly these keys: season (year), onset_window (\"<week> to <week>\", forecast not normal), "
    "onset_category (early/normal/late versus normal), early_dry_spell_days (\"<low> to <high>\"), "
    "early_dry_spell_category (short/normal/long), late_dry_spell_days, late_dry_spell_category, "
    "cessation_window, cessation_category (early/normal/late), crop_actions (an object with advice "
    "for maize, millet, sorghum, groundnut and cowpea) and message_en (an SMS of at most 320 "
    "characters for farmers). Weeks are written like \"3rd week of April\". Copy every date, week and "
    "number of days exactly as the forecast states it. Never add a fact that the forecast does not contain."
)


def narrative_text(fc: Forecast) -> str:
    text = (
        f"The onset of rain for the {fc.season} season in the Upper West is expected between the "
        f"{fc.onset.start} and the {fc.onset.end}, compared with a normal onset of "
        f"{fc.onset_ltm.start} to {fc.onset_ltm.end}. "
        f"{fc.early_dry_category.capitalize()} early dry spells of {fc.early_dry[0]}-{fc.early_dry[1]} days "
        f"are forecast in the Upper West (LTM {fc.early_dry_ltm[0]} days). "
        f"Late dry spells are expected to last {fc.late_dry[0]}-{fc.late_dry[1]} days "
        f"against a normal of {fc.late_dry_ltm[0]} days. "
        f"Cessation is expected between the {fc.cessation.start} and the {fc.cessation.end} "
        f"(normal {fc.cessation_ltm.start} to {fc.cessation_ltm.end})."
    )
    return text


def raw_pdf_text(fc: Forecast) -> str:
    # Mimics pdftotext output: LTM and forecast columns side by side, wrapped mid-phrase.
    def split_window(w):
        return f"{w.start} - {str(w.end).rsplit(' ', 1)[0]}", str(w.end).rsplit(" ", 1)[1]
    o_ltm_a, o_ltm_b = split_window(fc.onset_ltm)
    o_fc_a, o_fc_b = split_window(fc.onset)
    c_ltm_a, c_ltm_b = split_window(fc.cessation_ltm)
    c_fc_a, c_fc_b = split_window(fc.cessation)
    return "\n".join([
        f"Table 2: Onset Dates for {fc.season} Season & Long-Term Mean (Normal) of the Onset Dates",
        "ZONE        Normal Onset Dates (LTM)  Forecasted Onset Dates",
        f"Upper West  {o_ltm_a} {o_fc_a}",
        f"            {o_ltm_b}                       {o_fc_b}",
        "Table 6. LTM of First Dry Spell Days and Forecast of First Dry Spell Days",
        f"Upper West       {fc.early_dry_ltm[0]}            {fc.early_dry[0]}-{fc.early_dry[1]}",
        "Table 7. LTM for Second Dry Spell Days and Forecast of Late Dry Spell- Days",
        f"Upper West       {fc.late_dry_ltm[0]}              {fc.late_dry[0]}-{fc.late_dry[1]}",
        f"Table 8. Cessation Dates for {fc.season} Season & Long-Term Mean Cessation Dates",
        f"Upper West  {c_ltm_a} {c_fc_a}",
        f"            {c_ltm_b}                {c_fc_b}",
    ])


def render(fc: Forecast, style: str) -> str:
    return {"table": fc.source_text, "narrative": lambda: narrative_text(fc),
            "raw_pdf": lambda: raw_pdf_text(fc)}[style]()


def _vary_range(rng: random.Random, base: Range, lo_min: int, hi_max: int) -> Range:
    low = max(lo_min, base[0] + rng.randint(-3, 3))
    width = max(0, (base[1] - base[0]) + rng.randint(-2, 3))
    return (low, min(hi_max, low + width))


def variant(rng: random.Random, fc: Forecast) -> Forecast:
    start = fc.onset.start.shift(rng.randint(-2, 4))
    span = max(1, (fc.onset.end.index - fc.onset.start.index) + rng.randint(-1, 1))
    onset = Window(start, start.shift(span))
    cessation = fc.cessation.shift(rng.randint(-3, 2))
    return replace(
        fc,
        onset=onset,
        early_dry=_vary_range(rng, fc.early_dry, 5, 21),
        late_dry=_vary_range(rng, fc.late_dry, 7, 25),
        cessation=cessation,
        note="counterfactual variant",
    )


def build(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for fc in load_curated():
        split = SPLIT_BY_SEASON[fc.season]
        # The real forecast itself, in every style.
        for style in STYLES:
            rows.append(_example(fc, style, split, "real"))
        for i in range(VARIANTS_PER_SEASON[fc.season]):
            rows.append(_example(variant(rng, fc), STYLES[i % len(STYLES)], split, "counterfactual"))
    return rows


def _example(fc: Forecast, style: str, split: str, origin: str) -> dict:
    from .guard import source_facts

    adv = advisory.generate(fc)
    return {
        "source_facts": sorted(source_facts(fc)),
        "split": split,
        "origin": origin,
        "style": style,
        "season": fc.season,
        "instruction": INSTRUCTION,
        "input": render(fc, style),
        "output": adv.to_json(),
        "target_status": "draft, awaiting MoFA extension officer review",
    }


def write(rows: list[dict], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        counts[split] = len(subset)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return counts


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "processed" / "forecast_advisory"
    print(write(build(), out))
