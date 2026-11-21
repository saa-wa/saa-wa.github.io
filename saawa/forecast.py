"""Structured Upper West seasonal forecast, and how to read it from the curated CSV.

A forecast holds GMet's long-term mean (LTM) and forecast values for one season.
Week windows are kept as GMet writes them ("3rd week of April - 2nd week of May");
day and millimetre ranges are kept as (low, high) integer pairs.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
ORDINALS = {"1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3,
            "4th": 4, "fourth": 4, "last": 4}
ORDINAL_WORDS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}

CURATED_CSV = Path(__file__).resolve().parent.parent / "data" / "curated" / "upper_west_forecasts.csv"

WEEK_RE = re.compile(
    r"\b(1st|2nd|3rd|4th|first|second|third|fourth|last)\s+week\s+of\s+(" + "|".join(MONTHS) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class Week:
    """A week of the year as GMet names it: the n-th week (1-4) of a month."""
    month: int  # 1-12
    n: int      # 1-4

    @classmethod
    def parse(cls, text: str) -> "Week":
        m = WEEK_RE.search(text)
        if not m:
            raise ValueError(f"no week found in {text!r}")
        return cls(MONTHS.index(m.group(2).capitalize()) + 1, ORDINALS[m.group(1).lower()])

    @property
    def index(self) -> int:
        """Weeks since January, so windows can be compared and shifted."""
        return (self.month - 1) * 4 + (self.n - 1)

    @classmethod
    def from_index(cls, i: int) -> "Week":
        return cls(i // 4 + 1, i % 4 + 1)

    def shift(self, weeks: int) -> "Week":
        return Week.from_index(self.index + weeks)

    def __str__(self) -> str:
        return f"{ORDINAL_WORDS[self.n]} week of {MONTHS[self.month - 1]}"


@dataclass(frozen=True)
class Window:
    start: Week
    end: Week

    @classmethod
    def parse(cls, text: str) -> "Window":
        weeks = [Week(MONTHS.index(m.group(2).capitalize()) + 1, ORDINALS[m.group(1).lower()])
                 for m in WEEK_RE.finditer(text)]
        if len(weeks) != 2:
            raise ValueError(f"expected two weeks in {text!r}, found {len(weeks)}")
        return cls(weeks[0], weeks[1])

    def shift(self, weeks: int) -> "Window":
        return Window(self.start.shift(weeks), self.end.shift(weeks))

    def __str__(self) -> str:
        return f"{self.start} to {self.end}"


Range = tuple[int, int]


def parse_range(text: str) -> Range | None:
    text = (text or "").strip()
    if not text:
        return None
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (nums[0], nums[1])


@dataclass
class Forecast:
    season: int
    onset_ltm: Window
    onset: Window
    early_dry_ltm: Range
    early_dry: Range
    late_dry_ltm: Range
    late_dry: Range
    cessation_ltm: Window
    cessation: Window
    season_length_ltm: Range | None = None
    season_length: Range | None = None
    rain_mm: dict[str, Range] = field(default_factory=dict)       # forecast totals by period: MJJ, JJA, JAS
    rain_ltm_mm: dict[str, Range] = field(default_factory=dict)
    source: str = ""
    note: str = ""

    # ---- categories, derived by comparing the forecast with the long-term mean ----

    @property
    def onset_category(self) -> str:
        return _window_category(self.onset, self.onset_ltm, ("early", "normal", "late"))

    @property
    def cessation_category(self) -> str:
        return _window_category(self.cessation, self.cessation_ltm, ("early", "normal", "late"))

    @property
    def early_dry_category(self) -> str:
        return _range_category(self.early_dry, self.early_dry_ltm, ("short", "normal", "long"))

    @property
    def late_dry_category(self) -> str:
        return _range_category(self.late_dry, self.late_dry_ltm, ("short", "normal", "long"))

    def rain_category(self, period: str) -> str | None:
        if period not in self.rain_mm or period not in self.rain_ltm_mm:
            return None
        return _range_category(self.rain_mm[period], self.rain_ltm_mm[period],
                               ("below normal", "near normal", "above normal"))

    def source_text(self) -> str:
        """The forecast written out the way GMet's tables read, for one Upper West segment.

        This is the model input for Stage 1. It keeps GMet's wording and includes the LTM
        values, so the model has to pick out the forecast values itself.
        """
        lines = [
            f"Seasonal forecast for Northern Ghana {self.season}, Upper West zone.",
            f"Onset dates. Normal (LTM): {self.onset_ltm.start} - {self.onset_ltm.end}. "
            f"Forecasted: {self.onset.start} - {self.onset.end}.",
            f"First (early) dry spell. LTM: {_fmt_range(self.early_dry_ltm)} days. "
            f"Forecast: {_fmt_range(self.early_dry)} days.",
            f"Second (late) dry spell. LTM: {_fmt_range(self.late_dry_ltm)} days. "
            f"Forecast: {_fmt_range(self.late_dry)} days.",
            f"Cessation dates. Normal: {self.cessation_ltm.start} - {self.cessation_ltm.end}. "
            f"Forecasted: {self.cessation.start} - {self.cessation.end}.",
        ]
        if self.season_length and self.season_length_ltm:
            lines.append(f"Length of season. LTM: {_fmt_range(self.season_length_ltm)} days. "
                         f"{self.season} forecast: {_fmt_range(self.season_length)} days.")
        for period in ("MJJ", "JJA", "JAS"):
            if period in self.rain_mm and period in self.rain_ltm_mm:
                lines.append(f"{period} total rainfall. LTM: {_fmt_range(self.rain_ltm_mm[period])} mm. "
                             f"Forecast: {_fmt_range(self.rain_mm[period])} mm.")
        return "\n".join(lines)


def _fmt_range(r: Range) -> str:
    return str(r[0]) if r[0] == r[1] else f"{r[0]}-{r[1]}"


def _window_category(fc: Window, ltm: Window, labels: tuple[str, str, str]) -> str:
    # Compare the middle of the windows; a shift of one week or more counts as early or late.
    shift = ((fc.start.index + fc.end.index) - (ltm.start.index + ltm.end.index)) / 2
    if shift <= -1:
        return labels[0]
    if shift >= 1:
        return labels[2]
    return labels[1]


def _range_category(fc: Range, ltm: Range, labels: tuple[str, str, str]) -> str:
    fc_mid, ltm_mid = sum(fc) / 2, sum(ltm) / 2
    tolerance = max(1.0, 0.1 * ltm_mid)
    if fc_mid < ltm_mid - tolerance:
        return labels[0]
    if fc_mid > ltm_mid + tolerance:
        return labels[2]
    return labels[1]


def load_curated(path: Path = CURATED_CSV) -> list[Forecast]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rain, rain_ltm = {}, {}
            for period in ("mjj", "jja", "jas"):
                fc, ltm = parse_range(row[f"{period}_rain_forecast_mm"]), parse_range(row[f"{period}_rain_ltm_mm"])
                if fc and ltm:
                    rain[period.upper()], rain_ltm[period.upper()] = fc, ltm
            out.append(Forecast(
                season=int(row["season"]),
                onset_ltm=Window.parse(row["onset_ltm"]),
                onset=Window.parse(row["onset_forecast"]),
                early_dry_ltm=parse_range(row["early_dry_spell_ltm_days"]),
                early_dry=parse_range(row["early_dry_spell_forecast_days"]),
                late_dry_ltm=parse_range(row["late_dry_spell_ltm_days"]),
                late_dry=parse_range(row["late_dry_spell_forecast_days"]),
                cessation_ltm=Window.parse(row["cessation_ltm"]),
                cessation=Window.parse(row["cessation_forecast"]),
                season_length_ltm=parse_range(row["season_length_ltm_days"]),
                season_length=parse_range(row["season_length_forecast_days"]),
                rain_mm=rain, rain_ltm_mm=rain_ltm,
                source=row["source_file"], note=row["verification_note"],
            ))
    return out
