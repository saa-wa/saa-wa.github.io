"""Read the Upper West rows of a GMet Northern Ghana seasonal forecast PDF into a Forecast.

GMet's tables come out of the PDF text layer with the LTM and forecast columns run together and
wrapped across two lines. This parser handles the 2025 and 2026 layouts. When a row cannot be read
unambiguously (as with the 2024 onset table) it raises ExtractionError rather than guess, and the
curated CSV, checked by hand against the PDF, is used instead.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .forecast import Forecast, Window

# Dashes come through the text layer as "-", "–" or a replacement character, so any short
# non-digit run between two numbers counts as a range separator.
DRY_ROW = re.compile(r"Upper West\s+(\d{1,2})\s+(\d{1,2})\s*[^\d\s]{1,3}\s*(\d{1,2})\b")
LENGTH_ROW = re.compile(r"Upper West\s+(\d{3})\s*[^\d\s]{1,3}\s*(\d{3})\s+(\d{3})\s*[^\d\s]{1,3}\s*(\d{3})")


class ExtractionError(ValueError):
    pass


def pdf_text(path: Path) -> str:
    """Layout-preserving text: pdftotext when installed, pdfplumber otherwise."""
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                              check=True).stdout.decode("utf-8", errors="replace")
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text(layout=True) or "" for page in pdf.pages)


ZONES = ("Transition", "North", "Upper West", "Upper East")


def _week_columns(path: Path, heading: str) -> tuple[Window, Window]:
    """The Upper West (LTM, forecast) windows of a date table, read from word positions.

    GMet's date cells wrap over two lines and the columns interleave in the text layer, so each
    word is assigned to the zone label nearest it vertically, and to the LTM or forecast column
    by its horizontal position relative to the forecast column's header.
    """
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if not re.search(heading, text, flags=re.IGNORECASE) or "Upper West" not in text or "...." in text:
                continue
            words = page.extract_words()
            labels = {}
            for i, w in enumerate(words):
                name = w["text"] if w["text"] in ("Transition", "North") else (
                    f"Upper {words[i + 1]['text']}" if w["text"] == "Upper" and i + 1 < len(words) else None)
                if name in ZONES and name not in labels:
                    labels[name] = w
            if "Upper West" not in labels:
                continue
            first_label_top = min(w["top"] for w in labels.values())
            label_right = max(w["x1"] for w in labels.values()) + 4
            row = [w for w in words if w["x0"] > label_right and w["top"] > first_label_top - 40
                   and min(labels, key=lambda z: abs(labels[z]["top"] - w["top"])) == "Upper West"]
            if not row:
                raise ExtractionError(f"Upper West row of {heading!r} is empty")
            # Group words into visual lines (tops within 6 points; wrapped lines sit ~11 apart),
            # then order each line left to right.
            lines: list[list] = []
            for w in sorted(row, key=lambda w: w["top"]):
                if lines and w["top"] - lines[-1][0]["top"] <= 6:
                    lines[-1].append(w)
                else:
                    lines.append([w])
            lines = [sorted(line, key=lambda w: w["x0"]) for line in lines]
            row = [w for line in lines for w in line]
            # The two columns are separated by the widest horizontal gap on the row's fullest line.
            top_line = max(lines, key=len)
            gap, split_x = max((b["x0"] - a["x1"], (a["x1"] + b["x0"]) / 2)
                               for a, b in zip(top_line, top_line[1:]))
            cols = [" ".join(w["text"] for w in row if (w["x0"] >= split_x) == right) for right in (False, True)]
            try:
                return Window.parse(cols[0]), Window.parse(cols[1])
            except ValueError as e:
                raise ExtractionError(f"Upper West row of {heading!r} is ambiguous: {cols}") from e
    raise ExtractionError(f"table not found: {heading}")


def from_pdf(path: Path) -> Forecast:
    path = Path(path)
    text = pdf_text(path)
    season = re.search(r"\b(20\d\d)\s+Season\b", text, flags=re.IGNORECASE)
    if not season:
        raise ExtractionError("season year not found")
    onset_ltm, onset = _week_columns(path, r"Table \d+[:.]?\s+Onset Dates")
    cess_ltm, cess = _week_columns(path, r"Table \d+[:.]?\s+Cessation Dates")
    dry = DRY_ROW.findall(text)
    if len(dry) < 2:
        raise ExtractionError("dry spell rows not found")
    (e_ltm, e_lo, e_hi), (l_ltm, l_lo, l_hi) = dry[0], dry[1]
    length_tables = list(re.finditer(r"^\s*Table \d+[^\n]{0,60}Length", text, flags=re.IGNORECASE | re.MULTILINE))
    length = LENGTH_ROW.search(text, length_tables[-1].end()) if length_tables else None
    return Forecast(
        season=int(season.group(1)),
        onset_ltm=onset_ltm, onset=onset,
        early_dry_ltm=(int(e_ltm), int(e_ltm)), early_dry=(int(e_lo), int(e_hi)),
        late_dry_ltm=(int(l_ltm), int(l_ltm)), late_dry=(int(l_lo), int(l_hi)),
        cessation_ltm=cess_ltm, cessation=cess,
        season_length_ltm=(int(length.group(1)), int(length.group(2))) if length else None,
        season_length=(int(length.group(3)), int(length.group(4))) if length else None,
        source=f"extracted from {path.name}",
    )
