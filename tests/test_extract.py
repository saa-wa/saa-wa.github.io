"""The PDF extractor must reproduce the hand-checked values for every GMet forecast we hold."""

from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

from saawa import extract  # noqa: E402
from saawa.forecast import load_curated  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "gmet"
FIELDS = ("onset", "onset_ltm", "early_dry", "late_dry", "cessation", "cessation_ltm",
          "season_length", "season_length_ltm")


@pytest.mark.parametrize("pdf", ["gmet_2024_mjj_jja_jas_forecast.pdf", "gmet_2025_northern_forecast.pdf",
                                 "gmet_2026_northern_forecast.pdf"])
def test_extracted_matches_curated(pdf):
    got = extract.from_pdf(RAW / pdf)
    want = {fc.season: fc for fc in load_curated()}[got.season]
    for f in FIELDS:
        assert getattr(got, f) == getattr(want, f), f


def test_refuses_document_without_tables():
    with pytest.raises(extract.ExtractionError):
        extract.from_pdf(RAW / "gmet_2026_aso_update.pdf")
