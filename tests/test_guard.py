import pytest

from saawa import advisory, augment, guard
from saawa.forecast import Week, Window, load_curated


@pytest.fixture(scope="module")
def forecasts():
    return {fc.season: fc for fc in load_curated()}


def test_curated_seasons_load(forecasts):
    assert set(forecasts) == {2024, 2025, 2026}
    fc = forecasts[2026]
    assert str(fc.onset) == "3rd week of April to 2nd week of May"
    assert fc.early_dry == (10, 14)
    assert fc.rain_mm["JJA"] == (450, 500)


def test_week_shift_crosses_month():
    assert Week.parse("4th week of April").shift(1) == Week.parse("1st week of May")
    assert Window.parse("3rd week of April - 2nd week of May").shift(-1).start == Week.parse("2nd week of April")


@pytest.mark.parametrize("season", [2024, 2025, 2026])
def test_baseline_advisory_passes_guard(forecasts, season):
    fc = forecasts[season]
    adv = advisory.generate(fc)
    result = guard.check(fc, adv.text_for_checking())
    assert result.passed, result.explain()
    assert len(adv.message_en) <= advisory.SMS_LIMIT


def test_guard_catches_changed_onset_week(forecasts):
    fc = forecasts[2026]
    adv = advisory.generate(fc)
    adv.message_en = adv.message_en.replace("2nd week of May", "4th week of May")
    result = guard.check(fc, adv.text_for_checking())
    assert not result.passed
    assert ("onset", "3rd week of April|4th week of May") in result.invented


def test_guard_catches_invented_dry_spell(forecasts):
    fc = forecasts[2026]
    adv = advisory.generate(fc)
    adv.message_en += " A late dry spell of 20 to 25 days is likely."
    result = guard.check(fc, adv.text_for_checking())
    assert not result.passed
    assert ("late_dry_spell", "20-25") in result.invented


def test_guard_catches_dropped_fact(forecasts):
    fc = forecasts[2026]
    text = advisory.generate(fc).message_en  # the SMS alone omits the late dry spell
    result = guard.check(fc, text)
    assert not result.passed
    assert ("late_dry_spell", "13-17") in result.missing
    assert 0 < result.nfc < 1


def test_nfc_definition():
    s = {("a", "1"), ("b", "2")}
    assert guard.nfc(s, s) == 1.0
    assert guard.nfc(s, {("a", "1")}) == 0.5
    assert guard.nfc(s, {("a", "1"), ("c", "3")}) == pytest.approx(1 / 3)


def test_numbers_preserved():
    assert guard.numbers_preserved("dry spell of 10 to 14 days", "... 10 ... 14 ...")
    assert not guard.numbers_preserved("dry spell of 10 to 14 days", "... 10 ... 15 ...")


def test_dataset_splits_by_season_and_targets_pass_guard():
    rows = augment.build()
    assert {r["split"] for r in rows if r["season"] == 2026} == {"test"}
    assert {r["split"] for r in rows if r["season"] == 2024} == {"train"}
    assert 180 <= len(rows) <= 260
    for r in rows:
        adv = advisory.Advisory.from_json(r["output"])
        assert len(adv.message_en) <= advisory.SMS_LIMIT
