"""The template mechanism, tested with placeholder strings (not Dagaare) in a temporary folder."""

import csv

from saawa import advisory, guard, translate, worksheet
from saawa.forecast import load_curated


def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def test_render_dagaare_needs_approval_then_renders(tmp_path, monkeypatch):
    monkeypatch.setattr(worksheet, "DAGAARE_DIR", tmp_path)
    fc = [f for f in load_curated() if f.season == 2026][0]

    _write(tmp_path / "message_templates.csv", worksheet.TEMPLATE_FIELDS, [])
    _write(tmp_path / "glossary.csv", worksheet.GLOSSARY_FIELDS, [])
    result, missing = translate.render_dagaare(fc)
    assert result.text is None and "template:dry" in missing and "glossary:April" in missing

    templates = [{"key": k, "english": v, "dagaare": "X:" + v, "status": "approved"}
                 for k, v in advisory.TEMPLATES_EN.items()]
    words = [{"term_en": w, "dagaare": f"<{w}>", "status": "approved"}
             for w in ("1st", "2nd", "3rd", "4th", "April", "May", "October")]
    _write(tmp_path / "message_templates.csv", worksheet.TEMPLATE_FIELDS, templates)
    _write(tmp_path / "glossary.csv", worksheet.GLOSSARY_FIELDS, words)

    result, missing = translate.render_dagaare(fc)
    assert missing == []
    assert "<3rd> week of <April>" in result.text and "10 to 14" in result.text
    assert guard.numbers_preserved(advisory.build_message(fc), result.text)


def test_unapproved_rows_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(worksheet, "DAGAARE_DIR", tmp_path)
    _write(tmp_path / "glossary.csv", worksheet.GLOSSARY_FIELDS,
           [{"term_en": "May", "dagaare": "draft", "status": "to review"}])
    assert worksheet.approved(tmp_path / "glossary.csv", "term_en") == {}


def test_draft_rows_preview_but_never_release(tmp_path, monkeypatch):
    from saawa import pipeline

    monkeypatch.setattr(worksheet, "DAGAARE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "REVIEW_LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(translate, "has_finetuned_model", lambda: False)
    fc = [f for f in load_curated() if f.season == 2026][0]
    templates = [{"key": k, "english": v, "dagaare": "D:" + v, "status": "drafted"}
                 for k, v in advisory.TEMPLATES_EN.items()]
    _write(tmp_path / "message_templates.csv", worksheet.TEMPLATE_FIELDS, templates)
    _write(tmp_path / "glossary.csv", worksheet.GLOSSARY_FIELDS, [])

    result = pipeline.run(fc, "rule")
    assert result.is_draft and result.dagaare.text.startswith("D:[Saa Wa? 2026]")
    pipeline.record_review(result, "mofa", "A", "approve")
    pipeline.record_review(result, "dagaare", "B", "approve")
    assert not result.status.startswith("Released")
    assert "approval of the Dagaare templates" in result.status
