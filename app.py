"""Saa Wa? review dashboard for MoFA extension officers and Dagaare reviewers.

    .venv/Scripts/python app.py        then open http://127.0.0.1:7860

A small FastAPI backend over the pipeline, serving a single-page dashboard from web/index.html.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from saawa import extract, pipeline
from saawa.forecast import Forecast, load_curated

ROOT = Path(__file__).resolve().parent
SEASONS = {fc.season: fc for fc in load_curated()}
CURRENT: dict[str, pipeline.Result] = {}

app = FastAPI(title="Saa Wa?")


def _range(r) -> str:
    return "" if not r else (str(r[0]) if r[0] == r[1] else f"{r[0]} to {r[1]}")


def forecast_json(fc: Forecast) -> dict:
    rows = [
        {"name": "Rainfall onset", "normal": str(fc.onset_ltm), "forecast": str(fc.onset), "category": fc.onset_category},
        {"name": "Early dry spell", "normal": f"{_range(fc.early_dry_ltm)} days", "forecast": f"{_range(fc.early_dry)} days",
         "category": fc.early_dry_category},
        {"name": "Late dry spell", "normal": f"{_range(fc.late_dry_ltm)} days", "forecast": f"{_range(fc.late_dry)} days",
         "category": fc.late_dry_category},
        {"name": "End of rains", "normal": str(fc.cessation_ltm), "forecast": str(fc.cessation),
         "category": fc.cessation_category},
    ]
    if fc.season_length:
        rows.append({"name": "Season length", "normal": f"{_range(fc.season_length_ltm)} days",
                     "forecast": f"{_range(fc.season_length)} days", "category": ""})
    return {"season": fc.season, "rows": rows, "source": fc.source, "note": fc.note}


def result_json(r: pipeline.Result) -> dict:
    approvals = pipeline.approvals_for(r.key)
    return {
        "key": r.key,
        "generator": r.generator,
        "forecast": forecast_json(r.forecast),
        "advisory": asdict(r.advisory) if r.advisory else None,
        "raw_generation": r.raw_generation,
        "guard": {"passed": r.guard.passed, "nfc": r.guard.nfc,
                  "missing": [f"{k}: {v}" for k, v in r.guard.missing],
                  "invented": [f"{k}: {v}" for k, v in r.guard.invented]},
        "dagaare": {"text": r.dagaare.text if r.dagaare else None,
                    "status": r.dagaare.status if r.dagaare else "Not translated: the English advisory did not pass the guard.",
                    "missing": r.dagaare_missing, "numbers_ok": r.numbers_ok, "draft": r.is_draft},
        "approvals": sorted(approvals),
        "status": r.status,
        "released": r.status.startswith("Released"),
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/seasons")
def seasons():
    has_model = any((ROOT / "models").glob("stage1-*/adapter_config.json"))
    return {"seasons": sorted(SEASONS, reverse=True), "has_model": has_model}


@app.post("/api/generate")
def generate(season: int | None = Form(None), generator: str = Form("rule"), pdf: UploadFile | None = File(None)):
    try:
        if pdf is not None and pdf.filename:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                shutil.copyfileobj(pdf.file, tmp)
            fc = extract.from_pdf(Path(tmp.name))
            fc.source = pdf.filename
        elif season in SEASONS:
            fc = SEASONS[season]
        else:
            raise HTTPException(400, "Choose a season or upload a GMet PDF.")
        result = pipeline.run(fc, "model" if generator == "model" else "rule")
    except HTTPException:
        raise
    except Exception as e:  # shown to the reviewer
        raise HTTPException(422, str(e))
    CURRENT["result"] = result
    return result_json(result)


@app.post("/api/review")
def review(role: str = Form(...), reviewer: str = Form(...), decision: str = Form(...), note: str = Form("")):
    result = CURRENT.get("result")
    if result is None:
        raise HTTPException(400, "Generate an advisory first.")
    if role not in ("mofa", "dagaare") or decision not in ("approve", "reject") or not reviewer.strip():
        raise HTTPException(400, "Give a role, a reviewer name and a decision.")
    pipeline.record_review(result, role, reviewer.strip(), decision, note)
    return result_json(result)


@app.get("/api/history")
def history():
    if not pipeline.REVIEW_LOG.exists():
        return []
    entries = [json.loads(line) for line in open(pipeline.REVIEW_LOG, encoding="utf-8")]
    return [{k: e[k] for k in ("time", "season", "role", "reviewer", "decision", "note")} for e in entries[-20:]][::-1]


@app.get("/api/metrics")
def metrics():
    out = {}
    for name, label in (("zeroshot-qwen2.5-1.5b-instruct", "Zero-shot"), ("stage1-qwen2.5-1.5b-instruct", "Fine-tuned")):
        path = ROOT / "results" / f"stage1_{name}.json"
        if path.exists():
            out[label] = json.loads(path.read_text(encoding="utf-8"))["summary"]["all"]
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
