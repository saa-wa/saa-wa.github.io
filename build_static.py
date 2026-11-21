"""Build the static demo for GitHub Pages into site/.

    .venv/Scripts/python build_static.py --repo https://github.com/<owner>/<repo>

Runs the real pipeline for every curated GMet season with both advisory writers (the fine-tuned
model needs models/stage1-* and a GPU), stores the results in site/demo.json, and copies the
dashboard with a flag that makes it read that file instead of calling the Python API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="", help="GitHub repository URL for the footer links")
    ap.add_argument("--no-model", action="store_true", help="skip the fine-tuned model (rule-based only)")
    args = ap.parse_args()

    import app  # the dashboard backend; reuses its JSON shape exactly
    from saawa import pipeline

    generators = ["rule"] if args.no_model else ["rule", "model"]
    results = {}
    for season, fc in sorted(app.SEASONS.items()):
        for gen in generators:
            r = pipeline.run(fc, gen)
            data = app.result_json(r)
            data["approvals"], data["released"] = [], False  # every visitor starts unreviewed
            results[f"{season}-{gen}"] = data
            print(f"{season} {gen}: guard {'pass' if r.guard.passed else 'FAIL'} (NFC {r.guard.nfc:.2f})")

    SITE.mkdir(exist_ok=True)
    demo = {"seasons": sorted(app.SEASONS, reverse=True), "results": results, "metrics": app.metrics()}
    (SITE / "demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=1), encoding="utf-8")

    page = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    flags = f'<script>window.SAAWA_STATIC = true; window.SAAWA_REPO = {json.dumps(args.repo)};</script>\n'
    page = page.replace("<script>\nconst $ =", flags + "<script>\nconst $ =", 1)
    (SITE / "index.html").write_text(page, encoding="utf-8")
    (SITE / ".nojekyll").write_text("")
    print(f"wrote {SITE}")


if __name__ == "__main__":
    main()
