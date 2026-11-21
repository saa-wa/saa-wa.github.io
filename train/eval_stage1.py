"""Evaluate the Stage 1 Advisory Generator on the held-out 2026 season.

    python train/eval_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct                       # zero-shot baseline
    python train/eval_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter models/stage1-qwen2.5-1.5b-instruct

Reports, per input style and overall: valid-JSON rate, mean NFC, guard pass rate (NFC = 1),
SMS length compliance and ROUGE-L of the message against the reference advisory.
Results go to results/stage1_<name>.json, with every generation for inspection.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rouge_score import rouge_scorer  # noqa: E402

from saawa import guard, llm  # noqa: E402
from saawa.advisory import SMS_LIMIT, Advisory  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_advisory"


def score(rows: list[dict], generations: list[str]) -> dict:
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    per_style = defaultdict(list)
    details = []
    for row, raw in zip(rows, generations):
        adv = llm.parse_advisory(raw)
        reference = Advisory.from_json(row["output"])
        text = adv.text_for_checking() if adv else raw
        src = {tuple(f) for f in row["source_facts"]}
        n = guard.nfc(src, guard.extract_facts(text))
        rec = {
            "valid_json": adv is not None,
            "nfc": n,
            "guard_pass": n == 1.0,
            "sms_ok": bool(adv) and len(adv.message_en) <= SMS_LIMIT,
            "rougeL": rouge.score(reference.message_en, adv.message_en if adv else raw)["rougeL"].fmeasure,
        }
        per_style[row["style"]].append(rec)
        per_style["all"].append(rec)
        details.append({"style": row["style"], "origin": row["origin"], **rec, "generation": raw})

    def agg(recs):
        return {"n": len(recs),
                "valid_json": statistics.mean(r["valid_json"] for r in recs),
                "mean_nfc": statistics.mean(r["nfc"] for r in recs),
                "guard_pass_rate": statistics.mean(r["guard_pass"] for r in recs),
                "sms_within_320": statistics.mean(r["sms_ok"] for r in recs),
                "rougeL": statistics.mean(r["rougeL"] for r in recs)}
    return {"summary": {k: agg(v) for k, v in per_style.items()}, "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=llm.DEFAULT_BASE)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(DATA / f"{args.split}.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    tok, model = llm.load(args.model, args.adapter)
    generations = []
    for i, row in enumerate(rows, 1):
        generations.append(llm.generate(tok, model, row["input"]))
        print(f"\r{i}/{len(rows)}", end="", flush=True)
    print()

    result = score(rows, generations)
    name = Path(args.adapter).name if args.adapter else "zeroshot-" + args.model.split("/")[-1].lower()
    result["config"] = vars(args)
    out = ROOT / "results" / f"stage1_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
