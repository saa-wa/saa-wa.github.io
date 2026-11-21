# Saa Wa?

Saa Wa? is a Dagaare-first rainfall onset advisory system for smallholder farmers in Ghana's Upper West Region. It is an AIMS AI Research Foundations capstone (Ghana Cohort) by Edina Galaa and Dr. Issahaku Fadilul-lah Yassaanah of UBIDS, Wa.

**Live demo:** https://issakafadil.github.io/saa-wa/ (static build: pre-computed advisories for the 2024–2026 GMet seasons; run `app.py` for PDF upload and live AI generation).

The Ghana Meteorological Agency (GMet) publishes its Northern Ghana seasonal forecast as a long technical PDF in English. Saa Wa? turns the Upper West part of that forecast into a short planting advisory. The advisory is first checked against the forecast. It is then rendered in Dagaare and released only after two people approve it:
- a MoFA extension officer, who approves the English text
- a Dagaare reviewer, who approves the translation

```
GMet PDF ─► extractor ─► Stage 1 advisory ─► Fact-Consistency Guard ─► Stage 2 Dagaare ─► human approval ─► SMS / radio
            (extract.py)  (advisory.py or       (guard.py, NFC = 1)      (approved templates   (pipeline.py)
                           fine-tuned LLM)                                or fine-tuned NLLB)
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu124   # skip for CPU-only
.venv/Scripts/pip install -r requirements.txt

.venv/Scripts/python -m pytest -q          # 17 tests
.venv/Scripts/python app.py                # dashboard at http://127.0.0.1:7860
```

## Rebuild the data

```bash
python -m saawa.augment      # Forecast–Advisory corpus → data/processed/forecast_advisory/
python -m saawa.worksheet    # Dagaare worksheets → data/dagaare/ (keeps existing translations)
```

## Train and evaluate

```bash
# Stage 1. Gemma 3 1B is gated: run `huggingface-cli login` and accept the licence first.
python train/train_stage1.py                                       # google/gemma-3-1b-it (proposal)
python train/train_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct    # open stand-in
python train/eval_stage1.py --model <base>                          # zero-shot baseline
python train/eval_stage1.py --model <base> --adapter models/stage1-<name>

# Stage 2. This needs at least 50 approved rows in data/dagaare/parallel_worksheet.csv.
python train/train_stage2.py
```

Results are written to `results/`. If model downloads stall with "CAS Client Error", set `HF_HUB_DISABLE_XET=1` before running. A 16 GB GPU is enough for both stages, and so is a free Colab T4.

## Static demo (GitHub Pages)

```bash
python build_static.py --repo https://github.com/Issakafadil/saa-wa   # regenerates site/ with the real pipeline
```

Commit `site/`, then publish it to the `gh-pages` branch that GitHub Pages serves:

```bash
git subtree push --prefix site origin gh-pages
```

## Repository map

| Path | What it holds |
| --- | --- |
| `saawa/forecast.py` | The Forecast structure, week arithmetic, categories compared with the long-term mean, and the curated loader |
| `saawa/extract.py` | Reads Upper West values from a GMet PDF using word positions, and refuses any row it cannot read unambiguously |
| `saawa/advisory.py` | Advisory JSON schema, rule-based baseline, and SMS templates (320 characters at most) |
| `saawa/guard.py` | Fact extraction, the NFC score and the Fact-Consistency Guard |
| `saawa/augment.py` | Counterfactual augmentation and splits by season |
| `saawa/translate.py` | Dagaare from approved templates, NLLB-200 with a `dga_Latn` token initialised from Mooré, and the zero-shot baseline |
| `saawa/worksheet.py` | Translator worksheets (templates, glossary, parallel sentences) |
| `saawa/llm.py` | Stage 1 prompt format, model loading and generation |
| `saawa/pipeline.py` | The end-to-end run, release rules and review log |
| `app.py`, `web/index.html` | Review dashboard (FastAPI backend and single-page frontend) |
| `train/` | Stage 1 QLoRA training and evaluation, and Stage 2 NLLB training |
| `docs/` | Data card and model card |
| `build_static.py`, `site/` | Static demo for GitHub Pages |
| `models/stage1-qwen2.5-1.5b-instruct/` | Fine-tuned Stage 1 LoRA adapter (base: Qwen/Qwen2.5-1.5B-Instruct) |

## What is real and what is a draft

- **Real:** GMet forecasts for 2024–2026 (`data/raw/gmet/`); the hand-checked Upper West values, verified by the extractor and the tests.
- **Draft:** advisory targets and crop advice. They follow GMet's own recommendations but must be reviewed by MoFA officers.
- **Not yet written:** all Dagaare text. The worksheets are ready for native speakers. Saa Wa? never invents Dagaare, and the app reports what is still missing.
