# Data Card

Saa Wa? uses two datasets. The first, the Forecast–Advisory corpus, exists in draft. The second, the Dagaare Agro-Weather Parallel Corpus, has worksheets ready, but its Dagaare text has not been written yet.

## 1. Forecast–Advisory corpus

| Field | Detail |
| --- | --- |
| Purpose | Train and test the Stage 1 Advisory Generator, which turns a GMet forecast into a structured English advisory |
| Location | `data/processed/forecast_advisory/{train,validation,test}.jsonl`, built by `python -m saawa.augment` |
| Real source | Three GMet Northern Ghana seasonal forecasts for 2024, 2025 and 2026 (`data/raw/gmet/`), all public documents from meteo.gov.gh |
| Hand-checked record | `data/curated/upper_west_forecasts.csv` holds the Upper West values for each season. Every value was read from the PDF and cross-checked by the position-based extractor (`saawa/extract.py`; see `tests/test_extract.py`). Values that cannot be read unambiguously are left blank. |
| Size | 209 examples: 9 real (3 seasons × 3 input styles) and 200 counterfactual |
| Splits | By season, so the model is always tested on a season it has not seen. 2024 is used for training (133), 2025 for validation (38) and 2026 for testing (38). |
| Input styles | `table` (GMet table wording), `narrative` (GMet summary wording) and `raw_pdf` (the PDF text layer, with LTM and forecast columns run together) |
| Augmentation | Onset and cessation shifted by whole weeks, and dry-spell ranges moved and widened (seeded, see `variant()` in `saawa/augment.py`) |
| Targets | Advisory JSON from the rule-based generator. **These are drafts:** MoFA extension officers in Wa are to review and correct every target before the dataset is final. |
| Each row holds | `split`, `origin` (real or counterfactual), `style`, `season`, `instruction`, `input`, `output` (target JSON), `source_facts` (the facts the guard checks) and `target_status` |
| Personal data | None |
| Licence | CC BY 4.0 for the corpus. The GMet source documents are public and are credited to GMet. |

**Known limits.**
- There are only three real seasons. The counterfactual variants are more regular than real forecasts.
- Rainfall totals (mm) are missing for some seasons because the PDF tables are misaligned.
- The targets use one fixed writing style, so a model trained on them will copy that style.

## 2. Dagaare Agro-Weather Parallel Corpus

| Field | Detail |
| --- | --- |
| Purpose | Render advisories in Dagaare. Approved sentence templates are used now, and NLLB-200 will be fine-tuned once enough approved pairs exist. |
| Worksheets | `data/dagaare/message_templates.csv` (11 sentence templates with `{slots}`), `glossary.csv` (34 terms, including week ordinals and month names) and `parallel_worksheet.csv` (23 free sentences) |
| Status | **No Dagaare has been written yet.** The Dagaare columns are blank and must be filled by native speakers. Nothing in this repository is machine-generated or invented Dagaare. |
| Workflow | A translator fills in `dagaare` and their own name, and a second speaker checks the entry and sets `status` to `approved`. Only approved rows are ever used (`saawa/worksheet.approved`). |
| Orthography | Standard Dagaare orthography, as used in Ghanaian school materials |
| Target size | All templates and glossary terms first, so every advisory can be rendered. Then about 3,000 sentence pairs for NLLB, as set out in the proposal. |
| Consent and credit | Translators are paid and credited by name only with their consent |
| Licence | CC BY 4.0 |

**Known limits.**
- The corpus covers one main dialect.
- Northern Dagaare, Waali, Sisaala and Birifor speakers are not yet served.
