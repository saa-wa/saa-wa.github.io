# Model Card

## Overview

Saa Wa? has two model stages and a rule-based guard between them. No advisory is released without two human approvals.

| Component | Model | Status |
| --- | --- | --- |
| Stage 1 Advisory Generator | `google/gemma-3-1b-it` with QLoRA, as in the proposal. Prototype trained on `Qwen/Qwen2.5-1.5B-Instruct`, an open stand-in, because Gemma requires a Hugging Face licence login. | Prototype trained; results below |
| Rule-based baseline | `saawa/advisory.py` | Working; drafts the training targets |
| Fact-Consistency Guard | `saawa/guard.py` (rules, not learned) | Working; covered by tests |
| Stage 2 Dagaare Translator | `facebook/nllb-200-distilled-600M` with a new `dga_Latn` token initialised from Mooré (`mos_Latn`) | Code ready; waiting for approved Dagaare pairs |
| Dagaare template renderer | `translate.render_dagaare` | Working; waiting for approved template translations |

## Intended use

The system drafts planting advisories for Upper West smallholder farmers from GMet's published Northern Ghana seasonal forecast. Those drafts go to:
- MoFA extension officers, who approve the English text
- Dagaare reviewers, who approve the translation

It summarises and translates GMet's forecast only. It never makes forecasts of its own.

**Out of scope:**
- independent weather forecasting
- pests, fertiliser and market prices
- languages other than Dagaare (Waali, Sisaala and Birifor are not covered)
- releasing anything without human approval

## Stage 1 training

| Setting | Value |
| --- | --- |
| Method | QLoRA with a 4-bit NF4 base, LoRA rank 16 and alpha 32 on all linear layers, dropout 0.05 |
| Optimiser | AdamW, learning rate 2e-4, cosine schedule, 3 epochs, effective batch size 8 |
| Loss | Computed on the advisory JSON only (prompt tokens are masked) |
| Data | Forecast–Advisory corpus (see `DATA_CARD.md`): trained on 2024, validated on 2025, tested on 2026 |
| Hardware | One NVIDIA RTX 4070 Ti SUPER (16 GB) |

## Evaluation

The test set is the 2026 season: 38 examples in three input styles, none of them seen in training. The metrics are:

- **Valid JSON:** the output parses into the advisory schema.
- **Mean NFC:** numeric fact consistency against GMet's forecast. 1.0 means nothing was lost and nothing was invented.
- **Guard pass:** the share of examples with NFC exactly 1.0, the share that could go on to human review.
- **SMS within 320:** the share of messages that fit in two SMS.
- **ROUGE-L:** overlap of the SMS with the reference message.

The base model is Qwen2.5-1.5B-Instruct. Both rows use the same prompt, which lists the advisory schema. Full outputs are in `results/stage1_*.json`.

| Model | Valid JSON | Mean NFC | Guard pass | SMS within 320 | ROUGE-L |
| --- | --- | --- | --- | --- | --- |
| Zero-shot (no fine-tuning) | 66% | 0.36 | 21% | 5% | 0.20 |
| QLoRA fine-tuned | **100%** | **0.97** | **92%** | **100%** | **0.93** |

Results by input style for the fine-tuned model:
- **Table and narrative:** NFC 1.00, and all 26 examples pass the guard.
- **Raw PDF text:** NFC 0.90, and 9 of 12 pass the guard. In the other 3, the model misread an onset or cessation window from the jumbled columns. **The guard blocked all three**, so none could reach a reviewer as correct.
- **The real 2026 GMet forecast:** all three styles pass with NFC 1.00. Zero-shot passed only the narrative style.

**Training curve.** Validation loss on the unseen 2025 season was 0.030 after epoch 1, then 0.018 after epochs 2 and 3. Training loss fell from 0.81 to 0.012.

**What this does and does not show.**
- It shows that fine-tuning teaches the model the schema, the 320-character limit and exact copying of facts, and that it generalises to a season it never saw.
- It does not show agronomic quality. The targets are the rule-based drafts, so ROUGE-L mostly measures how closely the model copies their wording.
- The proposal's NFC target of 0.95 is met overall (0.97), but not yet on raw PDF text (0.90). In practice the extractor (`saawa/extract.py`) turns PDFs into the clean table style before the model sees them.

## Risks and limitations

- **Wrong planting advice is the main harm.** The guard blocks any advisory whose onset, dry-spell or cessation facts differ from GMet's. Two people must still approve every release.
- **The guard is only as good as its fact extraction.** It checks dates, weeks and day counts. It does not check the meaning of the advice text, which is why MoFA review is required.
- **The targets are drafts.** The model learns the baseline's wording until MoFA officers rewrite the targets.
- **Data is small.** There are three real seasons plus counterfactual variants. Performance on future GMet layouts must be re-checked each season.
- **Forecast uncertainty.** GMet's own verification shows low skill at Wa for some 2025 periods, so every advisory names GMet as its source and tells farmers to check with their extension officer.
- **Licences.**
  - NLLB-200 weights are CC BY-NC 4.0 (non-commercial).
  - Qwen2.5 is Apache 2.0.
  - Gemma is under the Gemma Terms of Use.
