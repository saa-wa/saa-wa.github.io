"""Stage 2: English-to-Dagaare translation with NLLB-200.

NLLB-200 has no Dagaare language code, so training adds a `dga_Latn` token whose embedding starts
as a copy of Mooré (`mos_Latn`), Dagaare's closest relative in the model. Until a fine-tuned model
exists under models/nllb-dagaare, translate() says so instead of guessing: nothing reaches a farmer
in Dagaare that a native speaker has not produced or approved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "facebook/nllb-200-distilled-600M"
FINETUNED_DIR = ROOT / "models" / "nllb-dagaare"
SRC_LANG, DGA_LANG, MOS_LANG = "eng_Latn", "dga_Latn", "mos_Latn"


@dataclass
class Translation:
    text: str | None
    status: str  # "translated", "baseline (Mooré, not Dagaare)" or "no Dagaare model yet"


def has_finetuned_model() -> bool:
    return (FINETUNED_DIR / "config.json").exists()


@lru_cache(maxsize=2)
def _load(model_dir: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tok, model


def sentences(text: str) -> list[str]:
    # Split only before a capital or "[", so "Saa Wa? 2026" stays whole.
    return [s for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z\[])", text.strip()) if s]


def _generate(model_dir: str, text: str, tgt_lang: str) -> str:
    import torch

    tok, model = _load(model_dir)
    tok.src_lang = SRC_LANG
    out = []
    for sent in sentences(text):
        batch = tok(sent, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids(tgt_lang),
                                 max_new_tokens=128, num_beams=4)
        out.append(tok.batch_decode(ids, skip_special_tokens=True)[0])
    return " ".join(out)


def translate(text: str) -> Translation:
    if not has_finetuned_model():
        return Translation(None, "no Dagaare model yet")
    return Translation(_generate(str(FINETUNED_DIR), text, DGA_LANG), "translated")


DRAFT_STATUS = "unverified draft: these Dagaare rows are not yet approved by a second speaker"


def render_dagaare(fc, include_drafts: bool = False) -> tuple[Translation, list[str]]:
    """Build the Dagaare SMS from translator-approved templates and glossary words.

    Returns the translation and the template or glossary entries still awaiting approval.
    With include_drafts, unapproved rows are used too and the result is marked as an unverified
    draft; untranslated glossary words then stay in English. A draft is never releasable.
    """
    from . import advisory, worksheet

    tpath, gpath = worksheet.DAGAARE_DIR / "message_templates.csv", worksheet.DAGAARE_DIR / "glossary.csv"
    templates, words = worksheet.approved(tpath, "key"), worksheet.approved(gpath, "term_en")
    parts = advisory.fit_parts(advisory.message_parts(fc))
    needed = {k for k, _ in parts} | {"week", "range"}
    week_words = {w for _, slots in parts for v in slots.values() if hasattr(v, "month")
                  for w in str(v).split(" week of ")}
    missing = sorted(f"template:{k}" for k in needed - templates.keys())
    missing += sorted(f"glossary:{w}" for w in week_words - words.keys())
    if not missing:
        return Translation(advisory.render_parts(parts, templates, words), "rendered from approved templates"), []
    if include_drafts:
        draft_templates = worksheet.drafted(tpath, "key")
        if needed <= draft_templates.keys():
            text = advisory.render_parts(parts, draft_templates, worksheet.drafted(gpath, "term_en"))
            return Translation(text, DRAFT_STATUS), missing
    return Translation(None, "Dagaare templates awaiting translation and approval"), missing


def zero_shot_baseline(text: str) -> Translation:
    """The proposal's Stage 2 baseline: untuned NLLB decoding through the Mooré tag."""
    return Translation(_generate(BASE_MODEL, text, MOS_LANG), "baseline (Mooré, not Dagaare)")


def add_dagaare_token(tokenizer, model) -> int:
    """Add dga_Latn to an NLLB tokenizer and model, initialised from mos_Latn. Returns its id."""
    if DGA_LANG not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [DGA_LANG]},
                                     replace_additional_special_tokens=False)
        model.resize_token_embeddings(len(tokenizer))
    dga_id, mos_id = tokenizer.convert_tokens_to_ids(DGA_LANG), tokenizer.convert_tokens_to_ids(MOS_LANG)
    emb = model.get_input_embeddings().weight
    emb.data[dga_id] = emb.data[mos_id].clone()
    return dga_id
