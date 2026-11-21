"""Fine-tune the Stage 2 Dagaare Translator (NLLB-200 distilled 600M).

    python train/train_stage2.py

Uses only rows of data/dagaare/parallel_worksheet.csv that a reviewer has marked "approved".
Adds the dga_Latn language token initialised from Mooré (mos_Latn), trains with AdamW at 1e-4,
and reports chrF++ and BLEU on a held-out 10% against the zero-shot Mooré-tag baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sacrebleu  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from saawa import translate, worksheet  # noqa: E402

MIN_PAIRS = 50


def batches(pairs, size):
    for i in range(0, len(pairs), size):
        yield pairs[i:i + size]


def encode(tok, src_texts, tgt_texts, dga_id, device):
    tok.src_lang = translate.SRC_LANG
    enc = tok(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    tgt = [[dga_id] + tok(t, add_special_tokens=False).input_ids[:126] + [tok.eos_token_id] for t in tgt_texts]
    n = max(map(len, tgt))
    labels = torch.tensor([t + [-100] * (n - len(t)) for t in tgt], device=device)
    return enc, labels


def decode(tok, model, texts, lang_id):
    tok.src_lang = translate.SRC_LANG
    out = []
    for chunk in batches(texts, 16):
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, forced_bos_token_id=lang_id, max_new_tokens=128, num_beams=4)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def metrics(hyp, ref):
    return {"chrF++": sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score,
            "BLEU": sacrebleu.corpus_bleu(hyp, [ref]).score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    approved = worksheet.approved(worksheet.DAGAARE_DIR / "parallel_worksheet.csv", "english")
    pairs = sorted(approved.items())
    if len(pairs) < MIN_PAIRS:
        sys.exit(f"Only {len(pairs)} approved English-Dagaare pairs; need at least {MIN_PAIRS}. "
                 f"Fill and approve rows in {worksheet.DAGAARE_DIR / 'parallel_worksheet.csv'} first.")
    random.Random(13).shuffle(pairs)
    n_test = n_val = max(5, len(pairs) // 10)
    test, val, train = pairs[:n_test], pairs[n_test:n_test + n_val], pairs[n_test + n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(translate.BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(translate.BASE_MODEL).to(device)

    mos_id = tok.convert_tokens_to_ids(translate.MOS_LANG)
    baseline = metrics(decode(tok, model, [e for e, _ in test], mos_id), [d for _, d in test])

    dga_id = translate.add_dagaare_token(tok, model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, out_dir = -1.0, translate.FINETUNED_DIR
    for epoch in range(args.epochs):
        model.train()
        random.Random(epoch).shuffle(train)
        for chunk in batches(train, args.batch):
            enc, labels = encode(tok, [e for e, _ in chunk], [d for _, d in chunk], dga_id, device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        val_chrf = metrics(decode(tok, model, [e for e, _ in val], dga_id), [d for _, d in val])["chrF++"]
        print(f"epoch {epoch + 1}: loss {loss.item():.3f}, validation chrF++ {val_chrf:.1f}")
        if val_chrf > best:
            best = val_chrf
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)

    model = AutoModelForSeq2SeqLM.from_pretrained(out_dir).to(device)
    finetuned = metrics(decode(tok, model, [e for e, _ in test], dga_id), [d for _, d in test])
    result = {"pairs": {"train": len(train), "validation": len(val), "test": len(test)},
              "zero_shot_mos_Latn": baseline, "finetuned_dga_Latn": finetuned}
    out = ROOT / "results" / "stage2_nllb.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
