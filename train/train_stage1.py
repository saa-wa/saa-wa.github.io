"""Fine-tune the Stage 1 Advisory Generator with QLoRA.

    python train/train_stage1.py                                   # Gemma 3 1B (needs a Hugging Face login)
    python train/train_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct  # open stand-in, no login needed

Settings follow the proposal: 4-bit NF4 base weights, LoRA rank 16, alpha 32, adapters on all
linear projections. Loss is computed on the advisory JSON only, not on the prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,  # noqa: E402
                          Trainer, TrainingArguments)

from saawa.llm import DEFAULT_BASE, chat  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_advisory"


def read(split: str) -> list[dict]:
    return [json.loads(line) for line in open(DATA / f"{split}.jsonl", encoding="utf-8")]


def encode(tok, row: dict, max_len: int) -> dict:
    prompt = tok.apply_chat_template(chat(tok, row["input"]), add_generation_prompt=True,
                                     tokenize=True, return_dict=True)["input_ids"]
    full = tok.apply_chat_template(chat(tok, row["input"], row["output"]),
                                   tokenize=True, return_dict=True)["input_ids"]
    full = full[:max_len]
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels[:len(full)]}


def collate(tok):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {k: [] for k in ("input_ids", "attention_mask", "labels")}
        for b in batch:
            gap = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad] * gap)
            out["attention_mask"].append(b["attention_mask"] + [0] * gap)
            out["labels"].append(b["labels"] + [-100] * gap)
        return {k: torch.tensor(v) for k, v in out.items()}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    name = args.model.split("/")[-1].lower()
    out_dir = ROOT / "models" / f"stage1-{name}"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train = Dataset.from_list([encode(tok, r, args.max_len) for r in read("train")])
    val = Dataset.from_list([encode(tok, r, args.max_len) for r in read("validation")])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=2, learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_steps=3, logging_steps=5, eval_strategy="epoch", save_strategy="epoch",
            save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
            bf16=True, gradient_checkpointing=True, report_to="none", seed=args.seed),
        train_dataset=train, eval_dataset=val, data_collator=collate(tok),
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    history = [h for h in trainer.state.log_history if "eval_loss" in h or "loss" in h]
    (out_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    print(f"saved adapter to {out_dir}")


if __name__ == "__main__":
    main()
