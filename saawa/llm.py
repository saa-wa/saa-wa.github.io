"""Stage 1 model: prompt format, loading (base model plus optional LoRA adapter) and generation.

The default base is google/gemma-3-1b-it as in the proposal. Gemma is gated on Hugging Face, so
any open instruction model (for example Qwen/Qwen2.5-1.5B-Instruct) can stand in with --model.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .advisory import Advisory
from .augment import INSTRUCTION

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "google/gemma-3-1b-it"
MODELS_DIR = ROOT / "models"


def user_prompt(forecast_text: str) -> str:
    return f"{INSTRUCTION}\n\nGMet forecast:\n{forecast_text}"


def chat(tokenizer, forecast_text: str, answer: str | None = None):
    messages = [{"role": "user", "content": user_prompt(forecast_text)}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def load(base: str = DEFAULT_BASE, adapter: str | None = None, four_bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(adapter or base)
    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@lru_cache(maxsize=1)
def load_finetuned():
    """The newest adapter under models/stage1-*, with its base model name from adapter_config.json."""
    adapters = sorted(MODELS_DIR.glob("stage1-*/adapter_config.json"), key=lambda p: p.stat().st_mtime)
    if not adapters:
        return None
    adapter_dir = adapters[-1].parent
    base = json.loads(adapters[-1].read_text())["base_model_name_or_path"]
    return (*load(base, str(adapter_dir)), adapter_dir.name)


def generate(tok, model, forecast_text: str, max_new_tokens: int = 640) -> str:
    import torch

    inputs = tok.apply_chat_template(chat(tok, forecast_text), add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_advisory(raw: str) -> Advisory | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return Advisory.from_json(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
