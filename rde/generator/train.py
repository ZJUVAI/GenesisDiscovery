"""Plan 11: LoRA fine-tuning and sampling for the feature-set generator (Qwen3-0.6B)."""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .data import Example

BASE = "Qwen/Qwen3-0.6B"
MAX_LEN = 256  # plan 11
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(path: str, device: str):
    """Tokenizer and base model in bf16 on one device (no LoRA yet)."""
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    return tok, model


def with_lora(model):
    """Plan 11: LoRA r=16, alpha=32, dropout 0.05 on the seven projection matrices."""
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def encode(tok, ex: Example) -> Tuple[List[int], List[int]]:
    """Input ids and labels: prompt tokens masked with -100, completion scored."""
    p = tok(ex.prompt, add_special_tokens=False)["input_ids"]
    c = tok(ex.completion, add_special_tokens=False)["input_ids"]
    ids = (p + c)[:MAX_LEN]
    labels = ([-100] * len(p) + c)[:MAX_LEN]
    return ids, labels


def batches(items: Sequence[Tuple[List[int], List[int]]], size: int, pad_id: int, rng: random.Random):
    """Shuffled right-padded batches; padding positions carry label -100."""
    order = list(range(len(items)))
    rng.shuffle(order)
    for s in range(0, len(order), size):
        chunk = [items[i] for i in order[s:s + size]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), width), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, (i, l) in enumerate(chunk):
            ids[r, :len(i)] = torch.tensor(i)
            lab[r, :len(l)] = torch.tensor(l)
            att[r, :len(i)] = 1
        yield ids, lab, att


@torch.no_grad()
def val_loss(model, items, pad_id: int, device: str, size: int = 64) -> float:
    """Mean token loss on held-out items."""
    model.eval()
    total, count = 0.0, 0
    for ids, lab, att in batches(items, size, pad_id, random.Random(0)):
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        n = int((lab[:, 1:] != -100).sum())
        total += float(out.loss) * n
        count += n
    model.train()
    return total / max(count, 1)


def train(base_path: str, examples: List[Example], device: str, out_dir: str, seed: int,
          epochs: int = 3, lr: float = 2e-4, batch: int = 64, holdout: float = 0.05, log=print):
    """Train a fresh LoRA adapter from the base model on the examples; returns (tok, model, history)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok, base = load(base_path, device)
    model = with_lora(base)
    encoded = [encode(tok, e) for e in examples]
    rng.shuffle(encoded)
    n_hold = max(1, round(holdout * len(encoded)))
    val, tr = encoded[:n_hold], encoded[n_hold:]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps_per_epoch = math.ceil(len(tr) / batch)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps_per_epoch * epochs), steps_per_epoch * epochs)
    history = []
    model.train()
    for epoch in range(epochs):
        for ids, lab, att in batches(tr, batch, tok.pad_token_id, rng):
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
        v = val_loss(model, val, tok.pad_token_id, device)
        history.append({"epoch": epoch + 1, "val_loss": v})
        log(f"epoch {epoch + 1} val_loss {v:.4f}")
    model.save_pretrained(out_dir)
    return tok, model, history


@torch.no_grad()
def sample(tok, model, prompts: List[str], device: str, batch: int = 64, max_new: int = 80,
           temperature: float = 1.0, top_p: float = 0.95, seed: int = 0) -> List[str]:
    """Completions for prompts at temperature 1.0 and top-p 0.95 (plan 11)."""
    model.eval()
    torch.manual_seed(seed)
    out: List[str] = []
    for s in range(0, len(prompts), batch):
        enc = tok(prompts[s:s + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temperature,
                             top_p=top_p, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        out += tok.batch_decode(new, skip_special_tokens=True)
    return out
