"""Step 6: the before/after table — pass rate and output length per
checkpoint per held-out split. grade at full credit is the pass
criterion; the diff-size penalty plays no part here.

  CKPTS="Qwen/Qwen2.5-Coder-1.5B-Instruct,sft-ckpt,grpo-ckpt" \
  SPLITS="heldout_yours.json" ROLLS=10 python eval_before_after.py

Run it before training (the base row), after SFT, and after GRPO —
without the base row the table proves nothing. Needs a CUDA GPU, the
clone with its task branches, and requirements-gpu.txt.
"""
import json
import os
import types
from statistics import mean

def load_defs():
    src = open("make_trajectories.py").read()
    src = src.split('\nif not os.path.exists("tasks.json")')[0]
    mod = types.ModuleType("mt")
    exec(compile(src, "make_trajectories.py", "exec"), mod.__dict__)
    return mod

mt = load_defs()                 # build_prompt, extract_diff, grade
CKPTS = [c.strip() for c in os.environ.get(
    "CKPTS", "Qwen/Qwen2.5-Coder-1.5B-Instruct,sft-ckpt,grpo-ckpt"
    ).split(",") if c.strip()]
SPLITS = [s.strip() for s in os.environ.get(
    "SPLITS", "heldout_yours.json").split(",") if s.strip()]
ROLLS = int(os.environ.get("ROLLS", "10"))

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def sample(model, tok, task):
    text = tok.apply_chat_template(
        [{"role": "user", "content": mt.build_prompt(task)}],
        add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=2048, do_sample=True,
                         temperature=0.8, top_p=0.95)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)

for ckpt in CKPTS:
    if "/" not in ckpt and not os.path.exists(ckpt):
        print("%s: not found, skipping (train it first)" % ckpt, flush=True)
        continue
    model, tok = FastLanguageModel.from_pretrained(
        ckpt, max_seq_length=4096, load_in_4bit=True)
    tok = get_chat_template(tok, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    for split in SPLITS:
        rates, lengths = [], []
        for task in json.load(open(split)):
            rolls = [sample(model, tok, task) for _ in range(ROLLS)]
            diffs = [mt.extract_diff(r) for r in rolls]
            rates.append(mean(d is not None and mt.grade(task, d) == 1.0
                              for d in diffs))
            lengths.append(mean(len(tok.encode(r)) for r in rolls))
        print("%-40s %-22s pass rate %.2f  mean output tokens %.0f"
              % (ckpt, split, mean(rates), mean(lengths)), flush=True)
