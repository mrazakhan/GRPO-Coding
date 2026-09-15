"""DPO from the SFT checkpoint, contrasted with GRPO on the same tasks.
DPO needs preference PAIRS, not a scalar reward, so we build them from the
grader: sample K completions per task, grade each with evaluate(), and pair
the best against the worst whenever their credit differs. Offline and
contrastive — no exploration during training, the key difference from GRPO.

  SFT_CKPT=sft-1p5b OUT=dpo-1p5b python dpo.py

Env: K (rollouts per task, default 8), PAIRS_FILE (skip regen), MAX_SEQ,
MAX_COMPLETION, BETA, EPOCHS. Needs a CUDA GPU + requirements-gpu.txt.
"""
import json
import os
import time
import types

def load_defs():
    src = open("make_trajectories.py").read()
    src = src.split('\nif not os.path.exists("tasks.json")')[0]
    mod = types.ModuleType("mt")
    exec(compile(src, "make_trajectories.py", "exec"), mod.__dict__)
    return mod

mt = load_defs()
TASKS = json.load(open(os.environ.get("TASKS_FILE", "tasks.json")))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "8192"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "1024"))
K = int(os.environ.get("K", "8"))
SFT_CKPT = os.environ.get("SFT_CKPT", "sft-1p5b")
OUT = os.environ.get("OUT", "dpo-1p5b")
PAIRS_FILE = os.environ.get("PAIRS_FILE", os.path.join("runs", "dpo_pairs.json"))

import prof
import trl
print("trl", getattr(trl, "__version__", "?"), flush=True)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import DPOConfig, DPOTrainer
from datasets import Dataset

model, tok = FastLanguageModel.from_pretrained(
    SFT_CKPT, max_seq_length=MAX_SEQ, load_in_4bit=True)
tok = get_chat_template(tok, chat_template="qwen-2.5")

def fmt(task):
    return tok.apply_chat_template(
        [{"role": "user", "content": mt.build_prompt(task)}],
        add_generation_prompt=True, tokenize=False)

def one(task):
    text = fmt(task)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=MAX_COMPLETION, do_sample=True,
                         temperature=0.8, top_p=0.95)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)

# --- build preference pairs from graded rollouts (unless cached) ----------
if os.path.exists(PAIRS_FILE):
    pairs = json.load(open(PAIRS_FILE))
    print("loaded %d cached pairs from %s" % (len(pairs), PAIRS_FILE), flush=True)
else:
    FastLanguageModel.for_inference(model)
    t0, pairs = time.time(), []
    for task in TASKS:
        if len(tok(fmt(task)).input_ids) >= MAX_SEQ - MAX_COMPLETION:
            continue
        rolls = [one(task) for _ in range(K)]
        scored = sorted(((mt.evaluate(r, task), r) for r in rolls),
                        key=lambda x: x[0])
        if scored[-1][0] > scored[0][0]:          # a real preference exists
            pairs.append({"prompt": fmt(task),
                          "chosen": scored[-1][1], "rejected": scored[0][1],
                          "margin": round(scored[-1][0] - scored[0][0], 3)})
        print("  %s: best %.2f worst %.2f %s" % (task["id"], scored[-1][0],
              scored[0][0], "PAIR" if scored[-1][0] > scored[0][0] else "skip"),
              flush=True)
    os.makedirs(os.path.dirname(PAIRS_FILE), exist_ok=True)
    json.dump(pairs, open(PAIRS_FILE, "w"), indent=1)
    print("built %d pairs in %.1f min -> %s"
          % (len(pairs), (time.time() - t0) / 60, PAIRS_FILE), flush=True)

if len(pairs) < 2:
    raise SystemExit("only %d preference pairs — the SFT model's rollouts "
                     "rarely differ in credit; raise K or mine more tasks"
                     % len(pairs))

# --- train ----------------------------------------------------------------
FastLanguageModel.for_training(model)
ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")}
                        for p in pairs])
trainer = DPOTrainer(
    model=model, ref_model=None, processing_class=tok, train_dataset=ds,
    args=DPOConfig(
        per_device_train_batch_size=int(os.environ.get("BATCH", "1")),
        gradient_accumulation_steps=int(os.environ.get("ACCUM", "4")),
        num_train_epochs=int(os.environ.get("EPOCHS", "3")),
        learning_rate=float(os.environ.get("LR", "5e-6")),
        beta=float(os.environ.get("BETA", "0.1")),
        max_length=MAX_SEQ, max_prompt_length=MAX_SEQ - MAX_COMPLETION,
        logging_steps=1, output_dir=OUT))
trainer.add_callback(prof.StepProfiler("dpo"))
trainer.train()
prof.summarize("dpo")
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("adapter saved to %s/" % OUT, flush=True)
