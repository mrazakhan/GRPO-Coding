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
import time
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

RESULTS = os.path.join("runs", "eval", "results.jsonl")

def record(row):
    """Append one measured row; the table accumulates across invocations
    (the base row from an earlier run stays)."""
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")

def _grid(headers, data):
    widths = [max(len(headers[i]), *(len(row[i]) for row in data)) if data
              else len(headers[i]) for i in range(len(headers))]
    fmt = "  ".join("%-" + str(w) + "s" for w in widths)
    print("\n" + fmt % tuple(headers))
    print(fmt % tuple("-" * w for w in widths))
    for row in data:
        print(fmt % tuple(row))

def print_table():
    rows = [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []
    latest = {}
    for r in rows:                       # last measurement per (ckpt, split)
        latest[(r["ckpt"], r["split"])] = r
    data = [[ckpt, split, "%.2f" % r["pass_rate"],
             "%.3f" % r.get("mean_credit", 0.0), "%.0f" % r["mean_tokens"],
             ("%s %s" % (r.get("ckpt_mtime", ""), r.get("label", ""))).strip()]
            for (ckpt, split), r in latest.items()]
    _grid(["checkpoint", "split", "pass", "mean-credit", "tokens",
           "trained at / label"], data)

MAX_SEQ = int(os.environ.get("MAX_SEQ", "20480"))  # prompts carry
# whole source files plus the grader's failing output — 4096 truncates
# them mid-prompt, silently

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
    if os.path.isdir(ckpt) and not os.path.exists(
            os.path.join(ckpt, "adapter_config.json")):
        # a run still in progress leaves checkpoint-N/ but no root adapter
        subs = sorted((d for d in os.listdir(ckpt)
                       if d.startswith("checkpoint-")),
                      key=lambda d: int(d.split("-")[1]))
        if subs:
            ckpt = os.path.join(ckpt, subs[-1])   # newest intermediate
            print("using latest checkpoint %s" % ckpt, flush=True)
        else:
            print("%s: no finished adapter yet, skipping" % ckpt, flush=True)
            continue
    try:
        model, tok = FastLanguageModel.from_pretrained(
            ckpt, max_seq_length=MAX_SEQ, load_in_4bit=True)
    except Exception as e:
        print("%s: could not load (%s), skipping"
              % (ckpt, type(e).__name__), flush=True)
        continue
    tok = get_chat_template(tok, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    for split in SPLITS:
        rates, partials, lengths = [], [], []
        for task in json.load(open(split)):
            rolls = [sample(model, tok, task) for _ in range(ROLLS)]
            diffs = [mt.extract_diff(r) for r in rolls]
            credits = [mt.grade(task, d) if d is not None else 0.0
                       for d in diffs]                 # fraction of grader
            rates.append(mean(c == 1.0 for c in credits))    # full solves
            partials.append(mean(credits))              # partial progress
            lengths.append(mean(len(tok.encode(r)) for r in rolls))
        row = {"ckpt": ckpt, "split": split, "rolls": ROLLS,
               "label": os.environ.get("EVAL_LABEL", ""),
               "ckpt_mtime": (time.strftime("%Y-%m-%d %H:%M:%S",
                              time.localtime(os.path.getmtime(ckpt)))
                              if os.path.exists(ckpt) else "hub"),
               "pass_rate": round(mean(rates), 3),
               "mean_credit": round(mean(partials), 3),
               "mean_tokens": round(mean(lengths), 1),
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        record(row)
        print("  %-42s pass %.2f  mean-credit %.3f  tokens %.0f"
              % (ckpt + " / " + split, row["pass_rate"],
                 row["mean_credit"], row["mean_tokens"]), flush=True)

print_table()
print("\nrows saved to %s" % RESULTS)
