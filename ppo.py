"""PPO from the SFT checkpoint, hand-rolled so it does not depend on TRL's
PPOTrainer (0.24 routes PPO through a reward-model, which cannot wrap a
git-grader). Genuine PPO mechanics: a clipped importance-ratio surrogate
over several inner epochs per rollout batch. The advantage baseline is the
group mean (no learned value net), so this is PPO's clipped objective with
a sampled baseline — the memory/complexity contrast with GRPO is the inner
epochs and per-token ratio, which prof captures.

  SFT_CKPT=sft OUT=ppo python ppo.py

Env: K (completions/step, 4), PPO_STEPS (60), INNER (2), CLIP (0.2),
LR (5e-6), MAX_SEQ (8192), MAX_COMPLETION (1024). CUDA GPU required.
"""
import unsloth                               # must precede trl/transformers
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
K = int(os.environ.get("K", "4"))
PPO_STEPS = int(os.environ.get("PPO_STEPS", "60"))
INNER = int(os.environ.get("INNER", "2"))
CLIP = float(os.environ.get("CLIP", "0.2"))
LR = float(os.environ.get("LR", "5e-6"))
SFT_CKPT = os.environ.get("SFT_CKPT", "sft")
OUT = os.environ.get("OUT", "ppo")

import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import prof

model, tok = FastLanguageModel.from_pretrained(
    SFT_CKPT, max_seq_length=MAX_SEQ, load_in_4bit=True)
tok = get_chat_template(tok, chat_template="qwen-2.5")
FastLanguageModel.for_training(model)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

def prompt_ids(task):
    text = tok.apply_chat_template(
        [{"role": "user", "content": mt.build_prompt(task)}],
        add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt").input_ids[0]

def completion_logprob(full_ids, plen):
    """sum log p(completion tokens) under the current policy."""
    logits = model(full_ids.unsqueeze(0)).logits[0]      # (T, V)
    sl = logits[plen - 1:-1].float()                     # only completion
    logp = F.log_softmax(sl, dim=-1)
    tgt = full_ids[plen:]
    return logp[torch.arange(len(tgt), device=logp.device), tgt].sum()

fits = [t for t in TASKS
        if len(prompt_ids(t)) < MAX_SEQ - MAX_COMPLETION]
print("%d of %d tasks fit the window" % (len(fits), len(TASKS)), flush=True)
prof_cb = prof.StepProfiler("ppo")
dev = model.device

class _S:                                    # tiny state stub for prof
    def __init__(self, i): self.global_step = i

for step in range(PPO_STEPS):
    prof_cb.on_step_begin(None, _S(step), None)
    task = fits[step % len(fits)]
    pids = prompt_ids(task).to(dev)
    plen = len(pids)
    gens, texts = [], []
    with torch.no_grad():
        for _ in range(K):
            out = model.generate(pids.unsqueeze(0), max_new_tokens=MAX_COMPLETION,
                                 do_sample=True, temperature=0.8, top_p=0.95)
            seq = out[0].tolist()            # plain list: not an inference tensor
            gens.append(seq)
            texts.append(tok.decode(seq[plen:], skip_special_tokens=True))
    rewards = torch.tensor([mt.evaluate(t, task) for t in texts])
    if float(rewards.std()) < 1e-6:          # no gradient in this group
        prof_cb.on_step_end(None, _S(step), None)
        print("[ppo] step %d  reward %.3f  (flat group, skipped)"
              % (step, float(rewards.mean())), flush=True)
        continue
    adv = (rewards - rewards.mean())
    adv = (adv / (adv.std() + 1e-6)).to(dev)
    with torch.no_grad():
        old_lp = torch.stack([completion_logprob(
            torch.tensor(g, device=dev, dtype=torch.long), plen) for g in gens])
    last = 0.0
    for _ in range(INNER):
        opt.zero_grad()
        loss = 0.0
        for i, g in enumerate(gens):
            new_lp = completion_logprob(
                torch.tensor(g, device=dev, dtype=torch.long), plen)
            ratio = torch.exp(new_lp - old_lp[i])
            loss = loss - torch.min(ratio * adv[i],
                                    torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[i])
        loss = loss / len(gens)
        loss.backward()
        opt.step()
        last = float(loss)
    prof_cb.on_step_end(None, _S(step), None)
    print("[ppo] step %d  reward %.3f  loss %.4f"
          % (step, float(rewards.mean()), last), flush=True)

prof.summarize("ppo")
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("adapter saved to %s/" % OUT, flush=True)
