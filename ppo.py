"""PPO from the SFT checkpoint, contrasted with GRPO on the same grader.
PPO is the predecessor GRPO improves on: it needs a value head (the critic)
and takes a scalar reward through a manual generate -> reward -> step loop.
This is the memory/complexity cost the comparison is meant to expose —
watch prof's peak VRAM against GRPO's.

  SFT_CKPT=sft-1p5b OUT=ppo-1p5b python ppo.py

Env: MAX_COMPLETION, BATCH, LR, PPO_STEPS, MAX_SEQ. CUDA GPU required.
Note: PPO's API churns across TRL releases and the value head does not ride
Unsloth's 4-bit path cleanly, so this uses plain HF + peft. If it trips on
your TRL version, paste the error and the printed version and I'll adapt.
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
SFT_CKPT = os.environ.get("SFT_CKPT", "sft-1p5b")
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
OUT = os.environ.get("OUT", "ppo-1p5b")
PPO_STEPS = int(os.environ.get("PPO_STEPS", "40"))

import prof
import torch
import trl
print("trl", getattr(trl, "__version__", "?"), flush=True)

# The classic manual-loop PPO API (generate -> step with scalar rewards) is
# the natural fit for a function reward. Guard for it explicitly.
try:
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler          # classic-API marker
    CLASSIC = hasattr(PPOTrainer, "step")
except Exception:
    CLASSIC = False
if not CLASSIC:
    raise SystemExit(
        "This TRL (%s) does not expose the classic PPOTrainer.step() loop "
        "this script uses (newer TRL routes PPO through a reward-model). "
        "Tell me the version and I'll adapt ppo.py to it."
        % getattr(trl, "__version__", "?"))

from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
lora = LoraConfig(r=16, lora_alpha=16, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
# value head on top of the policy; load the SFT adapter as the starting point
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    SFT_CKPT if os.path.exists(SFT_CKPT) else BASE,
    quantization_config=bnb, peft_config=lora)

ppo = PPOTrainer(PPOConfig(batch_size=int(os.environ.get("BATCH", "4")),
                           mini_batch_size=1,
                           learning_rate=float(os.environ.get("LR", "5e-6"))),
                 model, ref_model=None, tokenizer=tok)

def build(task):
    text = tok.apply_chat_template(
        [{"role": "user", "content": mt.build_prompt(task)}],
        add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt").input_ids[0]

fits = [t for t in TASKS
        if len(build(t)) < MAX_SEQ - MAX_COMPLETION]
print("%d of %d tasks fit the window" % (len(fits), len(TASKS)), flush=True)
profiler = prof.StepProfiler("ppo")

for step in range(PPO_STEPS):
    profiler.on_step_begin(None, type("S", (), {"global_step": step})(), None)
    batch = [fits[(step * ppo.config.batch_size + j) % len(fits)]
             for j in range(ppo.config.batch_size)]
    queries = [build(t).to(ppo.accelerator.device) for t in batch]
    responses = ppo.generate(queries, max_new_tokens=MAX_COMPLETION,
                             do_sample=True, top_p=0.95, temperature=0.8)
    texts = [tok.decode(r, skip_special_tokens=True) for r in responses]
    rewards = [torch.tensor(mt.evaluate(x, t), dtype=torch.float32)
               for x, t in zip(texts, batch)]
    ppo.step(queries, responses, rewards)
    profiler.on_step_end(None, type("S", (), {"global_step": step})(), None)
    print("[ppo] step %d  mean reward %.3f"
          % (step, sum(float(r) for r in rewards) / len(rewards)), flush=True)

prof.summarize("ppo")
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("adapter saved to %s/" % OUT, flush=True)
