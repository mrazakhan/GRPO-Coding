"""Step 5: GRPO from the SFT checkpoint, rewarded by the repository grader.
The reward is dense — evaluate() returns the fraction of grader tests the
model's diff passes, minus the diff-size penalty — so the group has signal
even before any rollout fully solves a task. Continues the SFT adapter.

  SFT_CKPT=sft-ckpt python grpo.py         # -> grpo-ckpt/

Needs a CUDA GPU, the clone with task branches, requirements-gpu.txt.
vLLM fast-inference is off by default (the pod's vLLM wheel mismatches
the driver); set USE_VLLM=1 once vLLM matches to sample far faster.
"""
import json
import os
import types

def load_defs():
    src = open("make_trajectories.py").read()
    src = src.split('\nif not os.path.exists("tasks.json")')[0]
    mod = types.ModuleType("mt")
    exec(compile(src, "make_trajectories.py", "exec"), mod.__dict__)
    return mod

mt = load_defs()                         # build_prompt, extract_diff, evaluate
TASKS = json.load(open(os.environ.get("TASKS_FILE", "tasks.json")))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "12288"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "2048"))
USE_VLLM = os.environ.get("USE_VLLM", "0") == "1"
OUT = os.environ.get("OUT", "grpo-ckpt")

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import prof

model, tok = FastLanguageModel.from_pretrained(
    os.environ.get("SFT_CKPT", "sft-ckpt"),
    max_seq_length=MAX_SEQ, load_in_4bit=True, fast_inference=USE_VLLM)
tok = get_chat_template(tok, chat_template="qwen-2.5")

def fmt(task):
    return tok.apply_chat_template(
        [{"role": "user", "content": mt.build_prompt(task)}],
        add_generation_prompt=True, tokenize=False)

# one row per task whose prompt leaves room for a completion; the reward
# needs the task, so carry its index and look it up.
rows = []
for i, t in enumerate(TASKS):
    if len(tok(fmt(t)).input_ids) < MAX_SEQ - MAX_COMPLETION:
        rows.append({"prompt": fmt(t), "task_idx": i})
print("%d of %d tasks fit the context window" % (len(rows), len(TASKS)),
      flush=True)
ds = Dataset.from_list(rows)

def reward_grader(prompts, completions, task_idx, **kw):
    # dense: the grader fraction the diff earns, penalty included
    return [mt.evaluate(c, TASKS[i]) for c, i in zip(completions, task_idx)]

trainer = GRPOTrainer(
    model=model, processing_class=tok, reward_funcs=[reward_grader],
    train_dataset=ds,
    args=GRPOConfig(
        per_device_train_batch_size=int(os.environ.get("BATCH", "1")),
        gradient_accumulation_steps=int(os.environ.get("ACCUM", "4")),
        num_generations=int(os.environ.get("NUM_GEN", "8")),
        max_prompt_length=MAX_SEQ - MAX_COMPLETION,
        max_completion_length=MAX_COMPLETION,
        learning_rate=float(os.environ.get("LR", "5e-6")),
        num_train_epochs=int(os.environ.get("EPOCHS", "3")),
        logging_steps=1, use_vllm=USE_VLLM,
        save_steps=int(os.environ.get("SAVE_STEPS", "10")),
        save_total_limit=2,              # keep last 2 — a late crash keeps one
        output_dir=OUT))
trainer.add_callback(prof.StepProfiler("grpo"))
trainer.train()
prof.summarize("grpo")
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("adapter saved to %s/" % OUT, flush=True)
