"""Step 4: SFT on the kept trajectories — the lesson's script, runnable.
Needs a CUDA GPU and requirements-gpu.txt. Reads the trajectory file the
teacher run wrote (TRAJECTORIES to override), writes the adapter to
sft-ckpt/.

  python sft.py
"""
import json
import os

TRAJ = os.environ.get("TRAJECTORIES",
                      "runs/anthropic-claude-sonnet-4.5/trajectories.json")
if not os.path.exists(TRAJ):
    raise SystemExit("%s not found — run make_trajectories.py first, or set "
                     "TRAJECTORIES to the run you want to train on" % TRAJ)
kept = json.load(open(TRAJ))
if len(kept) < 5:
    raise SystemExit("only %d trajectories in %s — mine more tasks "
                     "(WANTED=50 python mine_tasks.py) before spending GPU "
                     "time" % (len(kept), TRAJ))
OUT = os.environ.get("OUT", "sft-ckpt")
print("training on %d trajectories from %s -> %s" % (len(kept), TRAJ, OUT), flush=True)

MAX_SEQ = int(os.environ.get("MAX_SEQ", "20480"))  # prompts carry
# whole source files plus the grader's failing output — 4096 truncates
# them mid-prompt, silently

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import prof

model, tok = FastLanguageModel.from_pretrained(
    os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    max_seq_length=MAX_SEQ, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")   # long rows on a 24GB card
tok = get_chat_template(tok, chat_template="qwen-2.5")

rows = [{"text": tok.apply_chat_template(
            [{"role": "user", "content": t["prompt"]},
             {"role": "assistant", "content": t["completion"]}],
            tokenize=False)}
        for t in kept]
trainer = SFTTrainer(model=model, processing_class=tok,
    train_dataset=Dataset.from_list(rows),
    args=SFTConfig(max_length=MAX_SEQ,   # TRL's own row cap — defaults low
                   per_device_train_batch_size=1, gradient_accumulation_steps=8,
                   num_train_epochs=int(os.environ.get("EPOCHS", "3")),
                   learning_rate=2e-4, logging_steps=1,
                   output_dir=OUT))
trainer = train_on_responses_only(trainer)   # mask the prompt out of the loss
trainer.add_callback(prof.StepProfiler("sft"))
trainer.train()
prof.summarize("sft")
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("adapter saved to %s/" % OUT, flush=True)
