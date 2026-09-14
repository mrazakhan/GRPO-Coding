"""How big are the eval prompts, in tokens? A 1.5B model tops out near
32K context; a prompt above that cannot be seen, and generation crashes
or truncates. Uses the same build_prompt the eval uses."""
import json, os, types
def load_defs():
    src = open("make_trajectories.py").read().split('\nif not os.path.exists("tasks.json")')[0]
    m = types.ModuleType("mt"); exec(compile(src, "make_trajectories.py", "exec"), m.__dict__); return m
mt = load_defs()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
for split in os.environ.get("SPLITS", "heldout_yours.json").split(","):
    split = split.strip()
    if not os.path.exists(split): continue
    print("\n=== %s ===" % split)
    for t in json.load(open(split)):
        n = len(tok(mt.build_prompt(t)).input_ids)
        flag = "  <-- OVER 32K, unseeable" if n > 32768 else ("  <- tight" if n > 20000 else "")
        print("  %-10s %6d tokens%s" % (t["id"], n, flag))
