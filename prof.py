"""Per-step profiler shared by grpo/dpo/ppo: peak VRAM and wall-time per
optimizer step, printed live and appended to runs/profile/<algo>.jsonl so
the algorithms can be compared on the same axes. summarize() prints the
median step time and peak memory across a run."""
import json
import os
import time

try:
    import torch
    from transformers import TrainerCallback
except Exception:                        # allows import on a CPU box
    TrainerCallback = object

PROF_DIR = "runs/profile"

class StepProfiler(TrainerCallback):
    def __init__(self, algo):
        self.algo = algo
        os.makedirs(PROF_DIR, exist_ok=True)
        self.path = os.path.join(PROF_DIR, algo + ".jsonl")
        open(self.path, "w").close()     # fresh per run
        self.t0 = None

    def on_step_begin(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kw):
        dt = time.time() - self.t0 if self.t0 else 0.0
        peak = torch.cuda.max_memory_allocated() / 1e9 \
            if torch.cuda.is_available() else 0.0
        with open(self.path, "a") as f:
            f.write(json.dumps({"algo": self.algo, "step": state.global_step,
                                "step_seconds": round(dt, 2),
                                "peak_vram_gb": round(peak, 2)}) + "\n")
        print("[prof %s] step %d  %.1fs  peak %.2f GB"
              % (self.algo, state.global_step, dt, peak), flush=True)

def summarize(algo):
    p = os.path.join(PROF_DIR, algo + ".jsonl")
    if not os.path.exists(p):
        return
    rows = [json.loads(l) for l in open(p)]
    if not rows:
        return
    times = sorted(r["step_seconds"] for r in rows)
    peaks = [r["peak_vram_gb"] for r in rows]
    med = times[len(times) // 2]
    print("\n[prof %s] %d steps | median %.1fs/step | peak VRAM %.2f GB"
          % (algo, len(rows), med, max(peaks)), flush=True)
