"""Bench candidate teacher models on the first N tasks: keep rates and
failure reasons side by side. The five-task protocol as one command.

  BENCH_MODELS="google/gemini-3.8-flash,anthropic/claude-sonnet-4.5" \
  BENCH_TASKS=5 python teacher_bench.py

Verify each id on openrouter.ai/models. Needs OPENROUTER_API_KEY and the
same working directory as make_trajectories.py (tasks.json + the clone).
"""
import json
import os
import types

MODELS = [m.strip() for m in os.environ.get(
    "BENCH_MODELS",
    "google/gemini-3.8-flash,anthropic/claude-sonnet-4.5").split(",") if m.strip()]
N = int(os.environ.get("BENCH_TASKS", "5"))

def load_defs():
    src = open("make_trajectories.py").read()
    src = src.split('\nif not os.path.exists("tasks.json")')[0]  # defs only
    mod = types.ModuleType("mt")
    exec(compile(src, "make_trajectories.py", "exec"), mod.__dict__)
    return mod

tasks = json.load(open("tasks.json"))[:N]
results = {}
for model in MODELS:
    mod = load_defs()
    mod.TEACHER_MODEL = model
    mod.TASKS, mod.done, mod.BAR = tasks, [], None
    print("\n=== %s ===" % model, flush=True)
    kept, reasons = 0, {}
    for task in tasks:
        row = mod.solve(task)
        if row:
            kept += 1
    results[model] = kept

print("\n%-45s %s" % ("model", "kept/%d" % N))
for model, kept in results.items():
    print("%-45s %d  %s" % (model, kept,
          "<- viable" if kept * 5 >= N * 3 else ""))
print("\nRule: 3+ of 5 kept means the teacher is viable for the full run.")
