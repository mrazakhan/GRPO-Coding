"""Bench candidate teacher models on the first N tasks: keep rates and
failure reasons side by side. The five-task protocol as one command.

  BENCH_MODELS="google/gemini-3.8-flash,anthropic/claude-sonnet-4.5" \
  BENCH_TASKS=5 python teacher_bench.py

Models run in parallel, one thread each; every model writes its full
attempt log to runs/bench/<model-slug>.log, and the console shows one
prefixed line per finished task. Grading is serialized across models
(git worktrees on the one clone collide). Verify each id on
openrouter.ai/models. Needs OPENROUTER_API_KEY and the same working
directory as make_trajectories.py (tasks.json + the clone).
"""
import builtins
import json
import logging
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor

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
os.makedirs(os.path.join("runs", "bench"), exist_ok=True)
shared_grade_lock = threading.Lock()   # one clone, one grader at a time
console_lock = threading.Lock()

def bench_one(model):
    slug = model.replace("/", "-")
    mod = load_defs()
    mod.TEACHER_MODEL = model
    mod.TASKS, mod.done, mod.BAR = tasks, [], None
    mod.grade_lock = shared_grade_lock

    log = logging.getLogger("bench." + slug)
    log.setLevel(logging.DEBUG)
    log.propagate = False                # this model's file only
    handler = logging.FileHandler(os.path.join("runs", "bench", slug + ".log"),
                                  mode="w")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    log.handlers[:] = [handler]
    mod.LOG = log

    def console(*args, **kwargs):        # solve()'s progress line, prefixed
        with console_lock:
            builtins.print("[%s]" % slug, *args, flush=True)
    mod.print = console

    kept = 0
    for task in tasks:
        if mod.solve(task):
            kept += 1
    handler.close()
    return model, kept

print("Benching %d models on %d tasks; logs in runs/bench/" % (len(MODELS), N),
      flush=True)
with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
    results = dict(pool.map(bench_one, MODELS))

print("\n%-45s %s" % ("model", "kept/%d" % N))
for model in MODELS:
    kept = results[model]
    print("%-45s %d  %s" % (model, kept,
          "<- viable" if kept * 5 >= N * 3 else ""))
print("\nRule: 3+ of 5 kept means the teacher is viable for the full run.")
