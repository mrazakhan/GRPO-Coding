"""Positive control for the eval: for every task in a split, apply the
task's OWN real fix (the commit the defect branch was reverted from) and
require grade == 1.0. A task that fails here is unwinnable — no model can
ever pass it — so a 0.00 eval on such a split says nothing about the
model. Run before trusting any pass rate.

  SPLITS="heldout_yours.json,tasks.json" python check_tasks.py
"""
import json
import os
import subprocess
import types

def load_defs():
    src = open("make_trajectories.py").read()
    src = src.split('\nif not os.path.exists("tasks.json")')[0]
    mod = types.ModuleType("mt")
    exec(compile(src, "make_trajectories.py", "exec"), mod.__dict__)
    return mod

mt = load_defs()
SPLITS = [s.strip() for s in os.environ.get(
    "SPLITS", "heldout_yours.json").split(",") if s.strip()]

def real_fix_diff(task):
    # the defect branch is the fix commit reverted; branch~1 IS the fix.
    parent = subprocess.run(["git", "-C", task["repo"], "rev-parse",
                             task["branch"] + "~1"],
                            capture_output=True, text=True).stdout.strip()
    return subprocess.run(["git", "-C", task["repo"], "diff",
                           task["branch"], parent, "--"] + task["files"],
                          capture_output=True, text=True).stdout

for split in SPLITS:
    if not os.path.exists(split):
        print("%s: missing" % split); continue
    tasks = json.load(open(split))
    print("\n=== %s (%d tasks) ===" % (split, len(tasks)))
    winnable = 0
    for task in tasks:
        diff = real_fix_diff(task)
        if not diff.strip():
            print("  %-10s NO FIX DIFF (branch~1 empty?)" % task["id"]); continue
        credit = mt.grade(task, diff)
        tail = mt.last_test_output.get(task["id"], "")[-160:].replace("\n", " ")
        mark = "ok" if credit == 1.0 else "UNWINNABLE"
        print("  %-10s real-fix credit %.2f  %-11s %s"
              % (task["id"], credit, mark, "" if credit == 1.0 else tail))
        winnable += credit == 1.0
    print("  -> %d/%d tasks are winnable through the eval path"
          % (winnable, len(tasks)))
