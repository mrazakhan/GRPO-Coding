"""Replay realistic teacher replies through the real extract -> grade
path against a real mined task. Catches harness bugs the offline seam
tests cannot: anything between the model's raw reply text and the
credit. Needs tasks.json (or heldout_yours.json) and the clone with
its task branches — run after mine_tasks.py, before trusting any
bench number. Run: python test_grading_replay.py"""
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

for name in ("tasks.json", "heldout_yours.json"):
    if os.path.exists(name) and json.load(open(name)):
        task = json.load(open(name))[0]
        break
else:
    raise SystemExit("needs a mined task: run mine_tasks.py first")

mod = load_defs()
fence = chr(96) * 3

# Ground truth: the defect branch's own fix, recovered by diffing the
# defect state against the branch's parent (the original fix commit).
parent = subprocess.run(["git", "-C", task["repo"], "rev-parse",
                         task["branch"] + "~1"],
                        capture_output=True, text=True).stdout.strip()
true_diff = subprocess.run(["git", "-C", task["repo"], "diff",
                            task["branch"], parent, "--"] + task["files"],
                           capture_output=True, text=True).stdout
assert true_diff.strip(), "no diff between defect branch and its parent"

def as_reply(diff):
    return "Here is the fix.\n%sdiff\n%s%s\n" % (fence, diff, fence)

failures = []
def check(label, reply, want_full_credit):
    diff = mod.extract_diff(reply)
    credit = mod.grade(task, diff) if diff else 0.0
    ok = (credit == 1.0) if want_full_credit else (credit == 0.0)
    print("%-38s credit %.2f %s" % (label, credit, "ok" if ok else "FAIL"))
    if not ok:
        failures.append(label)

# Replies a correct teacher could realistically send — all must keep.
check("clean fenced reply", as_reply(true_diff), True)
check("fence strips trailing newline", as_reply(true_diff.rstrip("\n")), True)
check("blank context lines emptied",
      as_reply("\n".join("" if l == " " else l
                         for l in true_diff.splitlines())), True)
check("prose after the closing fence",
      as_reply(true_diff) + "\nThis resolves the failing tests.", True)

# Replies that must NOT earn credit — the negative controls.
check("no fence at all", "I would change the function to...", False)
check("wrong-path diff",
      as_reply("--- a/nope.py\n+++ b/nope.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"),
      False)

if failures:
    raise SystemExit("replay FAILURES: %s" % ", ".join(failures))
print("all %d replay cases behave against %s" % (6, task["id"]))
