'''Step 3: teacher trajectories — assembled from the lesson's step 5 (reward helpers)
and step 3 (OpenRouter pipeline). Run from /workspace after mine_tasks.py.'''
import os, re, subprocess, tempfile

FENCE = chr(96) * 3   # a triple-backtick string; write the literal in your file

def extract_diff(text):
    start = text.find(FENCE + "diff")
    if start < 0:
        return None
    start += len(FENCE) + len("diff")
    end = text.find(FENCE, start)
    return None if end < 0 else text[start:end].strip("\n")

def grade(task, diff):
    """Per-test credit in [0, 1]; 0.0 whenever the grader never ran."""
    workdir = tempfile.mkdtemp()
    try:
        subprocess.run(["git", "worktree", "add", "--detach", workdir,
                        task["branch"]], cwd=task["repo"],
                       check=True, capture_output=True)
        applied = subprocess.run(["git", "-C", workdir, "apply", "-"],
                                 input=diff.encode(), capture_output=True)
        if applied.returncode != 0:
            return 0.0                       # gate: the patch did not apply
        out = subprocess.run(task["grader"].split() + ["--tb=no"],
                             cwd=workdir, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": workdir},
                             timeout=120).stdout
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
        if passed + failed == 0:
            return 0.0                       # gate: no test ever executed
        return passed / (passed + failed)
    except subprocess.TimeoutExpired:
        return 0.0                           # gate: the grader hung
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", workdir],
                       cwd=task["repo"], capture_output=True)

def evaluate(completion_text, task):
    diff = extract_diff(completion_text)
    if diff is None:
        return 0.0
    credit = grade(task, diff)
    penalty = 0.05 * min(len(diff.splitlines()) / 100, 1.0)
    return max(credit - penalty, 0.0) if credit > 0 else 0.0

import json, os, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "google/gemini-3.8-flash")
WORKERS = int(os.environ.get("WORKERS", "3"))
OUT_DIR = os.path.join("runs", TEACHER_MODEL.replace("/", "-"))
grade_lock = Lock()
progress_lock = Lock()
done = []

def teacher(prompt):
    """One chat completion via OpenRouter -> (text, model that served it)."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": TEACHER_MODEL,
                         "max_tokens": 16384,
                         "messages": [{"role": "user", "content": prompt}]}
                        ).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    message = data["choices"][0]["message"]
    return (message.get("content") or "",   # reasoning models can send null
            data.get("model", TEACHER_MODEL))

def show(task, path):
    """A file's contents at the task's defect state — never the working tree."""
    return subprocess.run(["git", "-C", task["repo"], "show",
                           task["branch"] + ":" + path],
                          capture_output=True, text=True).stdout

def build_prompt(task):
    sources = "\n\n".join(show(task, f) for f in task["files"])
    return (f"{task['instruction']}\n\nRelevant files:\n{sources}\n\n"
            "Answer with a unified diff in a diff code fence.")

def solve(task):
    row, reason = None, "no reply"
    for attempt in range(3):
        reply, served_by = teacher(build_prompt(task))  # parallel: pure waiting
        diff = extract_diff(reply)
        if not reply:
            reason = "empty reply"
            continue
        if not diff:
            reason = "no diff fence"
            continue
        with grade_lock:                       # serialized: touches the repo
            credit = grade(task, diff)
        reason = "credit %.2f" % credit        # 0.00 usually: diff not applying
        if credit == 1.0:
            row = {"task": task["id"], "teacher": served_by,
                   "prompt": build_prompt(task), "completion": reply}
            break
    with progress_lock:
        done.append(task["id"])
        print("[%d/%d] %s %s" % (len(done), len(TASKS), task["id"],
              "kept (attempt %d)" % (attempt + 1) if row
              else "gave up (last: %s)" % reason), flush=True)
    return row

TASKS = json.load(open("tasks.json"))
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    kept = [r for r in pool.map(solve, TASKS) if r]

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "trajectories.json")
with open(out_path, "w") as f:
    json.dump(kept, f)
print(len(kept), "trajectories written to", out_path)
