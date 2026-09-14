'''Step 3: teacher trajectories — assembled from the lesson's step 5 (reward helpers)
and step 3 (OpenRouter pipeline). Run from /workspace after mine_tasks.py.
Env: OPENROUTER_API_KEY (required), TEACHER_MODEL, WORKERS, LOG_LEVEL=DEBUG for
reply snippets on failures.'''

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
        for extra in ([], ["--recount"], ["--3way"]):
            applied = subprocess.run(["git", "-C", workdir, "apply"] + extra
                                     + ["-"], input=diff.encode(),
                                     capture_output=True)
            if applied.returncode == 0:
                if extra:
                    LOG.debug("%s: patch needed git apply %s",
                              task["id"], extra[0])
                break
        if applied.returncode != 0:          # gate: the patch did not apply
            LOG.info("%s: patch rejected by git apply: %s", task["id"],
                     applied.stderr.decode(errors="replace").strip()[:200])
            return 0.0
        out = subprocess.run(task["grader"].split() + ["--tb=no"],
                             cwd=workdir, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": workdir},
                             timeout=120).stdout
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
        if passed + failed == 0:             # gate: no test ever executed
            LOG.info("%s: grader ran no tests: %s", task["id"],
                     out.strip()[-200:])
            return 0.0
        return passed / (passed + failed)
    except subprocess.TimeoutExpired:        # gate: the grader hung
        LOG.info("%s: grader timed out after 120s", task["id"])
        return 0.0
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

import logging
import time
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
LOG = logging.getLogger("trajectories")

try:
    from tqdm import tqdm
except ImportError:                      # optional: bar + ETA when installed
    tqdm = None
if tqdm:
    class _TqdmHandler(logging.Handler):
        def emit(self, record):
            tqdm.write(self.format(record))
    _h = _TqdmHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                      datefmt="%H:%M:%S"))
    logging.getLogger().handlers[:] = [_h]  # keep log lines off the bar
BAR = None
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
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read()[:300]
        if e.code in (401, 403):
            raise SystemExit("API auth failed (%d): %s — check "
                             "OPENROUTER_API_KEY" % (e.code, body))
        LOG.warning("API error %d, treating as failed attempt: %s",
                    e.code, body)
        time.sleep(5)
        return "", TEACHER_MODEL
    except urllib.error.URLError as e:
        LOG.warning("network error, treating as failed attempt: %s", e.reason)
        time.sleep(5)
        return "", TEACHER_MODEL
    choice = data["choices"][0]
    finish = choice.get("finish_reason")
    if finish not in (None, "stop"):
        LOG.warning("finish_reason=%s — reply may be truncated "
                    "(reasoning tokens eating the budget?)", finish)
    text = choice["message"].get("content") or ""  # reasoning models: null
    LOG.debug("API %.1fs, %d chars", time.time() - t0, len(text))
    return text, data.get("model", TEACHER_MODEL)

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
            LOG.info("%s attempt %d: empty reply", task["id"], attempt + 1)
            continue
        if not diff:
            reason = "no diff fence"
            LOG.info("%s attempt %d: %d chars, no diff fence",
                     task["id"], attempt + 1, len(reply))
            LOG.debug("reply head: %r", reply[:200])
            continue
        t0 = time.time()
        with grade_lock:                       # serialized: touches the repo
            credit = grade(task, diff)
        reason = "credit %.2f" % credit        # 0.00 usually: diff not applying
        LOG.info("%s attempt %d: %d chars, diff %d lines, credit %.2f "
                 "(grade %.1fs)", task["id"], attempt + 1, len(reply),
                 len(diff.splitlines()), credit, time.time() - t0)
        if credit == 1.0:
            row = {"task": task["id"], "teacher": served_by,
                   "prompt": build_prompt(task), "completion": reply}
            break
    with progress_lock:
        done.append(task["id"])
        line = "[%d/%d] %s %s" % (len(done), len(TASKS), task["id"],
               "kept (attempt %d)" % (attempt + 1) if row
               else "gave up (last: %s)" % reason)
        if BAR:
            BAR.update(1)
            tqdm.write(line)
        else:
            print(line, flush=True)
    return row

if not os.path.exists("tasks.json"):
    raise SystemExit("tasks.json not found in %s — run mine_tasks.py first, "
                     "or symlink the file here" % os.getcwd())
TASKS = json.load(open("tasks.json"))
missing = sorted({t["repo"] for t in TASKS if not os.path.isdir(t["repo"])})
if missing:
    raise SystemExit("task repo path(s) do not exist: %s — fix the repo "
                     "field in tasks.json" % ", ".join(missing))
LOG.info("teacher=%s workers=%d tasks=%d out=%s",
         TEACHER_MODEL, WORKERS, len(TASKS), OUT_DIR)
run_t0 = time.time()
if tqdm:
    BAR = tqdm(total=len(TASKS), unit="task", dynamic_ncols=True)
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    kept = [r for r in pool.map(solve, TASKS) if r]
if BAR:
    BAR.close()

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "trajectories.json")
with open(out_path, "w") as f:
    json.dump(kept, f)
print(len(kept), "trajectories written to", out_path)
LOG.info("kept %d/%d (%.0f%%) in %.1f min", len(kept), len(TASKS),
         100.0 * len(kept) / max(len(TASKS), 1),
         (time.time() - run_t0) / 60)
teachers = {}
for row in kept:
    teachers[row["teacher"]] = teachers.get(row["teacher"], 0) + 1
for name, count in sorted(teachers.items()):
    LOG.info("  served by %s: %d", name, count)
