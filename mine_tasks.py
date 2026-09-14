"""Mine verified defect tasks from a repository's bug-fix commits."""
import json
import os
import subprocess

REPO = os.path.abspath("sqlglot")
WANTED = int(os.environ.get("WANTED", "30"))
HELDOUT = int(os.environ.get("HELDOUT", "5"))

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args],
                          capture_output=True, text=True)

def grader_for(test_files):
    """A pytest command for the fix's tests: the test module itself, or
    test_optimizer.py filtered to the fixture's stem for fixture-only fixes."""
    module = next((f for f in test_files
                   if f.startswith("tests/test_") and f.endswith(".py")), None)
    if module:
        return ["python", "-m", "pytest", module, "-q", "--tb=no"]
    stem = test_files[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return ["python", "-m", "pytest", "tests/test_optimizer.py",
            "-k", stem, "-q", "--tb=no"]

def summary(grader):
    out = subprocess.run(grader, cwd=REPO, capture_output=True,
                         text=True, timeout=300).stdout.strip()
    return out.splitlines()[-1] if out else ""

log = git("log", "--first-parent", "-i", "--grep", "fix",
          "--pretty=%H %s", "-800").stdout.splitlines()
tasks = []
for line in log:
    if len(tasks) >= WANTED:
        break
    sha, msg = line.split(" ", 1)
    files = git("show", "--name-only", "--pretty=format:", sha).stdout.split()
    src = [f for f in files if f.startswith("sqlglot/") and f.endswith(".py")]
    test = [f for f in files if f.startswith("tests/")]
    if not (1 <= len(src) <= 2 and 1 <= len(test) <= 2):
        continue                       # not the one-fix-plus-its-tests shape
    branch = "task-%02d" % (len(tasks) + 1)
    git("checkout", "-q", "-B", branch, sha)
    if git("revert", "--no-commit", sha).returncode != 0:
        git("revert", "--abort")       # e.g. a merge commit: skip it
        continue
    git("checkout", sha, "--", *test)  # restore the grader
    git("commit", "-qm", branch + ": defect state")
    grader = grader_for(test)
    on_defect = summary(grader)
    git("checkout", "-q", sha)
    on_fix = summary(grader)
    accepted = ("failed" in on_defect and "passed" in on_fix
                and "failed" not in on_fix)
    print(branch, sha[:8], "ACCEPT" if accepted else "reject",
          "| defect:", on_defect[:50], "| fix:", on_fix[:50])
    if accepted:
        tasks.append({"id": branch, "branch": branch,
                      "repo": REPO,
                      "grader": " ".join(grader[:-2]),
                      "files": src, "instruction": msg})
    else:
        git("branch", "-qD", branch)

train, heldout = tasks[:-HELDOUT], tasks[-HELDOUT:]
with open("tasks.json", "w") as f:
    json.dump(train, f, indent=1)
with open("heldout_yours.json", "w") as f:
    json.dump(heldout, f, indent=1)
print(len(train), "training tasks in tasks.json;",
      len(heldout), "reserved in heldout_yours.json.",
      "Rewrite each instruction from its commit message before training.")
