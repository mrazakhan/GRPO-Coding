# GRPO-Coding — personalize a coder to one repository (RL Session 11.5)

The scripts behind the course's short project: personalize
Qwen2.5-Coder to one repository (SQLGlot by default), following
Bespoke Labs' recipe at weekend scale. The lesson explains every
step; this directory is the runnable form.

## Setup (once, on the training machine)

    git config --global user.email "you@example.com"   # commits fail without an identity
    git config --global user.name  "you"
    git clone https://github.com/tobymao/sqlglot
    pip install -r requirements.txt && (cd sqlglot && pip install -e .)
    pip install -r requirements-gpu.txt   # training machine only (CUDA)

## Scripts

- `mine_tasks.py` — turns SQLGlot's bug-fix history into verified
  defect tasks: revert a fix, restore its tests as the grader, keep
  only tasks whose grader fails on the defect and passes at the fix.
  Writes `tasks.json` (training) and `heldout_yours.json` (reserved).
  Run it from the directory containing the `sqlglot` clone.
  Env: `WANTED` (default 30), `HELDOUT` (default 5).
  After it runs: rewrite each `instruction` so it does not name the
  code change — a verbatim commit message hands the model the answer.

- `make_trajectories.py` — generates teacher trajectories for SFT:
  prompts the teacher with each task's instruction and defect-state
  files, keeps completions the grader passes at full credit. API
  calls run on a small worker pool; grading is serialized (concurrent
  `git worktree` operations on one repository collide). Writes
  `trajectories.json`.
  Env: `OPENROUTER_API_KEY` (required), `TEACHER_MODEL`
  (default `google/gemini-3.8-flash`), `WORKERS` (default 3).

Steps 4-6 (SFT, GRPO, the before/after table) are in the lesson; both
their code and the substitutions to `grpo_mbpp.py` (in the course practice repository)
are documented there.
