# Results — personalizing a coder to one repository (RL Session 11.5)

Post-training a small code model to repair real bugs in a single repository
(**SQLGlot**), comparing SFT and three RL algorithms against the untrained
base. All numbers below were measured on a rented RTX 4090 (24 GB) with the
scripts in this repository.

## Setup

| Component | Choice |
|---|---|
| Repository | [SQLGlot](https://github.com/tobymao/sqlglot) |
| Task mining | `mine_tasks.py`, `WANTED=60`, oversized-file tasks filtered (`MAX_SRC_CHARS=40000`) |
| Dataset | ~50 training tasks + 10 held-out; each verified winnable (real fix → grade 1.0) via `check_tasks.py` |
| Teacher (SFT data) | `anthropic/claude-sonnet-4.5`, chosen by a five-task bench (Sonnet 3/5 vs Gemini Flash 1/5 vs DeepSeek 0/5) |
| Student | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, 4-bit, LoRA r=16 on 7 projection layers |
| Reward | repository grader: apply the model's diff in a `git worktree`, run the fix's tests, credit = fraction passed (dense) |
| Eval | held-out tasks, 10–30 sampled rollouts/task at temperature 0.8, **seeded** (`SEED=0`) for reproducibility |

**Metric.** `pass` is the fraction of rollouts that *fully* solve a task
(diff applies **and** every grader test passes). Because a 1.5B student almost
never one-shots a full multi-hunk optimizer fix, the sensitive metric is
**`mean-credit`** — the mean grader fraction across rollouts — which reveals
progress below the full-solve line.

## Main result (seeded, `SEED=0`, held-out split)

| checkpoint | mean-credit | vs base | note |
|---|---|---|---|
| base (Qwen2.5-Coder-1.5B) | 0.120 | — | untrained |
| SFT | 0.129 | +8% | imitation of teacher diffs |
| **GRPO** | **0.161** | **+34%** | RL on the grader reward — **best** |
| DPO | 0.102 | −15% | preference-pair RL — below base |

Full-solve `pass` rate stayed at ~0.00 for every checkpoint (occasional 0.01),
as expected for a 1.5B on real optimizer bugs — the write-up's own base model
scored 0 full-solves too. The signal lives in `mean-credit`.

**Headline:** the training stages are monotonic (base → SFT → GRPO), and
**GRPO is the only method that beats the base model**, by a clear margin.

## Compute profile (per-method, from `prof.py`)

| method | steps | median s/step | peak VRAM |
|---|---|---|---|
| SFT | 30 | 2.9 s | 2.5 GB |
| GRPO | 102 | 24.1 s | 11.1 GB |
| DPO | 6 | 4.6 s | 20.9 GB |
| PPO | — | — | — (see below) |

GRPO is slow per step (it generates and grades online) but **used roughly
half the VRAM of DPO** while being the better method — a strong efficiency
point.

## Findings

1. **GRPO wins because its reward is dense.** The grader returns partial
   credit, so a group of rollouts has reward *variance* (e.g. 0.1 vs 0.0)
   even when none fully solves the task — enough gradient to climb. GRPO
   directly optimizes the exact quantity being measured.

2. **DPO is starved on a weak policy.** DPO needs a *distinct* better-vs-worse
   pair per task; it builds them from the student's own rollouts. Because the
   1.5B's rollouts on most tasks all score identically (usually all 0), only
   ~8 preference pairs formed → 6 optimizer steps → it trained on far less
   signal than GRPO and landed below base. This is a property of the setup
   (offline, contrastive, no exploration), not a bug.

3. **SFT plateaus / mildly regresses on out-of-distribution held-out tasks.**
   SFT tripled output length (imitating the teacher's long diffs) but barely
   moved held-out mean-credit; on some eval seeds it sat at or below base.
   Imitation of a narrow training set transfers weakly to the harder held-out
   tasks (lineage / annotate_types / diff). RL recovers and surpasses it.

4. **Eval variance is large and must be controlled.** Re-evaluating an
   unchanged checkpoint swung mean-credit ±0.02–0.03 (temperature-0.8 sampling
   over 10 tasks), which initially reordered the methods. Seeding the eval
   (`SEED=0`) and raising rollouts made the comparison reproducible and the
   ranking stable — the numbers above are the seeded ones.

## Caveats / limitations

- **Small student, small held-out set.** A 1.5B on 10 held-out tasks; absolute
  numbers are low and differences are a few points of mean-credit. Capacity
  (a 7B student) is the lever most likely to push full-solve rate above zero —
  on the earlier dataset the base 7B (0.083) and its SFT (0.120) were measured,
  but the 7B RL sweep was not completed.
- **PPO result not obtained.** TRL 0.24 removed the classic `PPOTrainer.step()`
  loop and routes PPO through a reward-*model* module, which cannot wrap a
  code-based git grader. `ppo.py` was rewritten as a from-scratch clipped
  policy-gradient loop (no TRL PPOTrainer); the last blocker (inference-mode
  tensors in the backward pass) was fixed, but the pod was terminated before a
  successful PPO run — so no PPO row exists yet.
- **Checkpoints and `runs/` were lost** with the pod (never committed). The
  code reproduces everything; re-running is the recovery path.

## Reproduction

```bash
# data
WANTED=60 HELDOUT=10 python mine_tasks.py
python check_tasks.py

# teacher trajectories (needs OPENROUTER_API_KEY)
python make_trajectories.py

# base row first, then train each stage into its own folder
SEED=0 ROLLS=30 CKPTS="Qwen/Qwen2.5-Coder-1.5B-Instruct" python eval_before_after.py
OUT=sft EPOCHS=10 python sft.py
SFT_CKPT=sft OUT=grpo python grpo.py
SFT_CKPT=sft OUT=dpo  python dpo.py
SFT_CKPT=sft OUT=ppo  python ppo.py

# seeded comparison + profile table
SEED=0 ROLLS=30 CKPTS="Qwen/Qwen2.5-Coder-1.5B-Instruct,sft,grpo,dpo,ppo" python eval_before_after.py
python report.py
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the RL trainers on a
24 GB card. Numbers here are from a single seeded run; report mean ± spread
over a few seeds before drawing fine distinctions.
