"""One comparison table across algorithms: held-out mean-credit and pass
rate from runs/eval/results.jsonl, median step time and peak VRAM from
runs/profile/<algo>.jsonl. Run after the eval so the rows are populated.

  python report.py            # optionally FAMILY=1p5b to filter labels
"""
import json
import os

FAMILY = os.environ.get("FAMILY", "")

def evals():
    p = os.path.join("runs", "eval", "results.jsonl")
    latest = {}
    if os.path.exists(p):
        for r in map(json.loads, open(p)):
            if FAMILY and FAMILY not in (r.get("label", "") or ""):
                continue
            latest[(r["ckpt"], r["split"])] = r
    return latest

def prof(algo):
    p = os.path.join("runs", "profile", algo + ".jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    if not rows:
        return None
    t = sorted(r["step_seconds"] for r in rows)
    return {"steps": len(rows), "median_s": t[len(t) // 2],
            "peak_gb": max(r["peak_vram_gb"] for r in rows)}

print("\n%-14s %-12s %-12s %-10s %-12s %s" % ("checkpoint", "mean-credit",
      "pass", "steps", "median s/step", "peak VRAM GB"))
for (ckpt, split), r in evals().items():
    algo = ("grpo" if "grpo" in ckpt else "dpo" if "dpo" in ckpt
            else "ppo" if "ppo" in ckpt else "")
    pr = prof(algo) if algo else None
    print("%-14s %-12.3f %-12.2f %-10s %-12s %s" % (
        ckpt, r.get("mean_credit", 0.0), r["pass_rate"],
        pr["steps"] if pr else "-", ("%.1f" % pr["median_s"]) if pr else "-",
        ("%.2f" % pr["peak_gb"]) if pr else "-"))
