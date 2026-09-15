"""One comparison table across algorithms: held-out mean-credit and pass
rate from runs/eval/results.jsonl, median step time and peak VRAM from
runs/profile/<algo>.jsonl. Columns auto-size to their contents.

  python report.py            # optionally FAMILY=1p5b to filter by label
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

def grid(headers, data):
    widths = [max(len(headers[i]), *(len(row[i]) for row in data)) if data
              else len(headers[i]) for i in range(len(headers))]
    fmt = "  ".join("%-" + str(w) + "s" for w in widths)
    print(fmt % tuple(headers))
    print(fmt % tuple("-" * w for w in widths))
    for row in data:
        print(fmt % tuple(row))

def algo_of(ckpt):
    for a in ("grpo", "dpo", "ppo", "sft"):
        if a in ckpt:
            return a
    return ""

data = []
for (ckpt, split), r in sorted(evals().items()):
    a = algo_of(ckpt)
    pr = prof(a) if a in ("grpo", "dpo", "ppo") else None
    data.append([ckpt, "%.3f" % r.get("mean_credit", 0.0),
                 "%.2f" % r["pass_rate"],
                 str(pr["steps"]) if pr else "-",
                 "%.1f" % pr["median_s"] if pr else "-",
                 "%.2f" % pr["peak_gb"] if pr else "-"])
grid(["checkpoint", "mean-credit", "pass", "steps", "median s/step",
      "peak VRAM GB"], data)
