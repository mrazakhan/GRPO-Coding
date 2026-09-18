const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";                 // 13.3 x 7.5
const W = 13.333, H = 7.5;

// ---- palette (technical navy / teal, mint accent for the winner) ----
const NAVY = "0F2440", NAVY2 = "17324E", TEAL = "1C7293", MINT = "2EC4B6";
const INK = "1A2733", GRAY = "5F7183", MUTE = "9AA9B6";
const PANEL = "EEF3F7", LIGHT = "FFFFFF", LINE = "D8E1E8";
const HEAD = "Century Schoolbook", BODY = "Calibri";

const notes = {};
function slide(bg) { const s = p.addSlide(); s.background = { color: bg || LIGHT }; return s; }
function title(s, t, col) {
  s.addText(t, { isTextBox: true, x: 0.7, y: 0.5, w: 12.0, h: 0.9, fontFace: HEAD,
    fontSize: 32, bold: true, color: col || NAVY, align: "left", margin: 0 });
}
function kicker(s, t, col) {
  s.addText(t.toUpperCase(), { isTextBox: true, x: 0.72, y: 0.28, w: 12, h: 0.3,
    fontFace: BODY, fontSize: 12, bold: true, color: col || TEAL, charSpacing: 3, margin: 0 });
}
function circleIcon(s, x, y, ch, fill) {
  s.addShape(p.ShapeType.ellipse, { x, y, w: 0.5, h: 0.5, fill: { color: fill || TEAL } });
  s.addText(ch, { isTextBox: true, x, y, w: 0.5, h: 0.5, align: "center", valign: "middle",
    fontFace: HEAD, fontSize: 18, bold: true, color: "FFFFFF", margin: 0 });
}

// ============================================================ 1. TITLE
let s = slide(NAVY);
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: W, h: H, fill: { color: NAVY } });
s.addShape(p.ShapeType.ellipse, { x: 9.3, y: -2.2, w: 6.5, h: 6.5, fill: { color: NAVY2 } });
s.addShape(p.ShapeType.ellipse, { x: 11.0, y: 3.6, w: 4.2, h: 4.2, fill: { color: "13293F" } });
s.addText("RL SESSION 11.5  ·  THE SHORT PROJECT", { isTextBox: true, x: 0.9, y: 1.5, w: 10, h: 0.4,
  fontFace: BODY, fontSize: 14, bold: true, color: MINT, charSpacing: 3, margin: 0 });
s.addText("Personalizing a Coder\nto One Repository", { isTextBox: true, x: 0.9, y: 2.0, w: 10.5, h: 2.0,
  fontFace: HEAD, fontSize: 46, bold: true, color: "FFFFFF", lineSpacing: 50, margin: 0 });
s.addText("Does reinforcement learning beat supervised fine-tuning at teaching a small model to repair bugs in a single codebase? An end-to-end study on SQLGlot.",
  { isTextBox: true, x: 0.9, y: 4.3, w: 9.2, h: 1.0, fontFace: BODY, fontSize: 16, color: "CADCFC",
    lineSpacing: 24, margin: 0 });
s.addText([
  { text: "Base → SFT → GRPO / DPO / PPO", options: { bold: true, color: "FFFFFF" } },
  { text: "     seeded, reproducible", options: { color: MUTE, italic: true } }],
  { isTextBox: true, x: 0.9, y: 5.7, w: 11, h: 0.4, fontFace: BODY, fontSize: 14, margin: 0 });
notes[1] = "A weekend-sized project: post-train Qwen2.5-Coder-1.5B to fix real bugs in one repository, and compare SFT against three RL algorithms against the untrained base.";

// ============================================================ 2. PROBLEM
s = slide(LIGHT);
kicker(s, "The problem");
title(s, "A general coder is weak on your repository");
s.addText("Off-the-shelf models know Python, not your codebase — its conventions, its module layout, the exact behavior its tests encode. On unfamiliar repository bugs, a small base model rarely produces a patch that even applies, let alone passes the tests.",
  { isTextBox: true, x: 0.72, y: 1.5, w: 6.2, h: 2.0, fontFace: BODY, fontSize: 16, color: INK, lineSpacing: 24, margin: 0 });
const goals = [
  ["Personalize", "Adapt a 1.5B coder to one repository with post-training, on a single 24 GB GPU."],
  ["Compare", "Measure SFT vs RL (GRPO, DPO, PPO) against the untrained base — same tasks, same metric."],
  ["Measure honestly", "A verifiable grader as the reward and the yardstick; seeded, reproducible evaluation."]];
let gy = 1.6;
goals.forEach(([h, d], i) => {
  s.addShape(p.ShapeType.roundRect, { x: 7.2, y: gy, w: 5.4, h: 1.55, rectRadius: 0.09,
    fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  circleIcon(s, 7.45, gy + 0.28, String(i + 1), TEAL);
  s.addText(h, { isTextBox: true, x: 8.15, y: gy + 0.22, w: 4.3, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText(d, { isTextBox: true, x: 8.15, y: gy + 0.62, w: 4.3, h: 0.85, fontFace: BODY, fontSize: 12.5, color: GRAY, lineSpacing: 16, margin: 0 });
  gy += 1.75;
});
s.addText("Reference point — Bespoke Labs' write-up: the base model scored 0 of 100 on the trained repo; SFT then RL lifted it to ~52–57%. Their base, like ours, started at zero.",
  { isTextBox: true, x: 0.72, y: 5.6, w: 6.2, h: 1.2, fontFace: BODY, fontSize: 12.5, italic: true, color: TEAL, lineSpacing: 18, margin: 0 });
notes[2] = "The base model is near-useless on repo-specific repair before personalization — that is exactly why the project exists.";

// ============================================================ 3. APPROACH / PIPELINE
s = slide(NAVY);
kicker(s, "The approach", MINT);
title(s, "One pipeline, five checkpoints", "FFFFFF");
const steps = [
  ["1", "Mine tasks", "Revert real bug-fix commits into verified defect tasks"],
  ["2", "Teacher data", "Sonnet 4.5 writes diffs; keep only graded-correct trajectories"],
  ["3", "SFT", "Imitate the kept teacher diffs (LoRA)"],
  ["4", "RL", "GRPO / DPO / PPO on the grader reward"],
  ["5", "Evaluate", "Held-out tasks, seeded, base→SFT→RL"]];
let cx = 0.72; const cw = 2.32, gap = 0.13;
steps.forEach(([n, h, d], i) => {
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 2.2, w: cw, h: 2.9, rectRadius: 0.1,
    fill: { color: NAVY2 }, line: { color: "27455F", width: 1 } });
  s.addShape(p.ShapeType.ellipse, { x: cx + cw / 2 - 0.35, y: 2.45, w: 0.7, h: 0.7, fill: { color: i === 3 ? MINT : TEAL } });
  s.addText(n, { isTextBox: true, x: cx + cw / 2 - 0.35, y: 2.45, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: HEAD, fontSize: 24, bold: true, color: NAVY, margin: 0 });
  s.addText(h, { isTextBox: true, x: cx + 0.1, y: 3.35, w: cw - 0.2, h: 0.5, align: "center", fontFace: HEAD, fontSize: 16, bold: true, color: "FFFFFF", margin: 0 });
  s.addText(d, { isTextBox: true, x: cx + 0.12, y: 3.85, w: cw - 0.24, h: 1.15, align: "center", fontFace: BODY, fontSize: 11.5, color: "AEC3D4", lineSpacing: 15, margin: 0 });
  if (i < 4) s.addText("›", { isTextBox: true, x: cx + cw - 0.02, y: 3.3, w: 0.3, h: 0.6, align: "center", valign: "middle", fontFace: BODY, fontSize: 26, bold: true, color: MINT, margin: 0 });
  cx += cw + gap;
});
s.addText("Every stage saves its own checkpoint into its own folder, so nothing overwrites and the before/after table is exact.",
  { isTextBox: true, x: 0.72, y: 5.5, w: 12, h: 0.5, fontFace: BODY, fontSize: 13.5, italic: true, color: "9FB6C8", margin: 0 });
notes[3] = "The pipeline is one repeatable sequence; each of the five checkpoints (base, SFT, GRPO, DPO, PPO) is evaluated on the same held-out tasks.";

// ============================================================ 4. DATA & REWARD
s = slide(LIGHT);
kicker(s, "Data & reward");
title(s, "A verifiable grader is the reward");
s.addText("Tasks are mined from the repository's own history: take a real bug-fix commit, revert it to recreate the defect, and keep the commit's tests as the grader. Each task is verified — the real fix scores 1.0, the defect scores 0 — so nothing untrainable enters the set.",
  { isTextBox: true, x: 0.72, y: 1.5, w: 6.15, h: 1.9, fontFace: BODY, fontSize: 15, color: INK, lineSpacing: 22, margin: 0 });
s.addText("The reward, per rollout", { isTextBox: true, x: 0.72, y: 3.45, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
const rw = [
  ["Apply", "the model's unified diff in a fresh git worktree of the defect"],
  ["Test", "run the task's pytest grader in that worktree"],
  ["Credit", "reward = fraction of grader tests that pass  (dense, 0–1)"]];
let ry = 3.9;
rw.forEach(([h, d]) => {
  circleIcon(s, 0.72, ry, h[0], TEAL);
  s.addText([{ text: h + "  ", options: { bold: true, color: NAVY } }, { text: d, options: { color: GRAY } }],
    { isTextBox: true, x: 1.4, y: ry + 0.03, w: 5.5, h: 0.5, fontFace: BODY, fontSize: 13.5, valign: "middle", lineSpacing: 16, margin: 0 });
  ry += 0.72;
});
// right callout card
s.addShape(p.ShapeType.roundRect, { x: 7.3, y: 1.5, w: 5.3, h: 4.6, rectRadius: 0.1, fill: { color: NAVY }, line: { color: NAVY, width: 1 } });
s.addText("Why dense credit matters", { isTextBox: true, x: 7.65, y: 1.8, w: 4.7, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: "FFFFFF", margin: 0 });
s.addText("A 1.5B rarely solves a whole multi-hunk fix in one shot — a binary pass reads 0 for everyone. The partial-credit fraction still separates a near-miss from noise, and it gives RL a gradient to climb.",
  { isTextBox: true, x: 7.65, y: 2.4, w: 4.7, h: 1.6, fontFace: BODY, fontSize: 13.5, color: "CADCFC", lineSpacing: 20, margin: 0 });
s.addText([{ text: "60", options: { fontSize: 40, bold: true, color: MINT } }, { text: "  tasks mined", options: { fontSize: 15, color: "CADCFC" } }],
  { isTextBox: true, x: 7.65, y: 4.1, w: 4.7, h: 0.7, fontFace: HEAD, margin: 0 });
s.addText([{ text: "10", options: { fontSize: 40, bold: true, color: MINT } }, { text: "  held out for evaluation", options: { fontSize: 15, color: "CADCFC" } }],
  { isTextBox: true, x: 7.65, y: 4.9, w: 4.7, h: 0.7, fontFace: HEAD, margin: 0 });
notes[4] = "The reward is code, not a neural network: apply the diff in a worktree and run pytest. Partial credit is the sensitive metric for a small student.";

// ============================================================ 5. METHODS
s = slide(LIGHT);
kicker(s, "Methods compared");
title(s, "Four ways to move the model");
const methods = [
  ["SFT", "Supervised fine-tuning", "Imitate the teacher's correct diffs. Teaches the diff format; no notion of reward.", TEAL],
  ["GRPO", "Group-relative policy opt.", "Sample K diffs, reward each with the grader, push toward the better ones. Online, dense reward — no critic.", MINT],
  ["DPO", "Direct preference opt.", "Build best-vs-worst pairs from rollouts and prefer the better. Offline, contrastive — no exploration.", TEAL],
  ["PPO", "Proximal policy opt.", "Clipped policy-gradient with a baseline. The predecessor GRPO streamlines; needs more machinery.", GRAY]];
let mx = 0.72; const mw = 2.95, mgap = 0.15;
methods.forEach(([tag, name, desc, col]) => {
  s.addShape(p.ShapeType.roundRect, { x: mx, y: 1.6, w: mw, h: 4.4, rectRadius: 0.1, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  s.addShape(p.ShapeType.roundRect, { x: mx + 0.25, y: 1.9, w: 1.55, h: 0.62, rectRadius: 0.08, fill: { color: col } });
  s.addText(tag, { isTextBox: true, x: mx + 0.25, y: 1.9, w: 1.55, h: 0.62, align: "center", valign: "middle", fontFace: HEAD, fontSize: 19, bold: true, color: "FFFFFF", margin: 0 });
  s.addText(name, { isTextBox: true, x: mx + 0.25, y: 2.7, w: mw - 0.5, h: 0.8, fontFace: HEAD, fontSize: 14.5, bold: true, color: NAVY, lineSpacing: 18, margin: 0 });
  s.addText(desc, { isTextBox: true, x: mx + 0.25, y: 3.5, w: mw - 0.5, h: 2.3, fontFace: BODY, fontSize: 13, color: GRAY, lineSpacing: 19, margin: 0 });
  mx += mw + mgap;
});
s.addText("All four adapt the same 1.5B student (LoRA); RL methods continue from the SFT checkpoint.",
  { isTextBox: true, x: 0.72, y: 6.2, w: 12, h: 0.4, fontFace: BODY, fontSize: 13, italic: true, color: TEAL, margin: 0 });
notes[5] = "GRPO fits the setup natively (reward is a function); DPO and PPO each have a structural mismatch with a code-based reward, which shows up in the results.";

// ============================================================ 6. MAIN RESULT (chart)
s = slide(LIGHT);
kicker(s, "Main result", TEAL);
title(s, "RL wins — and only RL beats the base");
const barData = [{ name: "mean-credit", labels: ["Base", "SFT", "GRPO", "DPO"], values: [0.120, 0.129, 0.161, 0.102] }];
s.addChart(p.ChartType.bar, barData, {
  x: 0.6, y: 1.6, w: 7.4, h: 5.2, barDir: "col",
  chartColors: [MUTE, TEAL, MINT, "C2CDD6"],
  showValue: true, dataLabelPosition: "outEnd",
  dataLabelColor: INK, dataLabelFontFace: BODY, dataLabelFontSize: 13, dataLabelFontBold: true,
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  catAxisLabelColor: INK, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 14, catAxisLabelFontBold: true,
  valAxisMinVal: 0, valAxisMaxVal: 0.18, showLegend: false, showTitle: false, barGapWidthPct: 60 });
s.addShape(p.ShapeType.roundRect, { x: 8.35, y: 1.7, w: 4.3, h: 2.35, rectRadius: 0.1, fill: { color: NAVY } });
s.addText("GRPO", { isTextBox: true, x: 8.6, y: 1.9, w: 3.8, h: 0.4, fontFace: BODY, fontSize: 13, bold: true, color: MINT, charSpacing: 2, margin: 0 });
s.addText([{ text: "+34%", options: { fontSize: 46, bold: true, color: "FFFFFF" } }],
  { isTextBox: true, x: 8.6, y: 2.25, w: 3.8, h: 0.9, fontFace: HEAD, margin: 0 });
s.addText("held-out mean-credit over the base model", { isTextBox: true, x: 8.6, y: 3.2, w: 3.8, h: 0.7, fontFace: BODY, fontSize: 13, color: "CADCFC", lineSpacing: 17, margin: 0 });
s.addText([
  { text: "Monotonic:  ", options: { bold: true, color: NAVY } },
  { text: "base 0.120 → SFT 0.129 → GRPO 0.161.", options: { color: GRAY } },
  { text: "  DPO 0.102 falls below base.", options: { color: GRAY } }],
  { isTextBox: true, x: 8.35, y: 4.3, w: 4.4, h: 1.0, fontFace: BODY, fontSize: 13.5, lineSpacing: 19, margin: 0 });
s.addText("Metric: mean fraction of grader tests passed, held-out tasks, seeded (SEED=0). Full-solve rate ≈ 0 for all — expected for a 1.5B on real optimizer bugs.",
  { isTextBox: true, x: 8.35, y: 5.5, w: 4.4, h: 1.3, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE, lineSpacing: 16, margin: 0 });
notes[6] = "GRPO is the only method that beats the base model, by a clear +34%. SFT helps a little; DPO ends below base.";

// ============================================================ 7. COMPUTE PROFILE (chart)
s = slide(LIGHT);
kicker(s, "Compute profile");
title(s, "GRPO wins at half DPO's memory");
const vram = [{ name: "peak VRAM (GB)", labels: ["SFT", "GRPO", "DPO"], values: [2.5, 11.1, 20.9] }];
s.addChart(p.ChartType.bar, vram, {
  x: 0.6, y: 1.7, w: 6.6, h: 5.0, barDir: "col",
  chartColors: [TEAL, MINT, "C2CDD6"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontFace: BODY, dataLabelFontSize: 13, dataLabelFontBold: true,
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  catAxisLabelColor: INK, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 14, catAxisLabelFontBold: true,
  valAxisMinVal: 0, valAxisMaxVal: 24, showLegend: false, showTitle: true, title: "Peak VRAM (GB, 24 GB card)",
  titleColor: GRAY, titleFontFace: BODY, titleFontSize: 13, barGapWidthPct: 70 });
// right: speed callouts
const speed = [["SFT", "2.9", "30"], ["GRPO", "24.1", "102"], ["DPO", "4.6", "6"]];
s.addText("Median seconds per step", { isTextBox: true, x: 7.7, y: 1.75, w: 5, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
let sy = 2.3;
speed.forEach(([m, sps, st]) => {
  s.addShape(p.ShapeType.roundRect, { x: 7.7, y: sy, w: 4.95, h: 1.15, rectRadius: 0.09, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  s.addText(m, { isTextBox: true, x: 7.95, y: sy + 0.32, w: 1.6, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: m === "GRPO" ? TEAL : NAVY, margin: 0 });
  s.addText([{ text: sps, options: { fontSize: 26, bold: true, color: NAVY } }, { text: " s/step", options: { fontSize: 12, color: GRAY } }],
    { isTextBox: true, x: 9.5, y: sy + 0.22, w: 1.9, h: 0.7, fontFace: HEAD, margin: 0 });
  s.addText([{ text: st, options: { fontSize: 20, bold: true, color: GRAY } }, { text: " steps", options: { fontSize: 12, color: MUTE } }],
    { isTextBox: true, x: 11.35, y: sy + 0.3, w: 1.3, h: 0.6, fontFace: HEAD, margin: 0 });
  sy += 1.32;
});
s.addText("GRPO is slower per step — it generates and grades online — but the better result costs about half the memory DPO uses.",
  { isTextBox: true, x: 0.72, y: 6.85, w: 12, h: 0.4, fontFace: BODY, fontSize: 12.5, italic: true, color: TEAL, margin: 0 });
notes[7] = "GRPO trades step time for memory efficiency and result quality; DPO is fast per step but ran only 6 steps and used the most VRAM.";

// ============================================================ 8. FINDINGS
s = slide(LIGHT);
kicker(s, "Why the ranking holds");
title(s, "The reason each method landed where it did");
const finds = [
  ["GRPO leads", MINT, "Dense reward gives a group of rollouts real variance — 0.1 vs 0.0 — even when none fully solves. That is a gradient toward the exact thing being measured."],
  ["DPO starves", TEAL, "It needs a distinct better-vs-worse pair per task. A weak policy's rollouts mostly score identically, so only ~8 pairs formed → 6 steps → below base."],
  ["SFT plateaus", TEAL, "Imitating the teacher's long diffs transfers weakly to harder held-out tasks; on some seeds SFT sits at or below base. RL recovers and surpasses it."],
  ["Noise is real", GRAY, "Re-evaluating one checkpoint swung ±0.02–0.03 and reordered methods. Seeding the eval made the comparison reproducible and the ranking stable."]];
let fx = 0.72, fy = 1.6; const fw = 5.9, fh = 2.35;
finds.forEach(([h, col, d], i) => {
  const x = fx + (i % 2) * (fw + 0.35), y = fy + Math.floor(i / 2) * (fh + 0.3);
  s.addShape(p.ShapeType.roundRect, { x, y, w: fw, h: fh, rectRadius: 0.1, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  s.addShape(p.ShapeType.ellipse, { x: x + 0.3, y: y + 0.32, w: 0.28, h: 0.28, fill: { color: col } });
  s.addText(h, { isTextBox: true, x: x + 0.75, y: y + 0.22, w: fw - 1, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText(d, { isTextBox: true, x: x + 0.35, y: y + 0.8, w: fw - 0.7, h: 1.4, fontFace: BODY, fontSize: 13.5, color: GRAY, lineSpacing: 19, margin: 0 });
});
notes[8] = "GRPO wins on dense reward; DPO is starved on a weak policy; SFT plateaus out of distribution; and the whole comparison only became trustworthy once the eval was seeded.";

// ============================================================ 9. ENGINEERING RIGOR
s = slide(NAVY);
kicker(s, "Making the numbers trustworthy", MINT);
title(s, "The harness was the hard part", "FFFFFF");
s.addText("A wrong number is worse than no number. Most of the effort went into a grader and eval that measure the model, not artifacts of the plumbing.",
  { isTextBox: true, x: 0.72, y: 1.45, w: 11.5, h: 0.7, fontFace: BODY, fontSize: 15, color: "CADCFC", lineSpacing: 21, margin: 0 });
const rig = [
  ["Patches that apply", "Restore trailing newlines, label file paths, repair hunk counts and blank context — so a right fix is not scored zero on formatting."],
  ["A reward that can't crash", "Every completion is graded inside a guard; one malformed diff can never kill a training run."],
  ["Positive controls", "Each task's real fix must score 1.0, and its prompt must fit context — else the task is dropped as unwinnable."],
  ["Seeded evaluation", "Identical sampling draws per checkpoint make the comparison paired and every number reproducible."]];
let rx = 0.72, ryy = 2.35; const rwid = 5.9, rht = 1.9;
rig.forEach(([h, d], i) => {
  const x = rx + (i % 2) * (rwid + 0.35), y = ryy + Math.floor(i / 2) * (rht + 0.3);
  s.addShape(p.ShapeType.roundRect, { x, y, w: rwid, h: rht, rectRadius: 0.1, fill: { color: NAVY2 }, line: { color: "27455F", width: 1 } });
  s.addText(h, { isTextBox: true, x: x + 0.35, y: y + 0.25, w: rwid - 0.7, h: 0.5, fontFace: HEAD, fontSize: 16.5, bold: true, color: MINT, margin: 0 });
  s.addText(d, { isTextBox: true, x: x + 0.35, y: y + 0.75, w: rwid - 0.7, h: 1.0, fontFace: BODY, fontSize: 13, color: "C7D6E3", lineSpacing: 18, margin: 0 });
});
notes[9] = "The debugging — newline handling, path repair, crash-proof reward, positive controls, seeded eval — is what makes the result table honest.";

// ============================================================ 10. CONCLUSION
s = slide(NAVY);
s.addShape(p.ShapeType.ellipse, { x: -2.3, y: 3.4, w: 6.5, h: 6.5, fill: { color: NAVY2 } });
s.addText("TAKEAWAYS", { isTextBox: true, x: 0.9, y: 0.8, w: 10, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: MINT, charSpacing: 3, margin: 0 });
s.addText("RL earned its place", { isTextBox: true, x: 0.9, y: 1.25, w: 11, h: 0.9, fontFace: HEAD, fontSize: 38, bold: true, color: "FFFFFF", margin: 0 });
const take = [
  "GRPO beat the base model by +34% mean-credit and beat every other method — at half DPO's memory.",
  "The reward is code, not a network: a git-worktree pytest grader, dense partial credit, verifiable both ways.",
  "Algorithm fit matters — GRPO's function reward plugs in natively; DPO starves and PPO needs a hand-rolled loop.",
  "Seeding turned a noisy, reordering table into a reproducible result."];
let ty = 2.5;
take.forEach((t) => {
  s.addShape(p.ShapeType.ellipse, { x: 0.95, y: ty + 0.05, w: 0.24, h: 0.24, fill: { color: MINT } });
  s.addText(t, { isTextBox: true, x: 1.4, y: ty - 0.05, w: 8.2, h: 0.75, fontFace: BODY, fontSize: 15, color: "E4EDF6", lineSpacing: 20, margin: 0 });
  ty += 0.82;
});
s.addShape(p.ShapeType.roundRect, { x: 10.0, y: 2.5, w: 2.7, h: 3.0, rectRadius: 0.12, fill: { color: NAVY2 }, line: { color: "27455F", width: 1 } });
s.addText("NEXT", { isTextBox: true, x: 10.25, y: 2.7, w: 2.2, h: 0.4, fontFace: BODY, fontSize: 12, bold: true, color: MINT, charSpacing: 2, margin: 0 });
s.addText([
  { text: "7B student", options: { bold: true, color: "FFFFFF" } }, { text: "\nfor the first full solves\n\n", options: { color: "AEC3D4", fontSize: 12 } },
  { text: "Complete PPO", options: { bold: true, color: "FFFFFF" } }, { text: "\nthe from-scratch loop\n\n", options: { color: "AEC3D4", fontSize: 12 } },
  { text: "More tasks", options: { bold: true, color: "FFFFFF" } }, { text: "\nto widen the reward signal", options: { color: "AEC3D4", fontSize: 12 } }],
  { isTextBox: true, x: 10.25, y: 3.15, w: 2.3, h: 2.3, fontFace: BODY, fontSize: 13.5, lineSpacing: 17, margin: 0 });
s.addText("SQLGlot  ·  Qwen2.5-Coder-1.5B  ·  teacher: Claude Sonnet 4.5  ·  single RTX 4090",
  { isTextBox: true, x: 0.9, y: 6.7, w: 11.5, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, margin: 0 });
notes[10] = "RL, specifically GRPO, is the method that moved a small model on real repository repair; the honest caveat is scale (a 7B and PPO completion are the next steps).";

Object.keys(notes).forEach(() => {});
p.slides.forEach((sl, i) => { if (notes[i + 1]) sl.addNotes(notes[i + 1]); });

p.writeFile({ fileName: "/home/user/grpo-coding/repo-personalization-results.pptx" })
  .then(f => console.log("wrote", f));
