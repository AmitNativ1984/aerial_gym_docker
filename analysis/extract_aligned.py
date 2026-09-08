"""Pick a reset-free window in which the drone is ALREADY POINTING AT ITS TARGET.

The earlier selection took the longest reset-free run, which for the mature policy caught
the initial turn-to-target: the drone starts a mean 88 deg off-heading and spends ~49 of
81 steps rotating, so yaw sits legitimately on the rail and dominates the picture.

Here the window is chosen for ALIGNMENT instead -- mean |bearing| below ALIGN_DEG across
the whole window -- so what is plotted is steady cruise toward a target the drone is
already facing, not the opening turn.
"""
import json, math, os

TMP = os.path.join(os.environ["CLAUDE_JOB_DIR"], "tmp")
DT = 0.03
CH = ["thrust", "roll", "pitch", "yaw_rate"]
WANT = 160          # ~4.8 s
MIN_LEN = 100       # >= 3 s, still a useful read
ALIGN_DEG = 20.0    # MEAN |bearing| ceiling
MAX_DEG = 35.0      # and the window must stay aligned THROUGHOUT, not on average --
                    # a mean-only test admits a window that is half a turn.


def runs(done_col):
    """All reset-free [start, end) spans."""
    out, start = [], 0
    for t, d in enumerate(done_col):
        if d:
            if t > start:
                out.append((start, t))
            start = t + 1
    if len(done_col) > start:
        out.append((start, len(done_col)))
    return out


def stats(seg):
    n = len(seg)
    d = [abs(seg[i + 1] - seg[i]) for i in range(n - 1)]
    flips = sum(1 for i in range(n - 1) if seg[i] * seg[i + 1] < 0 and abs(seg[i]) > 1e-3)
    rail = sum(1 for v in seg if abs(v) >= 0.999)
    return {"mean_abs": sum(abs(v) for v in seg) / n,
            "mean_step": sum(d) / len(d) if d else 0.0,
            "flips_per_s": flips / (n * DT),
            "pct_rail": 100.0 * rail / n}


out = {"dt": DT, "channels": CH, "align_deg": ALIGN_DEG, "policies": {}}
for key, fname, label in [
    ("mature", "trace2_mature_L0.json", "Mature (epoch 1800)"),
    ("early", "trace2_early_L0.json", "Curriculum 1 (epoch 50)"),
]:
    raw = json.load(open(os.path.join(TMP, fname)))
    cmd, done, bear = raw["cmd"], raw["done"], raw["bearing_rad"]
    T, E = len(cmd), len(cmd[0])

    # candidate windows: reset-free, long enough, and well aligned throughout
    best = None
    for e in range(E):
        for (s, en) in runs([done[t][e] for t in range(T)]):
            span = en - s
            if span < MIN_LEN:
                continue
            for L in (min(span, WANT), MIN_LEN):
                if L > span:
                    continue
                for off in range(0, span - L + 1, 2):
                    a, b = s + off, s + off + L
                    vals = [abs(bear[t][e]) for t in range(a, b)]
                    mb_deg = math.degrees(sum(vals) / L)
                    mx_deg = math.degrees(max(vals))
                    if mb_deg > ALIGN_DEG or mx_deg > MAX_DEG:
                        continue
                    # aligned first, length second: a well-aligned 3 s beats a
                    # 4.8 s window that smuggles in half a turn.
                    score = (-mb_deg, L)
                    if best is None or score > best[0]:
                        best = (score, e, a, b, mb_deg)
    if best is None:
        raise SystemExit(f"{key}: no window under {ALIGN_DEG} deg mean bearing")
    _, env, s0, s1, mb_deg = best

    series = {c: [round(cmd[t][env][i], 4) for t in range(s0, s1)] for i, c in enumerate(CH)}
    bseries = [round(math.degrees(bear[t][env]), 2) for t in range(s0, s1)]
    out["policies"][key] = {
        "label": label, "epoch": raw["epoch"], "env": env,
        "span": [s0, s1], "n": s1 - s0,
        "mean_bearing_deg": round(mb_deg, 1),
        "max_bearing_deg": round(max(abs(v) for v in bseries), 1),
        "series": series, "bearing": bseries,
        "stats": {c: stats(series[c]) for c in CH},
    }
    print(f"{key:7} env {env:2}  steps {s0}-{s1} ({s1-s0})  "
          f"mean |bearing| {mb_deg:.1f} deg  max {max(abs(v) for v in bseries):.1f} deg")

json.dump(out, open(os.path.join(TMP, "plot_data.json"), "w"))
print("\nwrote plot_data.json")
for k, v in out["policies"].items():
    print(f"\n{v['label']}  (aligned: mean |bearing| {v['mean_bearing_deg']} deg)")
    for c in CH:
        st = v["stats"][c]
        print(f"  {c:>9} mean|a| {st['mean_abs']:.3f}  step {st['mean_step']:.4f}  "
              f"flips/s {st['flips_per_s']:5.2f}  rail {st['pct_rail']:5.1f}%")
