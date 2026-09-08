"""Is the saturated yaw command PURPOSEFUL (turning to face the target) or DRIFT?

Three tests that separate them. The vehicle frame is yaw-only, so
bearing = atan2(n_y, n_x) over the unit direction-to-target is exactly the heading
error: 0 means the drone points at its target.

  1. SIGN AGREEMENT. A purposeful turn commands yaw in the direction that REDUCES the
     bearing error. Chance is 50%. A controller should be far above it.
  2. DOES THE ERROR ACTUALLY FALL? Compare |bearing| early vs late in each episode.
     A drone that turns to face its target ends up facing it.
  3. WHAT HAPPENS ONCE ALIGNED? If yaw saturation is the turn, it must STOP once the
     bearing is small. If yaw is still railed at |bearing| < 15 deg, it is not the turn.
"""
import json, math, os

TMP = os.path.join(os.environ["CLAUDE_JOB_DIR"], "tmp")
DT = 0.03
YAW_MAX_DEG = 60.0


def analyse(fname, label):
    d = json.load(open(os.path.join(TMP, fname)))
    cmd, done, bear = d["cmd"], d["done"], d["bearing_rad"]
    T, E = len(cmd), len(cmd[0])

    agree = tot = 0
    railed_aligned = aligned = 0
    railed_mis = mis = 0
    first, last = [], []
    ep_b, ep_ok = [], 0

    for e in range(E):
        seg = []
        for t in range(T):
            b = bear[t][e]                 # heading error, radians
            y = cmd[t][e][3]               # yaw-rate command
            # 1. sign agreement: reducing a POSITIVE bearing needs a POSITIVE yaw rate
            if abs(b) > math.radians(5) and abs(y) > 1e-3:
                tot += 1
                if (b > 0) == (y > 0):
                    agree += 1
            # 3. railed while already pointing at the target?
            if abs(b) < math.radians(15):
                aligned += 1
                if abs(y) >= 0.999:
                    railed_aligned += 1
            else:
                mis += 1
                if abs(y) >= 0.999:
                    railed_mis += 1
            seg.append(abs(b))
            if done[t][e]:
                if len(seg) >= 20:
                    ep_b.append((seg[:10], seg[-10:]))
                seg = []
        if len(seg) >= 20:
            ep_b.append((seg[:10], seg[-10:]))

    for a, b in ep_b:
        fa, la = sum(a) / len(a), sum(b) / len(b)
        first.append(fa); last.append(la)
        if la < fa:
            ep_ok += 1

    print(f"\n===== {label} =====")
    print(f"1. yaw sign agrees with reducing bearing error : "
          f"{100*agree/max(tot,1):5.1f}%   (chance = 50%, n={tot})")
    if first:
        print(f"2. mean |bearing| first 10 steps of episode   : {math.degrees(sum(first)/len(first)):5.1f} deg")
        print(f"   mean |bearing| last 10 steps of episode    : {math.degrees(sum(last)/len(last)):5.1f} deg")
        print(f"   episodes where the error FELL              : {100*ep_ok/len(ep_b):5.1f}%  (n={len(ep_b)})")
    print(f"3. yaw at rail while ALIGNED (|bearing|<15deg)  : "
          f"{100*railed_aligned/max(aligned,1):5.1f}%   ({aligned} such steps)")
    print(f"   yaw at rail while MISALIGNED                 : "
          f"{100*railed_mis/max(mis,1):5.1f}%   ({mis} such steps)")
    print(f"   time spent already aligned                   : {100*aligned/(T*E):5.1f}%")


analyse("trace2_mature_L0.json", "MATURE ep1800, open space")
analyse("trace2_early_L0.json", "CURRICULUM 1 ep50, open space")
print("\nREAD: if (1) is near 50%, (3) shows the rail held even when aligned, and the")
print("error does not fall, the yaw command is not a turn-to-target -- it is drift.")
