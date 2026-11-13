from decimal import Decimal, ROUND_HALF_UP
def rd(x, d=3):
    # Round once from the unrounded value, half up: 27/720 = 0.0375 gives 0.038 (a plain format call can give 0.037)
    return float(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
def rda(a, d=3):
    return np.array([rd(v, d) for v in np.atleast_1d(a)])
# Supplementary check for Section 5.2: selection on seeds 1-6 and evaluation on seeds 7-12, and the four simultaneous-injection combinations
# Usage: python seed_split_and_combined.py  (run from the top folder of the package so that the relative paths resolve)
# The thresholds are set in the same way as in run_experiment_v7_final.py (99th percentile of the per-window statistic in baseline_long.pkl).
import numpy as np, pandas as pd
from scipy import stats

cal_df = pd.read_pickle('baseline_long.pkl'); cal = cal_df.values; cols = cal_df.columns.tolist()
C1, C3T, C3S = cols.index('XMEAS(9)'), cols.index('XMEAS(13)'), cols.index('XMEAS(7)')
noise = cal[:, C1].std()
builds = [pd.read_pickle(f'tep2py/indep_builds/build_{i}/baseline.pkl').values for i in range(1, 13)]
sl3, ic3, *_ = stats.linregress(cal[:, C3S], cal[:, C3T])

def windows(bs, W):
    return [d[i*W:(i+1)*W] for d in bs for i in range(d.shape[0]//W)]
def slope(w, W): return stats.linregress(np.arange(W), w[:, C1]).slope
def maxres(w): return np.max(np.abs(w[:, C3T] - (sl3*w[:, C3S] + ic3)))
def thr1(W): return np.percentile(np.abs([slope(w, W) for w in windows([cal], W)]), 99)
def drift(w, W, mag): v = w.copy(); v[:, C1] += np.linspace(0, mag*noise, W); return v
def swap(w): v = w.copy(); v[:, C3T] = w[:, C3S]; return v

# ---- (1) Select W and MAG_S1 on seeds 1-6 ----
A = builds[:6]
print('[seeds 1-6] power by W (MAG 1.6x); thresholds from the calibration series')
for W in [10, 20, 30, 50, 75, 100]:
    t = thr1(W); ws = windows(A, W)
    print(f'  W={W:>3d} windows={len(ws)} power={rd(np.mean([abs(slope(drift(w,W,1.6),W))>t for w in ws])):.3f}')
W = 50; t1 = thr1(W); wA = windows(A, W)
print('[seeds 1-6] W=50, power by MAG')
mags = [round(0.5+0.1*i, 1) for i in range(16)]
for m in mags:
    print(f'  MAG={m:.1f} power={rd(np.mean([abs(slope(drift(w,W,m),W))>t1 for w in wA])):.3f}')

# ---- (2) Evaluate on seeds 7-12 (W=50, MAG 1.6, S2 omits 1 point) ----
t3 = np.percentile([maxres(w) for w in windows([cal], W)], 99)
def l2mask():
    m = np.zeros(W, bool); n = max(1, int(W*0.02)); s = W//2 - n//2; m[s:s+n] = True; return m
def checks(w, mask): return (abs(slope(w, W)) > t1, bool(mask.any()), maxres(w) > t3)
def run(Wins, s1, s2, s3):
    r = np.array([checks(drift(swap(w) if s3 else w, W, 1.6) if s1 else (swap(w) if s3 else w),
                         l2mask() if s2 else np.zeros(W, bool)) for w in Wins], float).mean(0)
    return r
print('\n[seeds 7-12] rows: injection / columns: S1_check, S2_check, S3_check')
B = windows(builds[6:], W)
for name, f in {'normal': (0,0,0), 'S1': (1,0,0), 'S2': (0,1,0), 'S3': (0,0,1)}.items():
    print(f'  {name:>3s}', rda(run(B, *f), 3), f'(windows={len(B)})')

# ---- (3) Simultaneous injection (seeds 1-12, 720 windows) ----
ALL = windows(builds, W)
print('\n[simultaneous injection, 720 windows]')
for name, f in {'S1+S2': (1,1,0), 'S1+S3': (1,0,1), 'S2+S3': (0,1,1), 'S1+S2+S3': (1,1,1)}.items():
    r = run(ALL, *f); inj = np.array(f, bool)
    print(f'  {name:>8s}', rda(r, 3), 'diagonal', rda(r[inj], 3), 'off-diagonal', rda(r[~inj], 3))
