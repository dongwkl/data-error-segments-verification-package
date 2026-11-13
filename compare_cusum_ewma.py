import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE   # top folder of the package (where baseline_long.pkl etc. are)
RES = os.path.join(ROOT, 'outputs')
os.makedirs(RES, exist_ok=True)
os.chdir(ROOT)
import numpy as np
import pandas as pd
from scipy import stats

# Data: same procedure as compare_3sigma_sweep.py (calibration R, 12 independent seeds, 720 windows)
df_cal = pd.read_pickle('baseline_long.pkl')
cal = df_cal.values
cols = df_cal.columns.tolist()
CH = cols.index('XMEAS(9)')

W = 50
x_idx = np.arange(W)
n_cal = cal.shape[0] // W
mu0 = cal[:, CH].mean()
sigma = cal[:, CH].std()


def inject(win, mag):
    w = win[:, CH].copy()
    w += np.linspace(0, mag * sigma, W)
    return w


# ---- Statistics: reset in every window; the maximum within the window is the decision statistic ----
def stat_slope(v):
    return abs(stats.linregress(x_idx, v).slope)


def make_cusum(k):
    def f(v):
        z = (v - mu0) / sigma
        up = lo = 0.0
        best = 0.0
        for zi in z:
            up = max(0.0, up + zi - k)
            lo = max(0.0, lo - zi - k)
            best = max(best, up, lo)
        return best
    return f


def make_ewma(lam):
    def f(v):
        z = (v - mu0) / sigma
        e = 0.0
        best = 0.0
        for zi in z:
            e = lam * zi + (1 - lam) * e
            best = max(best, abs(e))
        return best
    return f


methods = {'S1_check(slope)': stat_slope}
# Settings used in the manuscript: CUSUM k = 0.5 and EWMA lambda = 0.2
methods['CUSUM k=0.5'] = make_cusum(0.5)
methods['EWMA lam=0.2'] = make_ewma(0.2)

# Decision limit: same procedure as theta1 of S1_check (P99 of the calibration-window statistic)
theta = {}
for name, f in methods.items():
    vals = [f(cal[i * W:(i + 1) * W, CH]) for i in range(n_cal)]
    theta[name] = np.percentile(vals, 99)

# Windows of 12 independent seeds
wins = []
for run_id in range(1, 13):
    d = pd.read_pickle(f'tep2py/indep_builds/build_{run_id}/baseline.pkl').values
    for i in range(d.shape[0] // W):
        wins.append(d[i * W:(i + 1) * W, :])
n = len(wins)

mags = [0.5, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.4, 3.0]
print(f"n windows = {n}, calibration windows = {n_cal}")
print("Decision limit (P99 of the calibration windows):")
for name in methods:
    print(f"  {name}: {theta[name]:.4f}")

fpr = {}
for name, f in methods.items():
    fpr[name] = np.mean([f(w[:, CH]) > theta[name] for w in wins])
print("\nFalse positive rate (FPR, no injection):")
for name in methods:
    print(f"  {name}: {fpr[name]:.4f}")

print("\nDetection power (rows: m, columns: method)")
header = "m   " + " ".join(f"{nm:>18s}" for nm in methods)
print(header)
power = {nm: [] for nm in methods}
for mag in mags:
    row = []
    for name, f in methods.items():
        p = np.mean([f(inject(w, mag)) > theta[name] for w in wins])
        power[name].append(p)
        row.append(p)
    print(f"{mag:<4.1f}" + " ".join(f"{p:18.4f}" for p in row))

# Smallest m at which each method first exceeds 0.95 (same as the definition of m* in the manuscript)
print("\nm* = smallest m at which the power first reaches 0.95 (on the grid of the table)")
for name in methods:
    first = next((m for m, p in zip(mags, power[name]) if p >= 0.95), None)
    print(f"  {name}: {first}")


# ---- +-3 sigma rule for comparison (same definition as compare_3sigma_sweep.py: anomalous if any observation lies outside the calibration mean +- 3 calibration standard deviations) ----
def flag_3sigma(v):
    return bool(np.any(np.abs(v - mu0) > 3 * sigma))
S3SIG = '3sigma_rule'
fpr_3sigma = float(np.mean([flag_3sigma(w[:, CH]) for w in wins]))
power_3sigma = [float(np.mean([flag_3sigma(inject(w, mag)) for w in wins])) for mag in mags]
print(f"\n±3σ rule: FPR={fpr_3sigma:.4f}, m=1.6 power={power_3sigma[mags.index(1.6)]:.4f}")

# ---- Save the results ----
import csv
with open(os.path.join(RES, 'cusum_ewma_power.csv'), 'w', newline='', encoding='utf-8') as fh:
    wr = csv.writer(fh)
    wr.writerow(['m'] + list(methods) + [S3SIG])
    for i, mag in enumerate(mags):
        wr.writerow([mag] + [round(float(power[nm][i]), 4) for nm in methods] + [round(power_3sigma[i], 4)])
with open(os.path.join(RES, 'cusum_ewma_theta_fpr.csv'), 'w', newline='', encoding='utf-8') as fh:
    wr = csv.writer(fh)
    wr.writerow(['method', 'theta_P99_of_60_calibration_windows', 'FPR_720_windows', 'm_star'])
    for nm in methods:
        first = next((m for m, p in zip(mags, power[nm]) if p >= 0.95), '')
        wr.writerow([nm, round(float(theta[nm]), 6), round(float(fpr[nm]), 4), first])
    first3 = next((m for m, p in zip(mags, power_3sigma) if p >= 0.95), '')
    wr.writerow([S3SIG, '', round(fpr_3sigma, 4), first3])
