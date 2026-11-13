import numpy as np
from decimal import Decimal, ROUND_HALF_UP
def rd(x, d=3):
    # Round once from the unrounded value, half up: 27/720 = 0.0375 gives 0.038 (a plain format call can give 0.037)
    return float(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
import pandas as pd
from scipy import stats

df_cal = pd.read_pickle('baseline_long.pkl')
cal = df_cal.values
cols = df_cal.columns.tolist()
CH_S1 = cols.index('XMEAS(9)')

W = 50
x_idx = np.arange(W)
n_cal_windows = cal.shape[0] // W

slopes_null = [stats.linregress(x_idx, cal[i*W:(i+1)*W, CH_S1]).slope for i in range(n_cal_windows)]
theta1 = np.percentile(np.abs(slopes_null), 99)
noise_std_S1 = cal[:, CH_S1].std()
center_S1 = cal[:, CH_S1].mean()

def check_S1(win):
    slope, *_ = stats.linregress(x_idx, win[:, CH_S1])
    return abs(slope) > theta1

def inject_s1_drift(win, mag):
    w = win.copy(); w[:, CH_S1] += np.linspace(0, mag * noise_std_S1, W); return w

UCL = center_S1 + 3 * noise_std_S1
LCL = center_S1 - 3 * noise_std_S1
def check_3sigma(win):
    return np.any((win[:, CH_S1] > UCL) | (win[:, CH_S1] < LCL))

all_windows = []
for run_id in range(1, 13):
    d = pd.read_pickle(f'tep2py/indep_builds/build_{run_id}/baseline.pkl').values
    n_w = d.shape[0] // W
    for i in range(n_w):
        all_windows.append(d[i * W:(i + 1) * W, :])

n = len(all_windows)

# FPR (both methods, on clean/un-injected windows -- same set, no injection)
s1_fpr = sum(check_S1(win) for win in all_windows) / n
sigma_fpr = sum(check_3sigma(win) for win in all_windows) / n

# Power sweep across m values, both methods, real recomputation (not hardcoded)
mags = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
s1_powers = []
sigma_powers = []
for mag in mags:
    s1_hits = 0
    sigma_hits = 0
    for win in all_windows:
        w1 = inject_s1_drift(win, mag)
        if check_S1(w1):
            s1_hits += 1
        if check_3sigma(w1):
            sigma_hits += 1
    s1_powers.append(s1_hits / n)
    sigma_powers.append(sigma_hits / n)

print(f"n windows = {n}")
print(f"S1_check FPR (design) = {s1_fpr:.4f}")
print(f"3-sigma FPR = {sigma_fpr:.4f}")
print("mag, S1_check power, 3sigma power")
for mag, p1, p2 in zip(mags, s1_powers, sigma_powers):
    print(f"{mag:.1f}, {p1:.4f}, {p2:.4f}")

print("\n[Note] Values at m=1.6x (reference values for the drift-magnitude figure and table of the manuscript):")
idx_16 = mags.index(1.6)
print(f"  S1_check: power={rd(s1_powers[idx_16]):.3f}, FPR={rd(s1_fpr):.3f}")
print(f"  Classical ±3σ: power={rd(sigma_powers[idx_16]):.3f}, FPR={rd(sigma_fpr):.3f}")
