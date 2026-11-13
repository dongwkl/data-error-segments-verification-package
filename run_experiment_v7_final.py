# Correspondence between manuscript symbols and variable names
#   S1, S2, S3     : segments (sensing, transmission, mapping)   check_S1, check_S2, check_S3 : check procedures
#   theta1, theta3 : thresholds theta1, theta3                  alpha, gamma : regression slope alpha, intercept gamma
#   m_drift        : sensor drift multiple m                      omega_omit       : omission ratio omega
#   W              : window length                                (R = set of windows of W values cut from baseline_long.pkl)
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
def rd(x, d=3):
    # Round once from the unrounded value, half up: 27/720 = 0.0375 gives 0.038 (a plain format call can give 0.037)
    return float(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
def pfmt(p):
    return 'p < 0.001' if p < 0.001 else 'p = %.3f' % rd(p)
import pandas as pd
from scipy import stats
from scipy.stats import chi2

df_cal = pd.read_pickle('baseline_long.pkl')
cal = df_cal.values
cols = df_cal.columns.tolist()

CH_S1 = cols.index('XMEAS(9)')
CH_S3_TARGET = cols.index('XMEAS(13)')
CH_S3_SOURCE = cols.index('XMEAS(7)')

W = 50
x_idx = np.arange(W)
n_cal_windows = cal.shape[0] // W

# S1: one statistic per window (slope) -> 99th percentile of that statistic
slopes_null = [stats.linregress(x_idx, cal[i*W:(i+1)*W, CH_S1]).slope for i in range(n_cal_windows)]
theta1 = np.percentile(np.abs(slopes_null), 99)

# S3: same procedure as S1 -> one statistic per window (maximum absolute residual) -> 99th percentile of that statistic
alpha, gamma, *_ = stats.linregress(cal[:, CH_S3_SOURCE], cal[:, CH_S3_TARGET])
def window_max_resid(win):
    resid = win[:, CH_S3_TARGET] - (alpha * win[:, CH_S3_SOURCE] + gamma)
    return np.max(np.abs(resid))

maxresid_null = [window_max_resid(cal[i*W:(i+1)*W, :]) for i in range(n_cal_windows)]
theta3 = np.percentile(maxresid_null, 99)   # 99th percentile of the per-window statistic (same procedure as S1)

print('θ3 (99th percentile of the per-window maximum residual, same procedure as S1) = %.4f' % theta3)
noise_std_S1 = cal[:, CH_S1].std()

def check_S1(win):
    slope, *_ = stats.linregress(x_idx, win[:, CH_S1])
    return abs(slope) > theta1
def check_S3(win):
    return window_max_resid(win) > theta3
def check_S2(mask):
    return mask.any()
def inject_s1_drift(win, mag):
    w = win.copy(); w[:, CH_S1] += np.linspace(0, mag*noise_std_S1, W); return w
def inject_s2_omission(win, frac):
    n_missing = int(W*frac); mask = np.zeros(W, dtype=bool)
    start = W//2 - n_missing//2; mask[start:start+n_missing] = True
    return win.copy(), mask
def inject_s3_swap(win):
    w = win.copy(); w[:, CH_S3_TARGET] = win[:, CH_S3_SOURCE]; return w

all_windows = []
for run_id in range(1, 13):
    d = pd.read_pickle(f'tep2py/indep_builds/build_{run_id}/baseline.pkl').values
    n_w = d.shape[0] // W
    for i in range(n_w):
        all_windows.append((run_id, d[i*W:(i+1)*W, :]))

m_drift = 1.6
omega_omit = 0.02

results = {k: {'S1': [], 'S2': [], 'S3': []} for k in ['S1_drift', 'S2_omission', 'S3_swap', 'clean']}
for run_id, win in all_windows:
    results['clean']['S1'].append(check_S1(win)); results['clean']['S2'].append(False); results['clean']['S3'].append(check_S3(win))
    w1 = inject_s1_drift(win, m_drift)
    results['S1_drift']['S1'].append(check_S1(w1)); results['S1_drift']['S2'].append(False); results['S1_drift']['S3'].append(check_S3(w1))
    w2, mask = inject_s2_omission(win, omega_omit)
    results['S2_omission']['S1'].append(check_S1(w2)); results['S2_omission']['S2'].append(check_S2(mask)); results['S2_omission']['S3'].append(check_S3(w2))
    w3 = inject_s3_swap(win)
    results['S3_swap']['S1'].append(check_S1(w3)); results['S3_swap']['S2'].append(False); results['S3_swap']['S3'].append(check_S3(w3))

print('\n=== [v7 final, same procedure for S1 and S3 (99th percentile of the per-window statistic) + 12 independent repetitions (720 windows)] 3x3 matrix ===')
for k in ['S1_drift', 'S2_omission', 'S3_swap']:
    r = results[k]
    print(k, 'S1_check=%.3f' % rd(np.mean(r['S1'])), 'S2_check=%.3f' % rd(np.mean(r['S2'])), 'S3_check=%.3f' % rd(np.mean(r['S3'])))
print('clean(FPR) S1_check=%.3f S2_check=%.3f S3_check=%.3f' % (
    rd(np.mean(results['clean']['S1'])), rd(np.mean(results['clean']['S2'])), rd(np.mean(results['clean']['S3']))))

def mcnemar(a, b):
    a=np.array(a); b=np.array(b)
    n01=np.sum((~a)&b); n10=np.sum(a&(~b))
    if n01+n10==0: return 1.0,n01,n10
    stat=(abs(n01-n10)-1)**2/(n01+n10)
    return 1-chi2.cdf(stat,1), n01, n10

p1,n01,n10 = mcnemar(results['S1_drift']['S1'], results['S1_drift']['S3'])
print(f'\nMcNemar(S1-drift): n(S1 only)={n10}, n(S3 only)={n01}, {pfmt(p1)}')
p2,n01b,n10b = mcnemar(results['S3_swap']['S3'], results['S3_swap']['S1'])
print(f'McNemar(S3-swap): n(S3 only)={n10b}, n(S1 only)={n01b}, {pfmt(p2)}')
