import numpy as np
from decimal import Decimal, ROUND_HALF_UP
def rd(x, d=3):
    # Round once from the unrounded value, half up: 27/720 = 0.0375 gives 0.038 (a plain format call can give 0.037)
    return float(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from sklearn.decomposition import PCA

df_cal = pd.read_pickle('baseline_long.pkl')
cal = df_cal.values
cols = df_cal.columns.tolist()

CH_S1 = cols.index('XMEAS(9)')
CH_S3_TARGET = cols.index('XMEAS(13)')
CH_S3_SOURCE = cols.index('XMEAS(7)')

W = 50
x_idx = np.arange(W)
n_cal_windows = cal.shape[0] // W

# ---- S1_check / S3_check thresholds (reproduced exactly as in run_experiment_v7_final.py) ----
slopes_null = [stats.linregress(x_idx, cal[i*W:(i+1)*W, CH_S1]).slope for i in range(n_cal_windows)]
theta1 = np.percentile(np.abs(slopes_null), 99)

alpha, gamma, *_ = stats.linregress(cal[:, CH_S3_SOURCE], cal[:, CH_S3_TARGET])
def window_max_resid(win):
    resid = win[:, CH_S3_TARGET] - (alpha * win[:, CH_S3_SOURCE] + gamma)
    return np.max(np.abs(resid))
maxresid_null = [window_max_resid(cal[i*W:(i+1)*W, :]) for i in range(n_cal_windows)]
theta3 = np.percentile(maxresid_null, 99)

noise_std_S1 = cal[:, CH_S1].std()

def check_S1(win):
    slope, *_ = stats.linregress(x_idx, win[:, CH_S1])
    return abs(slope) > theta1
def check_S3(win):
    return window_max_resid(win) > theta3
def inject_s1_drift(win, mag):
    w = win.copy(); w[:, CH_S1] += np.linspace(0, mag*noise_std_S1, W); return w
def inject_s3_swap(win):
    w = win.copy(); w[:, CH_S3_TARGET] = win[:, CH_S3_SOURCE]; return w

# ---- PCA baseline (Jackson 1991 / Montgomery 2013 convention) ----
mu = cal.mean(axis=0)
sigma = cal.std(axis=0)
sigma[sigma == 0] = 1.0
cal_std = (cal - mu) / sigma

pca_full = PCA().fit(cal_std)
cpv = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = int(np.searchsorted(cpv, 0.90) + 1)
pca = PCA(n_components=n_comp).fit(cal_std)
print(f'PCA: {n_comp} components retain {cpv[n_comp-1]*100:.1f}% cumulative variance (CPV>=90% rule, Jackson 1991)')

eigvals = pca.explained_variance_
def t2_spe(win_std):
    scores = pca.transform(win_std)
    recon = pca.inverse_transform(scores)
    t2 = np.sum((scores**2) / eigvals, axis=1)
    spe = np.sum((win_std - recon)**2, axis=1)
    return t2, spe

def window_stat(win_raw):
    win_std = (win_raw - mu) / sigma
    t2, spe = t2_spe(win_std)
    return t2.max(), spe.max()

cal_t2_max, cal_spe_max = [], []
for i in range(n_cal_windows):
    t2m, spem = window_stat(cal[i*W:(i+1)*W, :])
    cal_t2_max.append(t2m); cal_spe_max.append(spem)
T2_THRESH = np.percentile(cal_t2_max, 99)
SPE_THRESH = np.percentile(cal_spe_max, 99)
print(f'T2_THRESH(P99)={T2_THRESH:.3f}  SPE_THRESH(P99)={SPE_THRESH:.3f}')

def check_T2(win):
    t2m, _ = window_stat(win); return t2m > T2_THRESH
def check_SPE(win):
    _, spem = window_stat(win); return spem > SPE_THRESH

all_windows = []
for run_id in range(1, 13):
    d = pd.read_pickle(f'tep2py/indep_builds/build_{run_id}/baseline.pkl').values
    n_w = d.shape[0] // W
    for i in range(n_w):
        all_windows.append(d[i*W:(i+1)*W, :])

m_drift = 1.6
res = {k: {'S1':[], 'S3':[], 'T2':[], 'SPE':[]} for k in ['clean','S1_drift','S3_swap']}
for win in all_windows:
    res['clean']['S1'].append(check_S1(win)); res['clean']['S3'].append(check_S3(win))
    res['clean']['T2'].append(check_T2(win)); res['clean']['SPE'].append(check_SPE(win))
    w1 = inject_s1_drift(win, m_drift)
    res['S1_drift']['S1'].append(check_S1(w1)); res['S1_drift']['S3'].append(check_S3(w1))
    res['S1_drift']['T2'].append(check_T2(w1)); res['S1_drift']['SPE'].append(check_SPE(w1))
    w3 = inject_s3_swap(win)
    res['S3_swap']['S1'].append(check_S1(w3)); res['S3_swap']['S3'].append(check_S3(w3))
    res['S3_swap']['T2'].append(check_T2(w3)); res['S3_swap']['SPE'].append(check_SPE(w3))

print('\n=== S1_check/S3_check (proposed) vs PCA T2/SPE (baseline), detection rate ===')
print(f'{"condition":>10s} {"S1_check":>9s} {"S3_check":>9s} {"PCA_T2":>9s} {"PCA_SPE":>9s}')
for k in ['clean','S1_drift','S3_swap']:
    r = res[k]
    print(f'{k:>10s} {rd(np.mean(r["S1"])):>9.3f} {rd(np.mean(r["S3"])):>9.3f} {rd(np.mean(r["T2"])):>9.3f} {rd(np.mean(r["SPE"])):>9.3f}')

n = len(all_windows)
print(f'\nn_windows per condition = {n}')
