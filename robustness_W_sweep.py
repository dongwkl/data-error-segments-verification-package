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
CH_S3_TARGET = cols.index('XMEAS(13)')
CH_S3_SOURCE = cols.index('XMEAS(7)')

m_drift = 1.6
noise_std_S1 = cal[:, CH_S1].std()
alpha_full, gamma_full, *_ = stats.linregress(cal[:, CH_S3_SOURCE], cal[:, CH_S3_TARGET])

all_builds = [pd.read_pickle(f'tep2py/indep_builds/build_{i}/baseline.pkl').values for i in range(1, 13)]

print(f'{"W":>5s} {"n_win":>6s} {"S1|drift":>9s} {"S3|drift":>9s} {"S1|swap":>9s} {"S3|swap":>9s} {"S1|clean":>9s} {"S3|clean":>9s}')
for W in [30, 50, 70, 100]:
    x_idx = np.arange(W)
    n_cal_windows = cal.shape[0] // W

    slopes_null = [stats.linregress(x_idx, cal[i*W:(i+1)*W, CH_S1]).slope for i in range(n_cal_windows)]
    theta1 = np.percentile(np.abs(slopes_null), 99)

    def window_max_resid(win):
        resid = win[:, CH_S3_TARGET] - (alpha_full * win[:, CH_S3_SOURCE] + gamma_full)
        return np.max(np.abs(resid))
    maxresid_null = [window_max_resid(cal[i*W:(i+1)*W, :]) for i in range(n_cal_windows)]
    theta3 = np.percentile(maxresid_null, 99)

    def check_S1(win):
        slope, *_ = stats.linregress(x_idx, win[:, CH_S1])
        return abs(slope) > theta1
    def check_S3(win):
        return window_max_resid(win) > theta3
    def inject_s1_drift(win, mag):
        w = win.copy(); w[:, CH_S1] += np.linspace(0, mag*noise_std_S1, W); return w
    def inject_s3_swap(win):
        w = win.copy(); w[:, CH_S3_TARGET] = win[:, CH_S3_SOURCE]; return w

    all_windows = []
    for d in all_builds:
        n_w = d.shape[0] // W
        for i in range(n_w):
            all_windows.append(d[i*W:(i+1)*W, :])

    s1_drift, s3_drift, s1_swap, s3_swap, s1_clean, s3_clean = [], [], [], [], [], []
    for win in all_windows:
        s1_clean.append(check_S1(win)); s3_clean.append(check_S3(win))
        w1 = inject_s1_drift(win, m_drift)
        s1_drift.append(check_S1(w1)); s3_drift.append(check_S3(w1))
        w3 = inject_s3_swap(win)
        s1_swap.append(check_S1(w3)); s3_swap.append(check_S3(w3))

    print(f'{W:>5d} {len(all_windows):>6d} '
          f'{rd(np.mean(s1_drift)):>9.3f} {rd(np.mean(s3_drift)):>9.3f} '
          f'{rd(np.mean(s1_swap)):>9.3f} {rd(np.mean(s3_swap)):>9.3f} '
          f'{rd(np.mean(s1_clean)):>9.3f} {rd(np.mean(s3_clean)):>9.3f}')
