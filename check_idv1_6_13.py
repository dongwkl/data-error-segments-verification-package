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

W = 50
x_idx = np.arange(W)
n_cal_windows = cal.shape[0] // W

slopes_null = [stats.linregress(x_idx, cal[i*W:(i+1)*W, CH_S1]).slope for i in range(n_cal_windows)]
theta1 = np.percentile(np.abs(slopes_null), 99)

alpha, gamma, *_ = stats.linregress(cal[:, CH_S3_SOURCE], cal[:, CH_S3_TARGET])
def window_max_resid(win):
    resid = win[:, CH_S3_TARGET] - (alpha * win[:, CH_S3_SOURCE] + gamma)
    return np.max(np.abs(resid))
maxresid_null = [window_max_resid(cal[i*W:(i+1)*W, :]) for i in range(n_cal_windows)]
theta3 = np.percentile(maxresid_null, 99)

def check_S1(win):
    slope, *_ = stats.linregress(x_idx, win[:, CH_S1])
    return abs(slope) > theta1
def check_S3(win):
    return window_max_resid(win) > theta3

print('=== False positive rate under real process disturbances IDV(1/6/13) (thresholds of the final v7) ===')
print('(Selection criteria already established in an earlier experiment: absorbed by the control loop / safety interlock / quality degradation only)\n')
for k, desc in [(1,'absorbed by the control loop'), (6,'safety interlock triggered'), (13,'quality degradation only, no interlock')]:
    d = np.load(f'idv{k}.npy')
    n_w = d.shape[0] // W
    wins = [d[i*W:(i+1)*W, :] for i in range(n_w)]
    s1_fp = np.mean([check_S1(w) for w in wins])
    s3_fp = np.mean([check_S3(w) for w in wins])
    print(f'IDV({k}) [{desc}]: S1_check FPR={rd(s1_fp,2):.2f}  S3_check FPR={rd(s3_fp,2):.2f}  (n_windows={n_w})')
