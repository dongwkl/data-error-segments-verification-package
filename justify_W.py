import numpy as np
from decimal import Decimal, ROUND_HALF_UP
def rd(x, d=3):
    # Round once from the unrounded value, half up: 27/720 = 0.0375 gives 0.038 (a plain format call can give 0.037)
    return float(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
import pandas as pd
from scipy import stats

df = pd.read_pickle('baseline_long.pkl')
data = df.values
cols = df.columns.tolist()
CH_S1 = cols.index('XMEAS(9)')
noise_std = data[:, CH_S1].std()
N = data.shape[0]

print('=== Varying the window length W: recompute the threshold -> inject the representative magnitude (1.6x) -> trade-off between detection power and number of windows ===\n')
print(f'{"W":>5s} {"win(min)":>8s} {"windows":>6s} {"θ1":>12s} {"power(1.6x)":>12s} {"FPR":>6s}')
for W in [10, 20, 30, 50, 75, 100, 150, 200]:
    n_windows = N // W
    windows = [data[i*W:(i+1)*W, CH_S1] for i in range(n_windows)]
    x_idx = np.arange(W)
    slopes_null = [stats.linregress(x_idx, w).slope for w in windows]
    thresh = np.percentile(np.abs(slopes_null), 99)

    def detect(w, mag):
        ww = w + np.linspace(0, mag*noise_std, W)
        slope, *_ = stats.linregress(x_idx, ww)
        return abs(slope) > thresh

    power = np.mean([detect(w, 1.6) for w in windows])
    fpr = np.mean([abs(stats.linregress(x_idx, w).slope) > thresh for w in windows])
    print(f'{W:>5d} {3*W:>8d} {n_windows:>6d} {thresh:>12.2e} {rd(power,2):>12.2f} {rd(fpr):>6.3f}')
