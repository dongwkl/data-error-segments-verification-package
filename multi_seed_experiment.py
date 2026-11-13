# Multi-seed experiment: S1_check vs CUSUM (k = 0.5) and EWMA (lambda = 0.2) (calibration seeds 1-6, evaluation seeds 7-12) and false positive rates under process disturbances on 12 seeds
# The data path is passed as an argument: python multi_seed_experiment.py <work_dir>
import sys
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
def rd(x, d=3):
    # Round once from the unrounded value, half up: 27/720 = 0.0375 gives 0.038 (a plain format call can give 0.037)
    return float(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
from scipy import stats
from scipy.stats import binomtest

D = sys.argv[1].rstrip('/') + '/'
rng = np.random.default_rng(20261030)

cal = np.load(D + 'cal.npy')
cols = open(D + 'cols.txt').read().split('\n')
CH1 = cols.index('XMEAS(9)')
CH3T, CH3S = cols.index('XMEAS(13)'), cols.index('XMEAS(7)')
W = 50
x_idx = np.arange(W)


def windows(a):
    n = a.shape[0] // W
    return a[:n * W].reshape(n, W, a.shape[1])


calw = windows(cal)                                    # 60 windows (R)
seed_w = {s: windows(np.load(D + f'seed{s}.npy')) for s in range(1, 13)}
SEL = np.concatenate([seed_w[s] for s in range(1, 7)])    # selection seeds 1-6 (360)
EVA = np.concatenate([seed_w[s] for s in range(7, 13)])   # evaluation seeds 7-12 (360)

mu0, sigma = cal[:, CH1].mean(), cal[:, CH1].std()


# ---------- Statistics (window array (n,W) -> (n,)) ----------
def st_slope(v):
    xc = x_idx - x_idx.mean()
    return np.abs((v * xc).sum(1) / (xc ** 2).sum())


def make_cusum(k):
    def f(v):
        z = (v - mu0) / sigma
        up = np.zeros(len(z)); lo = np.zeros(len(z)); best = np.zeros(len(z))
        for t in range(W):
            up = np.maximum(0, up + z[:, t] - k); lo = np.maximum(0, lo - z[:, t] - k)
            best = np.maximum(best, np.maximum(up, lo))
        return best
    return f


def make_ewma(lam):
    def f(v):
        z = (v - mu0) / sigma
        e = np.zeros(len(z)); best = np.zeros(len(z))
        for t in range(W):
            e = lam * z[:, t] + (1 - lam) * e
            best = np.maximum(best, np.abs(e))
        return best
    return f


def st_3sigma(v):  # Largest |z| of a single observation beyond +-3 sigma; the limit is fixed at 3
    return np.abs((v - mu0) / sigma).max(1)


# Settings used in the manuscript: CUSUM k = 0.5 and EWMA lambda = 0.2
methods = {'S1_check': st_slope, 'CUSUM k=0.5': make_cusum(0.5), 'EWMA l=0.2': make_ewma(0.2)}


def ch(w):  # S1 channel of the window array (n,W)
    return w[:, :, CH1]


def inj(v, m):  # Linear drift 0 -> m*sigma
    return v + np.linspace(0, m * sigma, W)[None, :] * 1.0


MS = [0.5, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.4, 3.0]
DESIGN_M = 1.6

# ---------- Precompute the statistics ----------
cache = {}
for name, f in methods.items():
    cache[name] = {
        'cal': f(ch(calw)),
        'SEL': {None: f(ch(SEL)), **{m: f(inj(ch(SEL), m)) for m in MS}},
        'EVA': {None: f(ch(EVA)), **{m: f(inj(ch(EVA), m)) for m in MS}},
    }
th_paper = {n: np.percentile(cache[n]['cal'], 99) for n in methods}   # Procedure of the manuscript: P99 of the 60 calibration windows


def rate(arr, th):
    return float(np.mean(arr > th))


print('=' * 70)
print('[0] Data: %d calibration windows, %d selection (seeds 1-6), %d evaluation (seeds 7-12)' % (len(calw), len(SEL), len(EVA)))
# Reproduction check: compare with Appendix A (all 720 windows of seeds 1-12, m=1.6)
ALL = np.concatenate([SEL, EVA])
sl_ref = np.array([abs(stats.linregress(x_idx, w[:, CH1]).slope) for w in ALL[:20]])
assert np.allclose(sl_ref, st_slope(ch(ALL[:20]))), 'slope implementation mismatch'
print('[0] Reproduction check (reference values: S1 FPR 0.0375/power 0.9847, CUSUM k=0.5 0.011/1.000, EWMA l=0.2 0.028/1.000)')
for n in ('S1_check', 'CUSUM k=0.5', 'EWMA l=0.2'):
    f = methods[n]
    print('    %-12s FPR %.4f  pw@1.6 %.4f' % (n, rate(f(ch(ALL)), th_paper[n]), rate(f(inj(ch(ALL), 1.6)), th_paper[n])))

# ---------- (1) Chart settings: one conventional setting per chart (k = 0.5, lambda = 0.2) ----------
# The values are the conventional settings of the literature (Montgomery, 2013) and are not selected from the data.
SEL_MS = [0.8, 1.0, 1.2, 1.4, 1.6]
print('\n[1] Normal-window FPR and mean power on the selection seeds (1-6) for each statistic (manuscript-procedure limit: calibration P99)')
print('%-14s %8s %10s' % ('method', 'FPR_sel', 'meanPow'))
for n in methods:
    fpr = rate(cache[n]['SEL'][None], th_paper[n])
    mp = np.mean([rate(cache[n]['SEL'][m], th_paper[n]) for m in SEL_MS])
    print('%-14s %8.4f %10.4f' % (n, fpr, mp))

SEL_CUSUM, SEL_EWMA = 'CUSUM k=0.5', 'EWMA l=0.2'
print('Fixed settings ->', SEL_CUSUM, ',', SEL_EWMA)
main = ['S1_check', SEL_CUSUM, SEL_EWMA]

# ---------- (2) Evaluation seeds 7-12: limit from the manuscript procedure ----------
def table_eval(th, title):
    print('\n' + title)
    print('%-14s %7s   m*(>=0.95)' % ('method', 'FPR'))
    for n in methods:
        fpr = rate(cache[n]['EVA'][None], th[n])
        first = next((m for m in MS if rate(cache[n]['EVA'][m], th[n]) >= 0.95), None)
        print(' %-13s %7.4f   %s' % (n, fpr, first))
    z = np.abs((ch(EVA) - mu0) / sigma).max(1)
    print(' ±3σ rule      %7.4f' % np.mean(z > 3))


table_eval(th_paper, '[2] Evaluation seeds 7-12, decision limit = calibration-window P99 (manuscript procedure)')

# ---------- (3) Same-FPR condition: set the limit from quantiles of the normal statistics of selection seeds 1-6 ----------
alpha = 0.05
th_eq = {n: np.quantile(cache[n]['SEL'][None], 1 - alpha) for n in methods}
table_eval(th_eq, '[3] Matched-FPR condition (limit = quantile of the normal statistics of seeds 1-6 that leaves %d%% above it -> target FPR=%.2f), evaluation seeds 7-12' % (alpha * 100, alpha))

# ---------- (4) Bootstrap of the calibration windows: instability of the limit ----------
B = 2000
print('\n[4] Bootstrap of the 60 calibration windows (B=%d): effect of the uncertainty of the P99 limit on the FPR on evaluation seeds 7-12' % B)
print('%-14s %-26s' % ('method', 'FPR median [95%PI]'))
for n in main:
    f_ = []
    for _ in range(B):
        idx = rng.integers(0, len(calw), len(calw))
        th = np.percentile(cache[n]['cal'][idx], 99)
        f_.append(rate(cache[n]['EVA'][None], th))
    fmt = lambda a: '%.3f [%.3f, %.3f]' % (np.median(a), np.percentile(a, 2.5), np.percentile(a, 97.5))
    print('%-14s %-26s' % (n, fmt(f_)))

# ---------- (5) Level-shift check (constant +0.8 sigma shift, evaluation seeds 7-12) ----------
print('\n[5] Level-shift check: constant +0.8σ shift (evaluation seeds 7-12, manuscript-procedure limit)')
for n in main:
    v = ch(EVA) + 0.8 * sigma
    print('%-14s detection rate %.3f' % (n, rd(rate(methods[n](v), th_paper[n]))))

# ---------- (6) Tables 5 and S5: FPR under process disturbances, multiple seeds ----------
alpha_s, gamma_s = stats.linregress(cal[:, CH3S], cal[:, CH3T])[:2]


def st_s3(w):
    return np.abs(w[:, :, CH3T] - (alpha_s * w[:, :, CH3S] + gamma_s)).max(1)


th3 = np.percentile(st_s3(calw), 99)


print('\n[6] Anomaly rate in process-disturbance windows (16 windows per seed)')
idv = {k: {s: windows(np.load(D + f'idv{k}_seed{s}.npy')) for s in range(1, 13)} for k in (1, 6, 13)}
print('%-9s %-14s %6s %6s  %s' % ('disturbance', 'method', 'k/n', 'rate', 'range over seeds'))
seeds = range(1, 13)
for k in (1, 6, 13):
    w = np.concatenate([idv[k][s] for s in seeds])
    for n in main + ['S3_check']:
        if n == 'S3_check':
            hit = st_s3(w) > th3
            per = [np.mean(st_s3(idv[k][s]) > th3) for s in seeds]
        else:
            hit = methods[n](ch(w)) > th_paper[n]
            per = [np.mean(methods[n](ch(idv[k][s])) > th_paper[n]) for s in seeds]
        print('IDV(%-2d)   %-14s %3d/%-3d %5.3f  %.2f~%.2f' % (k, n, hit.sum(), len(hit), rd(hit.mean()), rd(min(per), 2), rd(max(per), 2)))

# ---------- (7) Isolation matrix: does it hold when the S1 check statistic is replaced? ----------
print('\n[7] Isolation matrix (evaluation seeds 7-12, manuscript-procedure limit). Only the S1 check statistic is replaced; the S2/S3 checks are unchanged')
omega = 0.02


def run_matrix(s1_name):
    f1 = methods[s1_name]; t1 = th_paper[s1_name]
    out = {}
    clean = EVA
    w1 = EVA.copy(); w1[:, :, CH1] = inj(ch(EVA), DESIGN_M)
    w3 = EVA.copy(); w3[:, :, CH3T] = EVA[:, :, CH3S]
    n_miss = int(W * omega)
    for lab, w, s2 in (('S1_drift', w1, np.zeros(len(EVA), bool)), ('S2_omission', EVA, np.ones(len(EVA), bool) if n_miss > 0 else np.zeros(len(EVA), bool)),
                       ('S3_swap', w3, np.zeros(len(EVA), bool)), ('clean(FPR)', clean, np.zeros(len(EVA), bool))):
        out[lab] = (np.mean(f1(ch(w)) > t1), np.mean(s2), np.mean(st_s3(w) > th3))
    return out


for n in main:
    print('\nS1 statistic = %s' % n)
    for lab, (a, b, c) in run_matrix(n).items():
        print('  %-12s S1_check=%.3f  S2_check=%.3f  S3_check=%.3f' % (lab, rd(a), rd(b), rd(c)))

# ---------- (8) Seed-level check (windows of one seed are correlated, so aggregate again with the seed as the unit) ----------
print('\n[8] Seed-level aggregation: paired difference of the per-seed rates (16 windows) (other method - S1_check), all seeds 1-12')
for k in (1, 6, 13):
    for other in (SEL_CUSUM, SEL_EWMA):
        d = np.array([np.mean(methods[other](ch(idv[k][s])) > th_paper[other]) - np.mean(methods['S1_check'](ch(idv[k][s])) > th_paper['S1_check']) for s in range(1, 13)])
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        p = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        print('IDV(%-2d) %-12s mean diff %+.3f  95%%CI (seed bootstrap) [%+.3f, %+.3f]  other>S1 in %d seeds, other<S1 in %d seeds, sign test p=%.3f' % (k, other, rd(d.mean()), np.percentile(bs, 2.5), np.percentile(bs, 97.5), pos, neg, rd(p)))
