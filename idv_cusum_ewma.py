from decimal import Decimal, ROUND_HALF_UP
def rd(x, d=3):
    # Round once from the unrounded value, half up: 27/720 = 0.0375 gives 0.038 (a plain format call can give 0.037)
    return float(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
import os, io, runpy, contextlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):   # the thresholds and methods of compare_cusum_ewma.py are reused; its own printout is not repeated
    ns = runpy.run_path(os.path.join(HERE, 'compare_cusum_ewma.py'))
methods, theta, CH, W = ns['methods'], ns['theta'], ns['CH'], ns['W']
print("\n=== False positive rate of the windows under IDV disturbances (XMEAS(9), calibrated thresholds unchanged) ===")
for k in (1, 6, 13):
    d = np.load(f'idv{k}.npy')
    wins = [d[i*W:(i+1)*W, :] for i in range(d.shape[0]//W)]
    out = [f"IDV({k}) n={len(wins)}"]
    for name, f in methods.items():
        out.append(f"{name}={rd(np.mean([f(w[:, CH]) > theta[name] for w in wins]),2):.2f}")
    print(" | ".join(out))
