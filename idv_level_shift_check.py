# Post hoc check: does XMEAS(9) behave like a mean shift in the windows of the process disturbances (IDV)? (Section 5.3 of the manuscript)
# Usage: cd work; python ../idv_level_shift_check.py  (run in the folder that holds the arrays: work/ of the package, or a folder regenerated with prep_npy.py and gen_idv.py)
# Inputs (working directory): cal.npy (3000 x 52 calibration data), cols.txt (channel names, one per line),
#                             idv{1,6,13}_seed{1..12}.npy (800 x 52 each; produced by gen_idv.py)
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
def rd(x, d=3):
    # Round once from the unrounded value, half up: 27/720 = 0.0375 gives 0.038 (a plain format call can give 0.037)
    return float(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
cal = np.load('cal.npy')
names = [c.strip() for c in open('cols.txt').read().split('\n') if c.strip()]
c9 = names.index('XMEAS(9)')
mu0, sd = cal[:, c9].mean(), cal[:, c9].std()
for k in (1, 6, 13):
    shifts = []
    for s in range(1, 13):
        d = np.load(f'idv{k}_seed{s}.npy')[:, c9]
        shifts.append((d[:800].reshape(16, 50).mean(axis=1) - mu0) / sd)   # window mean minus mu0, in units of sigma
    a = np.array(shifts)
    print(f'IDV({k}): windows={a.size} min={a.min():.2f} max={a.max():.2f} share(|shift|>1 sigma)={rd(np.mean(np.abs(a) > 1),2):.2f}')
