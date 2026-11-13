# Convert the pkl files of the verification package to npy (the pkl files were saved with NumPy 2 and cannot be read in the NumPy 1.26 environment needed by the simulator)
# Usage: python prep_npy.py <verification_package path> <output folder>
import sys, numpy as np, pandas as pd
root, out = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'
df = pd.read_pickle(root + 'baseline_long.pkl')
np.save(out + 'cal.npy', df.values); open(out + 'cols.txt', 'w').write('\n'.join(df.columns))
for i in range(1, 13):
    np.save(out + f'seed{i}.npy', pd.read_pickle(root + f'tep2py/indep_builds/build_{i}/baseline.pkl').values)
