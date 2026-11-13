# For each of the 12 independent TEP seeds (build_1..12), generate 800 rows (3-minute interval) with IDV(1), (6), (13) switched on from the start of the scenario
import sys
sys.path.insert(0,'.')
import numpy as np
import temain_mod
seed = int(sys.argv[1]); out = sys.argv[2]
nx = 800
for k in (1, 6, 13):
    idata = np.zeros((nx, 20), dtype=np.int32, order='F'); idata[:, k-1] = 1
    x = temain_mod.temain(nx*180, nx, idata, 1)
    np.save(f'{out}/idv{k}_seed{seed}.npy', x)
