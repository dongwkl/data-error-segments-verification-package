# Verification package

Scripts, data, and outputs that reproduce the numbers reported in the accompanying manuscript.
All text files are UTF-8.

## 1. Layout

```
verification_package/
├─ README.md
├─ LICENSE, THIRD_PARTY_NOTICES.md
├─ *.py                         13 scripts (Section 2)
├─ baseline_long.pkl            calibration data, 3,000 samples x 52 channels
├─ idv1.npy, idv6.npy, idv13.npy   process disturbance runs IDV(1), IDV(6), IDV(13)
├─ work/                        arrays of the 12 seeds for multi_seed_experiment.py and idv_level_shift_check.py
├─ outputs/                     stored outputs (CSV files and multi_seed_out.txt)
└─ tep2py/indep_builds/build_1 ... build_12/   simulator build and baseline.pkl of each seed
```

## 2. Scripts and the manuscript

| Script | Reads | Produces | Used for |
|---|---|---|---|
| `justify_W.py` | `baseline_long.pkl` | W, θ1, power at 1.6×, FPR | Window length (Section 5.1; Table S1) |
| `compare_3sigma_sweep.py` | `baseline_long.pkl`; 12 `baseline.pkl` | FPR; power for m = 0.5 to 3.0 | Drift magnitude; ±3σ comparison (Section 5.1; Table S2) |
| `run_experiment_v7_final.py` | `baseline_long.pkl`; 12 `baseline.pkl` | θ3; 3 × 3 matrix; McNemar tests | Isolation (Figure 3) |
| `robustness_W_sweep.py` | `baseline_long.pkl`; 12 `baseline.pkl` | Matrix for W = 30, 50, 70, 100 | Window length robustness (Section 5.2; Table S3) |
| `check_idv1_6_13.py` | `baseline_long.pkl`; `idv1/6/13.npy` | FPR of S1_check and S3_check, 16 windows | Process disturbances, single run (Table S5) |
| `pca_baseline_v1.py` | `baseline_long.pkl`; 12 `baseline.pkl` | 31-component PCA; T² and SPE rates | PCA comparison (Section 5.4; Table 6) |
| `compare_cusum_ewma.py` | `baseline_long.pkl`; 12 `baseline.pkl` | `outputs/cusum_ewma_power.csv`, `outputs/cusum_ewma_theta_fpr.csv`; thresholds, FPR, power, m* | CUSUM and EWMA, all 12 seeds (Table 2; Table S2) |
| `idv_cusum_ewma.py` | `idv1/6/13.npy` | FPR of all statistics, 16 windows | Process disturbances, single run (Table S5) |
| `prep_npy.py`, `gen_idv.py` | 12 seeds (build_1 to build_12) | Arrays of 12 seeds in `work/`; IDV(1), IDV(6), IDV(13) runs of 800 samples per seed (Linux only; optional, see Section 4) | Multi-seed disturbance data |
| `multi_seed_experiment.py` | arrays of the above | `outputs/multi_seed_out.txt`: seed-split evaluation, matched FPR, bootstrap of the FPR, mean-shift check, 12-seed disturbances, replaced-statistic matrix, per-seed paired tests | Tables 4 and 5; Tables S4 and S6 |
| `seed_split_and_combined.py` | `baseline_long.pkl`; 12 `baseline.pkl` | Selection on seeds 1 to 6, evaluation on seeds 7 to 12, four simultaneous-injection combinations | Section 5.2 |
| `idv_level_shift_check.py` | arrays of `gen_idv.py` | Shift of the XMEAS(9) window mean in sigma units, share above 1 sigma | Section 5.3 |

## 3. Symbols of the manuscript and variable names

| Manuscript | Script |
|---|---|
| S1, S2, S3 | `S1`, `S2`, `S3` (segments); `check_S1`, `check_S2`, `check_S3` (check procedures); `inject_s1_drift`, `inject_s2_omission`, `inject_s3_swap` (injection) |
| θ1, θ3 | `theta1`, `theta3` (thresholds) |
| α, γ | `alpha`, `gamma` (slope and intercept of the S3_check regression) |
| m | `m_drift` (drift multiple, 1.6) |
| ω | `omega_omit` (omission ratio, 0.02) |
| W | `W` (window length, 50) |
| R | set of non-overlapping windows of W values cut from `baseline_long.pkl` |

## 4. How to run

Requirements: Python 3 with pandas, numpy, scipy, scikit-learn.
The pickle files were saved with NumPy 2, so NumPy must be 2.0 or later:
`pip install -U numpy pandas scipy scikit-learn`. On Windows without `python3`, use `python`.
Run every script from the top folder of the package, except `idv_level_shift_check.py`, which is run in `work/` (see below).

Core scripts:

```
python justify_W.py
python compare_3sigma_sweep.py
python run_experiment_v7_final.py
python robustness_W_sweep.py
python check_idv1_6_13.py
python pca_baseline_v1.py
```

CUSUM and EWMA comparison with the two settings used in the manuscript (CUSUM k = 0.5 and EWMA λ = 0.2; a few seconds; writes `outputs/cusum_ewma_power.csv` and `outputs/cusum_ewma_theta_fpr.csv`):

```
python compare_cusum_ewma.py
python idv_cusum_ewma.py
```

Seed split and simultaneous injection:

```
python seed_split_and_combined.py
```

Multi-seed disturbance data and experiment. The folder `work/` holds the generated arrays, so any operating system with Python, NumPy, and SciPy can run these two scripts:

```
python multi_seed_experiment.py work
cd work
python ../idv_level_shift_check.py
cd ..
```

The output of `multi_seed_experiment.py` is identical to `outputs/multi_seed_out.txt`.

Optional regeneration of `work/` (Linux or WSL, Python 3.11, NumPy 1.26; the compiled modules are cpython-311 x86-64 Linux):

1. `python prep_npy.py <package folder> <work folder>` converts the pickle files to `.npy` (the simulator needs NumPy 1.26, which cannot read NumPy 2 pickles).
2. In each folder `tep2py/indep_builds/build_i` (i = 1 to 12), run
   `python <package folder>/gen_idv.py i <work folder>`.
   The simulator must be called with `npts = nx*180` so that the records are 3 minutes apart. The output agrees with the stored `baseline.pkl` with an error of 0.0.

Then run the two scripts above with `<work folder>` in place of `work`.

The stored `idv1/6/13.npy` come from seeds other than these 12 and cannot be reproduced; Table 5, Table S6, and the mean-shift check of Section 5.3 are therefore computed from the generated data in `work/`; Table S5 uses the stored files.

## 5. Notes

- The manuscript calls the 3 × 3 table of detection rates a confusion matrix. The output of `multi_seed_experiment.py` labels the same table "Isolation matrix".
- The ±3σ m* is 2.4 on the grid of `compare_cusum_ewma.py` (m = 0.5, 0.8, ..., 3.0) and 2.2 on the finer grid of `compare_3sigma_sweep.py`.
- Rounding: rates over 720, 360, or 192 windows have 3 decimals, rates over 16 windows 2 decimals, and p values 3 decimals (p < 0.001 for smaller values). Each value is rounded once from the unrounded value, and a tie rounds up (27/720 = 0.0375 gives 0.038). Function `rd` in each script does this.
- Pickle files can run code when opened. Open only the `.pkl` files of this package, which the author created.

## 6. Supplementary tables of the manuscript

Section 5 of the manuscript cites these tables as Table S1 to S6. The values are those printed by the scripts of Section 2, rounded as described in Section 5 (Notes).

### Table S1. Window length selection (calibration windows; drift of 1.6×)

| W | Window (min) | Windows | θ1 (°C per sample) | Power at 1.6× | FPR |
|---|---|---|---|---|---|
| 10 | 30 | 300 | 4.77 × 10⁻³ | 0.25 | 0.010 |
| 20 | 60 | 150 | 1.57 × 10⁻³ | 0.50 | 0.013 |
| 30 | 90 | 100 | 9.30 × 10⁻⁴ | 0.66 | 0.010 |
| 50 | 150 | 60 | 3.07 × 10⁻⁴ | 0.98 | 0.017 |
| 75 | 225 | 40 | 1.62 × 10⁻⁴ | 1.00 | 0.025 |
| 100 | 300 | 30 | 9.55 × 10⁻⁵ | 1.00 | 0.033 |
| 150 | 450 | 20 | 4.32 × 10⁻⁵ | 1.00 | 0.050 |
| 200 | 600 | 15 | 2.59 × 10⁻⁵ | 1.00 | 0.067 |

Source: justify_W.py output.

### Table S2. Sensor drift detection, all 12 seeds (720 windows)

| Method | θ | FPR | π(1.6) | m∗ | Note |
|---|---|---|---|---|---|
| S1_check (slope) | 3.07 × 10⁻⁴ | 0.038 | 0.985 | 1.5 |  |
| CUSUM k = 0.5 | 5.83 | 0.011 | 1.000 | 1.2 |  |
| EWMA λ = 0.2 | 1.01 | 0.028 | 1.000 | 1.2 |  |
| ±3σ rule | μ0 ± 3σ | 0.131 | 0.674 | 2.4 | grid 0.5, 0.8, 1.0, … |

Source: compare_cusum_ewma.py (CSV files cusum_ewma_power.csv and cusum_ewma_theta_fpr.csv) and compare_3sigma_sweep.py. The values here use all 12 seeds and are not a held-out evaluation. m∗ is read on the grid of m.

### Table S3. Isolation results for window lengths of 30 to 100 (all 12 seeds)

| W | Windows | S1 \| drift | S3 \| drift | S1 \| swap | S3 \| swap | S1 \| normal | S3 \| normal |
|---|---|---|---|---|---|---|---|
| 30 | 1200 | 0.656 | 0.021 | 0.007 | 1.000 | 0.007 | 0.021 |
| 50 | 720 | 0.985 | 0.018 | 0.038 | 1.000 | 0.038 | 0.018 |
| 70 | 504 | 0.994 | 0.020 | 0.014 | 1.000 | 0.014 | 0.020 |
| 100 | 360 | 1.000 | 0.028 | 0.022 | 1.000 | 0.022 | 0.028 |

Source: robustness_W_sweep.py output. Notation: S1 | drift is the detection rate of S1_check under the drift condition; swap is the channel mapping error. The regression (α, γ) is fixed and θ1 and θ3 are recomputed for each W.

### Table S4. Detection rates of the confusion matrix with three statistics for S1_check (seeds 7 to 12, 360 windows)

| S1 statistic | Condition | S1_check | S2_check | S3_check |
|---|---|---|---|---|
| Slope | S1 drift | 0.989 | 0.000 | 0.008 |
| Slope | S2 omission | 0.028 | 1.000 | 0.008 |
| Slope | S3 channel mapping error | 0.028 | 0.000 | 1.000 |
| Slope | Normal (FPR) | 0.028 | 0.000 | 0.008 |
| CUSUM (k = 0.5) | S1 drift | 1.000 | 0.000 | 0.008 |
| CUSUM (k = 0.5) | S2 omission | 0.014 | 1.000 | 0.008 |
| CUSUM (k = 0.5) | S3 channel mapping error | 0.014 | 0.000 | 1.000 |
| CUSUM (k = 0.5) | Normal (FPR) | 0.014 | 0.000 | 0.008 |
| EWMA (λ = 0.2) | S1 drift | 1.000 | 0.000 | 0.008 |
| EWMA (λ = 0.2) | S2 omission | 0.022 | 1.000 | 0.008 |
| EWMA (λ = 0.2) | S3 channel mapping error | 0.022 | 0.000 | 1.000 |
| EWMA (λ = 0.2) | Normal (FPR) | 0.022 | 0.000 | 0.008 |

Source: multi_seed_experiment.py. The 720-window matrix of the slope statistic is in Figure 3.

### Table S5. False positive rate under process disturbances, single run of the package (16 windows per disturbance)

| IDV | Description | S1_check | CUSUM (k = 0.5) | EWMA (λ = 0.2) | S3_check |
|---|---|---|---|---|---|
| IDV(1) | A/C ratio in the combined A and C feed, step | 0.13 | 0.25 | 0.19 | 0.38 |
| IDV(6) | A feed loss | 0.06 | 0.94 | 0.94 | 1.00 |
| IDV(13) | Reaction kinetics, slow drift | 0.31 | 0.50 | 0.50 | 0.94 |

Source: check_idv1_6_13.py and idv_cusum_ewma.py. The files idv1.npy, idv6.npy, and idv13.npy come from a seed other than these 12. The 12-seed results are Table 5 of the manuscript.

### Table S6. Per-seed paired comparison with S1_check (12 seeds; difference = rate of the chart minus rate of S1_check)

| IDV | Chart | Mean difference | 95% interval (bootstrap over seeds) | Seeds chart higher / lower | Sign test p |
|---|---|---|---|---|---|
| IDV(1) | CUSUM k = 0.5 | −0.057 | −0.083 to −0.026 | 1 / 9 | 0.021 |
| IDV(1) | EWMA λ = 0.2 | −0.047 | −0.078 to −0.010 | 2 / 9 | 0.065 |
| IDV(6) | CUSUM k = 0.5 | +0.505 | +0.292 to +0.703 | 9 / 1 | 0.021 |
| IDV(6) | EWMA λ = 0.2 | +0.385 | +0.161 to +0.615 | 8 / 3 | 0.227 |
| IDV(13) | CUSUM k = 0.5 | +0.068 | −0.068 to +0.203 | 6 / 5 | 1.000 |
| IDV(13) | EWMA λ = 0.2 | +0.031 | −0.073 to +0.135 | 5 / 4 | 1.000 |

Source: multi_seed_experiment.py. Each seed is one unit, with 16 windows per seed. The Bonferroni threshold of the six sign tests is 0.05/6 ≈ 0.008.

## 7. Simulation setup

**Simulator and software.** The simulator is the TEP Fortran code (Downs and Vogel, 1993) with the closed-loop control program modified by Russell et al. (2000) and Chiang et al. (2001), and with an interface subroutine added by Xavier and de Seixas (2018). The files of the tep2py repository were compiled with f2py and gfortran 13.3.0 into Python 3.11 extension modules for x86-64 Linux. The analysis was run with Python 3.13.18, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, and scikit-learn 1.9.1. NumPy 2.0 or later is needed to load the stored pickle files. The scripts read the stored data files and do not rerun the simulator, except `gen_idv.py`, which runs the simulator builds.

**Data.** Calibration: `baseline_long.pkl` (3,000 samples × 52 channels, 60 windows). Evaluation: `baseline.pkl` of each of the 12 builds (3,000 × 52 each, 720 windows). Process disturbances IDV(1), IDV(6), IDV(13), 800 samples (16 windows) each: `idv1.npy`, `idv6.npy`, `idv13.npy` (one run, from a seed other than these 12) and the 12-seed arrays in `work/` (192 windows per disturbance).

**Random-number seed.** Seed i is the constant G in `teprob.f` (line 1187), compiled into its own build `tep2py/indep_builds/build_i`. The builds differ only in G. Each build reproduces its stored `baseline.pkl` with a maximum difference of 0.0.

| Seed (build_i) | G | Seed (build_i) | G |
|---|---|---|---|
| 1 | 1431655765 | 7 | 6678322168 |
| 2 | 4243534565 | 8 | 7984782901 |
| 3 | 3456432354 | 9 | 8934302332 |
| 4 | 1731738903 | 10 | 9873223412 |
| 5 | 4346024432 | 11 | 1089278833 |
| 6 | 5784921734 | 12 | 5687912315 |
