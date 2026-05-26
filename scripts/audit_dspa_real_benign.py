"""Stage 3: protocol-comparable QWK vs DSPA-MIL with REAL benign predictions.

Replaces the synthetic-benign assumption in audit_dspa_final_comparable.py
(lines 117-121) with actual model predictions on 118 TCGA-PRAD Solid-Tissue-
Normal slides extracted in Stage 2.

Inputs:
  G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv          (n=426 cancer)
  G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv   (n=118 benign)

Reports:
  - Real-benign QWK on the 20-random subset (apples-to-apples vs DSPA-MIL)
  - Real-benign QWK on the full 118-slide cohort (robustness check)
  - Decomposition delta vs the synthetic-95%-correct estimate (0.71)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

N_CLASSES = 6
DRIVE = Path(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD')
CANCER_PRED = DRIVE / 'predictions_20x.csv'
BENIGN_PRED = DRIVE / 'predictions_20x_benign.csv'

prob_cols = [f'prob_isup_{i}' for i in range(N_CLASSES)]
class_idx = np.arange(N_CLASSES)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                             labels=list(range(N_CLASSES)))


def e_isup_decision(probs, T=1.0):
    sl = np.log(np.clip(probs, 1e-10, 1.0)) / T
    sl -= sl.max(axis=1, keepdims=True)
    p = np.exp(sl)
    p /= p.sum(axis=1, keepdims=True)
    e = (p * class_idx).sum(axis=1)
    return np.clip(np.round(e).astype(int), 0, N_CLASSES - 1)


# ----------------------------------------------------------------------
#  Load cancer + benign predictions
# ----------------------------------------------------------------------
cancer_df = pd.read_csv(CANCER_PRED)
print(f'Loaded cancer:  {len(cancer_df):4d} slide-level predictions')
if not BENIGN_PRED.exists():
    sys.exit(f'\nERROR: {BENIGN_PRED} not found. Run Stage 2 (Colab benign '
             f'extraction) first.')
benign_df = pd.read_csv(BENIGN_PRED)
print(f'Loaded benign:  {len(benign_df):4d} slide-level predictions')


# ----------------------------------------------------------------------
#  Patient-level aggregation (both cohorts)
# ----------------------------------------------------------------------
def aggregate_by_patient(df):
    df = df.copy()
    df['patient_id'] = df['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')
    agg = df.groupby('patient_id').agg({
        'true_isup': lambda s: int(s.mode().iloc[0]),
        **{c: 'mean' for c in prob_cols},
    }).reset_index()
    return agg


cancer_p = aggregate_by_patient(cancer_df)
benign_p = aggregate_by_patient(benign_df)
print(f'Patient-level: cancer {len(cancer_p)} pts  /  benign {len(benign_p)} pts')


# ----------------------------------------------------------------------
#  5-fold CV temperature scaling on CANCER patients only
#  (matches the Stage 2 protocol so CV-T is not contaminated by benign)
# ----------------------------------------------------------------------
T_grid = np.linspace(0.5, 15.0, 100)
probs_c = np.asarray(cancer_p[prob_cols].values)
y_true_c = np.asarray(cancer_p['true_isup'].values, dtype=int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_qwks, cv_Ts = [], []
for ti, vi in skf.split(np.arange(len(cancer_p)), y_true_c):
    best_T, best_q = T_grid[0], -np.inf
    for T in T_grid:
        q = qwk(y_true_c[ti], e_isup_decision(probs_c[ti], T))
        if q > best_q:
            best_T, best_q = T, q
    pred = e_isup_decision(probs_c[vi], best_T)
    cv_qwks.append(qwk(y_true_c[vi], pred))
    cv_Ts.append(best_T)

T_global = float(np.mean(cv_Ts))
cancer_only_qwk_cvt = float(np.mean(cv_qwks))
print(f'\n[Cancer only, patient-level, CV-T + E[ISUP]]: '
      f'QWK = {cancer_only_qwk_cvt:.4f} +/- {np.std(cv_qwks):.4f}')
print(f'  per-fold T*: {[f"{t:.2f}" for t in cv_Ts]}  mean T* = {T_global:.2f}')


# ----------------------------------------------------------------------
#  Apply T_global to BOTH cancer (entire) and benign (entire) probs
# ----------------------------------------------------------------------
preds_c = e_isup_decision(probs_c, T_global)

probs_b = np.asarray(benign_p[prob_cols].values)
y_true_b = np.asarray(benign_p['true_isup'].values, dtype=int)
preds_b = e_isup_decision(probs_b, T_global)

# Benign-side classification accuracy at ISUP=0
benign_correct = int((preds_b == 0).sum())
benign_n = len(preds_b)
print(f'\n[Benign-side under same T*]:')
print(f'  patients predicted ISUP=0:   {benign_correct}/{benign_n}  '
      f'({100.0*benign_correct/benign_n:.1f}%)')
print(f'  predicted ISUP distribution:')
for k in range(N_CLASSES):
    n = int((preds_b == k).sum())
    print(f'    ISUP {k}: {n:3d} ({100.0*n/benign_n:.1f}%)')


# ----------------------------------------------------------------------
#  Protocol-comparable QWK vs DSPA-MIL
#  (A) full-benign cohort (118 benign + cancer) -- robustness check
#  (B) random-20 benign subset (apples-to-apples vs DSPA-MIL test composition)
# ----------------------------------------------------------------------
print('\n' + '=' * 75)
print('PROTOCOL-COMPARABLE QWK vs DSPA-MIL  (REAL benign, no synthesis)')
print('=' * 75)

y_full_A = np.concatenate([y_true_c, y_true_b])  # pyright: ignore[reportArgumentType]
p_full_A = np.concatenate([preds_c, preds_b])    # pyright: ignore[reportArgumentType]
qwk_full = qwk(y_full_A, p_full_A)
print(f'\n(A) FULL benign cohort  : n_cancer={len(y_true_c)}  n_benign={len(y_true_b)}')
print(f'    Protocol-comparable QWK = {qwk_full:.4f}')

# (B) random-20 benign sub-sample, seeds 1..20 for sensitivity
RANDOM_SEEDS = list(range(1, 21))
qwk_B_samples = []
for seed in RANDOM_SEEDS:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true_b), size=20, replace=False)
    y_sub = y_true_b[idx]
    p_sub = preds_b[idx]
    y_full_B = np.concatenate([y_true_c, y_sub])  # pyright: ignore[reportArgumentType]
    p_full_B = np.concatenate([preds_c, p_sub])   # pyright: ignore[reportArgumentType]
    qwk_B_samples.append(qwk(y_full_B, p_full_B))

qwk_B_mean = float(np.mean(qwk_B_samples))
qwk_B_std = float(np.std(qwk_B_samples))
qwk_B_lo, qwk_B_hi = float(np.percentile(qwk_B_samples, 2.5)), float(np.percentile(qwk_B_samples, 97.5))
print(f'\n(B) RANDOM 20 benign (seeds 1..20): n_cancer={len(y_true_c)}  n_benign=20')
print(f'    Protocol-comparable QWK = {qwk_B_mean:.4f} +/- {qwk_B_std:.4f}')
print(f'    95% range across seeds  : [{qwk_B_lo:.4f}, {qwk_B_hi:.4f}]')


# ----------------------------------------------------------------------
#  Compare to DSPA-MIL + synthetic-95% baseline
# ----------------------------------------------------------------------
print('\n' + '=' * 75)
print('COMPARISON to DSPA-MIL (Hao et al. MICCAI 2025)')
print('=' * 75)
print(f'  DSPA-MIL AB-MIL baseline      : 0.6770 +/- 0.021')
print(f'  DSPA-MIL CLAM-SB              : 0.6890 +/- 0.022')
print(f'  DSPA-MIL CLAM-MB              : 0.7270 +/- 0.020')
print(f'  DSPA-MIL (their proposed best): 0.7420 +/- 0.037')
print()
print(f'  Synthetic-95% benign estimate (S118):  0.71  (replaced)')
print(f'  REAL-benign full cohort  (A)  : {qwk_full:.4f}')
print(f'  REAL-benign random-20    (B)  : {qwk_B_mean:.4f}  '
      f'(95% range across seeds [{qwk_B_lo:.4f}, {qwk_B_hi:.4f}])')
print()
print(f'  Residual gap vs CLAM-MB (using B mean): {0.727 - qwk_B_mean:+.4f}')
print(f'  Residual gap vs AB-MIL  (using B mean): {0.677 - qwk_B_mean:+.4f}')
