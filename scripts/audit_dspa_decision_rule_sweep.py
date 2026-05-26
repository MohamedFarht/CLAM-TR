"""Decision-rule sensitivity sweep over the real-benign + cancer cohort.

Tries 7 decision rules and reports protocol-comparable QWK for each.
Helps choose the most defensible headline before drafting.

Rules tested:
  R1: Slide-level argmax (no aggregation, no temperature)
  R2: Slide-level E[ISUP] with T=1 (no temperature)
  R3: Patient-level argmax (mean probs per patient)
  R4: Patient-level E[ISUP] with T=1 (no temperature)
  R5: Patient-level E[ISUP] with cancer-only CV-T (current audit)
  R6: Patient-level E[ISUP] with FULL-COHORT CV-T (includes benign)
  R7: Patient-level argmax with class-balanced prior re-weighting
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

prob_cols = [f'prob_isup_{i}' for i in range(N_CLASSES)]
class_idx = np.arange(N_CLASSES)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                             labels=list(range(N_CLASSES)))


def e_isup(probs, T=1.0):
    sl = np.log(np.clip(probs, 1e-10, 1.0)) / T
    sl -= sl.max(axis=1, keepdims=True)
    p = np.exp(sl)
    p /= p.sum(axis=1, keepdims=True)
    e = (p * class_idx).sum(axis=1)
    return np.clip(np.round(e).astype(int), 0, N_CLASSES - 1)


def aggregate_by_patient(df):
    df = df.copy()
    df['patient_id'] = df['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')
    agg = df.groupby('patient_id').agg({
        'true_isup': lambda s: int(s.mode().iloc[0]),
        **{c: 'mean' for c in prob_cols},
    }).reset_index()
    return agg


def cv_tune_T(probs, y_true, n_splits=5, seed=42):
    """Fit temperature via 5-fold stratified CV maximizing QWK."""
    T_grid = np.linspace(0.5, 15.0, 100)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_qwks, cv_Ts = [], []
    for ti, vi in skf.split(np.arange(len(y_true)), y_true):
        best_T, best_q = T_grid[0], -np.inf
        for T in T_grid:
            q = qwk(y_true[ti], e_isup(probs[ti], T))
            if q > best_q:
                best_T, best_q = T, q
        cv_qwks.append(qwk(y_true[vi], e_isup(probs[vi], best_T)))
        cv_Ts.append(best_T)
    return float(np.mean(cv_Ts)), float(np.mean(cv_qwks)), float(np.std(cv_qwks))


# ----------------------------------------------------------------------
#  Load
# ----------------------------------------------------------------------
cancer_df = pd.read_csv(DRIVE / 'predictions_20x.csv')
benign_df = pd.read_csv(DRIVE / 'predictions_20x_benign.csv')
print(f'Loaded: cancer={len(cancer_df)} slides  benign={len(benign_df)} slides')

cancer_p = aggregate_by_patient(cancer_df)
benign_p = aggregate_by_patient(benign_df)
print(f'Patient-level: cancer={len(cancer_p)} pts  benign={len(benign_p)} pts\n')

# Pre-compute slide-level + patient-level arrays
probs_c_slide = np.asarray(cancer_df[prob_cols].values)
y_c_slide = np.asarray(cancer_df['true_isup'].values, dtype=int)
probs_b_slide = np.asarray(benign_df[prob_cols].values)
y_b_slide = np.asarray(benign_df['true_isup'].values, dtype=int)

probs_c_p = np.asarray(cancer_p[prob_cols].values)
y_c_p = np.asarray(cancer_p['true_isup'].values, dtype=int)
probs_b_p = np.asarray(benign_p[prob_cols].values)
y_b_p = np.asarray(benign_p['true_isup'].values, dtype=int)


# ----------------------------------------------------------------------
#  Tune CV-T variants
# ----------------------------------------------------------------------
T_cancer, _, _ = cv_tune_T(probs_c_p, y_c_p)
# Full-cohort CV-T: combine cancer + benign at patient level
probs_all_p = np.concatenate([probs_c_p, probs_b_p], axis=0)  # pyright: ignore[reportArgumentType]
y_all_p = np.concatenate([y_c_p, y_b_p])                       # pyright: ignore[reportArgumentType]
T_full, _, _ = cv_tune_T(probs_all_p, y_all_p)
print(f'CV-T (cancer-only):      T*={T_cancer:.2f}')
print(f'CV-T (full cohort):      T*={T_full:.2f}\n')


# ----------------------------------------------------------------------
#  Apply rules + report QWK on full + random-20 subset
# ----------------------------------------------------------------------
def rule_qwk(preds_c, y_c, preds_b, y_b):
    """Return (full_cohort_qwk, random20_mean_qwk, random20_std)."""
    full_q = qwk(np.concatenate([y_c, y_b]),                # pyright: ignore[reportArgumentType]
                 np.concatenate([preds_c, preds_b]))          # pyright: ignore[reportArgumentType]
    qwk_samples = []
    for seed in range(1, 21):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_b), size=20, replace=False)
        qwk_samples.append(qwk(np.concatenate([y_c, y_b[idx]]),    # pyright: ignore[reportArgumentType]
                              np.concatenate([preds_c, preds_b[idx]])))  # pyright: ignore[reportArgumentType]
    return full_q, float(np.mean(qwk_samples)), float(np.std(qwk_samples))


print('=' * 95)
print(f'{"Rule":55s}  {"Full":>8s}  {"Rand20 mean +/- std":>22s}  {"vs CLAM-MB":>11s}')
print('=' * 95)

# R1: slide-level argmax
preds_c = cancer_df['pred_isup'].values.astype(int)
preds_b = benign_df['pred_isup'].values.astype(int)
f, m, s = rule_qwk(preds_c, y_c_slide, preds_b, y_b_slide)
print(f'{"R1: slide-level argmax (no agg, no T)":55s}  {f:8.4f}  {m:8.4f} +/- {s:6.4f}     {0.727-m:+8.4f}')

# R2: slide-level E[ISUP] T=1
preds_c = e_isup(probs_c_slide, T=1.0)
preds_b = e_isup(probs_b_slide, T=1.0)
f, m, s = rule_qwk(preds_c, y_c_slide, preds_b, y_b_slide)
print(f'{"R2: slide-level E[ISUP] T=1":55s}  {f:8.4f}  {m:8.4f} +/- {s:6.4f}     {0.727-m:+8.4f}')

# R3: patient-level argmax
preds_c = probs_c_p.argmax(axis=1)
preds_b = probs_b_p.argmax(axis=1)
f, m, s = rule_qwk(preds_c, y_c_p, preds_b, y_b_p)
print(f'{"R3: patient-level argmax":55s}  {f:8.4f}  {m:8.4f} +/- {s:6.4f}     {0.727-m:+8.4f}')

# R4: patient-level E[ISUP] T=1
preds_c = e_isup(probs_c_p, T=1.0)
preds_b = e_isup(probs_b_p, T=1.0)
f, m, s = rule_qwk(preds_c, y_c_p, preds_b, y_b_p)
print(f'{"R4: patient-level E[ISUP] T=1":55s}  {f:8.4f}  {m:8.4f} +/- {s:6.4f}     {0.727-m:+8.4f}')

# R5: patient-level E[ISUP] cancer-CV-T
preds_c = e_isup(probs_c_p, T=T_cancer)
preds_b = e_isup(probs_b_p, T=T_cancer)
f, m, s = rule_qwk(preds_c, y_c_p, preds_b, y_b_p)
print(f'{f"R5: patient-level E[ISUP] cancer-CV-T (T={T_cancer:.2f})":55s}  {f:8.4f}  {m:8.4f} +/- {s:6.4f}     {0.727-m:+8.4f}')

# R6: patient-level E[ISUP] full-cohort CV-T
preds_c = e_isup(probs_c_p, T=T_full)
preds_b = e_isup(probs_b_p, T=T_full)
f, m, s = rule_qwk(preds_c, y_c_p, preds_b, y_b_p)
print(f'{f"R6: patient-level E[ISUP] full-cohort CV-T (T={T_full:.2f})":55s}  {f:8.4f}  {m:8.4f} +/- {s:6.4f}     {0.727-m:+8.4f}')

# R7: patient-level argmax with uniform-prior reweighting (counteract cancer-class skew in PANDA training)
# Apply per-class prior correction: new_prob = prob / panda_prior; renormalize.
# PANDA ISUP distribution: roughly [0:0.34, 1:0.25, 2:0.13, 3:0.10, 4:0.10, 5:0.08]
# (approximate from CLAM-TR PANDA Subset 9128 noise-cleaned: equivalent to v20)
panda_prior = np.array([0.34, 0.25, 0.13, 0.10, 0.10, 0.08])
uniform_prior = np.full(N_CLASSES, 1.0 / N_CLASSES)
ratio = uniform_prior / panda_prior

probs_c_p_rw = probs_c_p * ratio[np.newaxis, :]
probs_c_p_rw = probs_c_p_rw / probs_c_p_rw.sum(axis=1, keepdims=True)
probs_b_p_rw = probs_b_p * ratio[np.newaxis, :]
probs_b_p_rw = probs_b_p_rw / probs_b_p_rw.sum(axis=1, keepdims=True)
preds_c = probs_c_p_rw.argmax(axis=1)
preds_b = probs_b_p_rw.argmax(axis=1)
f, m, s = rule_qwk(preds_c, y_c_p, preds_b, y_b_p)
print(f'{"R7: patient-level argmax + prior-correction":55s}  {f:8.4f}  {m:8.4f} +/- {s:6.4f}     {0.727-m:+8.4f}')

print()
print('=' * 95)
print('Reference:')
print(f'  DSPA-MIL AB-MIL baseline      : 0.6770 +/- 0.021')
print(f'  DSPA-MIL CLAM-SB              : 0.6890 +/- 0.022')
print(f'  DSPA-MIL CLAM-MB              : 0.7270 +/- 0.020')
print(f'  DSPA-MIL (their proposed best): 0.7420 +/- 0.037')
