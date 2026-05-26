"""Protocol comparability audit against DSPA-MIL (Hao et al. MICCAI 2025).

Compares our predictions_20x.csv against DSPA-MIL's reported setup:
  DSPA-MIL: n=449 TCGA-PRAD, includes 20 BENIGN, patient-level prediction, 20x
  Ours:     n=426 TCGA-PRAD, no benign, per-slide prediction, 20x

Tests four protocol axes locally (no Colab):
  1. Patient-level aggregation vs per-slide  (group by patient ID, average probs)
  2. Multiple-slide aggregation rules (mean, max, first-DX-only)
  3. Synthetic benign inclusion (counterfactual: add benign cases the model already
     gets ~right; estimate QWK boost from class-space alignment)
  4. Decision rules under each above (argmax vs E[ISUP])

Goal: quantify how much of the 0.74 vs 0.62 gap is PROTOCOL vs METHODOLOGY.
"""
import pandas as pd
import numpy as np
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

N_CLASSES = 6
df = pd.read_csv(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv')
print(f'Loaded: {len(df)} per-slide predictions')

# Extract patient ID from slide_id
# TCGA slide IDs: TCGA-2A-A8VL-01Z-00-DX1.XXX -> patient = TCGA-2A-A8VL
df['patient_id'] = df['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')
df['dx_suffix'] = df['slide_id'].str.extract(r'-(DX\d+)\.')
print(f'Unique patients: {df["patient_id"].nunique()}')
print(f'Slides per patient distribution:')
print(df.groupby('patient_id').size().value_counts().sort_index())
print()

prob_cols = [f'prob_isup_{i}' for i in range(N_CLASSES)]
class_idx = np.arange(N_CLASSES)

def qwk(y_t, y_p):
    return cohen_kappa_score(y_t, y_p, weights='quadratic', labels=list(range(N_CLASSES)))

# ── Baseline: per-slide ──
print('=' * 70)
print('BASELINE: per-slide evaluation (our current setup, n=426)')
print('=' * 70)
y_argmax = df['pred_isup'].values
y_true = df['true_isup'].values
probs = df[prob_cols].values
e_isup = np.clip(np.round((probs * class_idx).sum(axis=1)).astype(int), 0, 5)

qwk_argmax = qwk(y_true, y_argmax)
qwk_eisup = qwk(y_true, e_isup)
print(f'  per-slide argmax: QWK = {qwk_argmax:.4f}')
print(f'  per-slide E[ISUP]: QWK = {qwk_eisup:.4f}')

# ── Audit 1: Patient-level aggregation ──
print()
print('=' * 70)
print('AUDIT 1: PATIENT-LEVEL AGGREGATION (matches DSPA-MIL "patient-level")')
print('=' * 70)

# Strategy A: average probabilities across patient's slides, then argmax/E[ISUP]
patient_groups = df.groupby('patient_id')
print(f'  Aggregating {len(df)} slides into {len(patient_groups)} patients...')

agg_records = []
for pid, group in patient_groups:
    # Average probabilities across this patient's slides
    avg_probs = group[prob_cols].mean(axis=0).values
    # True label: should be consistent across patient's slides; take the first
    true_isups = group['true_isup'].unique()
    if len(true_isups) > 1:
        # Different labels across slides — this WOULD be a data issue. Take mode.
        true = int(group['true_isup'].mode().iloc[0])
    else:
        true = int(true_isups[0])
    pred_argmax = int(np.argmax(avg_probs))
    pred_eisup = int(np.clip(round((avg_probs * class_idx).sum()), 0, 5))
    agg_records.append({
        'patient_id': pid,
        'n_slides': len(group),
        'true_isup': true,
        'pred_argmax': pred_argmax,
        'pred_eisup': pred_eisup,
        **{f'avg_prob_{i}': avg_probs[i] for i in range(N_CLASSES)},
    })
agg_df = pd.DataFrame(agg_records)
y_true_p = agg_df['true_isup'].values
y_argmax_p = agg_df['pred_argmax'].values
y_eisup_p = agg_df['pred_eisup'].values
qwk_argmax_p = qwk(y_true_p, y_argmax_p)
qwk_eisup_p = qwk(y_true_p, y_eisup_p)
print(f'  patient-level (mean-prob aggregation): n={len(agg_df)}')
print(f'  patient-level argmax: QWK = {qwk_argmax_p:.4f}  (delta vs slide: {qwk_argmax_p - qwk_argmax:+.4f})')
print(f'  patient-level E[ISUP]: QWK = {qwk_eisup_p:.4f}  (delta vs slide: {qwk_eisup_p - qwk_eisup:+.4f})')

# Strategy B: use only the DX1 slide (canonical "primary" diagnostic slide)
print()
dx1_mask = df['dx_suffix'] == 'DX1'
dx1_df = df[dx1_mask].copy()
print(f'  DX1-only subset: n={len(dx1_df)}')
y_argmax_dx1 = dx1_df['pred_isup'].values
y_true_dx1 = dx1_df['true_isup'].values
probs_dx1 = dx1_df[prob_cols].values
e_isup_dx1 = np.clip(np.round((probs_dx1 * class_idx).sum(axis=1)).astype(int), 0, 5)
qwk_argmax_dx1 = qwk(y_true_dx1, y_argmax_dx1)
qwk_eisup_dx1 = qwk(y_true_dx1, e_isup_dx1)
print(f'  DX1-only argmax: QWK = {qwk_argmax_dx1:.4f}')
print(f'  DX1-only E[ISUP]: QWK = {qwk_eisup_dx1:.4f}')

# ── Audit 2: Including synthetic benign cases ──
print()
print('=' * 70)
print('AUDIT 2: SYNTHETIC BENIGN INCLUSION (matches DSPA-MIL test set)')
print('=' * 70)
print('  DSPA-MIL has 20 benign cases in TCGA-PRAD. Our test set has none.')
print('  Hypothesis: PANDA-trained models reliably predict benign tissue as ISUP=0.')
print('  If true, adding 20 correctly-predicted benign cases boosts QWK.')
print()

# We cannot ADD real benign cases (we filtered them out), but we can SIMULATE
# the boost by assuming the model would predict 20 benign cases correctly
# (which is reasonable because PANDA training included extensive benign tissue).
#
# Method: take the patient-level results, append 20 synthetic (true=0, pred=0) cases,
# recompute QWK. This is an UPPER bound on the boost from benign inclusion.

for n_synth in [10, 20, 30]:
    # Append synthetic benign cases
    synth_true = np.array([0] * n_synth)
    synth_pred = np.array([0] * n_synth)  # assume model correctly predicts benign

    # Patient-level + synthetic benign + argmax
    y_t_aug = np.concatenate([y_true_p, synth_true])
    y_p_aug = np.concatenate([y_argmax_p, synth_pred])
    qwk_aug_a = qwk(y_t_aug, y_p_aug)
    # E[ISUP]
    y_p_aug_e = np.concatenate([y_eisup_p, synth_pred])
    qwk_aug_e = qwk(y_t_aug, y_p_aug_e)
    print(f'  + {n_synth} synthetic benign (perfectly classified):')
    print(f'    patient-level argmax: QWK = {qwk_aug_a:.4f}  (delta vs patient-only: {qwk_aug_a - qwk_argmax_p:+.4f})')
    print(f'    patient-level E[ISUP]: QWK = {qwk_aug_e:.4f}  (delta vs patient-only: {qwk_aug_e - qwk_eisup_p:+.4f})')

# Realistic benign: PANDA in-distribution shows ~95% correct on ISUP=0.
# So 20 benign with ~95% correct (19 → 0, 1 → 1).
print()
print('  Realistic case: PANDA in-dist ISUP=0 recall ~95% (1 false +1 prediction):')
synth_true = np.array([0] * 20)
synth_pred = np.array([0] * 19 + [1])
y_t_aug = np.concatenate([y_true_p, synth_true])
y_p_aug = np.concatenate([y_argmax_p, synth_pred])
qwk_aug = qwk(y_t_aug, y_p_aug)
print(f'    patient-level argmax: QWK = {qwk_aug:.4f}  (delta vs patient-only: {qwk_aug - qwk_argmax_p:+.4f})')

# ── Audit 3: TCGA-PRAD subset that matches DSPA-MIL count (449 vs our 426) ──
print()
print('=' * 70)
print('AUDIT 3: COHORT-SIZE INSPECTION')
print('=' * 70)
print(f'  DSPA-MIL TCGA-PRAD: 449 total')
print(f'      Benign 20 + GG1 44 + GG2 126 + GG3 92 + GG4 65 + GG5 123')
print(f'      Cancer-only sum: 450')
print(f'  Mohamed TCGA-PRAD:  426 total')
print(f'      No benign, ISUP1 42 + ISUP2 120 + ISUP3 89 + ISUP4 61 + ISUP5 114')
print(f'      Cancer-only sum: 426')
print(f'  Gap: DSPA-MIL has ~24 more cancer cases.')
print(f'        Plausibly: looser tissue/quality filter at our extraction step.')
print(f'        These are likely BORDERLINE slides; including them adds noise')
print(f'        but does not necessarily change QWK direction.')

# ── Audit 4: combine ALL adjustments to give upper-bound comparable QWK ──
print()
print('=' * 70)
print('AUDIT 4: ALL ADJUSTMENTS COMBINED (upper-bound comparable QWK)')
print('=' * 70)
print('  Patient-level aggregation + synthetic benign (realistic) + E[ISUP] decision rule')

synth_true = np.array([0] * 20)
synth_pred_e = np.array([0] * 19 + [1])

y_t_full = np.concatenate([y_true_p, synth_true])
y_p_full = np.concatenate([y_eisup_p, synth_pred_e])
qwk_full = qwk(y_t_full, y_p_full)

print(f'  Combined: QWK = {qwk_full:.4f}')
print(f'  vs DSPA-MIL best (DSPA-MIL): 0.7420')
print(f'  vs DSPA-MIL CLAM-MB:         0.7270')
print(f'  vs DSPA-MIL AB-MIL:          0.6770')
print()
print(f'  Residual gap to DSPA-MIL best:    {0.742 - qwk_full:+.4f}')
print(f'  Residual gap to DSPA-MIL CLAM-MB: {0.727 - qwk_full:+.4f}')
print(f'  Residual gap to DSPA-MIL AB-MIL:  {0.677 - qwk_full:+.4f}')

print()
print('=' * 70)
print('INTERPRETATION:')
print('=' * 70)
print('  If residual gap > 0.10: methodological gap is REAL; Karolinska retraining justified.')
print('  If residual gap 0.03-0.10: protocol explains most of gap; retraining marginal value.')
print('  If residual gap < 0.03: gap is FULLY PROTOCOL; no retraining needed.')
