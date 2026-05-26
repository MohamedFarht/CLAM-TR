"""Fig 4: Benign-cohort predicted-ISUP distribution under R2 vs R5 decoding.

Two-panel side-by-side bar chart.
Panel A: R2 (calibration-free) — diverse low-grade distribution, 33% correct at ISUP=0
Panel B: R5 (cancer-CV-T T*=4.86) — collapsed to ISUP=2, 0% correct at ISUP=0
Highlight ISUP=0 bars (correct benign) in distinctive colour.
"""
from __future__ import annotations
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, '.claude/bin')
from q1_fig_style import apply_style, COLORS, FIG_DOUBLE, save_fig

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
apply_style()

N_CLASSES = 6
prob_cols = [f'prob_isup_{i}' for i in range(N_CLASSES)]
class_idx = np.arange(N_CLASSES)


def e_isup(probs, T=1.0):
    sl = np.log(np.clip(probs, 1e-10, 1.0)) / T
    sl -= sl.max(axis=1, keepdims=True)
    p = np.exp(sl); p /= p.sum(axis=1, keepdims=True)
    e = (p * class_idx).sum(axis=1)
    return np.clip(np.round(e).astype(int), 0, N_CLASSES - 1)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                             labels=list(range(N_CLASSES)))


def cv_tune_T(probs, y_true, n_splits=5, seed=42):
    T_grid = np.linspace(0.5, 15.0, 100)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    Ts = []
    for ti, vi in skf.split(np.arange(len(y_true)), y_true):
        best_T, best_q = T_grid[0], -np.inf
        for T in T_grid:
            q = qwk(y_true[ti], e_isup(probs[ti], T))
            if q > best_q:
                best_T, best_q = T, q
        Ts.append(best_T)
    return float(np.mean(Ts))


cancer_df = pd.read_csv(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv')
benign_df = pd.read_csv(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv')

# Patient-level aggregation for benign (cancer-tuned T uses cancer-patient-level)
cancer_df['patient_id'] = cancer_df['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')
benign_df['patient_id'] = benign_df['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')

cancer_p = cancer_df.groupby('patient_id').agg({
    'true_isup': lambda s: int(s.mode().iloc[0]),
    **{c: 'mean' for c in prob_cols},
}).reset_index()
benign_p = benign_df.groupby('patient_id').agg({
    'true_isup': lambda s: int(s.mode().iloc[0]),
    **{c: 'mean' for c in prob_cols},
}).reset_index()

probs_c_p = np.asarray(cancer_p[prob_cols].values)
y_c_p = np.asarray(cancer_p['true_isup'].values, dtype=int)
probs_b_p = np.asarray(benign_p[prob_cols].values)

# Fit T on cancer-only patient-level (this matches Section 3.7 narrative)
T_cancer = cv_tune_T(probs_c_p, y_c_p)
print(f'T* (cancer-CV-T): {T_cancer:.2f}')

# R2 (T=1) on benign
preds_b_r2 = e_isup(probs_b_p, T=1.0)
# R5 (T*=4.86) on benign
preds_b_r5 = e_isup(probs_b_p, T=T_cancer)

# Build bar counts
def bar_counts(preds):
    counts = np.zeros(N_CLASSES, dtype=int)
    for k in range(N_CLASSES):
        counts[k] = int(np.sum(preds == k))
    return counts


counts_r2 = bar_counts(preds_b_r2)
counts_r5 = bar_counts(preds_b_r5)
n_total = len(preds_b_r2)
print(f'R2 distribution: {dict(zip(range(6), counts_r2))}')
print(f'R5 distribution: {dict(zip(range(6), counts_r5))}')

fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharey=True)

x = np.arange(N_CLASSES)
xlabels = [f'ISUP\n{k}' for k in range(N_CLASSES)]

# Panel A: R2
ax = axes[0]
bar_colors = [COLORS['teal'] if k == 0 else COLORS['blue'] for k in range(N_CLASSES)]
bars = ax.bar(x, counts_r2, color=bar_colors, edgecolor='black', linewidth=0.6)
for k, c in enumerate(counts_r2):
    if c > 0:
        ax.text(k, c + 2, f'{c}\n({100*c/n_total:.0f}%)',
                ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(xlabels)
ax.set_ylabel(f'Benign patients (n={n_total})')
ax.set_title('(a) R2 — calibration-free ($T=1$)', fontsize=9)
ax.set_ylim(0, max(max(counts_r2), max(counts_r5)) * 1.25)

# Panel B: R5
ax = axes[1]
bar_colors = [COLORS['teal'] if k == 0 else COLORS['magenta'] for k in range(N_CLASSES)]
bars = ax.bar(x, counts_r5, color=bar_colors, edgecolor='black', linewidth=0.6)
for k, c in enumerate(counts_r5):
    if c > 0:
        ax.text(k, c + 2, f'{c}\n({100*c/n_total:.0f}%)',
                ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(xlabels)
ax.set_title(f'(b) R5 — cancer-tuned CV-T ($T^* = {T_cancer:.2f}$)', fontsize=9)

# Add a single legend for the highlight
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS['teal'], edgecolor='black',
                         label='ISUP 0 (correct benign)')]
fig.legend(handles=legend_elements, loc='upper center', ncol=1,
           bbox_to_anchor=(0.5, 1.04), framealpha=0.9, fontsize=8)

fig.suptitle(r'Predicted ISUP distribution on TCGA-PRAD benign cohort ($n=118$ patients) under two decoding rules',
             fontsize=10, y=1.10)

save_fig(fig, 'fig4_benign_softmax_r2_vs_r5')
plt.close(fig)
print('\nDONE: Fig 4 benign softmax saved.')
