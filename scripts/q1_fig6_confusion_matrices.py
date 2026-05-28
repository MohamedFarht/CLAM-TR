"""Q1 F1: per-cohort confusion matrices (3 panels).

Bulten Fig 2 / DSPA Fig 5 style. Paired count + row-% cells, diverging
colormap, kappa_w annotated bottom-right.

Three panels:
  (a) PANDA 5-fold CV mean confusion (RetNet-bestHP-UNI v1 — headline cfg)
  (b) TCGA-PRAD R2 (slide-level expected-value, T=1, no calibration)
  (c) TCGA-PRAD R5 (patient-level expected-value, cancer-CV-tuned T)

Inputs:
  G:/My Drive/CLAM_TR_Results/v20_fullscale/UNI/predictions/RetNet_bestHP_fold{0-4}_predictions.json
  G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv
  G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv
Writes papers/NCA 2026/figures/fig6_confusion_matrices.{pdf,png}.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, '.claude/bin')
from q1_fig_style import apply_style, COLORS, save_fig

apply_style()

N_CLASSES = 6
class_idx = np.arange(N_CLASSES)
prob_cols = [f'prob_isup_{i}' for i in range(N_CLASSES)]

PANDA_PRED_DIR = Path(r'G:/My Drive/CLAM_TR_Results/v20_fullscale/UNI/predictions')
TCGA_CANCER = Path(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv')
TCGA_BENIGN = Path(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv')


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


def load_panda_confusion():
    """5-fold CV — concatenate all folds, build single confusion."""
    y_true_all, y_pred_all = [], []
    for fold in range(5):
        fp = PANDA_PRED_DIR / f'RetNet_bestHP_fold{fold}_predictions.json'
        with open(fp) as f:
            d = json.load(f)
        y_true_all.extend(d['labels'])
        y_pred_all.extend(d['predictions'])
    y_true = np.asarray(y_true_all)
    y_pred = np.asarray(y_pred_all)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    k = qwk(y_true, y_pred)
    n = len(y_true)
    return cm, k, n


def load_tcga_r2():
    """R2: slide-level E[ISUP] at T=1, full cohort (cancer + benign)."""
    c = pd.read_csv(TCGA_CANCER)
    b = pd.read_csv(TCGA_BENIGN)
    probs_c = np.asarray(c[prob_cols].values, dtype=float)
    probs_b = np.asarray(b[prob_cols].values, dtype=float)
    y_c = np.asarray(c['true_isup'].values, dtype=int)
    y_b = np.zeros(len(b), dtype=int)
    probs = np.vstack([probs_c, probs_b])
    y_true = np.concatenate([y_c, y_b])
    y_pred = e_isup(probs, T=1.0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    k = qwk(y_true, y_pred)
    return cm, k, len(y_true)


def load_tcga_r5():
    """R5: patient-level E[ISUP] with cancer-CV-tuned T."""
    c = pd.read_csv(TCGA_CANCER)
    b = pd.read_csv(TCGA_BENIGN)
    c['patient_id'] = c['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')
    b['patient_id'] = b['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')
    c_p = c.groupby('patient_id').agg({
        'true_isup': lambda s: int(s.mode().iloc[0]),
        **{col: 'mean' for col in prob_cols}}).reset_index()
    b_p = b.groupby('patient_id').agg({
        'true_isup': lambda s: int(s.mode().iloc[0]),
        **{col: 'mean' for col in prob_cols}}).reset_index()
    probs_c_p = np.asarray(c_p[prob_cols].values, dtype=float)
    probs_b_p = np.asarray(b_p[prob_cols].values, dtype=float)
    y_c_p = np.asarray(c_p['true_isup'].values, dtype=int)
    y_b_p = np.zeros(len(b_p), dtype=int)
    T_star = cv_tune_T(probs_c_p, y_c_p)
    print(f'R5 T*: {T_star:.2f}')
    probs = np.vstack([probs_c_p, probs_b_p])
    y_true = np.concatenate([y_c_p, y_b_p])
    y_pred = e_isup(probs, T=T_star)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    k = qwk(y_true, y_pred)
    return cm, k, len(y_true), T_star


def draw_panel(ax, cm, k, n, title):
    cm = cm.astype(int)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum_safe = np.where(row_sum == 0, 1, row_sum)
    pct = cm / row_sum_safe * 100.0
    im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val_color = 'white' if pct[i, j] > 55 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({pct[i,j]:.0f}%)',
                    ha='center', va='center', fontsize=7.5,
                    color=val_color)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([f'{k}' for k in range(N_CLASSES)])
    ax.set_yticklabels([f'{k}' for k in range(N_CLASSES)])
    ax.set_xlabel('Predicted ISUP')
    ax.set_ylabel('True ISUP')
    ax.set_title(title, fontsize=9.5)
    ax.text(0.98, 0.04, f'$\\kappa_w = {k:.4f}$\n$n = {n}$',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, color='black',
            bbox=dict(facecolor='white', edgecolor='gray',
                      boxstyle='round,pad=0.3', alpha=0.9))
    return im


def main():
    print('Loading PANDA 5-fold...')
    cm_p, k_p, n_p = load_panda_confusion()
    print(f'PANDA: k={k_p:.4f} n={n_p}')

    print('Loading TCGA-PRAD R2...')
    cm_r2, k_r2, n_r2 = load_tcga_r2()
    print(f'R2: k={k_r2:.4f} n={n_r2}')

    print('Loading TCGA-PRAD R5...')
    cm_r5, k_r5, n_r5, T_star = load_tcga_r5()
    print(f'R5: k={k_r5:.4f} n={n_r5} T*={T_star:.2f}')

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    im0 = draw_panel(axes[0], cm_p, k_p, n_p,
                     '(a) PANDA 5-fold CV — RetNet-bestHP / UNI v1')
    im1 = draw_panel(axes[1], cm_r2, k_r2, n_r2,
                     '(b) TCGA-PRAD R2 — slide-level $E[$ISUP$]$, $T = 1$')
    im2 = draw_panel(axes[2], cm_r5, k_r5, n_r5,
                     f'(c) TCGA-PRAD R5 — patient-level, $T^* = {T_star:.2f}$ (cancer-CV)')

    cbar = fig.colorbar(im2, ax=axes, fraction=0.020, pad=0.02,
                        shrink=0.95)
    cbar.set_label('Row-normalised %', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    save_fig(fig, 'fig6_confusion_matrices')
    plt.close(fig)
    print('DONE: fig6_confusion_matrices saved.')


if __name__ == '__main__':
    main()
