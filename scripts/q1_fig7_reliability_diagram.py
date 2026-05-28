"""Q1 F5: reliability / calibration diagram with ECE + Brier score.

Standard 2-panel reliability diagram for the headline configuration on
TCGA-PRAD under R2 (calibration-free) vs R5 (cancer-CV-tuned T) — directly
visualises the calibration-collapse story currently told in §4.5.

We use the "ordinal-confidence" framing: the top-1 softmax probability vs
empirical accuracy (top-1 prediction equals true), binned into 10 confidence
bins. Reports Expected Calibration Error (ECE) and Brier score (one-vs-rest
multiclass averaging).

Inputs:
  G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv
  G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv
Writes:
  papers/NCA 2026/figures/fig7_reliability_diagram.{pdf,png}
  papers/NCA 2026/figures/calibration_metrics.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, '.claude/bin')
from q1_fig_style import apply_style, COLORS, save_fig

apply_style()

N_CLASSES = 6
N_BINS = 10
class_idx = np.arange(N_CLASSES)
prob_cols = [f'prob_isup_{i}' for i in range(N_CLASSES)]

TCGA_CANCER = Path(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv')
TCGA_BENIGN = Path(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv')
OUT_METRICS = Path('papers/NCA 2026/figures/calibration_metrics.csv')


def e_isup(probs, T=1.0):
    sl = np.log(np.clip(probs, 1e-10, 1.0)) / T
    sl -= sl.max(axis=1, keepdims=True)
    p = np.exp(sl)
    p /= p.sum(axis=1, keepdims=True)
    e = (p * class_idx).sum(axis=1)
    return np.clip(np.round(e).astype(int), 0, N_CLASSES - 1), p


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
            preds, _ = e_isup(probs[ti], T)
            q = qwk(y_true[ti], preds)
            if q > best_q:
                best_T, best_q = T, q
        Ts.append(best_T)
    return float(np.mean(Ts))


def reliability(probs, y_true, n_bins=N_BINS):
    """top-1 confidence vs accuracy in n_bins equal-width bins."""
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y_true).astype(int)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lo = bin_edges[:-1]
    bin_hi = bin_edges[1:]
    centers = (bin_lo + bin_hi) / 2
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    weights = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (conf >= bin_lo[i]) & (conf < bin_hi[i]) if i < n_bins - 1 \
            else (conf >= bin_lo[i]) & (conf <= bin_hi[i])
        n_in = int(mask.sum())
        weights[i] = n_in
        if n_in > 0:
            accs[i] = correct[mask].mean()
            confs[i] = conf[mask].mean()
    ece = float((np.abs(accs - confs) * (weights / weights.sum())).sum())
    return centers, accs, confs, weights, ece


def brier_multiclass(probs, y_true, n_classes=N_CLASSES):
    """One-vs-rest macro Brier."""
    oh = np.zeros((len(y_true), n_classes))
    oh[np.arange(len(y_true)), y_true] = 1
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def panel(ax, centers, accs, weights, ece, brier, title, colour):
    bar_w = 1.0 / N_BINS * 0.85
    # diagonal
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8, alpha=0.7,
            label='Perfect calibration')
    ax.bar(centers, accs, width=bar_w, color=colour, edgecolor='black',
           linewidth=0.5, alpha=0.85, label='Observed accuracy')
    # Sample-size annotations
    for c, a, w in zip(centers, accs, weights):
        if w > 0:
            ax.text(c, a + 0.02, f'{int(w)}', ha='center', va='bottom',
                    fontsize=6.5, color='black')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Top-1 softmax confidence')
    ax.set_ylabel('Top-1 accuracy')
    ax.set_title(title, fontsize=9.5)
    ax.text(0.02, 0.97, f'ECE = {ece:.3f}\nBrier = {brier:.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
            bbox=dict(facecolor='white', edgecolor='gray',
                      boxstyle='round,pad=0.3', alpha=0.9))
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.95)


def main():
    c = pd.read_csv(TCGA_CANCER)
    b = pd.read_csv(TCGA_BENIGN)
    probs_c = np.asarray(c[prob_cols].values, dtype=float)
    probs_b = np.asarray(b[prob_cols].values, dtype=float)
    y_c = np.asarray(c['true_isup'].values, dtype=int)
    y_b = np.zeros(len(b), dtype=int)
    # Full-cohort slide-level (R2 framing)
    probs_all = np.vstack([probs_c, probs_b])
    y_all = np.concatenate([y_c, y_b])

    # R5 T* on cancer-only patient-level (matches §3.7 narrative)
    c['patient_id'] = c['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')
    c_p = c.groupby('patient_id').agg({
        'true_isup': lambda s: int(s.mode().iloc[0]),
        **{col: 'mean' for col in prob_cols}}).reset_index()
    probs_c_p = np.asarray(c_p[prob_cols].values, dtype=float)
    y_c_p = np.asarray(c_p['true_isup'].values, dtype=int)
    T_star = cv_tune_T(probs_c_p, y_c_p)
    print(f'T*: {T_star:.2f}')

    # Calibrated probs for R5 on full slide-level
    _, probs_all_T = e_isup(probs_all, T=T_star)

    centers_r2, accs_r2, confs_r2, w_r2, ece_r2 = reliability(probs_all, y_all)
    brier_r2 = brier_multiclass(probs_all, y_all)
    centers_r5, accs_r5, confs_r5, w_r5, ece_r5 = reliability(probs_all_T, y_all)
    brier_r5 = brier_multiclass(probs_all_T, y_all)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    panel(axes[0], centers_r2, accs_r2, w_r2, ece_r2, brier_r2,
          '(a) R2 — calibration-free ($T = 1$)', COLORS['blue'])
    panel(axes[1], centers_r5, accs_r5, w_r5, ece_r5, brier_r5,
          f'(b) R5 — cancer-CV-tuned ($T^* = {T_star:.2f}$)', COLORS['magenta'])

    fig.suptitle('Reliability diagram on TCGA-PRAD full cohort '
                 r'(cancer + benign; $n_{\mathrm{slides}} = 544$)',
                 fontsize=10, y=1.02)

    save_fig(fig, 'fig7_reliability_diagram')
    plt.close(fig)

    # Save metrics CSV
    pd.DataFrame([
        {'rule': 'R2', 'T': 1.0, 'ECE': ece_r2, 'Brier': brier_r2,
         'n_bins': N_BINS, 'n_total': len(y_all)},
        {'rule': 'R5', 'T': T_star, 'ECE': ece_r5, 'Brier': brier_r5,
         'n_bins': N_BINS, 'n_total': len(y_all)},
    ]).to_csv(OUT_METRICS, index=False, float_format='%.4f')
    print(f'Wrote {OUT_METRICS}')
    print('DONE: fig7_reliability_diagram saved.')


if __name__ == '__main__':
    main()
