"""Study TF3 / extra Q1 supp: inter-decoding-rule Cohen's kappa heatmap.

Neidlinger Ext Fig 10 style. Shows which slides each decoding rule grades
identically vs differently on TCGA-PRAD. Diagonal = 1.0 by construction.

Five decoding rules from §3.5:
  R1: argmax (slide-level), T=1
  R2: E[ISUP] (slide-level), T=1            <- primary headline
  R3: argmax (patient-level), T=1
  R4: E[ISUP] (patient-level), T=1
  R5: E[ISUP] (patient-level), cancer-CV-tuned T
  R6: E[ISUP] (slide-level), full-cohort-CV-tuned T
  R7: argmax (patient-level, class-balanced prior reweighted), T=1

For the heatmap we use full-cohort slide-level evaluations (or patient-level
where the rule mandates) and report SLIDE-LEVEL agreement (mapping patient
predictions back to slide ids when the rule is patient-level).

Reads TCGA-PRAD CSVs.
Writes papers/NCA 2026/figures/fig_inter_rule_kappa.{pdf,png} +
       inter_rule_kappa.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, '.claude/bin')
from q1_fig_style import apply_style, COLORS, save_fig

apply_style()

N_CLASSES = 6
class_idx = np.arange(N_CLASSES)
prob_cols = [f'prob_isup_{i}' for i in range(N_CLASSES)]

TCGA_CANCER = Path(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv')
TCGA_BENIGN = Path(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv')
OUT_CSV = Path('papers/NCA 2026/figures/inter_rule_kappa.csv')


def e_isup(probs, T=1.0):
    sl = np.log(np.clip(probs, 1e-10, 1.0)) / T
    sl -= sl.max(axis=1, keepdims=True)
    p = np.exp(sl)
    p /= p.sum(axis=1, keepdims=True)
    return np.clip(np.round((p * class_idx).sum(axis=1)).astype(int),
                   0, N_CLASSES - 1)


def argmax_pred(probs):
    return probs.argmax(axis=1)


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


def main():
    c = pd.read_csv(TCGA_CANCER)
    b = pd.read_csv(TCGA_BENIGN)
    c['patient_id'] = c['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')
    b['patient_id'] = b['slide_id'].str.extract(r'^(TCGA-[A-Z0-9]+-[A-Z0-9]+)')

    # Build slide-level full cohort with patient mapping
    full = pd.concat([c.assign(cohort='cancer'),
                      b.assign(cohort='benign', true_isup=0)],
                     ignore_index=True)
    full_probs = np.asarray(full[prob_cols].values, dtype=float)
    full_y = np.asarray(full['true_isup'].values, dtype=int)
    full_patient = full['patient_id'].fillna(full['slide_id']).values

    # Patient-level
    pat = full.groupby(full_patient).agg({
        'true_isup': lambda s: int(s.mode().iloc[0]),
        **{col: 'mean' for col in prob_cols}})
    pat_probs = np.asarray(pat[prob_cols].values, dtype=float)
    pat_y = np.asarray(pat['true_isup'].values, dtype=int)
    pat_to_slides = {p: list(np.where(full_patient == p)[0])
                     for p in pat.index}

    # Tune T (cancer-only patient + full-cohort patient)
    c_p = c.groupby('patient_id').agg({
        'true_isup': lambda s: int(s.mode().iloc[0]),
        **{col: 'mean' for col in prob_cols}})
    T_cancer = cv_tune_T(np.asarray(c_p[prob_cols].values, dtype=float),
                         np.asarray(c_p['true_isup'].values, dtype=int))
    T_full = cv_tune_T(pat_probs, pat_y)
    print(f'T_cancer (R5): {T_cancer:.2f}, T_full (R6): {T_full:.2f}')

    # Rule predictions at SLIDE level
    rules = {}
    rules['R1: argmax slide T=1'] = argmax_pred(full_probs)
    rules['R2: E[ISUP] slide T=1'] = e_isup(full_probs, T=1.0)

    def lift_pat_to_slide(pat_pred):
        out = np.zeros(len(full), dtype=int)
        for i, p in enumerate(pat.index):
            for s in pat_to_slides[p]:
                out[s] = pat_pred[i]
        return out

    rules['R3: argmax patient T=1'] = lift_pat_to_slide(argmax_pred(pat_probs))
    rules['R4: E[ISUP] patient T=1'] = lift_pat_to_slide(e_isup(pat_probs, T=1.0))
    rules['R5: E[ISUP] patient T*=' + f'{T_cancer:.2f}'] = lift_pat_to_slide(
        e_isup(pat_probs, T=T_cancer))
    rules['R6: E[ISUP] slide T*=' + f'{T_full:.2f}'] = e_isup(full_probs, T=T_full)

    # Compute pairwise QWK between rules
    names = list(rules.keys())
    n_rules = len(names)
    K = np.zeros((n_rules, n_rules))
    for i in range(n_rules):
        for j in range(n_rules):
            K[i, j] = qwk(rules[names[i]], rules[names[j]])

    # Also annotate each rule with its kappa vs ground truth
    gt_qwk = {n: qwk(full_y, rules[n]) for n in names}

    # CSV out
    df = pd.DataFrame(K, index=names, columns=names)
    df.to_csv(OUT_CSV, float_format='%.4f')
    print(f'Wrote {OUT_CSV}')
    print(df.round(3).to_string())

    # Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    short = [n.split(':')[0] for n in names]
    full_lab = [n.replace(': ', ':\n') for n in names]
    im = ax.imshow(K, cmap='RdYlBu_r', vmin=0.5, vmax=1.0, aspect='auto')
    for i in range(n_rules):
        for j in range(n_rules):
            color = 'white' if K[i, j] < 0.72 else 'black'
            ax.text(j, i, f'{K[i,j]:.3f}', ha='center', va='center',
                    fontsize=8, color=color)
    ax.set_xticks(range(n_rules))
    ax.set_yticks(range(n_rules))
    ax.set_xticklabels(short, rotation=0)
    ax.set_yticklabels(full_lab, fontsize=7.5)
    # Append ground-truth kappa on a side annotation column
    for i, n in enumerate(names):
        ax.text(n_rules - 0.3, i, f'$\\kappa_w$ vs truth: {gt_qwk[n]:.3f}',
                ha='left', va='center', fontsize=7,
                color='#222222',
                bbox=dict(facecolor='#f5f5f5', edgecolor='gray',
                          boxstyle='round,pad=0.15', alpha=0.85))
    ax.set_title('Inter-decoding-rule agreement on TCGA-PRAD\n'
                 r'(slide-level $\kappa_w$ between rule pairs; $n = 544$)',
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label=r'Inter-rule $\kappa_w$')
    plt.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.10)

    save_fig(fig, 'fig_inter_rule_kappa')
    plt.close(fig)
    print('DONE: fig_inter_rule_kappa saved.')


if __name__ == '__main__':
    main()
