"""Fig 2: Forest plot comparing our 3 R2-decoded results against DSPA-MIL's 4 baselines.

X-axis: QWK
Y-axis: 7 rows (our 3 + DSPA-MIL 4), sorted by QWK
Each row: point estimate + 95% CI horizontal bar
Our results highlighted (different colour); DSPA-MIL baselines in neutral grey.
Reference vertical line at our R2 full-cohort headline.
"""
from __future__ import annotations
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

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


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ks.append(qwk(y_true[idx], y_pred[idx]))
    lo = np.percentile(ks, 100 * alpha / 2)
    hi = np.percentile(ks, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# Load + compute our 3 numbers
cancer_df = pd.read_csv(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv')
benign_df = pd.read_csv(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x_benign.csv')

probs_c = np.asarray(cancer_df[prob_cols].values)
y_c = np.asarray(cancer_df['true_isup'].values, dtype=int)
probs_b = np.asarray(benign_df[prob_cols].values)
y_b = np.asarray(benign_df['true_isup'].values, dtype=int)

# R2 = slide-level E[ISUP] T=1
preds_c_r2 = e_isup(probs_c, T=1.0)
preds_b_r2 = e_isup(probs_b, T=1.0)

# (a) Cancer-only
kappa_cancer = qwk(y_c, preds_c_r2)
ci_cancer = bootstrap_ci(y_c, preds_c_r2)

# (b) Full cohort
y_full = np.concatenate([y_c, y_b])  # pyright: ignore[reportArgumentType]
p_full = np.concatenate([preds_c_r2, preds_b_r2])  # pyright: ignore[reportArgumentType]
kappa_full = qwk(y_full, p_full)
ci_full = bootstrap_ci(y_full, p_full)

# (c) Random-20 benign across seeds 1..20 (mean ± across-seed std)
kappas_rand20 = []
for seed in range(1, 21):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_b), size=20, replace=False)
    y_sub = np.concatenate([y_c, y_b[idx]])  # pyright: ignore[reportArgumentType]
    p_sub = np.concatenate([preds_c_r2, preds_b_r2[idx]])  # pyright: ignore[reportArgumentType]
    kappas_rand20.append(qwk(y_sub, p_sub))
kappa_rand20 = float(np.mean(kappas_rand20))
ci_rand20 = (float(np.percentile(kappas_rand20, 2.5)),
             float(np.percentile(kappas_rand20, 97.5)))

print(f'Cancer-only: {kappa_cancer:.4f} [{ci_cancer[0]:.4f}, {ci_cancer[1]:.4f}]')
print(f'Full cohort: {kappa_full:.4f} [{ci_full[0]:.4f}, {ci_full[1]:.4f}]')
print(f'Random-20  : {kappa_rand20:.4f} [{ci_rand20[0]:.4f}, {ci_rand20[1]:.4f}] (across-seed)')

# DSPA-U-MIL published numbers (Hao et al. Medical Image Analysis 2026; updated S121 from MIA 2026 PDF)
# Mean ± 1.96σ for 95% CI from reported σ
def dspa_ci(mean, sigma):
    return mean - 1.96 * sigma, mean + 1.96 * sigma


# Inter-pathologist Gleason QWK reference range (R009 PANDA challenge + R001 ISUP consensus):
# typical inter-pathologist kappa for Gleason grading sits in 0.65-0.85 range.
# This is the noise ceiling for any single-cohort external-validation kappa.
PATHOLOGIST_RANGE_LO = 0.65
PATHOLOGIST_RANGE_HI = 0.85

rows = [
    ('Ours: cancer-only ($n=426$)',                kappa_cancer, ci_cancer,    COLORS['blue']),
    ('Ours: random-20 benign ($n=446$)',           kappa_rand20, ci_rand20,    COLORS['blue']),
    ('Ours: full cohort ($n=544$) [primary]',      kappa_full,   ci_full,      COLORS['magenta']),
    ('DSPA-U-MIL AB-MIL baseline ($n=446$)',       0.6770,       dspa_ci(0.6770, 0.023), COLORS['gray']),
    ('DSPA-U-MIL CLAM-SB ($n=446$)',               0.6880,       dspa_ci(0.6880, 0.025), COLORS['gray']),
    ('DSPA-U-MIL CLAM-MB ($n=446$)',               0.7270,       dspa_ci(0.7270, 0.020), COLORS['gray']),
    ('DSPA-U-MIL (proposed, $n=446$)',             0.7940,       dspa_ci(0.7940, 0.042), COLORS['gray']),
]

# Sort by kappa ascending so highest kappa is at the top
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=FIG_DOUBLE)
y_pos = np.arange(len(rows))
labels = [r[0] for r in rows]
kappas = [r[1] for r in rows]
errlo = [r[1] - r[2][0] for r in rows]
errhi = [r[2][1] - r[1] for r in rows]
colors = [r[3] for r in rows]

# Pathologist reference cloud (noise ceiling) — Bejnordi Fig 3A inspired
ax.axvspan(PATHOLOGIST_RANGE_LO, PATHOLOGIST_RANGE_HI,
           color=COLORS['teal'], alpha=0.10, zorder=0,
           label=f'Inter-pathologist $\\kappa_w$ range '
                 f'[{PATHOLOGIST_RANGE_LO:.2f}, {PATHOLOGIST_RANGE_HI:.2f}] '
                 f'(R009, R001)')

ax.errorbar(kappas, y_pos, xerr=[errlo, errhi], fmt='none', color='#666666',
            elinewidth=1.0, capsize=3, zorder=2)
for i, (k, c) in enumerate(zip(kappas, colors)):
    ax.scatter(k, i, s=42, color=c, edgecolors='black', linewidths=0.6, zorder=3)

# Reference line at our full-cohort headline
ax.axvline(kappa_full, color=COLORS['magenta'], alpha=0.35, linewidth=1.0,
           linestyle='--', zorder=1, label=f'Our full-cohort headline ($\\kappa_w = {kappa_full:.3f}$)')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(r'Quadratic-weighted $\kappa$ (95% CI)')
ax.set_xlim(0.40, 0.90)
ax.set_title(r'External validation on TCGA-PRAD: protocol-comparable QWK vs published baselines',
             fontsize=10, pad=10)
ax.legend(loc='lower right', framealpha=0.9, fontsize=7)
ax.grid(True, axis='x', alpha=0.3)

save_fig(fig, 'fig2_forestplot_vs_dspa')
plt.close(fig)
print('\nDONE: Fig 2 forest plot saved.')
