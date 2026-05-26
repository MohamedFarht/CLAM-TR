"""Fig 3: Per-site cancer-only κ forest plot for TCGA-PRAD's 19 sites (≥4 slides).

Y-axis: 19 sites, sorted by κ
X-axis: QWK
Each row: bootstrap mean + 95% CI horizontal bar; dot size proportional to n.
Reference vertical line at overall cancer-only κ under R2.
Highlight catastrophic-failure sites (κ < 0.10) in red.
"""
from __future__ import annotations
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, '.claude/bin')
from q1_fig_style import apply_style, COLORS, FIG_DOUBLE_TALL, save_fig

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
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        return float('nan')
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                             labels=list(range(N_CLASSES)))


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 4:
        return float('nan'), float('nan'), float('nan')
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        k = qwk(y_true[idx], y_pred[idx])
        if not np.isnan(k):
            ks.append(k)
    if not ks:
        return float('nan'), float('nan'), float('nan')
    lo = np.percentile(ks, 100 * alpha / 2)
    hi = np.percentile(ks, 100 * (1 - alpha / 2))
    return float(np.mean(ks)), float(lo), float(hi)


df = pd.read_csv(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD/predictions_20x.csv')
df['site_code'] = df['slide_id'].str.extract(r'^TCGA-([A-Z0-9]{2})-')

probs = np.asarray(df[prob_cols].values)
y_true = np.asarray(df['true_isup'].values, dtype=int)
df['pred_r2'] = e_isup(probs, T=1.0)

overall_qwk = qwk(y_true, df['pred_r2'].values.astype(int))
print(f'Overall cancer-only QWK (R2): {overall_qwk:.4f}')

results = []
for site, gdf in df.groupby('site_code'):
    n = len(gdf)
    if n < 4:
        continue
    yt = gdf['true_isup'].values.astype(int)
    yp = gdf['pred_r2'].values.astype(int)
    mean_k, lo, hi = bootstrap_ci(yt, yp)
    results.append({
        'site': site,
        'n': n,
        'kappa': mean_k,
        'lo': lo,
        'hi': hi,
    })

site_df = pd.DataFrame(results).sort_values('kappa', ascending=True).reset_index(drop=True)
print(f'Sites with >=4 slides: {len(site_df)}')

# Build figure
fig, ax = plt.subplots(figsize=FIG_DOUBLE_TALL)
y_pos = np.arange(len(site_df))
labels = [f'{row["site"]} (n={row["n"]:>3d})' for _, row in site_df.iterrows()]

errlo = (site_df['kappa'] - site_df['lo']).values
errhi = (site_df['hi'] - site_df['kappa']).values

ax.errorbar(site_df['kappa'].values, y_pos, xerr=[errlo, errhi], fmt='none',
            color='#888888', elinewidth=0.8, capsize=2, zorder=2)

# Colour code: red for catastrophic (kappa < 0.10), orange (< 0.30), blue (>= 0.30)
def site_colour(k):
    if k < 0.10:
        return COLORS['magenta']
    if k < 0.30:
        return COLORS['orange']
    return COLORS['blue']

n_arr = site_df['n'].values
sizes = 10 + 1.2 * n_arr  # marker size proportional to n
for i, row in site_df.iterrows():
    c = site_colour(row['kappa'])
    ax.scatter(row['kappa'], i, s=sizes[i], color=c, edgecolors='black',
               linewidths=0.5, zorder=3)

# Reference: overall cancer-only headline
ax.axvline(overall_qwk, color=COLORS['black'], alpha=0.4, linewidth=1.0,
           linestyle='--', zorder=1, label=f'Overall cancer-only ($\\kappa_w = {overall_qwk:.3f}$)')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel(r'Per-site quadratic-weighted $\kappa$ (bootstrap mean, 95% CI)')
ax.set_xlim(-0.40, 1.05)
ax.set_title(r'Per-site cancer-only $\kappa$ on TCGA-PRAD under R2 decoding (19 sites, $\geq 4$ slides)',
             fontsize=10, pad=10)

# Legend for colour coding
import matplotlib.patches as mpatches
red_patch    = mpatches.Patch(color=COLORS['magenta'], label=r'Catastrophic ($\kappa < 0.10$)')
orange_patch = mpatches.Patch(color=COLORS['orange'],  label=r'Weak ($0.10 \leq \kappa < 0.30$)')
blue_patch   = mpatches.Patch(color=COLORS['blue'],    label=r'Adequate ($\kappa \geq 0.30$)')
ax.legend(handles=[red_patch, orange_patch, blue_patch], loc='lower right',
          framealpha=0.9, fontsize=7)
ax.grid(True, axis='x', alpha=0.3)

save_fig(fig, 'fig3_per_site_forest')
plt.close(fig)
print('\nDONE: Fig 3 per-site forest saved.')
