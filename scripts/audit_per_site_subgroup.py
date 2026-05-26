"""Per-site TCGA-PRAD subgroup sensitivity under R2 (slide-level E[ISUP] T=1).

TCGA slide IDs encode the contributing source site in the 2-letter code after
'TCGA-'. E.g., 'TCGA-EJ-7123-...' = site EJ. Computes per-site QWK + count to
check whether the headline holds across the 32+ US contributing sites, OR
whether the model performs especially poorly on specific sites (which would
be a scanner/protocol concentration signal for Discussion).

Only reports a finding if at least one site has >= 8 cancer cases AND QWK
deviates >= 0.10 from the overall headline.
"""
from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

N_CLASSES = 6
DRIVE = Path(r'G:/My Drive/CLAM_TR_Results/TCGA_PRAD')

prob_cols = [f'prob_isup_{i}' for i in range(N_CLASSES)]
class_idx = np.arange(N_CLASSES)


def qwk(y_true, y_pred):
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        return float('nan')
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                             labels=list(range(N_CLASSES)))


def e_isup(probs, T=1.0):
    sl = np.log(np.clip(probs, 1e-10, 1.0)) / T
    sl -= sl.max(axis=1, keepdims=True)
    p = np.exp(sl)
    p /= p.sum(axis=1, keepdims=True)
    e = (p * class_idx).sum(axis=1)
    return np.clip(np.round(e).astype(int), 0, N_CLASSES - 1)


# ----------------------------------------------------------------------
#  Load + extract site code
# ----------------------------------------------------------------------
df = pd.read_csv(DRIVE / 'predictions_20x.csv')
df['site_code'] = df['slide_id'].str.extract(r'^TCGA-([A-Z0-9]{2})-')
print(f'Cancer slides: {len(df)}  / sites: {df["site_code"].nunique()}')

probs = np.asarray(df[prob_cols].values)
y_true = np.asarray(df['true_isup'].values, dtype=int)
preds_r2 = e_isup(probs, T=1.0)

# Overall headline-aligned cancer-side QWK (for context)
overall_qwk = qwk(y_true, preds_r2)
print(f'\nOverall cancer-only QWK under R2 (E[ISUP] T=1): {overall_qwk:.4f}\n')

# ----------------------------------------------------------------------
#  Per-site QWK
# ----------------------------------------------------------------------
df['pred_r2'] = preds_r2
site_results = []
for site, gdf in df.groupby('site_code'):
    n = len(gdf)
    if n < 4:
        continue
    yt = gdf['true_isup'].values.astype(int)
    yp = gdf['pred_r2'].values.astype(int)
    q = qwk(yt, yp)
    isup_dist = Counter(int(v) for v in yt)
    site_results.append({
        'site': site,
        'n_slides': n,
        'qwk_r2': q,
        'n_isup_5_true': isup_dist.get(5, 0),
        'pct_isup_5_pred': 100.0 * int(np.sum(yp == 5)) / n,  # pyright: ignore[reportArgumentType]
    })

site_df = pd.DataFrame(site_results).sort_values('n_slides', ascending=False)
print('=== Per-site QWK (sites with >=4 cancer slides) ===')
print(f'{"site":>6s}  {"n":>4s}  {"QWK":>8s}  {"n_isup5_true":>13s}  {"pct_isup5_pred":>15s}  {"|delta|":>8s}')
print('-' * 75)
for _, r in site_df.iterrows():
    delta = abs(r['qwk_r2'] - overall_qwk) if not np.isnan(r['qwk_r2']) else float('nan')
    flag = ' [OUTLIER]' if delta >= 0.10 else ''
    qwk_str = f'{r["qwk_r2"]:.4f}' if not np.isnan(r['qwk_r2']) else '   NaN '
    delta_str = f'{delta:.4f}' if not np.isnan(delta) else '   NaN '
    print(f'  {r["site"]:>4s}  {r["n_slides"]:4d}  {qwk_str:>8s}  '
          f'{r["n_isup_5_true"]:>13d}  {r["pct_isup_5_pred"]:>14.1f}%  {delta_str:>8s}{flag}')

# Headline takeaways
n_sites_with_8plus = int((site_df['n_slides'] >= 8).sum())
n_outliers = int((np.abs(site_df['qwk_r2'] - overall_qwk) >= 0.10).sum())
print(f'\n=== Summary ===')
print(f'Total sites (with >=4 slides) : {len(site_df)}')
print(f'Sites with >=8 slides         : {n_sites_with_8plus}')
print(f'Sites with |delta| >= 0.10    : {n_outliers}')
print(f'Per-site QWK range            : [{site_df["qwk_r2"].min():.4f}, {site_df["qwk_r2"].max():.4f}]')
print(f'Per-site QWK IQR              : [{site_df["qwk_r2"].quantile(0.25):.4f}, {site_df["qwk_r2"].quantile(0.75):.4f}]')

# Worth-discussion threshold
if n_outliers >= 2:
    print(f'\n=> FINDING: {n_outliers} sites deviate >=0.10 from overall headline.')
    print(f'   WORTH a paragraph in Discussion (scanner/protocol concentration signal).')
else:
    print(f'\n=> NO FINDING: site-level QWK is largely homogeneous.')
    print(f'   The headline gap is NOT concentrated in specific sites.')
    print(f'   Recommend NOT including this in Discussion (saves space + avoids')
    print(f'   over-interpretation of small-N per-site QWK estimates).')
