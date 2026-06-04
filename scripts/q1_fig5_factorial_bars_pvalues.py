"""Q1 F3: grouped bar chart for the 6 x 2 factorial.

GigaPath Fig 2-3 / Kurata Fig 2-3 style:
- 6 aggregators on x-axis, grouped by 2 encoders (UNI vs UNI2-h).
- Bars show mean QWK with 95% bootstrap CI error bars.
- Individual 5-fold dots overlaid on each bar (paired by fold).
- Paired Wilcoxon p-value on top of each aggregator-vs-ABMIL contrast.

Reads study/main/figures/retention_ablation_per_fold.csv.
Writes papers/NCA 2026/figures/fig5_factorial_bars_pvalues.{pdf,png}.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, '.claude/bin')
from q1_fig_style import apply_style, COLORS, FIG_DOUBLE, save_fig

apply_style()

INPUT_CSV = Path('study/main/figures/retention_ablation_per_fold.csv')
HP = 'best'
N_BOOT = 10000
SEED = 42

AGG_ORDER = ['ABMIL', 'DualBranch', 'SpatialMult', 'SpatialAdd', 'RetNet',
             'SpatialMultRetNet']
ENC_ORDER = ['UNI', 'UNI2-h']
ENC_LABEL = {'UNI': 'UNI v1', 'UNI2-h': 'UNI2-h'}
ENC_COLOR = {'UNI': COLORS['blue'], 'UNI2-h': COLORS['orange']}


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    bm = values[idx].mean(axis=1)
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[df['hp'] == HP].copy()

    # Build per-(encoder, aggregator) per-fold arrays
    data = {}
    for enc in ENC_ORDER:
        data[enc] = {}
        for agg in AGG_ORDER:
            sub = df[(df['encoder'] == enc) & (df['variant'] == agg)]
            data[enc][agg] = sub['full_qwk'].values

    n_aggs = len(AGG_ORDER)
    n_encs = len(ENC_ORDER)
    width = 0.36
    x = np.arange(n_aggs)
    offsets = np.linspace(-(width/2 * (n_encs - 1)), (width/2 * (n_encs - 1)),
                          n_encs)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    for i_enc, enc in enumerate(ENC_ORDER):
        means = np.array([data[enc][a].mean() if len(data[enc][a]) > 0 else np.nan
                          for a in AGG_ORDER])
        ci_lo = np.zeros(n_aggs)
        ci_hi = np.zeros(n_aggs)
        for i_a, a in enumerate(AGG_ORDER):
            vals = data[enc][a]
            if len(vals) >= 2:
                lo, hi = bootstrap_ci(vals)
            else:
                lo, hi = means[i_a], means[i_a]
            ci_lo[i_a] = max(0.0, means[i_a] - lo)
            ci_hi[i_a] = max(0.0, hi - means[i_a])
        bar_x = x + offsets[i_enc]
        ax.bar(bar_x, means, width, label=ENC_LABEL[enc],
               color=ENC_COLOR[enc], edgecolor='black', linewidth=0.6,
               alpha=0.85)
        ax.errorbar(bar_x, means, yerr=[ci_lo, ci_hi], fmt='none',
                    color='black', capsize=2.5, linewidth=0.8)
        # Fold dots
        for i_a, a in enumerate(AGG_ORDER):
            vals = data[enc][a]
            if len(vals) > 0:
                jitter = (np.random.RandomState(SEED + i_a + i_enc * 7)
                          .uniform(-width*0.18, width*0.18, size=len(vals)))
                ax.scatter(np.full_like(vals, bar_x[i_a]) + jitter, vals,
                           s=10, color='white', edgecolor='black',
                           linewidth=0.5, zorder=3, alpha=0.95)

    # Paired Wilcoxon vs ABMIL (within encoder) on top of bars
    y_max = 0.97
    for i_enc, enc in enumerate(ENC_ORDER):
        base = data[enc]['ABMIL']
        if len(base) < 3:
            continue
        for i_a, a in enumerate(AGG_ORDER):
            if a == 'ABMIL':
                continue
            v = data[enc][a]
            if len(v) != len(base):
                continue
            try:
                stat, p = wilcoxon(v, base, zero_method='wilcox')
            except ValueError:
                continue
            ann = p_to_stars(p)
            if ann == 'ns':
                continue
            xpos = x[i_a] + offsets[i_enc]
            ax.text(xpos, y_max, ann, ha='center', va='bottom',
                    fontsize=7.5, color='black')

    # Significance legend
    sig_legend = ('Paired Wilcoxon vs ABMIL (same encoder, 5 folds): '
                  '* $p<0.05$, ** $p<0.01$, *** $p<0.001$')
    ax.text(0.5, -0.27, sig_legend, ha='center', va='top',
            transform=ax.transAxes, fontsize=7.5, color='#555555')

    ax.set_xticks(x)
    ax.set_xticklabels(AGG_ORDER, rotation=20, ha='right')
    ax.set_ylabel(r'Quadratic-weighted $\kappa$ (5-fold CV)')
    ax.set_ylim(0.90, 0.99)
    ax.set_xlabel('Aggregator')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
              title='Encoder', title_fontsize=8)
    ax.set_title('Within-distribution PANDA QWK by encoder and aggregator',
                 fontsize=10)
    ax.grid(axis='x', visible=False)

    save_fig(fig, 'fig5_factorial_bars_pvalues')
    plt.close(fig)
    print('DONE: fig5_factorial_bars_pvalues saved.')


if __name__ == '__main__':
    main()
