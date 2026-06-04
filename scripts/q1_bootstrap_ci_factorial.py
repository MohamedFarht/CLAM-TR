"""Q1 T1: bootstrap 95% CI from 5-fold per-fold QWK for the 6 × 2 factorial.

Reads study/main/figures/retention_ablation_per_fold.csv.
Outputs:
  - papers/NCA 2026/figures/factorial_ci_inline.csv  (one row per encoder x aggregator)
  - papers/NCA 2026/figures/factorial_ci_inline.tex  (LaTeX snippet with inline CI subscripts)

The CI is computed via percentile bootstrap on the 5 per-fold QWK values
(B=10000 resamples, q025/q975 percentiles). This is the standard "fold-level
bootstrap" reported in MIL-aggregator papers (e.g. DTFD-MIL Tab 1 with 95% CI
in cell subscripts, Kurata Tab 2 with 95% CI on bars).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

INPUT_CSV = Path('study/main/figures/retention_ablation_per_fold.csv')
OUT_CSV = Path('papers/NCA 2026/figures/factorial_ci_inline.csv')
OUT_TEX = Path('papers/NCA 2026/figures/factorial_ci_inline.tex')

N_BOOT = 10000
SEED = 42
HP = 'best'  # the headline configuration (gamma bestHP per CODE_INDEX)

AGG_ORDER = ['ABMIL', 'DualBranch', 'SpatialMult', 'SpatialAdd', 'RetNet',
             'SpatialMultRetNet']
ENC_ORDER = ['UNI', 'UNI2-h']


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile bootstrap CI for the mean of a 5-element array."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[df['hp'] == HP].copy()

    rows = []
    tex_rows = []
    for enc in ENC_ORDER:
        for agg in AGG_ORDER:
            sub = df[(df['encoder'] == enc) & (df['variant'] == agg)]
            if sub.empty:
                # ABMIL UNI2-h only has fold 0 — fall back gracefully
                rows.append({'encoder': enc, 'aggregator': agg, 'n_folds': 0,
                             'mean': np.nan, 'std': np.nan,
                             'ci_lo': np.nan, 'ci_hi': np.nan,
                             'inline': 'n/a'})
                continue
            qwks = sub['full_qwk'].values
            mean = float(qwks.mean())
            std = float(qwks.std(ddof=1)) if len(qwks) > 1 else 0.0
            if len(qwks) >= 3:
                ci_lo, ci_hi = bootstrap_ci(qwks)
            else:
                ci_lo, ci_hi = mean, mean
            inline = f'{mean:.4f}_{{[{ci_lo:.4f},{ci_hi:.4f}]}}'
            rows.append({'encoder': enc, 'aggregator': agg, 'n_folds': len(qwks),
                         'mean': mean, 'std': std,
                         'ci_lo': ci_lo, 'ci_hi': ci_hi, 'inline': inline})
            tex_rows.append(
                f'% {enc:7s} {agg:18s}  '
                f'$\\kappa_w = {mean:.4f}\\ [{ci_lo:.4f},\\ {ci_hi:.4f}]$  '
                f'($\\pm{std:.4f}$, $n={len(qwks)}$)'
            )

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f'Wrote {OUT_CSV}')
    print(out_df.to_string(index=False))

    with open(OUT_TEX, 'w', encoding='utf-8') as f:
        f.write('% Factorial 6-aggregator x 2-encoder mean[95% CI] from 5-fold CV\n')
        f.write(f'% B={N_BOOT} percentile bootstrap, seed={SEED}, hp={HP}\n%\n')
        for r in tex_rows:
            f.write(r + '\n')
    print(f'Wrote {OUT_TEX}')


if __name__ == '__main__':
    main()
