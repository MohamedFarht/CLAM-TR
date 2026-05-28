"""Q1 T5: per-class quadratic-weighted kappa + per-class precision/recall/F1.

Ren Tab 2-3 / Bulten Tab 1 style. Breaks the aggregate kappa down into
per-class quality (one-vs-rest framing for kappa; standard sklearn for P/R/F1).

Reads PANDA 5-fold JSON predictions for RetNet-bestHP / UNI v1.
Writes:
  - papers/NCA 2026/figures/tab_per_class_kappa.csv
  - papers/NCA 2026/figures/tab_per_class_kappa.tex (LaTeX tabular snippet)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

PANDA_PRED_DIR = Path(r'G:/My Drive/CLAM_TR_Results/v20_fullscale/UNI/predictions')
OUT_CSV = Path('papers/NCA 2026/figures/tab_per_class_kappa.csv')
OUT_TEX = Path('papers/NCA 2026/figures/tab_per_class_kappa.tex')

N_CLASSES = 6
GRADE_LABELS = {0: 'Benign', 1: 'ISUP 1', 2: 'ISUP 2', 3: 'ISUP 3',
                4: 'ISUP 4', 5: 'ISUP 5'}


def load_concat(agg='RetNet', hp='bestHP'):
    yt, yp = [], []
    for fold in range(5):
        fp = PANDA_PRED_DIR / f'{agg}_{hp}_fold{fold}_predictions.json'
        with open(fp) as f:
            d = json.load(f)
        yt.extend(d['labels'])
        yp.extend(d['predictions'])
    return np.asarray(yt), np.asarray(yp)


def per_class_kappa_ovr(y_true, y_pred, k):
    """One-vs-rest binary kappa for grade k.

    'Quadratic' on a 2-class problem reduces to unweighted Cohen kappa.
    """
    yt = (y_true == k).astype(int)
    yp = (y_pred == k).astype(int)
    return cohen_kappa_score(yt, yp)


def main():
    y_true, y_pred = load_concat()

    rows = []
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0)
    overall_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic',
                                    labels=list(range(N_CLASSES)))

    for k in range(N_CLASSES):
        rows.append({
            'grade': GRADE_LABELS[k],
            'n': int(support[k]),
            'kappa_ovr': float(per_class_kappa_ovr(y_true, y_pred, k)),
            'precision': float(p[k]),
            'recall': float(r[k]),
            'f1': float(f1[k]),
        })
    rows.append({'grade': 'OVERALL ($\\kappa_w$)', 'n': int(support.sum()),
                 'kappa_ovr': float(overall_qwk),
                 'precision': float(p.mean()), 'recall': float(r.mean()),
                 'f1': float(f1.mean())})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format='%.4f')
    print(f'Wrote {OUT_CSV}')
    print(df.to_string(index=False))

    # LaTeX tabular snippet
    lines = [
        '% Per-class kappa + P/R/F1 breakdown — RetNet-bestHP / UNI v1, PANDA 5-fold CV',
        '\\begin{table}[t]',
        '\\caption{Per-class breakdown of the headline RetNet-bestHP on UNI~v1, '
        'concatenated across all five PANDA cross-validation folds ($n = 9{,}128$). '
        'Per-class $\\kappa$ is computed one-vs-rest (unweighted Cohen). '
        'Precision, recall, and F1 are sklearn defaults with macro-averaged '
        'aggregate. Overall $\\kappa_w$ is the quadratic-weighted six-class kappa.}',
        '\\label{tab:per_class_kappa}',
        '\\begin{tabular}{lcccccc}',
        '\\toprule',
        'Grade & $n$ & $\\kappa$ (OvR) & Precision & Recall & F1 \\\\',
        '\\midrule',
    ]
    for r in rows[:-1]:
        lines.append(f'{r["grade"]:9s} & {r["n"]:5d} & '
                     f'{r["kappa_ovr"]:.4f} & {r["precision"]:.4f} & '
                     f'{r["recall"]:.4f} & {r["f1"]:.4f} \\\\')
    lines.append('\\midrule')
    last = rows[-1]
    lines.append(f'\\textbf{{{last["grade"]}}} & \\textbf{{{last["n"]}}} & '
                 f'\\textbf{{{last["kappa_ovr"]:.4f}}} & '
                 f'\\textbf{{{last["precision"]:.4f}}} & '
                 f'\\textbf{{{last["recall"]:.4f}}} & '
                 f'\\textbf{{{last["f1"]:.4f}}} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    with open(OUT_TEX, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {OUT_TEX}')


if __name__ == '__main__':
    main()
