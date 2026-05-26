"""Q1 NCA figure style — apply at top of every figure-generation script.

Imports as a module: `from q1_fig_style import apply_style, COLORS, FIG_SINGLE, FIG_DOUBLE, save_fig`
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path('papers/NCA 2026/figures')

COLORS = {
    'blue':    '#648FFF',
    'purple':  '#785EF0',
    'magenta': '#DC267F',
    'orange':  '#FE6100',
    'yellow':  '#FFB000',
    'teal':    '#009E73',
    'gray':    '#999999',
    'black':   '#000000',
}
COLOR_LIST = list(COLORS.values())

FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.0, 3.5)
FIG_DOUBLE_TALL = (7.0, 5.0)


def apply_style():
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['figure.titlesize'] = 11
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 5
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['ytick.major.width'] = 0.8
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLOR_LIST)


def save_fig(fig, name, formats=('pdf', 'png')):
    """Save to papers/NCA 2026/figures/{name}.{fmt}."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = FIG_DIR / f'{name}.{fmt}'
        fig.savefig(out, format=fmt, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f'  Saved: {out}')
