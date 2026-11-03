"""Shared matplotlib style so every notebook figure reads as one system."""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Fixed categorical order: slot 1 is always control, slot 2 always treatment.
CONTROL = "#2a78d6"
TREATMENT = "#eb6834"
SERIES = (CONTROL, TREATMENT, "#1baf7a", "#eda100")
ARM_COLORS = {"control": CONTROL, "treatment": TREATMENT}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "figure.figsize": (8, 4),
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
