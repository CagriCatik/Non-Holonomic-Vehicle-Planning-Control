"""Unified Dark Automotive Telemetry Theme for Non-Holonomic RRT Visualizations."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

BG_COLOR: str = "#0b1120"        # Deep Space Slate (Figure background)
AXES_COLOR: str = "#0f172a"      # Dark Slate (Axes background)
CARD_COLOR: str = "#1e293b"      # Surface / Legend / Obstacle fill
OBSTACLE_EDGE: str = "#475569"   # Slate border for obstacles
GRID_COLOR: str = "#334155"      # Subtle Slate Grid lines

TEXT_PRIMARY: str = "#f8fafc"    # Bright Off-White for titles and major text
TEXT_MUTED: str = "#94a3b8"      # Slate for labels, ticks, and secondary info
TEXT_FAINT: str = "#64748b"      # Muted slate for reference lines

ACCENT_CYAN: str = "#38bdf8"     # Cyan / Sky for vehicle chassis & forward motion
ACCENT_BLUE: str = "#3b82f6"     # Vivid blue
ACCENT_EMERALD: str = "#10b981"  # Emerald green for Goal / Forward arrow
ACCENT_AMBER: str = "#f59e0b"    # Amber / Orange for Reverse gear / Steered wheels
ACCENT_ROSE: str = "#f43f5e"     # Rose / Red for Start / Reverse arrow / Limits
ACCENT_PURPLE: str = "#a855f7"   # Violet for MPC predicted horizon

WHEEL_FACE: str = "#e2e8f0"      # High-contrast light slate wheel face on dark background
WHEEL_FRONT_FACE: str = "#38bdf8"  # Distinct steered front wheel highlight
WHEEL_EDGE: str = "#0f172a"      # Dark wheel border

DARK_THEME_RC = {
    "figure.facecolor": BG_COLOR,
    "figure.edgecolor": BG_COLOR,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "figure.autolayout": False,
    "axes.facecolor": AXES_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.linewidth": 1.0,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "text.color": TEXT_PRIMARY,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.40,
    "grid.linestyle": "--",
    "grid.linewidth": 0.75,
    "legend.facecolor": CARD_COLOR,
    "legend.edgecolor": GRID_COLOR,
    "legend.labelcolor": TEXT_PRIMARY,
    "legend.fontsize": 8.0,
    "legend.framealpha": 0.92,
    "savefig.facecolor": BG_COLOR,
    "savefig.edgecolor": BG_COLOR,
    "savefig.dpi": 140,
}


def apply_dark_theme() -> None:
    """Apply the unified dark telemetry theme to matplotlib globally."""
    mpl.rcParams.update(DARK_THEME_RC)


def style_axis(
    ax: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Style a single matplotlib Axes according to the dark telemetry theme."""
    ax.set_facecolor(AXES_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.40, color=GRID_COLOR, linewidth=0.75)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.0, color=TEXT_MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.0, color=TEXT_MUTED)

