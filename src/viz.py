"""Shared chart style so every figure in the project looks like one system.

Palette: validated reference palette (blue <-> red diverging pair, neutral gray midpoint).
Sentiment is a polarity, so Negative=red, Neutral=gray, Positive=blue. Every mark is
direct-labelled so colour is never the only way to read a chart.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import FIG_DIR, ensure_dirs  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GRAY = "#898781"

SENTIMENT_COLORS = {"Negative": RED, "Neutral": GRAY, "Positive": BLUE}
# one-hue sequential ramp (blue) for heatmaps, light -> dark
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
        }
    )


def finish(fig, name: str, subtitle: str | None = None) -> str:
    """Save to reports/figures/<name>.png and return the path."""
    ensure_dirs()
    if subtitle:
        fig.text(0.0, -0.02, subtitle, fontsize=8.5, color=MUTED, ha="left", va="top")
    path = FIG_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)
