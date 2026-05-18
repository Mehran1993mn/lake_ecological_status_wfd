"""
Figure 1: Ecological-status differences in lake water-quality and morphometric variables.

This script creates a 6 x 2 grid of half-violin and percentile-box plots for selected
lake variables grouped by ecological status class. The figure is saved as PNG and SVG.

Expected input:
    data/imputedmulti.txt

Default output:
    images/figure1_status_variable_distributions_6x2_grid.png
    images/figure1_status_variable_distributions_6x2_grid.svg

Run from the project folder:
    python scripts/figure1_status_variable_distributions.py

Optional arguments:
    python scripts/figure1_status_variable_distributions.py --data data/imputedmulti.txt --out-dir images
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.patches import Rectangle
from scipy.stats import kruskal, mannwhitneyu

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# DEFAULT USER SETTINGS
# =========================================================
DEFAULT_DATA_PATH = Path("data") / "imputedmulti.txt"
DEFAULT_STATUS_COLUMN = "ecologystatus"
DEFAULT_OUTPUT_DIR = Path("images")

STATUS_ORDER = ["Bad/Poor", "Moderate", "Good", "High"]
STATUS_COLORS = {
    "Bad/Poor": "#ef4444",  # red
    "Moderate": "#f59e0b",  # orange
    "Good": "#4caf50",      # green
    "High": "#3b82f6",      # blue
}

VARIABLES = [
    "meanTP", "meanTN", "meanTurb", "meanWC", "meanCond",
    "bottom_DO_min", "surface_DO_min", "bottom_pH_min",
    "surface_pH_max", "area_m2", "SECCIDEPTH", "NEWDEPTH",
]

TITLE_MAP = {
    "meanTP": "Total phosphorus (micrograms/L)",
    "meanTN": "Total nitrogen (micrograms/L)",
    "meanTurb": "Turbidity (FNU)",
    "meanWC": "Colour (mg Pt)",
    "meanCond": "Conductivity (mS/m)",
    "bottom_DO_min": "Bottom-layer minimum DO (mg/L)",
    "surface_DO_min": "Surface-layer minimum DO (mg/L)",
    "bottom_pH_min": "Bottom-layer minimum pH",
    "surface_pH_max": "Surface-layer maximum pH",
    "area_m2": "Area (m2)",
    "SECCIDEPTH": "Secchi depth (m)",
    "NEWDEPTH": "Maximum depth (m)",
}

# Figure style
FIGSIZE = (15, 26)
DPI = 300
FONT_FAMILY = "Times New Roman"
AXIS_LINEWIDTH = 1.0
BOX_LINEWIDTH = 1.0
WHISKER_LINEWIDTH = 1.0
MEDIAN_LINEWIDTH = 1.2
MEAN_MARKER_SIZE = 4.0
Y_LABEL_SIZE = 14
Y_TICK_SIZE = 12
KW_TEXT_SIZE = 12


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_status(value):
    """Convert ecological status labels or numeric codes to four standard classes."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())

    mapping = {
        "bad": "Bad/Poor",
        "poor": "Bad/Poor",
        "bad poor": "Bad/Poor",
        "poor bad": "Bad/Poor",
        "bad/poor": "Bad/Poor",
        "poor/bad": "Bad/Poor",
        "moderate": "Moderate",
        "good": "Good",
        "high": "High",
        "0": "Bad/Poor",
        "1": "Bad/Poor",
        "2": "Moderate",
        "3": "Good",
        "4": "High",
    }

    if text in mapping:
        return mapping[text]
    if "bad" in text or "poor" in text:
        return "Bad/Poor"
    if "moderate" in text:
        return "Moderate"
    if text == "good":
        return "Good"
    if text == "high":
        return "High"
    return np.nan


def read_table(path):
    """Read a tabular text file using common separators."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Place the data file in the data folder or pass --data PATH."
        )

    for sep in ["\t", ";", ",", None]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    raise ValueError("Could not read the file. Check the separator or file path.")


def lighten_color(color, blend=0.62):
    """Blend a color with white."""
    rgb = np.array(mcolors.to_rgb(color))
    mixed = rgb * (1 - blend) + np.array([1.0, 1.0, 1.0]) * blend
    return tuple(mixed)


def setup_axis_style(ax, is_top=False):
    """Apply consistent axis formatting."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LINEWIDTH)

    ax.set_facecolor("white")
    ax.grid(False)

    if is_top:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(axis="y", labelsize=Y_TICK_SIZE, width=AXIS_LINEWIDTH, length=4)
        ax.tick_params(axis="x", bottom=True, top=False, labelsize=Y_TICK_SIZE)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xticks([1.0, 2.0, 3.0, 4.0])
        ax.set_xticklabels(STATUS_ORDER, rotation=0, ha="center")


def configure_numeric_axis(ax, variable):
    """Use scientific notation for lake area."""
    if variable == "area_m2":
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)


def draw_half_violin(ax, values, pos, color, width=0.5):
    """Draw the right half of a violin plot for one ecological status group."""
    vals = pd.Series(values).dropna().astype(float).values
    if len(vals) < 2:
        return

    vp = ax.violinplot(
        [vals],
        positions=[pos],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = vp["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(1.0)

    vertices = body.get_paths()[0].vertices
    center = np.mean(vertices[:, 0])
    vertices[:, 0] = np.maximum(vertices[:, 0], center)


def summarize_percentile_box(vals):
    """Calculate percentiles and mean for the custom box overlay."""
    vals = pd.Series(vals).dropna().astype(float).values
    if len(vals) == 0:
        return None

    return {
        "p05": np.percentile(vals, 5),
        "q25": np.percentile(vals, 25),
        "median": np.percentile(vals, 50),
        "q75": np.percentile(vals, 75),
        "p95": np.percentile(vals, 95),
        "mean": np.mean(vals),
    }


def draw_percentile_box(ax, x, stats, color, box_width=0.25, cap_width=0.1):
    """Draw a percentile box showing 5th, 25th, 50th, 75th, 95th percentile and mean."""
    if stats is None:
        return

    gap = 0.02
    right_edge = x - gap
    left_edge = right_edge - box_width
    center_x = (left_edge + right_edge) / 2

    ax.plot(
        [center_x, center_x],
        [stats["p05"], stats["p95"]],
        color="black",
        lw=WHISKER_LINEWIDTH,
        zorder=3,
    )
    ax.plot(
        [center_x - cap_width / 2, center_x + cap_width / 2],
        [stats["p05"], stats["p05"]],
        color="black",
        lw=WHISKER_LINEWIDTH,
        zorder=3,
    )
    ax.plot(
        [center_x - cap_width / 2, center_x + cap_width / 2],
        [stats["p95"], stats["p95"]],
        color="black",
        lw=WHISKER_LINEWIDTH,
        zorder=3,
    )

    rect = Rectangle(
        (left_edge, stats["q25"]),
        box_width,
        stats["q75"] - stats["q25"],
        facecolor=lighten_color(color, 0.65),
        edgecolor="black",
        linewidth=BOX_LINEWIDTH,
        zorder=4,
    )
    ax.add_patch(rect)

    ax.plot(
        [left_edge, right_edge],
        [stats["median"], stats["median"]],
        color="black",
        lw=MEDIAN_LINEWIDTH,
        zorder=5,
    )
    ax.plot(
        center_x,
        stats["mean"],
        marker="o",
        markersize=MEAN_MARKER_SIZE,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.5,
        linestyle="None",
        zorder=6,
    )


def compute_axis_limits_from_percentiles(grouped_values):
    """Set axis limits from 5th and 95th percentiles to reduce influence of extreme outliers."""
    minima, maxima = [], []
    for vals in grouped_values:
        vals = pd.Series(vals).dropna().astype(float).values
        if len(vals) == 0:
            continue
        minima.append(np.percentile(vals, 5))
        maxima.append(np.percentile(vals, 95))

    if not minima or not maxima:
        return None, None

    ymin, ymax = float(min(minima)), float(max(maxima))
    yrng = ymax - ymin
    if not np.isfinite(yrng) or yrng <= 0:
        yrng = max(abs(ymax), 1.0) * 0.1

    return ymin - 0.03 * yrng, ymax + 0.04 * yrng


def get_kw_stats(grouped_values):
    """Run Kruskal-Wallis test across ecological status groups."""
    valid_groups = [np.asarray(g, dtype=float) for g in grouped_values if len(g) > 0]
    if len(valid_groups) < 2:
        return "K-W: not available", 1.0

    _, p_val = kruskal(*valid_groups)
    text = "K-W p < 0.001" if p_val < 0.001 else f"K-W p = {p_val:.3f}"
    return text, p_val


def compute_pairwise_stats(grouped_values):
    """Run pairwise Mann-Whitney U tests and calculate absolute Cliff-type delta."""
    comparisons = []
    n_groups = len(grouped_values)

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            vals1 = pd.Series(grouped_values[i]).dropna().astype(float).values
            vals2 = pd.Series(grouped_values[j]).dropna().astype(float).values

            if len(vals1) < 2 or len(vals2) < 2:
                continue

            _, p_val = mannwhitneyu(vals1, vals2, alternative="two-sided")

            mat = np.sign(np.subtract.outer(vals1, vals2))
            delta = np.abs(np.mean(mat))

            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                continue

            comparisons.append({
                "x1": i + 1.0,
                "x2": j + 1.0,
                "text": f"{stars} |delta|={delta:.2f}",
                "dist": j - i,
            })

    comparisons.sort(key=lambda x: x["dist"])
    levels = {}
    for comp in comparisons:
        level = 1
        while True:
            overlap = False
            for existing in levels.get(level, []):
                if not (comp["x2"] <= existing["x1"] or comp["x1"] >= existing["x2"]):
                    overlap = True
                    break
            if not overlap:
                levels.setdefault(level, []).append(comp)
                comp["level"] = level
                break
            level += 1

    return comparisons


def add_significance_brackets(ax_top, comparisons, kw_text):
    """Add Kruskal-Wallis text and pairwise significance brackets above each panel."""
    if not comparisons:
        ax_top.set_ylim(0, 1)
        ax_top.text(4.55, 0.5, kw_text, ha="right", va="center", fontsize=KW_TEXT_SIZE)
        return

    max_level = max(c["level"] for c in comparisons)
    ax_top.set_ylim(0, max_level + 1)

    kw_y = max_level + 0.2
    ax_top.text(4.55, kw_y, kw_text, ha="right", va="center", fontsize=KW_TEXT_SIZE)

    for comp in comparisons:
        x1, x2, y, text = comp["x1"], comp["x2"], comp["level"], comp["text"]
        tick_len = 0.25
        ax_top.plot([x1, x1, x2, x2], [y - tick_len, y, y, y - tick_len], color="black", lw=0.8)

        bbox_props = dict(boxstyle="square,pad=0.1", fc="white", ec="none")
        ax_top.text(
            (x1 + x2) / 2,
            y,
            text,
            ha="center",
            va="center",
            fontsize=Y_TICK_SIZE,
            color="black",
            bbox=bbox_props,
        )


def make_figure(data_path, status_column, output_dir):
    """Load data, make the 6 x 2 plot grid, and save output files."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(data_path)
    if status_column not in df.columns:
        raise KeyError(f"Column '{status_column}' not found in the input data.")

    missing_variables = [col for col in VARIABLES if col not in df.columns]
    if missing_variables:
        raise KeyError("These required columns are missing: " + ", ".join(missing_variables))

    df = df.copy()
    df[status_column] = df[status_column].apply(normalize_status)
    df = df[df[status_column].isin(STATUS_ORDER)].copy()

    for col in VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig = plt.figure(figsize=FIGSIZE)
    gs_outer = fig.add_gridspec(nrows=6, ncols=2, hspace=0.25, wspace=0.15)

    positions = [1.0, 2.0, 3.0, 4.0]
    colors = [STATUS_COLORS[s] for s in STATUS_ORDER]

    for idx, variable in enumerate(VARIABLES):
        row, col = idx // 2, idx % 2

        gs_inner = gs_outer[row, col].subgridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.0)
        ax_top = fig.add_subplot(gs_inner[0])
        ax = fig.add_subplot(gs_inner[1])

        sub = df[[status_column, variable]].dropna().copy()
        grouped = [
            sub.loc[sub[status_column] == status, variable].dropna().astype(float).values
            for status in STATUS_ORDER
        ]

        for vals, pos, color in zip(grouped, positions, colors):
            draw_half_violin(ax, vals, pos=pos, color=color, width=0.5)
            stats = summarize_percentile_box(vals)
            draw_percentile_box(ax, x=pos, stats=stats, color=color)

        setup_axis_style(ax, is_top=False)
        setup_axis_style(ax_top, is_top=True)
        configure_numeric_axis(ax, variable)

        ax.set_xlim(0.5, 4.6)
        ax_top.set_xlim(0.5, 4.6)
        ax.set_ylabel(TITLE_MAP.get(variable, variable), fontsize=Y_LABEL_SIZE)

        lower, upper = compute_axis_limits_from_percentiles(grouped)
        if lower is not None and upper is not None:
            ax.set_ylim(lower, upper)

        kw_text, kw_pval = get_kw_stats(grouped)

        comparisons = []
        if kw_pval < 0.05:
            comparisons = compute_pairwise_stats(grouped)

        add_significance_brackets(ax_top, comparisons, kw_text)

    final_output_png = output_dir / "figure1_status_variable_distributions_6x2_grid.png"
    final_output_svg = output_dir / "figure1_status_variable_distributions_6x2_grid.svg"

    fig.savefig(final_output_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(final_output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Combined figure saved successfully:")
    print(final_output_png)
    print(final_output_svg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create ecological-status distribution figure for lake variables."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to input table. Default: data/imputedmulti.txt",
    )
    parser.add_argument(
        "--status-column",
        default=DEFAULT_STATUS_COLUMN,
        help="Name of ecological status column. Default: ecologystatus",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for figures. Default: images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    make_figure(
        data_path=args.data,
        status_column=args.status_column,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
