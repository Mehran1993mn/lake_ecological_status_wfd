"""
Figure 5: BNN uncertainty decomposition across ecological-status classes.

This script creates a three-panel violin/box/scatter figure for total, aleatoric, and
epistemic uncertainty from the Bayesian Neural Network (BNN) multi-class output.

Expected input:
    data/bnn/multi.txt

Default output:
    images/figure5_bnn_multiclass_uncertainty_3panel_normalized.png
    images/figure5_bnn_multiclass_uncertainty_3panel_normalized.svg
    images/figure5_bnn_multiclass_uncertainty_3panel_normalized.pdf

Run from the project folder:
    python scripts/figure5_bnn_uncertainty_decomposition.py

Optional arguments:
    python scripts/figure5_bnn_uncertainty_decomposition.py --data data/bnn/multi.txt --out-dir images
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# DEFAULT SETTINGS
# =========================================================
DEFAULT_DATA_PATH = Path("data") / "bnn" / "multi.txt"
DEFAULT_OUTPUT_DIR = Path("images")

SEP = "\t"
DECIMAL = ","

CLASS_COL = "true_label"
COL_TOTAL = "entropy_norm"
COL_ALEA = "aleatoric_entropy_norm"
COL_EPI = "epistemic_mi"
COL_EPI_NORM_MULTI = "epistemic_mi_norm"

USE_NORMALIZED = True

FONT_FAMILY = "Times New Roman"
FONT_BASE = 14
FONT_TITLE = 22
FONT_AXIS = 18
FONT_TICKS = 16
FONT_MEAN = 14

FIG_W, FIG_H = 13, 5.8
DPI = 300

YMIN = None
YMAX = None

VIOLIN_WIDTH = 0.85
BOX_WIDTH = 0.22
POINT_SIZE = 16
MEAN_MARKER_SIZE = 90
JITTER = 0.06
POINT_ALPHA = 0.18
VIOLIN_ALPHA = 0.30

CLASS_COLORS = {
    "Bad/Poor": "#c44e52",
    "Moderate": "#dd8452",
    "Good": "#55a868",
    "High": "#4c72b0",
}

PANEL_TITLES = {
    "Total uncertainty": "Total",
    "Aleatoric": "Aleatoric",
    "Epistemic": "Epistemic",
}

CLASS_ORDER = ["Bad/Poor", "Moderate", "Good", "High"]


def configure_style():
    """Apply publication-style typography and output settings."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_AXIS,
        "axes.titlesize": FONT_TITLE,
        "xtick.labelsize": FONT_TICKS,
        "ytick.labelsize": FONT_TICKS,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_uncertainty_file(path, dataset_name, epi_norm_col=None):
    """Load and clean the BNN uncertainty table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. Place the file in the expected data folder or pass --data PATH.")

    df = pd.read_csv(path, sep=SEP, decimal=DECIMAL, low_memory=False)

    needed = [CLASS_COL, COL_TOTAL, COL_ALEA, COL_EPI]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if epi_norm_col is not None and epi_norm_col in df.columns:
        df[epi_norm_col] = pd.to_numeric(df[epi_norm_col], errors="coerce")

    df = df.dropna(subset=needed).copy()
    df["dataset"] = dataset_name
    return df


def draw_violin_box_scatter(ax, data_dict, ylabel=None, title=""):
    """Draw one panel with violin, boxplot, jittered points, and a mean marker."""
    positions = np.arange(1, len(CLASS_ORDER) + 1)

    violin_data = [data_dict[cls] for cls in CLASS_ORDER]
    vp = ax.violinplot(
        violin_data,
        positions=positions,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, cls in zip(vp["bodies"], CLASS_ORDER):
        body.set_facecolor(CLASS_COLORS[cls])
        body.set_edgecolor("black")
        body.set_alpha(VIOLIN_ALPHA)
        body.set_linewidth(0.9)

    bp = ax.boxplot(
        violin_data,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=1.3, color="black"),
        whiskerprops=dict(linewidth=1.0, color="black"),
        capprops=dict(linewidth=1.0, color="black"),
    )

    for box, cls in zip(bp["boxes"], CLASS_ORDER):
        box.set_facecolor(CLASS_COLORS[cls])
        box.set_edgecolor("black")
        box.set_alpha(0.75)
        box.set_linewidth(1.0)

    rng = np.random.default_rng(42)
    for i, cls in enumerate(CLASS_ORDER, start=1):
        vals = np.asarray(data_dict[cls], dtype=float)
        if len(vals) == 0:
            continue

        jitter = rng.uniform(-JITTER, JITTER, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=POINT_SIZE,
            color=CLASS_COLORS[cls],
            alpha=POINT_ALPHA,
            edgecolors="none",
            zorder=2,
        )

        ax.scatter(
            i,
            np.mean(vals),
            s=MEAN_MARKER_SIZE,
            marker="D",
            color="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(CLASS_ORDER, rotation=0, ha="center")
    ax.set_title(title, pad=10)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)


def make_figure(data_path=DEFAULT_DATA_PATH, output_dir=DEFAULT_OUTPUT_DIR):
    """Create and save the BNN uncertainty-decomposition figure."""
    configure_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_mul = load_uncertainty_file(data_path, "Multi", epi_norm_col=COL_EPI_NORM_MULTI)

    mul_map = {
        1: "Bad/Poor",
        2: "Moderate",
        3: "Good",
        4: "High",
    }

    df_mul["class_name"] = df_mul[CLASS_COL].map(mul_map)
    df_mul = df_mul.dropna(subset=["class_name"]).copy()

    df_plot = df_mul.copy()

    if USE_NORMALIZED:
        if COL_EPI_NORM_MULTI in df_plot.columns:
            df_plot["epi_plot"] = pd.to_numeric(df_plot[COL_EPI_NORM_MULTI], errors="coerce")
        else:
            df_plot["epi_plot"] = df_plot[COL_EPI] / np.log(4)

        ylabel = "Normalized uncertainty"
        save_stub = "figure5_bnn_multiclass_uncertainty_3panel_normalized"
    else:
        df_plot["epi_plot"] = df_plot[COL_EPI]
        ylabel = "Uncertainty"
        save_stub = "figure5_bnn_multiclass_uncertainty_3panel_raw"

    long_df = pd.concat([
        df_plot[["class_name", COL_TOTAL]].rename(columns={COL_TOTAL: "value"}).assign(kind="Total uncertainty"),
        df_plot[["class_name", COL_ALEA]].rename(columns={COL_ALEA: "value"}).assign(kind="Aleatoric"),
        df_plot[["class_name", "epi_plot"]].rename(columns={"epi_plot": "value"}).assign(kind="Epistemic"),
    ], ignore_index=True)

    long_df["class_name"] = pd.Categorical(long_df["class_name"], categories=CLASS_ORDER, ordered=True)
    long_df = long_df.dropna(subset=["class_name", "value"]).copy()

    panel_kinds = ["Total uncertainty", "Aleatoric", "Epistemic"]
    panel_data = {}
    for kind in panel_kinds:
        sub = long_df[long_df["kind"] == kind].copy()
        panel_data[kind] = {
            cls: sub.loc[sub["class_name"] == cls, "value"].to_numpy()
            for cls in CLASS_ORDER
        }

    if YMIN is None or YMAX is None:
        all_vals = long_df["value"].to_numpy()
        auto_min = np.nanmin(all_vals)
        auto_max = np.nanmax(all_vals)
        pad = 0.06 * (auto_max - auto_min if auto_max > auto_min else 1.0)

        y_min = max(0, auto_min - pad) if YMIN is None else YMIN
        y_max = auto_max + pad if YMAX is None else YMAX
    else:
        y_min, y_max = YMIN, YMAX

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), dpi=DPI, sharey=True)

    for ax, kind in zip(axes, panel_kinds):
        draw_violin_box_scatter(
            ax,
            panel_data[kind],
            ylabel=ylabel if ax is axes[0] else None,
            title=PANEL_TITLES[kind],
        )
        ax.set_ylim(y_min, y_max)

    fig.text(
        0.5, 0.02,
        "Violin = distribution, box = IQR and median, white diamond = mean",
        ha="center",
        fontsize=FONT_MEAN,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_png = output_dir / f"{save_stub}.png"
    out_svg = output_dir / f"{save_stub}.svg"
    out_pdf = output_dir / f"{save_stub}.pdf"

    fig.canvas.draw()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(out_png)
    print(out_svg)
    print(out_pdf)


def parse_args():
    parser = argparse.ArgumentParser(description="Create Figure 5 BNN uncertainty-decomposition figure.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to the BNN uncertainty table.")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where figures are saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.data, args.out_dir)
