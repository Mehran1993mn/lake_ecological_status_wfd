"""
Figure 3: Overall calibration-performance heatmap across ecological-status classification models.

This script creates a heatmap comparing probability calibration using Expected Calibration
Error (ECE) and the Brier score scaled as Brier/10. The values are entered directly from
the model-summary results, so no external data file is required.

Default output:
    images/figure3_calibration_ece_brier_heatmap.png
    images/figure3_calibration_ece_brier_heatmap.svg

Run from the project folder:
    python scripts/figure3_calibration_heatmap.py
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize


# =========================================================
# DEFAULT SETTINGS
# =========================================================
DEFAULT_OUTPUT_DIR = Path("images")
DPI = 300

MODELS = ["XGBoost", "RF", "SVM", "BNN", "ANN", "TabNet"]
METRICS = ["ECE", "Brier / 10"]

# Original order in each list:
# [Bad/Poor, Moderate, Good, High, Overall, Binary]
ECE_TABLE = {
    "TabNet":  [0.018822, 0.043347, 0.092468, 0.035392, 0.083469, 0.018973],
    "ANN":     [0.039799, 0.036276, 0.145900, 0.067284, 0.062346, 0.012878],
    "XGBoost": [0.018270, 0.035208, 0.045916, 0.025227, 0.048142, 0.068659],
    "SVM":     [0.019229, 0.022607, 0.032672, 0.035230, 0.052608, 0.017172],
    "RF":      [0.068642, 0.057356, 0.113160, 0.035658, 0.041004, 0.033292],
    "BNN":     [0.009509, 0.015719, 0.025856, 0.024685, 0.023065, 0.014228],
}

BRIER_TABLE = {
    "TabNet":  [0.029090, 0.110291, 0.204668, 0.120455, 0.464504, 0.185515],
    "ANN":     [0.029970, 0.109316, 0.215513, 0.127927, 0.482726, 0.194819],
    "XGBoost": [0.026845, 0.111338, 0.194663, 0.117399, 0.450245, 0.193806],
    "SVM":     [0.026149, 0.105315, 0.201455, 0.129665, 0.462585, 0.170244],
    "RF":      [0.032260, 0.110232, 0.207650, 0.119945, 0.470086, 0.203956],
    "BNN":     [0.024124, 0.099754, 0.192002, 0.123834, 0.439713, 0.164367],
}


def configure_style():
    """Apply publication-style typography and output settings."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.titleweight": "normal",
        "axes.labelsize": 18,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "svg.fonttype": "none",
    })


def get_text_color(value, cmap, norm):
    """Return white or black text depending on the heatmap cell luminance."""
    rgba = cmap(norm(value))
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.55 else "white"


def make_figure(output_dir=DEFAULT_OUTPUT_DIR):
    """Create and save the calibration heatmap."""
    configure_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_index = 4
    ece_overall = np.array([ECE_TABLE[m][overall_index] for m in MODELS], dtype=float)
    brier_overall_div10 = np.array([BRIER_TABLE[m][overall_index] / 10.0 for m in MODELS], dtype=float)
    data = np.column_stack([ece_overall, brier_overall_div10])

    norm = Normalize(vmin=float(np.min(data)), vmax=float(np.max(data)))
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)

    ax.set_title("Overall calibration performance across models", pad=18)
    ax.set_ylabel("Models", labelpad=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(
                j, i, f"{value:.3f}",
                ha="center", va="center",
                fontsize=15,
                color=get_text_color(value, cmap, norm),
            )

    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Displayed value", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    fig.text(
        0.5, 0.02,
        "Shown metrics are overall ECE and overall Brier score scaled as Brier/10. Lower values indicate better calibration.",
        ha="center", fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    png_file = output_dir / "figure3_calibration_ece_brier_heatmap.png"
    svg_file = output_dir / "figure3_calibration_ece_brier_heatmap.svg"
    fig.savefig(png_file, format="png", bbox_inches="tight", facecolor="white")
    fig.savefig(svg_file, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:")
    print(png_file)
    print(svg_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Create Figure 3 calibration heatmap.")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where figures are saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.out_dir)
