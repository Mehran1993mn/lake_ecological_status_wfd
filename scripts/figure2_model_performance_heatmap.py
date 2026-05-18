"""
Figure 2: Overall test-performance heatmap across ecological-status classification models.

This script creates a heatmap comparing model performance using F1-score, Matthews
correlation coefficient (MCC), and accuracy. The values are entered directly from the
model-summary results, so no external data file is required.

Default output:
    images/figure2_overall_test_performance_heatmap.png
    images/figure2_overall_test_performance_heatmap.svg

Run from the project folder:
    python scripts/figure2_model_performance_heatmap.py
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

MODELS = ["XGBoost", "RF", "SVM", "BNN", "ANN", "TabNet", "Ensemble"]
METRICS = ["F1", "MCC", "Accuracy"]

SUMMARY_TEST = {
    "XGBoost":  {"F1": 0.66, "MCC": 0.50, "Accuracy": 0.67},
    "RF":       {"F1": 0.64, "MCC": 0.47, "Accuracy": 0.65},
    "SVM":      {"F1": 0.66, "MCC": 0.49, "Accuracy": 0.68},
    "BNN":      {"F1": 0.66, "MCC": 0.50, "Accuracy": 0.64},
    "ANN":      {"F1": 0.64, "MCC": 0.47, "Accuracy": 0.68},
    "TabNet":   {"F1": 0.68, "MCC": 0.52, "Accuracy": 0.69},
    "Ensemble": {"F1": 0.69, "MCC": 0.53, "Accuracy": 0.70},
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
        "axes.labelsize": 19,
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
    """Create and save the model-performance heatmap."""
    configure_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.array([
        [SUMMARY_TEST[m]["F1"], SUMMARY_TEST[m]["MCC"], SUMMARY_TEST[m]["Accuracy"]]
        for m in MODELS
    ], dtype=float)

    norm = Normalize(vmin=float(np.min(data)), vmax=float(np.max(data)))
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)

    ax.set_title("Overall test performance across models", pad=18)
    ax.set_ylabel("Models", labelpad=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center",
                fontsize=15,
                color=get_text_color(data[i, j], cmap, norm),
            )

    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Score", fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    fig.text(
        0.5, 0.015,
        "Higher values indicate better model performance.",
        ha="center", fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.035, 1, 1])

    png_file = output_dir / "figure2_overall_test_performance_heatmap.png"
    svg_file = output_dir / "figure2_overall_test_performance_heatmap.svg"
    fig.savefig(png_file, format="png", bbox_inches="tight", facecolor="white")
    fig.savefig(svg_file, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:")
    print(png_file)
    print(svg_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Create Figure 2 model-performance heatmap.")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where figures are saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.out_dir)
