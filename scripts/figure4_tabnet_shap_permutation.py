"""
Figure 4: TabNet feature importance from SHAP and permutation importance.

This script creates a mirrored horizontal bar plot comparing normalized mean absolute
SHAP importance with permutation importance for the TabNet model. The script expects
two tab-separated input files.

Expected inputs:
    data/permutation/Permutation.txt
    data/shap/SHAP.txt

Default output:
    images/figure4_tabnet_shap_permutation_combined_normalized.png
    images/figure4_tabnet_shap_permutation_combined_normalized.svg

Run from the project folder:
    python scripts/figure4_tabnet_shap_permutation.py

Optional arguments:
    python scripts/figure4_tabnet_shap_permutation.py --perm data/permutation/Permutation.txt --shap data/shap/SHAP.txt --out-dir images
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# DEFAULT SETTINGS
# =========================================================
DEFAULT_PERM_PATH = Path("data") / "permutation" / "Permutation.txt"
DEFAULT_SHAP_PATH = Path("data") / "shap" / "SHAP.txt"
DEFAULT_OUTPUT_DIR = Path("images")

FONT_FAMILY = "Times New Roman"
FONT_BASE = 13
FONT_TITLE = 22
FONT_AXIS = 18
FONT_TICKS = 15
FONT_FEAT = 18
FIG_W, FIG_H = 14, 10
DPI = 300

TOP_N_FEATURES = 12
NORMALIZE_SHAP = True
PERM_XMAX = None
SHAP_XMAX = None

FEATURE_NAME_MAP = {
    "meanTP": "TP",
    "meanTN": "TN",
    "tyyppi (lake type)": "NLT",
    "meanTurb": "Turbidity",
    "meanWC": "WC",
    "SECCIDEPTH": "SD",
    "surface_pH_max": r"$\mathrm{pH}_{S,\max}$",
    "bottom_DO_min": r"$\mathrm{DO}_{B,\min}$",
    "thermal_class": "TS",
    "area_m2": "Area",
    "meanCond": "Conductivity",
    "NEWDEPTH": "MD",
    "surface_DO_min": r"$\mathrm{DO}_{S,\min}$",
    "bottom_pH_min": r"$\mathrm{pH}_{B,\min}$",
}


def configure_style():
    """Apply publication-style typography and output settings."""
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = FONT_BASE
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["svg.fonttype"] = "none"


def load_and_clean_table(path):
    """Load a tab-separated table and strip whitespace from text columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. Place the file in the expected data folder or pass the path as an argument.")

    df = pd.read_csv(path, sep="\t", dtype=str)
    return df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)


def numeric_column(series):
    """Convert strings to numeric values while accepting comma or dot decimal separators."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def extract_importance_table(raw, column_map, table_name):
    """Extract feature, importance, and confidence-interval columns."""
    missing = [v for v in column_map.values() if v not in raw.columns]
    if missing:
        raise KeyError(f"Missing {table_name} columns: {missing}")

    df = raw[[column_map["feature"], column_map["mean"], column_map["low"], column_map["high"]]].copy()
    df.columns = ["Feature", "Importance", "CI_Low", "CI_High"]

    for col in ["Importance", "CI_Low", "CI_High"]:
        df[col] = numeric_column(df[col])

    df["Feature"] = df["Feature"].astype(str).str.strip()
    df = df.dropna(subset=["Feature", "Importance", "CI_Low", "CI_High"]).copy()
    df = df[df["Feature"] != ""]
    df = df[df["Feature"].str.lower() != "nan"]

    df = (
        df.groupby("Feature", as_index=False)
          .agg({"Importance": "mean", "CI_Low": "mean", "CI_High": "mean"})
    )
    return df


def make_figure(perm_path=DEFAULT_PERM_PATH, shap_path=DEFAULT_SHAP_PATH, output_dir=DEFAULT_OUTPUT_DIR):
    """Create and save the TabNet SHAP-vs-permutation figure."""
    configure_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perm_raw = load_and_clean_table(perm_path)
    shap_raw = load_and_clean_table(shap_path)

    perm_cols = {
        "feature": "feature_or_groupTAB",
        "mean": "importance_meanTAB",
        "low": "ci95_lowTAB",
        "high": "ci95_highTAB",
    }

    shap_cols = {
        "feature": "featureTAB",
        "mean": "mean_abs_shapTAB",
        "low": "ci95_lowTAB",
        "high": "ci95_highTAB",
    }

    perm_df = extract_importance_table(perm_raw, perm_cols, "permutation")
    shap_df = extract_importance_table(shap_raw, shap_cols, "SHAP")

    if NORMALIZE_SHAP:
        total_shap = shap_df["Importance"].sum()
        if total_shap <= 0 or pd.isna(total_shap):
            raise ValueError("Normalization failed: total SHAP importance is invalid.")
        shap_df["Importance"] = shap_df["Importance"] / total_shap
        shap_df["CI_Low"] = shap_df["CI_Low"] / total_shap
        shap_df["CI_High"] = shap_df["CI_High"] / total_shap

    # Rank features by the average rank from both methods.
    perm_rank = perm_df[["Feature", "Importance"]].copy()
    perm_rank["perm_rank"] = perm_rank["Importance"].rank(ascending=False, method="average")

    shap_rank = shap_df[["Feature", "Importance"]].copy()
    shap_rank["shap_rank"] = shap_rank["Importance"].rank(ascending=False, method="average")

    rank_df = pd.merge(
        perm_rank[["Feature", "perm_rank"]],
        shap_rank[["Feature", "shap_rank"]],
        on="Feature",
        how="outer",
    )

    rank_df["perm_rank"] = rank_df["perm_rank"].fillna(rank_df["perm_rank"].max())
    rank_df["shap_rank"] = rank_df["shap_rank"].fillna(rank_df["shap_rank"].max())
    rank_df["mean_rank"] = rank_df[["perm_rank", "shap_rank"]].mean(axis=1)

    feature_order = rank_df.sort_values("mean_rank")["Feature"].tolist()
    if TOP_N_FEATURES is not None:
        feature_order = feature_order[:TOP_N_FEATURES]

    perm_df = perm_df[perm_df["Feature"].isin(feature_order)].copy()
    shap_df = shap_df[shap_df["Feature"].isin(feature_order)].copy()

    perm_df = perm_df.set_index("Feature").reindex(feature_order).reset_index()
    shap_df = shap_df.set_index("Feature").reindex(feature_order).reset_index()

    display_feature_order = [FEATURE_NAME_MAP.get(f, f) for f in feature_order]
    y = np.arange(len(feature_order))

    perm_vals = perm_df["Importance"].fillna(0).values
    perm_low = perm_df["CI_Low"].fillna(0).values
    perm_high = perm_df["CI_High"].fillna(0).values

    shap_vals = shap_df["Importance"].fillna(0).values
    shap_low = shap_df["CI_Low"].fillna(0).values
    shap_high = shap_df["CI_High"].fillna(0).values

    shap_plot = -shap_vals
    shap_err_low = np.abs(shap_vals - shap_low)
    shap_err_high = np.abs(shap_high - shap_vals)

    perm_err_low = np.abs(perm_vals - perm_low)
    perm_err_high = np.abs(perm_high - perm_vals)

    left_max = np.max(shap_vals) if SHAP_XMAX is None else SHAP_XMAX
    right_max = np.max(perm_vals) if PERM_XMAX is None else PERM_XMAX
    left_lim = left_max * 1.20
    right_lim = right_max * 1.20

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    color_shap = "#6a51a3"
    color_perm = "#2b8cbe"

    ax.barh(
        y, shap_plot,
        height=0.72,
        color=color_shap,
        edgecolor="black",
        linewidth=0.6,
        xerr=np.vstack([shap_err_high, shap_err_low]),
        capsize=2,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
    )

    ax.barh(
        y, perm_vals,
        height=0.72,
        color=color_perm,
        edgecolor="black",
        linewidth=0.6,
        xerr=np.vstack([perm_err_low, perm_err_high]),
        capsize=2,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
    )

    ax.axvline(0, color="black", linewidth=1.2)

    for i in range(len(feature_order) - 1):
        ax.axhline(i + 0.5, color="lightgray", linewidth=0.8, zorder=0)

    ax.set_yticks([])
    for yi, label in zip(y, display_feature_order):
        ax.text(
            0, yi, label,
            ha="center", va="center",
            fontsize=FONT_FEAT,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
            zorder=5,
        )

    ax.set_xlim(-left_lim, right_lim)
    ax.set_ylim(-0.7, len(feature_order) - 0.3)
    ax.invert_yaxis()

    left_ticks = np.linspace(0, left_lim, 4)
    right_ticks = np.linspace(0, right_lim, 4)
    xticks = np.concatenate([-left_ticks[::-1][:-1], right_ticks])
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):.2f}" for t in xticks], fontsize=FONT_TICKS)

    ax.set_xlabel("")
    ax.set_title("TabNet feature importance: SHAP vs permutation", fontsize=FONT_TITLE, pad=18)

    ax.text(
        -left_lim * 0.72, -1.0, "SHAP",
        ha="center", va="bottom",
        fontsize=FONT_AXIS, fontweight="bold", color=color_shap,
    )
    ax.text(
        right_lim * 0.72, -1.0, "Permutation",
        ha="center", va="bottom",
        fontsize=FONT_AXIS, fontweight="bold", color=color_perm,
    )

    note = "Left: normalized mean |SHAP| (95% CI)" if NORMALIZE_SHAP else "Left: mean |SHAP| (95% CI)"
    ax.text(
        0.5, -0.10,
        note + "    |    Right: mean change in F1-score (95% CI)",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=13,
    )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.1)

    plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.97])

    suffix = "normalized" if NORMALIZE_SHAP else "raw"
    png_file = output_dir / f"figure4_tabnet_shap_permutation_combined_{suffix}.png"
    svg_file = output_dir / f"figure4_tabnet_shap_permutation_combined_{suffix}.svg"

    fig.canvas.draw()
    fig.savefig(svg_file, format="svg", bbox_inches="tight")
    fig.savefig(png_file, format="png", bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(png_file)
    print(svg_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Create Figure 4 TabNet SHAP-vs-permutation importance figure.")
    parser.add_argument("--perm", default=DEFAULT_PERM_PATH, help="Path to the permutation-importance table.")
    parser.add_argument("--shap", default=DEFAULT_SHAP_PATH, help="Path to the SHAP-importance table.")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where figures are saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.perm, args.shap, args.out_dir)
