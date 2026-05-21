"""
Reproducibility wrapper for the Finnish lake ecological-status project.

This script checks required folders/files and runs the reproducible figure scripts.
Large raw data files must be placed manually in inputs/raw_data/.
"""

import subprocess
from pathlib import Path


def run_step(command, description):
    print(f"\n=== {description} ===")
    subprocess.run(command, shell=True, check=True)


def main():
    print("Starting reproducibility workflow...")

    Path("images").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    raw_data_files = [
        "inputs/raw_data/station_with_ecology_values.csv",
        "inputs/raw_data/tn323.csv",
        "inputs/raw_data/tp315.csv",
        "inputs/raw_data/turbidity7677.csv",
    ]

    missing_raw = [f for f in raw_data_files if not Path(f).exists()]

    if missing_raw:
        print("\nLarge raw data files are missing.")
        print("This is expected when the repository is cloned from GitHub.")
        print("Place the following files in inputs/raw_data/ before running the full data-processing workflow:")
        for f in missing_raw:
            print(f" - {f}")

    required_figure_input = Path("data/imputedmulti.txt")
    if not required_figure_input.exists():
        raise FileNotFoundError(
            "Missing required file: data/imputedmulti.txt. "
            "Add this file before running the visualization workflow."
        )

    run_step(
        "python scripts/figure1_status_variable_distributions.py",
        "SE4 Figure 1: ecological-status variable distributions",
    )

    run_step(
        "python scripts/figure2_model_performance_heatmap.py",
        "SE4 Figure 2: model-performance heatmap",
    )

    run_step(
        "python scripts/figure3_calibration_heatmap.py",
        "SE4 Figure 3: calibration heatmap",
    )

    optional_steps = [
        {
            "files": ["data/permutation/Permutation.txt", "data/shap/SHAP.txt"],
            "command": "python scripts/figure4_tabnet_shap_permutation.py",
            "description": "SE4 Figure 4: TabNet SHAP and permutation importance",
        },
        {
            "files": ["data/bnn/multi.txt"],
            "command": "python scripts/figure5_bnn_uncertainty_decomposition.py",
            "description": "SE4 Figure 5: BNN uncertainty decomposition",
        },
    ]

    for step in optional_steps:
        missing = [f for f in step["files"] if not Path(f).exists()]

        if missing:
            print(f"\nSkipping {step['description']}.")
            print("Missing optional input file(s):")
            for f in missing:
                print(f" - {f}")
        else:
            run_step(step["command"], step["description"])

    print("\nReproducibility workflow finished.")


if __name__ == "__main__":
    main()
