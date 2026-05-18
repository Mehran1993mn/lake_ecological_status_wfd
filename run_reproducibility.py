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


def check_file(path, required=True):
    path = Path(path)
    if not path.exists() and required:
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.exists()


def main():
    print("Starting reproducibility workflow...")

    # Create output folders if they do not exist
    Path("images").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    # Check large raw-data folder
    raw_data_files = [
        "inputs/raw_data/station_with_ecology_values.csv",
        "inputs/raw_data/tn323.csv",
        "inputs/raw_data/tp315.csv",
        "inputs/raw_data/turbidity7677.csv",
    ]

    missing_raw = [f for f in raw_data_files if not Path(f).exists()]

    if missing_raw:
        print("\nLarge raw data files are missing.")
        print("This is expected if the repository was cloned from GitHub.")
        print("Place the following files in inputs/raw_data/ before running the full data-processing workflow:")
        for f in missing_raw:
            print(f" - {f}")

    # Check Assignment 6 figure inputs
    figure_inputs = [
        "data/imputedmulti.txt",
    ]

    for f in figure_inputs:
        check_file(f, required=True)

    # Run reproducible visualization scripts
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

    # These require additional files.
    optional_steps = [
        (
            "data/permutation/Permutation.txt",
            "data/shap/SHAP.txt",
            "python scripts/figure4_tabnet_shap_permutation.py",
            "SE4 Figure 4: TabNet SHAP and permutation importance",
        ),
        (
            "data/bnn/multi.txt",
            None,
            "python scripts/figure5_bnn_uncertainty_decomposition.py",
            "SE4 Figure 5: BNN uncertainty decomposition",
        ),
    ]

    for file1, file2, command, description in optional_steps:
        file1_exists = Path(file1).exists()
        file2_exists = True if file2 is None else Path(file2).exists()

        if file1_exists and file2_exists:
            run_step(command, description)
        else:
            print(f"\nSkipping {description}.")
            print("Missing required input file(s):")
            if not file1_exists:
                print(f" - {file1}")
            if file2 is not None and not file2_exists:
                print(f" - {file2}")

    print("\nReproducibility workflow finished.")


if __name__ == "__main__":
    main()
