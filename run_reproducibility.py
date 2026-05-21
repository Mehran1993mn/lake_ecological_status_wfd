"""
Reproducibility wrapper for the Finnish lake ecological-status project.

This script checks required folders/files and runs the reproducible figure scripts.
Large raw data files must be placed manually in inputs/raw_data/.
"""

import subprocess
from pathlib import Path


def run_step(command, description):
    """Run one workflow step and stop if it fails."""
    print(f"\n=== {description} ===")
    subprocess.run(command, shell=True, check=True)


def main():
    print("Starting reproducibility workflow...")

    # Create output folders if they do not exist
    Path("images").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    # Large raw data files are documented but not stored in GitHub
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

    # Required input for Figure 1
    required_file = Path("data/imputedmulti.txt")
    if not required_file.exists():
        raise FileNotFoundError(
            "Missing required file: data/imputedmulti.txt. "
            "Please place this file in the data/ folder before running."
        )

    # Main reproducible visualization scripts
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

    # Optional scripts requiring additional data files
    optional_steps = [
        {
            "description": "SE4 Figure 4: TabNet SHAP and permutation importance",
            "command": "python scripts/figure4_tabnet_shap_permutation.py",
            "files": [
                "data/permutation/Permutation.txt",
                "data/shap/SHAP.txt",
            ],
        },
        {
            "description": "SE4 Figure 5: BNN uncertainty decomposition",
            "command": "python scripts/figure5_bnn_uncertainty_decomposition.py",
            "files": [
                "data/bnn/multi.txt",
            ],
        },
    ]

    for step in optional_steps:
        missing_files = [f for f in step["files"] if not Path(f).exists()]

        if missing_files:
            print(f"\nSkipping {step['description']}.")
            print("Missing optional input file(s):")
            for f in missing_files:
                print(f" - {f}")
        else:
            run_step(step["command"], step["description"])

    print("\nReproducibility workflow finished.")


if __name__ == "__main__":
    main()
