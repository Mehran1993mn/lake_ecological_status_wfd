# 🇫🇮 National-Scale Machine Learning Assessment of Lake Ecological Status under the EU WFD

**Author:** Mehran Mahdian  
**Contact:** Mehran.mahdian@uef.fi  
**Organization:** University of Eastern Finland  

**Profiles:**  
- [LinkedIn](https://www.linkedin.com/in/mehran-mahdian1993/)  
- [Google Scholar](https://scholar.google.com/citations?user=GWxu7rQAAAAJ&hl=en&oi=ao)

---

## 🌍 Project Overview

### 📌 Problem Statement

Lakes provide critical ecosystem services including biodiversity support, drinking water, fisheries, agriculture, recreation, and climate regulation. However, anthropogenic pressures and climate change are accelerating eutrophication, browning, oxygen depletion, harmful algal blooms, and ecological degradation.

Under the **European Union Water Framework Directive (EU WFD)**, lake ecological status is assessed using biological quality elements and classified into ecological-status classes. These assessments are scientifically important but often expensive, labor intensive, temporally sparse, and difficult to update in near real time.

This project develops a national-scale machine-learning workflow for predicting Finnish lake ecological status using routinely monitored water-quality and morphometric variables.

---

## ⚠️ Challenge Statement

WFD ecological classification is difficult to model because:

- Ecological classes overlap, especially the **Moderate** class.
- Class imbalance exists, especially for **Bad/Poor** lakes.
- Lake ecological responses are nonlinear and multivariate.
- Boundaries between ecological classes are transitional and noisy.
- Routine water-quality variables are easier to monitor than biological quality elements, but their predictive reliability must be tested.

---

## 💡 Solution Statement

This repository provides a reproducible machine-learning framework that:

- Uses routine water-quality and morphometric variables.
- Compares machine-learning and deep-learning models.
- Includes uncertainty-aware ecological-status prediction.
- Provides explainable-AI outputs using feature-importance methods.
- Generates publication-style figures.
- Includes a reproducibility wrapper and software environment files.

---

## 🎯 Objectives

The main objectives are to:

1. Predict Finnish lake ecological status using routine monitoring variables.
2. Compare model performance across machine-learning and deep-learning approaches.
3. Identify dominant environmental drivers of ecological classification.
4. Quantify uncertainty in ecological-status prediction.
5. Provide a reproducible repository for rerunning the analysis.

---

## 🔎 Research Questions

**RQ1.** Can routine water-quality and morphometric variables predict WFD ecological status across Finnish lakes?

**RQ2.** Which physicochemical and morphometric variables contribute most strongly to ecological classification?

**RQ3.** Does Bayesian modelling improve reliability by quantifying predictive uncertainty?

**RQ4.** Is predictive uncertainty mainly aleatoric or epistemic?

---

## 📊 Data Sources

### Study Area

- **Country:** Finland  
- **Number of lakes:** 2,487  
- **Monitoring period:** 2012–2017  
- **Sampling depth:** surface samples, mainly 0–2 m  
- **Aggregation:** lake-wise period means  
- **Target:** WFD ecological-status class  

### Main Predictors

The modelling workflow uses routine water-quality and morphometric variables, including:

- Total phosphorus, TP
- Total nitrogen, TN
- Turbidity
- Conductivity
- Water temperature
- pH
- Water colour
- Dissolved oxygen
- Secchi depth
- Maximum depth
- Surface area
- Natural lake type variables

### Published Data Source

| Dataset | Source | Description | Access |
|---|---|---|---|
| Hertta / SYKE environmental data | Finnish Environment Institute, SYKE | Water quality, lake information, and ecological-status data | Public environmental data services |

Large raw SYKE/VESLA files are **not included** in this repository because of file-size limitations. Instructions for placing raw files locally are provided in [`inputs/README_inputs.md`](inputs/README_inputs.md).

---

## ⚙️ Methods Summary

### Preprocessing

The workflow includes:

- Surface-water filtering
- Lake-wise aggregation for 2012–2017
- Missing-value handling
- Class-label harmonization
- Preparation of modelling-ready feature tables
- Cross-validation setup

### Models

The project framework includes:

- Random Forest
- XGBoost
- Support Vector Machine
- Artificial Neural Network / MLP
- TabNet
- Bayesian Neural Network
- Ensemble summaries

### Interpretability and Statistical Analysis

The workflow includes:

- SHAP analysis
- Permutation importance
- Kruskal-Wallis tests
- Pairwise Mann-Whitney U tests
- Feature-importance visualization

### Evaluation Metrics

Model performance is evaluated using:

- Macro-F1 score
- Accuracy
- Matthews correlation coefficient, MCC
- Brier score
- Expected Calibration Error, ECE
- Confusion matrices
- Train/test cross-validation summaries

---

## 📁 Repository Structure

```text
lake_ecological_status_wfd/
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
├── .gitignore
├── run_reproducibility.py
│
├── notebooks/
│   ├── SE1_data_source_vesla_api.ipynb
│   ├── SE2_data_processing.ipynb
│   ├── SE3_xgboost_modelling.ipynb
│   ├── SE4_figure1_status_variable_distributions.ipynb
│   ├── SE4_figure2_model_performance_heatmap.ipynb
│   ├── SE4_figure3_calibration_heatmap.ipynb
│   ├── SE4_figure4_tabnet_shap_permutation.ipynb
│   └── SE4_figure5_bnn_uncertainty_decomposition.ipynb
│
├── scripts/
│   ├── figure1_status_variable_distributions.py
│   ├── figure2_model_performance_heatmap.py
│   ├── figure3_calibration_heatmap.py
│   ├── figure4_tabnet_shap_permutation.py
│   └── figure5_bnn_uncertainty_decomposition.py
│
├── inputs/
│   ├── README_inputs.md
│   └── raw_data/
│
├── data/
│   ├── imputedmulti.txt
│   ├── permutation/
│   ├── shap/
│   └── bnn/
│
├── images/
└── outputs/
```

---

## ✅ Standard Elements

## ✅ Standard Elements

| Standard element | Repository file or folder | Description |
|---|---|---|
| SE1: Data access script | `notebooks/SE1_data_source_vesla_api.ipynb` | Retrieves water-quality data from the SYKE/VESLA API. |
| SE2: Data processing script | `notebooks/SE2_data_processing.ipynb` | Processes raw lake and water-quality data into modelling-ready form. |
| SE3: Model train and validation script | `notebooks/SE3_xgboost_modelling.ipynb` | Trains and validates the XGBoost ecological-status model. |
| SE4: Data and results visualization script | `scripts/figure*.py` and SE4 notebooks | Generates data-story and results figures. |
| SE5: Software environment or image | `requirements.txt` and `Dockerfile` | Defines the software environment required to run the workflow. |
| SE6: Reproducibility wrapper | `run_reproducibility.py` | Runs the reproducible workflow from the project root. |

---

## 🔁 How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/Mehran1993mn/lake_ecological_status_wfd.git
cd lake_ecological_status_wfd
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

### 3. Add raw input data

Large raw SYKE/VESLA files are not included in this repository.

Before running the full workflow, place the following files in:

```text
inputs/raw_data/
```

Required raw files:

```text
station_with_ecology_values.csv
tn323.csv
tp315.csv
turbidity7677.csv
```

These files are documented in:

```text
inputs/README_inputs.md
```

### 4. Run the reproducibility wrapper

```bash
python run_reproducibility.py
```

### 5. Expected outputs

The workflow generates or checks outputs in:

```text
images/
outputs/
```

Some optional figures require additional input files:

```text
data/permutation/Permutation.txt
data/shap/SHAP.txt
data/bnn/multi.txt
```

If these optional files are not present, the reproducibility wrapper skips the corresponding optional figures.

---

## 🐳 Docker Usage

A Dockerfile is included as the software-image option.

Build the image with:

```bash
docker build -t lake-ecological-status-wfd .
```

Run the container with:

```bash
docker run --rm lake-ecological-status-wfd
```

For the full workflow, raw input files still need to be available in the expected folder structure.

---

## 💻 Computational Requirements

Recommended requirements:

- Python 3.10 or newer
- 16 GB RAM recommended
- Optional GPU for Bayesian/deep-learning workflows
- Operating system: Windows, Linux, or macOS

---

## 📈 Main Results Summary

The modelling framework achieved robust national-scale ecological-status classification performance:

- Best multi-class macro-F1 approximately **0.65**
- Binary classification F1 approximately **0.84**
- Best calibration achieved by the Bayesian Neural Network, with ECE approximately **0.016**
- Nutrient variables, especially TP, TN, and turbidity, were dominant predictors
- The Moderate class was the most difficult ecological-status class to classify
- Predictive uncertainty was mainly aleatoric

---

## 🖼️ Example Figures

### Multi-Class Classification Performance

![Model Performance](images/fig1.png)

**Figure 1.** Model performance metrics based on F1-score, accuracy, and Matthews correlation coefficient.

### Feature Importance

![SHAP Results](images/fig2.png)

**Figure 2.** Feature-importance analysis showing dominant environmental drivers of ecological-status classification.

Additional generated figures are stored in the `images/` folder.

---

## 📖 Citation

**Mahdian, M., Abolfathi, S., Kukkonen, J. V. K., & Kolehmainen, M.**  
*Explainable and Uncertainty-Aware AI for National-Scale Ecological Status Assessment of Lakes under the EU Water Framework Directive.*  
Manuscript under review at *Water Research*.

---

## 📜 License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

## 🤝 Contribution Guidelines

Contributions that improve scientific quality, clarity, and reproducibility are welcome.

Please follow these principles:

- Open an issue before making major methodological or result-affecting changes.
- Keep pull requests focused and clearly describe what changed.
- Do not modify scripts used to reproduce reported results without documenting the change.
- Keep random seeds fixed where possible.
- Document hyperparameter changes.
- Update environment/dependency files when packages change.
- Clearly separate exploratory notebooks from production-ready scripts.
- Do not commit large raw SYKE/VESLA datasets.
- Respect all data licenses and ethical guidelines.

---

## 📝 Notes

This repository emphasizes:

- Reproducibility
- Transparent modelling
- Uncertainty-aware environmental decision support
- Clear scientific communication
