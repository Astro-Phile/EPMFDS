# EPFMDS
EPMFDS (Equipment Predictive Maintenance &amp; Fault Diagnosis System):A hybrid predictive maintenance system combining classical machine learning techniques for early fault detection and diagnosis in industrial equipment. 
Data sourced from :https://data.nasa.gov/d/ff5v-kuh6


---

```markdown
# Ensemble Predictive Maintenance & Fault Diagnosis System (EPMFDS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/travis/Astro-Phile/EPMFDS/master.svg)](https://travis-ci.org/Astro-Phile/EPMFDS)
[![GitHub stars](https://img.shields.io/github/stars/Astro-Phile/EPMFDS.svg)](https://github.com/Astro-Phile/EPMFDS/stargazers)

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Modeling Approach](#modeling-approach)
  - [Implemented Models](#implemented-models)
  - [Detailed SVM Pipeline](#detailed-svm-pipeline)
- [Evaluation & Comparison](#evaluation--comparison)
- [Results & Analysis](#results--analysis)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

**Ensemble Predictive Maintenance & Fault Diagnosis System (EPMFDS)** is a research-driven project focused on predicting the Remaining Useful Life (RUL) of turbofan engines using classical machine learning techniques. Built upon the NASA C-MAPSS FD001 dataset, the project leverages an ensemble of models—including K-Nearest Neighbors (KNN), Decision Trees, Linear Regression, Support Vector Machine (SVM), and Bayesian Learning—to generate robust diagnostic insights and forecast potential equipment failures.

---

## Motivation

In industrial settings, unscheduled downtime can be incredibly costly. This project addresses this critical issue by:
- Reducing maintenance costs through early fault detection.
- Enhancing system reliability via predictive diagnostics.
- Demonstrating that classical machine learning methods, when combined and properly tuned, can deliver high-performance predictive maintenance solutions without relying on deep learning techniques.

---

## Project Goals

- **Develop a robust preprocessing pipeline**: Clean, normalize, and engineer features from sensor data.
- **Implement multiple classical models**: Use traditional techniques (KNN, Decision Tree, Linear Regression, SVM, Bayesian Learning) for forecasting RUL.
- **Optimize model performance**: Employ hyperparameter tuning (e.g., GridSearchCV) and dimensionality reduction (PCA) to refine predictions.
- **Perform in-depth evaluation and failure analysis**: Use comprehensive metrics (MAE, MSE, RMSE, R²) and visualizations (residual plots, scatter plots) to understand model performance.
- **Build a scalable framework**: Ensure the pipeline can be adapted for other datasets (FD002–FD004) and extended to real-world industrial applications.

---

## Dataset

The project is primarily based on the **NASA C-MAPSS FD001 dataset** which includes:
- **Training data**: Contains engine run-to-failure cycles.
- **Test data**: Contains engine operational cycles with provided RUL values in a separate file (e.g., `RUL_FD001.txt`).
- **Sensor Data**: Multiple sensor readings (vibration, temperature, pressure, etc.) and operational settings.

Additional datasets (FD002, FD003, FD004) are supported by the modular design for future extension.

---

## Preprocessing Pipeline

The preprocessing component encompasses the following steps:
- **Data Cleaning**: Removal of constant and near-constant features using variance threshold techniques.
- **Normalization**: Standardizing sensor readings with MinMaxScaler.
- **Feature Engineering**: Calculation of RUL (Remaining Useful Life) for training engines.
- **Visualization**: Outlier detection, sensor trend analysis, and correlation heatmaps to ensure data quality.

A comprehensive notebook ([Preprocessing.ipynb](notebooks/Preprocessing.ipynb)) details these steps and provides visual diagnostics.

---

## Modeling Approach

### Implemented Models

The following classical models have been implemented as part of the ensemble approach:
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Linear Regression**
- **Support Vector Machine (SVM)**
- **Bayesian Learning**

Each model is developed in individual notebooks, and model performance is evaluated using a standardized set of regression metrics.

### Detailed SVM Pipeline

The SVM model (implemented in [SVM_Model.ipynb](notebooks/SVM_Model.ipynb)) is enhanced with:
- **PCA Integration**: Reducing feature dimensionality (95% variance retained) to remove noise and improve training efficiency.
- **Hyperparameter Tuning**: Using GridSearchCV (with 3-fold cross-validation) to optimize parameters (`C`, `ε`, `gamma`).
- **Comprehensive Evaluation**: Model performance is assessed via MAE, MSE, RMSE, and R², along with residual analysis (histogram and scatter plots).

---

## Evaluation & Comparison

All models are evaluated using the same data splits and metrics to ensure a fair comparison. Key evaluation strategies include:
- **Cross-Validation**: Consistent use of k-fold (typically 3- or 5-fold) cross-validation for robust performance metrics.
- **Performance Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score (Coefficient of Determination)
- **Visual Diagnostics**: Residual plots and predicted vs actual RUL comparisons help identify model strengths and weaknesses.

A summary table is generated to compare all models, helping determine which method best fits the predictive maintenance requirements.

---

## Results & Analysis

Initial evaluations indicate:
- **SVM with PCA** provides robust non-linear modeling capabilities, though it requires longer tuning times.
- **Bayesian Learning** offers fast, interpretable results, making it a good baseline.
- **KNN and Decision Trees** present competitive performance with varying trade-offs in accuracy and interpretability.
- Comparative visualizations (e.g., scatter plots for predicted vs actual RUL) are included in the notebooks for detailed failure case analysis.

These insights support iterative model improvements and ensemble strategies for further boosting predictive performance.

---

## Repository Structure

```plaintext
EPMFDS/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── [additional datasets: FD002.txt, FD003.txt, FD004.txt]
├── notebooks/
│   ├── Preprocessing.ipynb        # Data cleaning, normalization, and visualization
│   ├── SVM_Model.ipynb            # Implementation of the SVM model with PCA & tuning
│   ├── Bayesian.ipynb             # Bayesian Learning model implementation
│   ├── KNN_Model.ipynb            # KNN model as a baseline
│   ├── Decision_Tree.ipynb        # Decision Tree model with visual diagnostics
│   └── Linear_Regression.ipynb    # Linear Regression baseline model
├── src/
│   ├── preprocessing.py           # Modular data preprocessing functions
│   ├── models.py                  # Model training and tuning functions
│   └── evaluation.py              # Functions for performance metric computations & plotting
├── requirements.txt               # Package dependencies
├── README.md                      # This file
└── LICENSE
```

---

## Installation

### Prerequisites

- **Python 3.8+**
- **Git**

Clone the repository and install dependencies:

```bash
git clone https://github.com/Astro-Phile/EPMFDS.git
cd EPMFDS
pip install -r requirements.txt
```

---

## Usage

1. **Data Preparation**  
   Place all NASA C-MAPSS data files in the `data/` folder. Ensure proper file naming (e.g., `train_FD001.txt`, `RUL_FD001.txt`, etc.).

2. **Running Notebooks**  
   Use Jupyter Notebook or Google Colab to run the provided notebooks sequentially:
   - Start with **Preprocessing.ipynb** for cleaning and data visualization.
   - Proceed with model-specific notebooks (e.g., **SVM_Model.ipynb**) for training and evaluation.
   
   To launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. **Extending to Additional Datasets**  
   The pipeline is modular—re-run preprocessing and model training notebooks for FD002–FD004 by updating file paths and variable configurations.

---

## Contributing

Contributions and suggestions are welcome! Please follow these guidelines:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a Pull Request with clear descriptions of your changes.
4. Follow PEP8 standards for Python code and maintain clear documentation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Acknowledgements

- **NASA Prognostics Data Repository**: For the C-MAPSS datasets.
- **Project Team Members**:
  - Aditya Kashyap (Repository & SVM Model)
  - Asadullah Faisal (Bayesian Learning)
  - Sunny Meena (KNN Model)
  - Faiz Imdad (Decision Tree Model)
  - Vishal Jharwal (Linear Regression Model)
- Inspired by best practices in predictive maintenance research and classical machine learning literature.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Future Work

- **Model Ensemble**: Combine multiple models to further improve predictive accuracy.
- **Extended Dataset Analysis**: Adapt the pipeline for FD002–FD004, analyzing performance differences.
- **Real-Time Implementation**: Explore integration with IoT systems for live predictive maintenance.
- **Enhanced Reporting**: Develop interactive dashboards for maintenance scheduling based on model predictions.

---

Thank you for checking out the EPMFDS project. For any questions or feedback, please open an issue or contact the maintainers.
---
