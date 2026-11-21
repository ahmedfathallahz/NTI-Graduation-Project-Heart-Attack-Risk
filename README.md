# NTI Graduation Project — Heart Attack Risk

### End-to-End Machine Learning & Data Analysis

A complete workflow from raw data to a deployed app: data inspection and quality checks, exploratory data analysis, feature engineering, supervised and unsupervised machine learning, model evaluation, and deployment as an interactive bilingual (Arabic / English) Streamlit dashboard with a risk-prediction demo and a health chat assistant.

**Live demo:** [nabd-ml.streamlit.app](https://nabd-ml.streamlit.app)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [End-to-End Workflow](#end-to-end-workflow)
4. [Dataset](#dataset)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
7. [Supervised Learning](#supervised-learning)
8. [Unsupervised Learning](#unsupervised-learning)
9. [Neural Network](#neural-network)
10. [Results](#results)
11. [Streamlit Dashboard (NABD)](#streamlit-dashboard-nabd)
12. [Getting Started](#getting-started)
13. [Troubleshooting](#troubleshooting)
14. [Tech Stack](#tech-stack)
15. [Limitations](#limitations)
---

## Project Overview

| | |
|---|---|
| **Goal** | Understand which factors are associated with a recorded heart-attack history and build models that rank patients by risk |
| **Dataset** | `MedicalDataset.xlsx` — 10,000 patients, 35 columns |
| **Target** | `HeartAttackRisk` — `1` = recorded heart-attack history, `0` = no recorded history (about 5.6 % positive) |
| **Task** | Binary classification on an imbalanced dataset, plus unsupervised patient profiling |
| **Best model** | Logistic Regression (selected by validation PR-AUC) |
| **App** | NABD — Streamlit dashboard (`app.py`), live at [nabd-ml.streamlit.app](https://nabd-ml.streamlit.app) |

The target describes **recorded history** in the data. The models are learning tools, not a diagnosis or a confirmed estimate of future heart-attack probability.

---

## Repository Structure

```
├── .devcontainer/             # Dev Container configuration
├── .gitignore
├── HeartAttackRisk.ipynb      # Full analysis and modelling notebook
├── MedicalDataset.xlsx        # Dataset
├── app.py                     # Streamlit dashboard
├── health_assistant.py        # Health chat assistant (Google Gemini)
├── requirements.txt           # Dependencies for the Streamlit app
└── requirements-notebook.txt  # Dependencies for the notebook
```

---

## End-to-End Workflow

```
Raw data (MedicalDataset.xlsx)
   -> 1. Data inspection and quality checks
   -> 2. Exploratory data analysis (~50 interactive charts)
   -> 3. Feature engineering and leakage-free preprocessing
   -> 4. Supervised learning: tuning, stacking, threshold selection
   -> 5. Unsupervised learning: clustering, anomalies, association rules
   -> 6. Neural network (Keras MLP)
   -> 7. Model comparison and evaluation on a held-out test set
   -> 8. Deployment: Streamlit dashboard (NABD) + Gemini health assistant
```

| Stage | Data analysis | Machine learning |
|-------|---------------|------------------|
| Understand | Shape, types, missing values, summary statistics, class balance | - |
| Explore | Prevalence, trends by age / sex / BMI / smoking, comorbidity matrix, relative risk, correlations | - |
| Prepare | Outlier clipping, log transform, robust scaling, encoding | Stratified 70/15/15 split, class weights |
| Model | - | Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Stacking, Keras MLP |
| Profile | Patient segments and cluster profiles | PCA, K-Means, GMM, t-SNE, Isolation Forest, Apriori |
| Evaluate | - | ROC-AUC, PR-AUC, recall / precision, calibration, threshold analysis |
| Deploy | KPI dashboard with filters | Risk-prediction demo, model explorer |

---

## Dataset

The dataset contains 10,000 patients and 35 columns, grouped as follows:

| Group | Columns |
|-------|---------|
| **Identifiers / geography** | `PatientID`, `State` (used for geographic EDA only, never as a model input) |
| **Demographics** | `Sex`, `AgeCategory`, `RaceEthnicityCategory` |
| **Body measurements** | `HeightInMeters`, `WeightInKilograms`, `BMI` |
| **Medical history** | `HadAngina`, `HadStroke`, `HadAsthma`, `HadSkinCancer`, `HadCOPD`, `HadDepressiveDisorder`, `HadKidneyDisease`, `HadArthritis`, `HadDiabetes` |
| **Functional difficulties** | `DeafOrHardOfHearing`, `BlindOrVisionDifficulty`, `DifficultyConcentrating`, `DifficultyWalking`, `DifficultyDressingBathing`, `DifficultyErrands` |
| **Lifestyle** | `SmokerStatus`, `ECigaretteUsage`, `AlcoholDrinkers`, `HighRiskLastYear` |
| **Prevention and tests** | `ChestScan`, `HIVTesting`, `FluVaxLast12`, `PneumoVaxEver`, `TetanusLast10Tdap`, `CovidPos` |
| **General health** | `GeneralHealth` |
| **Target** | `HeartAttackRisk` |

Data quality checks (missing values, duplicates, data types) are performed at the start of the notebook, and every yes/no column is inspected for class balance.

---

## Exploratory Data Analysis

About 50 interactive Plotly figures, each followed by a short interpretation. Colours are semantic: red = higher risk, yellow = moderate, green = more favourable.

| Theme | Examples |
|-------|----------|
| **Population** | Prevalence of all conditions, age and sex distribution, general-health composition, race / ethnicity comparison, preventive-care rates |
| **Body and BMI** | BMI distribution and categories, heart attack / stroke / diabetes by BMI, age x BMI diabetes heatmap |
| **Heart and stroke** | Heart attack and stroke by age and sex, top 20 highest-risk groups, relative risk against a baseline (never smoked + normal BMI) |
| **Chronic conditions** | Chronic-condition counts, comorbidity matrix P(B given A), effect of depression on other conditions, kidney disease by diabetes status |
| **Lifestyle** | Smoking status, e-cigarette use, alcohol, high-risk behaviour |
| **Function and mobility** | Walking difficulty by age and arthritis, functional-decline cascade |
| **Geography** | Top states by heart attack, diabetes, depression and smoking |
| **Correlations** | Correlation heatmap, correlation with a general-health score, age-standardised prevalence |

The notebook also includes an interactive, filterable EDA dashboard (sex, age, BMI range and more).

---

## Preprocessing & Feature Engineering

The pipeline is **leakage-free**: all transformations are fitted on the training set only, then applied to the validation and test sets.

| Feature type | Steps |
|--------------|-------|
| Continuous | Median imputation, IQR outlier clipping, log transform, robust scaling |
| Binary | Most-frequent imputation |
| Ordinal (`GeneralHealth`, `AgeCategory`) | Most-frequent imputation, ordered encoding |
| Nominal | Most-frequent imputation, one-hot encoding (drop first) |

- **Split:** stratified 70 % train / 15 % validation / 15 % test.
- **Engineered features** (target-independent): `AgeMidpoint`, `ComorbidityCount`, `LifestyleRiskScore`, `BMI_Category`.
- **Excluded from the model:** `PatientID`, `State`, and any column derived from the target.
- **Class imbalance:** handled with class weights only (no resampling), so probabilities stay better calibrated and thresholds are chosen on the untouched validation set.

---

## Supervised Learning

| Model | Tuning | Best CV PR-AUC |
|-------|--------|----------------|
| Logistic Regression | GridSearchCV (best `C = 10`) | **0.4988** |
| LightGBM | Optuna, 25 trials | 0.4891 |
| CatBoost | Optuna, 12 trials | 0.4881 |
| XGBoost | Optuna, 25 trials | 0.4859 |
| Random Forest | GridSearchCV | 0.4692 |
| Stacking ensemble | Logistic Regression, Random Forest and XGBoost as base learners with a Logistic Regression meta-learner | compared on validation |

- The best model is selected by **validation PR-AUC**, the most informative metric under class imbalance.
- **Threshold selection** is done on the validation set at several operating points (default 0.50, F1-optimal, F2-optimal, and minimum-recall), then applied once to the test set.
- The default clinical operating point is **minimum recall of 80 %**, because missing cases is the costly error in a screening scenario.

---

## Unsupervised Learning

- **Dimensionality reduction:** PCA (cumulative explained variance) and t-SNE projection.
- **Clustering:** K-Means (elbow and silhouette to choose k), Gaussian Mixture Models (BIC curve) and Agglomerative clustering, compared with internal metrics.
- **Cluster interpretation:** heart-attack rate per cluster and a cluster-profile heatmap.
- **Anomaly detection:** Isolation Forest and Local Outlier Factor.
- **Unsupervised features:** cluster and anomaly features are engineered and tested to see whether they improve the supervised models.
- **Association rules:** Apriori (minimum support 0.003, minimum confidence 0.10, maximum length 3).

---

## Neural Network

A Keras feed-forward network (MLP) on the same processed features, with L2 regularisation, Dropout, Batch Normalisation and early stopping on validation PR-AUC. A small grid over layer sizes, dropout and learning rate (12 configurations) is searched, and the configuration with the best validation PR-AUC is kept. Class imbalance is handled with class weights.

---

## Results

Held-out **test set** (n = 1,500, 83 positive cases). Thresholds are fixed on the validation set.

| Metric | Value |
|---|---|
| ROC-AUC | **0.871** |
| PR-AUC | **0.407** |

| Operating point | Threshold | Recall | Precision |
|---|---|---|---|
| Default | 0.50 | 75.9 % | 21.7 % |
| F1-optimal | 0.81 | 54.2 % | 40.5 % |
| **Minimum recall >= 80 % (chosen)** | **0.36** | **83.1 %** | **16.8 %** |

At the chosen operating point the model deliberately favours **catching cases (recall)** over precision, so many flagged patients are false alarms. This is an expected trade-off for a screening-style threshold on a rare outcome.

---

## Streamlit Dashboard (NABD)

**Live:** [nabd-ml.streamlit.app](https://nabd-ml.streamlit.app)

A dashboard-style app with KPI cards, icon navigation, a **dark / light mode toggle**, and a full **Arabic (RTL) / English** switch.

| Page | Description |
|------|-------------|
| **Overview** | Headline KPIs, auto-generated insights, key charts |
| **Analysis** | All EDA charts grouped by category (Population, Body & BMI, Heart & Stroke, Chronic Conditions, Lifestyle, Mobility, Geography, Correlations) with live filters |
| **Insights** | Key findings drawn from the analysis |
| **Segments** | PCA, K-Means (choose K), elbow / silhouette, GMM BIC, cluster profiles, t-SNE, Isolation-Forest anomalies |
| **Models** | ROC, PR, confusion matrix, calibration, interactive decision-threshold explorer, coefficient drivers |
| **Predict** | Inputs organised in blocks; the result card is **red** for high-risk indicators and **green** for low-risk, with a gauge and the factors that moved the score |
| **Recommendations** | General healthy-lifestyle recommendations |
| **Assistant** | Health chat assistant powered by Google Gemini |

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/ahmedfathallahz/NTI-Graduation-Project-Heart-Attack-Risk.git
cd NTI-Graduation-Project-Heart-Attack-Risk
```

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

`MedicalDataset.xlsx` is loaded from the project folder (or can be uploaded from inside the app). If the saved notebook artifacts (`artifacts/preprocessor.joblib` and `artifacts/best_heart_attack_model.joblib`) are present, the app uses them; otherwise it re-trains the same Logistic Regression pipeline automatically.

### 4. Run the notebook

```bash
pip install -r requirements-notebook.txt
jupyter notebook HeartAttackRisk.ipynb
```

You can also open the repository in **GitHub Codespaces** or **VS Code Dev Containers**; the `.devcontainer` folder sets up the environment.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `MedicalDataset.xlsx not found` | Keep the file next to `app.py` (or in a `data/` folder), or upload it in the app |
| Pickle / scikit-learn version warnings when loading artifacts | Delete the `artifacts/` folder to let the app re-train, or install the same scikit-learn version used in the notebook |
| Icons show as plain text | The app loads its icon font from Google Fonts; check your internet connection |
| Optional notebook packages missing (XGBoost, LightGBM, CatBoost, Optuna, mlxtend, TensorFlow) | Install `requirements-notebook.txt`; the notebook skips a section when its package is unavailable |

---

## Tech Stack

**Python** · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · TensorFlow / Keras · mlxtend · Plotly · Streamlit · Google Gemini API

---

## Limitations

- Trained on a **sample dataset**; results are not validated for clinical use.
- The target reflects **recorded history**, not a future outcome.
- Low precision at the high-recall threshold means many false alarms.
- Observed associations are **descriptive and not causal**.


