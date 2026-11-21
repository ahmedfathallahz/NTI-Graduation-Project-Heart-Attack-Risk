# NTI Graduation Project — Heart Attack Risk

### End-to-End Machine Learning & Data Analysis

A complete workflow from raw data to a deployed app: data inspection and cleaning, exploratory data analysis, feature engineering, supervised and unsupervised machine learning, model evaluation, and deployment as an interactive bilingual (Arabic / English) Streamlit dashboard with a risk-prediction demo and a health chat assistant.

**Live demo:** [nabd-ml.streamlit.app](https://nabd-ml.streamlit.app)

---

## Overview

| | |
|---|---|
| **Dataset** | `MedicalDataset.xlsx` — 10,000 patients, 35 columns (demographics, body measurements, medical history, lifestyle, functional difficulties, preventive care) |
| **Target** | `HeartAttackRisk` — `1` = recorded heart-attack history, `0` = no recorded history (~5.6 % positive) |
| **Task** | Binary classification on an imbalanced dataset + unsupervised patient profiling |
| **Best model** | Logistic Regression (selected by validation PR-AUC) |
| **App** | NABD — Streamlit dashboard (`app.py`), live at [nabd-ml.streamlit.app](https://nabd-ml.streamlit.app) |

> The target describes **recorded history** in the data. The models are predictive tools for learning purposes, not a diagnosis or a confirmed estimate of future heart-attack probability.

---

## Repository Structure

```
├── .devcontainer/             # Dev Container configuration
├── .gitignore
├── HeartAttackRisk.ipynb      # Full analysis & modelling notebook
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
   -> 1. Data inspection & quality checks
   -> 2. Exploratory data analysis (~50 interactive charts)
   -> 3. Feature engineering & leakage-free preprocessing
   -> 4. Supervised learning: tuning, stacking, threshold selection
   -> 5. Unsupervised learning: clustering, anomalies, association rules
   -> 6. Neural network (Keras MLP)
   -> 7. Model comparison & evaluation on a held-out test set
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

## Notebook Contents

| # | Section | What it covers |
|---|---------|----------------|
| 1 | **Data Loading & Inspection** | Shape, types, missing values, summary statistics, feature engineering |
| 2 | **EDA & Visualizations** | ~50 interactive Plotly figures: demographics, BMI, heart & stroke, chronic conditions, lifestyle, mobility, geography, correlations, relative risk |
| 3 | **Preprocessing** | Leakage-free pipeline, outlier clipping, log transform, robust scaling, ordinal / one-hot encoding, class weights |
| 4 | **Supervised Learning** | GridSearch + Optuna tuning, Stacking ensemble, validation leaderboard, test evaluation, clinical thresholds |
| 5 | **Unsupervised Learning** | PCA, K-Means, GMM, t-SNE, Isolation Forest, association rules (Apriori) |
| 6 | **Neural Network** | Keras MLP with L2, Dropout, BatchNorm, early stopping and a tuning grid |
| 7 | **Models Comparison** | Tables, train / validation / test charts, ROC & PR curves |
| 8 | **LLM Chat Assistant** | General health assistant powered by Google Gemini |

### Preprocessing & modelling highlights

- **No data leakage:** the pipeline is fitted on the training set only (70 % train / 15 % validation / 15 % test, stratified). Target-derived columns are excluded from the features.
- **Class imbalance** is handled with **class weights only** (no resampling), which keeps probabilities better calibrated.
- **Engineered features:** `AgeMidpoint`, `ComorbidityCount`, `LifestyleRiskScore`, `BMI_Category`.
- **Excluded from the model:** `PatientID` and `State` (the state is used for geographic EDA only).
- **Models compared:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Stacking ensemble, Keras MLP.
- **Threshold selection** is done on the validation set with an operating point that guarantees a minimum recall (≥ 80 %), because missing cases is the costly error in a screening scenario.

### Results (held-out test set, n = 1,500)

| Metric | Value |
|---|---|
| ROC-AUC | **0.871** |
| PR-AUC | **0.407** |
| Decision threshold (min-recall ≥ 80 % on validation) | 0.36 |
| Recall at that threshold | 83.1 % |
| Precision at that threshold | 16.8 % |

At this operating point the model deliberately favours **catching cases (recall)** over precision, so many flagged patients are false alarms — an expected trade-off for a screening-style threshold on a rare outcome.

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

`MedicalDataset.xlsx` is loaded from the project folder (or can be uploaded from inside the app).
If the saved notebook artifacts (`artifacts/preprocessor.joblib`, `artifacts/best_heart_attack_model.joblib`) are present, the app uses them; otherwise it re-trains the same Logistic Regression pipeline automatically.

### 4. Run the notebook

```bash
pip install -r requirements-notebook.txt
jupyter notebook HeartAttackRisk.ipynb
```

Or open the repo in **GitHub Codespaces / VS Code Dev Containers** — the `.devcontainer` folder sets up the environment.

### 5. Health assistant (Gemini)

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey), then either paste it in the Assistant page or set it as a secret:

```toml
# .streamlit/secrets.toml   (never commit this file)
GEMINI_API_KEY = "your-key"
```

---

## Tech Stack

**Python** · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · TensorFlow / Keras · mlxtend · Plotly · Streamlit · Google Gemini API

---

## Limitations

- Trained on a **sample dataset**; results are not validated for clinical use.
- The target reflects **recorded history**, not a future outcome.
- Low precision at the high-recall threshold means many false alarms.
- Observed associations are **descriptive and not causal**.

---

## Author

**Ahmed Fathallah** — NTI Graduation Project
