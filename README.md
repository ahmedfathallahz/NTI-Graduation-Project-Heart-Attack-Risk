<div align="center">

# NABD &nbsp;|&nbsp; نبض

### Heart-Attack Risk Analytics — from raw data to an interactive bilingual dashboard

**Live demo: [nabd-ml.streamlit.app](https://nabd-ml.streamlit.app/)**

*NTI graduation project*

</div>

> **Educational project only.** NABD is not a medical device and does not replace a doctor. The dataset is a sample and the model is not clinically validated. For any health concern, consult a qualified physician.

---

<a id="table-of-contents"></a>

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dashboard pages](#dashboard-pages)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Tech stack](#tech-stack)
8. [Project structure](#project-structure)
9. [Run it locally](#run-it-locally)
10. [Deploy on Streamlit Cloud](#deploy-on-streamlit-cloud)
11. [Limitations](#limitations)

---

<a id="overview"></a>

## Overview

NABD (نبض, Arabic for *pulse*) explores a medical dataset of 10,000 patients to understand who reports a heart-attack history and why. The project covers the full data-science workflow:

**EDA → leakage-free preprocessing → supervised models → unsupervised learning → neural network → model comparison → interactive dashboard → health chatbot.**

The notebook is the research; the Streamlit app is the product built on top of it, usable in **English and Arabic** (with full RTL support).

<a id="features"></a>

## Features

- **49 interactive charts** in 8 groups, with live filters (sex, age range, BMI, smoking)
- **Auto-generated insights** and a risk-factor ratio table
- **Patient segmentation**: PCA, K-Means, GMM, t-SNE and Isolation-Forest anomalies
- **Model lab**: ROC, Precision–Recall, calibration, confusion matrix and a **threshold slider**
- **Risk-prediction demo** with a colour-coded result, gauge and "what pushed the score up or down"
- **Personalised recommendations** based on your demo result, plus general healthy-habit advice
- **Health chatbot** with safety rules, emergency warnings and dashboard context
- **Bilingual UI** (EN / العربية) with one-click switch, dark theme, animated ECG header

<a id="dashboard-pages"></a>

## Dashboard pages

| Page | What you get |
|------|--------------|
| **Overview** | KPI cards, headline insights, key charts, quick navigation |
| **Analysis** | 49 charts: Population, Body & BMI, Heart & Stroke, Chronic Conditions, Lifestyle, Function & Mobility, Geography, Correlations & Risk |
| **Insights** | Strongest associations, age/smoking/weight/geography findings, with-vs-without ratio chart |
| **Segments** | Choose K (2–8), elbow and silhouette, cluster profiles, PCA space, anomaly scores, t-SNE |
| **Models** | ROC-AUC / PR-AUC, recall and precision at any threshold, calibration, model leaderboard, top drivers |
| **Predict** | Enter demographics, body, lifestyle, history and function data and get a risk score |
| **Recommendations** | Tips tailored to your result and general guidance with data-backed notes |
| **Assistant** | Ask general health questions in Arabic or English |

<a id="dataset"></a>

## Dataset

- **10,000 patients × 35 columns**, target `HeartAttackRisk` (5.55% positive, 555 cases)
- Demographics, body measures, chronic conditions, lifestyle, daily function, preventive care and US state
- `PatientID` and `State` are excluded from model inputs
- Features derived from the target (`ChronicCount`, `Has3PlusChronic`) are used for EDA only, never for modelling

<a id="methodology"></a>

## Methodology

**Preprocessing** (fitted on the training set only, so no leakage)

| Feature type | Steps |
|--------------|-------|
| Continuous | median imputation → IQR outlier clipping → log transform → robust scaling |
| Binary | most-frequent imputation |
| Ordinal (general health, age) | ordinal encoding |
| Nominal | one-hot encoding |

Engineered features: age midpoint, comorbidity count, lifestyle risk score, BMI category.

**Split:** stratified 70% train / 15% validation / 15% test.

**Class imbalance:** class weights only (≈ 17:1), no resampling, so probabilities are not distorted twice.

**Supervised models** (tuned on the training set with stratified 5-fold CV, scored by PR-AUC): Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, a Stacking ensemble and a Keras MLP. Model selection uses validation PR-AUC; thresholds are fixed on validation and applied once to the test set.

**Unsupervised learning:** PCA, K-Means, GMM, hierarchical clustering, t-SNE, Isolation Forest, LOF and association rules (Apriori). Cluster, anomaly and rule features were tested as extra model inputs and **not** added, because they did not improve PR-AUC for both models.

<a id="results"></a>

## Results

Held-out **test set (n = 1,500)**. Selected model: **Logistic Regression** — ROC-AUC **0.871**, PR-AUC **0.407**.

| Operating point | Threshold | Recall | Precision | Missed cases |
|-----------------|:---------:|:------:|:---------:|:------------:|
| Default | 0.50 | 0.759 | 0.217 | 20 of 83 |
| F1-optimal | 0.813 | 0.542 | 0.405 | 38 of 83 |
| **Min recall ≥ 80%** (screening) | **0.360** | **0.831** | 0.168 | **14 of 83** |

At the screening threshold the model catches **69 of 83** true cases (TN 1075 · FP 342 · FN 14 · TP 69).

| Model | Test PR-AUC | Test ROC-AUC |
|-------|:-----------:|:------------:|
| Neural Network (MLP) | 0.441 | 0.861 |
| XGBoost | 0.415 | 0.872 |
| CatBoost | 0.414 | 0.872 |
| Stacking Ensemble | 0.410 | 0.873 |
| Logistic Regression | 0.407 | 0.871 |
| Random Forest | 0.401 | 0.871 |
| LightGBM | 0.397 | 0.871 |

> For screening, missing a real case costs more than a false alarm, so the dashboard defaults to the recall-oriented threshold and lets you move it.

<a id="tech-stack"></a>

## Tech stack

| Area | Tools |
|------|-------|
| Data | pandas, NumPy, openpyxl |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, TensorFlow / Keras, mlxtend |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit with custom CSS (dark theme, RTL) |
| Chatbot | Google `google-genai` client |

<a id="project-structure"></a>

## Project structure