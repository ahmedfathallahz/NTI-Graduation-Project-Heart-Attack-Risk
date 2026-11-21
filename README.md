<div align="center">

# <img src="https://api.iconify.design/mdi:heart-pulse.svg?color=%23ef4444" width="38" height="38" align="absmiddle"> NABD &nbsp;|&nbsp; نبض

### Heart-Attack Risk Analytics — from raw data to an interactive bilingual dashboard

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nabd-ml.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-charts-3F4F75?logo=plotly&logoColor=white)
![Languages](https://img.shields.io/badge/UI-English%20%7C%20العربية-22d3ee)

**<img src="https://api.iconify.design/mdi:link-variant.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> Live demo: [nabd-ml.streamlit.app](https://nabd-ml.streamlit.app/)**

*NTI graduation project*

</div>

> <img src="https://api.iconify.design/mdi:alert.svg?color=%23f59e0b" width="18" height="18" align="absmiddle"> **Educational project only.** NABD is not a medical device and does not replace a doctor. The dataset is a sample and the model is not clinically validated. For any health concern, consult a qualified physician.

---

<a id="table-of-contents"></a>

## <img src="https://api.iconify.design/mdi:format-list-bulleted.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Table of contents

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

## <img src="https://api.iconify.design/mdi:magnify.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Overview

NABD (نبض, Arabic for *pulse*) explores a medical dataset of 10,000 patients to understand who reports a heart-attack history and why. The project covers the full data-science workflow:

**EDA → leakage-free preprocessing → supervised models → unsupervised learning → neural network → model comparison → interactive dashboard → health chatbot.**

The notebook is the research; the Streamlit app is the product built on top of it, usable in **English and Arabic** (with full RTL support).

<a id="features"></a>

## <img src="https://api.iconify.design/mdi:star-four-points.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Features

- **49 interactive charts** in 8 groups, with live filters (sex, age range, BMI, smoking)
- **Auto-generated insights** and a risk-factor ratio table
- **Patient segmentation**: PCA, K-Means, GMM, t-SNE and Isolation-Forest anomalies
- **Model lab**: ROC, Precision–Recall, calibration, confusion matrix and a **threshold slider**
- **Risk-prediction demo** with a colour-coded result, gauge and "what pushed the score up or down"
- **Personalised recommendations** based on your demo result, plus general healthy-habit advice
- **Health chatbot** with safety rules, emergency warnings and dashboard context
- **Bilingual UI** (EN / العربية) with one-click switch, dark theme, animated ECG header

<a id="dashboard-pages"></a>

## <img src="https://api.iconify.design/mdi:view-dashboard.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Dashboard pages

| Page | What you get |
|------|--------------|
| <img src="https://api.iconify.design/mdi:home.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> **Overview** | KPI cards, headline insights, key charts, quick navigation |
| <img src="https://api.iconify.design/mdi:chart-line.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> **Analysis** | 49 charts: Population, Body & BMI, Heart & Stroke, Chronic Conditions, Lifestyle, Function & Mobility, Geography, Correlations & Risk |
| <img src="https://api.iconify.design/mdi:lightbulb-on.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> **Insights** | Strongest associations, age/smoking/weight/geography findings, with-vs-without ratio chart |
| <img src="https://api.iconify.design/mdi:puzzle.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> **Segments** | Choose K (2–8), elbow and silhouette, cluster profiles, PCA space, anomaly scores, t-SNE |
| <img src="https://api.iconify.design/mdi:flask.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> **Models** | ROC-AUC / PR-AUC, recall and precision at any threshold, calibration, model leaderboard, top drivers |
| <img src="https://api.iconify.design/mdi:stethoscope.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> **Predict** | Enter demographics, body, lifestyle, history and function data and get a risk score |
| <img src="https://api.iconify.design/mdi:check-circle.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> **Recommendations** | Tips tailored to your result and general guidance with data-backed notes |
| <img src="https://api.iconify.design/mdi:chat.svg?color=%2322d3ee" width="18" height="18" align="absmiddle"> **Assistant** | Ask general health questions in Arabic or English |

<a id="dataset"></a>

## <img src="https://api.iconify.design/mdi:database.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Dataset

- **10,000 patients × 35 columns**, target `HeartAttackRisk` (5.55% positive, 555 cases)
- Demographics, body measures, chronic conditions, lifestyle, daily function, preventive care and US state
- `PatientID` and `State` are excluded from model inputs
- Features derived from the target (`ChronicCount`, `Has3PlusChronic`) are used for EDA only, never for modelling

<a id="methodology"></a>

## <img src="https://api.iconify.design/mdi:brain.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Methodology

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

## <img src="https://api.iconify.design/mdi:trophy.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Results

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

## <img src="https://api.iconify.design/mdi:tools.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Tech stack

| Area | Tools |
|------|-------|
| Data | pandas, NumPy, openpyxl |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, TensorFlow / Keras, mlxtend |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit with custom CSS (dark theme, RTL) |
| Chatbot | Google `google-genai` client |

<a id="project-structure"></a>

## <img src="https://api.iconify.design/mdi:folder-outline.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Project structure

```
NTI-Graduation-Project-Heart-Attack-Risk/
├── HeartAttackRisk.ipynb       # main notebook: EDA, preprocessing, models, unsupervised, NN
├── app.py                      # Streamlit dashboard (8 pages, EN/AR)
├── health_assistant.py         # chatbot logic (used by the app and as a terminal chat)
├── MedicalDataset.xlsx         # dataset (10,000 × 35)
├── requirements.txt            # app dependencies
├── requirements-notebook.txt   # notebook dependencies
├── artifacts/                  # optional: saved preprocessor + best model from the notebook
└── .gitignore
```

If `artifacts/` is missing, the app re-trains the same Logistic Regression pipeline automatically on first load.

<a id="run-it-locally"></a>

## <img src="https://api.iconify.design/mdi:rocket-launch.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Run it locally

```bash
# 1. clone
git clone https://github.com/ahmedfathallahz/NTI-Graduation-Project-Heart-Attack-Risk.git
cd NTI-Graduation-Project-Heart-Attack-Risk

# 2. create an environment and install
python -m venv env
env\Scripts\activate            # macOS / Linux: source env/bin/activate
pip install -r requirements.txt

# 3. run
streamlit run app.py
```

Place `MedicalDataset.xlsx` next to `app.py`, or upload it from the app's first screen.

### <img src="https://api.iconify.design/mdi:chat.svg?color=%2322d3ee" width="20" height="20" align="absmiddle"> Enable the chatbot

Create `.streamlit/secrets.toml` (it is git-ignored):

```toml
GEMINI_API_KEY = "your-key-here"
```

or set the `GEMINI_API_KEY` environment variable. **Never commit your key.**

Terminal version: `python health_assistant.py`

### <img src="https://api.iconify.design/mdi:notebook.svg?color=%2322d3ee" width="20" height="20" align="absmiddle"> Re-run the notebook

```bash
pip install -r requirements-notebook.txt
jupyter notebook HeartAttackRisk.ipynb
```

The notebook writes its artifacts to `artifacts/`, and the app picks them up automatically.

<a id="deploy-on-streamlit-cloud"></a>

## <img src="https://api.iconify.design/mdi:cloud-upload.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Deploy on Streamlit Cloud

1. Push the repo to GitHub (without `env/`, `__pycache__/` or `.streamlit/secrets.toml`).
2. On [share.streamlit.io](https://share.streamlit.io) choose the repo, branch `main`, main file `app.py`.
3. In **Advanced settings** pick Python **3.12** and paste your key into **Secrets**:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Deploy. Live version: **[nabd-ml.streamlit.app](https://nabd-ml.streamlit.app/)**

<a id="limitations"></a>

## <img src="https://api.iconify.design/mdi:scale-balance.svg?color=%2322d3ee" width="22" height="22" align="absmiddle"> Limitations

### <img src="https://api.iconify.design/mdi:database-alert.svg?color=%23f59e0b" width="20" height="20" align="absmiddle"> 1. The data is a small sample, not real clinical data

- **Size:** 10,000 patients with 555 heart-attack cases (5.550%). The test set holds only 1,500 patients, of whom **83 are positive** (5.533%).
- **Label quality:** heart-attack history is *self-reported*. It is not confirmed by medical records, ECG or lab tests.
- **Single source:** the model was never checked on another hospital, country or time period, so there is no external validation.
- **Associations, not causes:** age mixes into most of the gaps seen in the EDA, and group comparisons (race, state) are unadjusted.
- **Uncertainty:** with only 83 positives, recall is not a precise number. At the screening threshold it is **0.831** (69 of 83), and its 95% Wilson confidence interval is **0.737 – 0.897**.

**Next step:** look for realistic clinical data (hospital or EHR records with confirmed outcomes, several sites, a larger number of positive cases), retrain on it, and validate externally before drawing any real-world conclusion.

### <img src="https://api.iconify.design/mdi:heart-pulse.svg?color=%23ef4444" width="20" height="20" align="absmiddle"> 2. Why recall matters most in medical screening

**Recall (sensitivity) = TP ÷ (TP + FN)** is the share of real cases the model catches.

- A **false negative** is a patient with a history who is told "low risk". They may skip a check-up and lose time.
- A **false alarm** costs an extra visit or test.

Missing a case is much more costly than an extra check, so the project chooses the threshold by **minimum recall ≥ 80%** on the validation set (threshold fixed there, then reported once on the test set).

Test set, n = 1,500 (83 positives, 1,417 negatives):

| Operating point | Threshold | Recall | Missed (FN) | Precision | False alarms (FP) | Patients flagged |
|-----------------|:---------:|:------:|:-----------:|:---------:|:-----------------:|:----------------:|
| Default | 0.500 | 0.759 | 20 | 0.217 | 227 | 19.333% |
| F1-optimal | 0.813 | 0.542 | 38 | 0.405 | 66 | 7.400% |
| **Min recall ≥ 80% (screening)** | **0.360** | **0.831** | **14** | 0.168 | 342 | 27.400% |

**What the trade-off means in numbers**

- Moving from the F1-optimal to the screening threshold catches **24 more cases** (38 → 14 missed) at the price of **276 more false alarms** (66 → 342).
- At the screening threshold, 411 patients are flagged and 342 of them (**0.832**) have no recorded history, so precision is only **0.168** and specificity is **0.759**.
- Negative predictive value is **0.987** (1,075 of the 1,089 patients labelled low-risk are truly negative), but the **14 missed cases** show that "low risk" is never a clearance.

### <img src="https://api.iconify.design/mdi:alert-circle-outline.svg?color=%23f59e0b" width="20" height="20" align="absmiddle"> 3. Other limits

- The risk score is a learning tool, **not a diagnosis**, and the model is not clinically validated.
- Low precision is the deliberate cost of high recall; a real deployment would need a follow-up test to filter false alarms.
- Test metrics are reported to three decimals (ROC-AUC 0.871, PR-AUC 0.407) and come from one random split, so small differences between models (for example 0.397 to 0.415 PR-AUC) may well be within noise (no confidence intervals were computed for them).

---

<div align="center">

Made with care for the NTI graduation project · <a href="https://nabd-ml.streamlit.app/">Try the live app</a>

</div>
