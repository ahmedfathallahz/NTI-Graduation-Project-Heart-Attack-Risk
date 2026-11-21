# NABD | نبض — Heart-Attack Risk Analytics

NTI graduation project. An end-to-end analysis of a medical dataset (EDA, preprocessing, supervised and unsupervised learning, a neural network) and a bilingual (English / Arabic) **Streamlit dashboard** built on top of it, with a Gemini-powered general health assistant.

> **Educational project only.** It is not a medical device and does not replace a doctor. The dataset is a sample and the model is not clinically validated.

## What is inside

| Part | Description |
|------|-------------|
| `HeartAttackRisk.ipynb` | The main notebook: data inspection, 50 EDA figures, leakage-free preprocessing, model suite, unsupervised learning, neural network, model comparison |
| `app.py` | Streamlit dashboard (Overview, Analysis, Insights, Segments, Models, Predict, Recommendations, Assistant) |
| `health_assistant.py` | Shared Gemini assistant logic (used by the app, and runnable as a terminal chat) |
| `artifacts/` | Optional. `preprocessor.joblib` and `best_heart_attack_model.joblib` saved by the notebook. If missing, the app re-trains the same Logistic Regression pipeline |
| `MedicalDataset.xlsx` | The dataset (10,000 patients, 35 columns). Place it next to `app.py`, or upload it in the app |

## The data

- 10,000 patients, 35 columns, target `HeartAttackRisk` (5.55% positive).
- Stratified split: 70% train / 15% validation / 15% test.
- `PatientID` and `State` are excluded from model inputs. Features derived from the target (`ChronicCount`, `Has3PlusChronic`) are used for EDA only, never for modelling.

## Method

**Preprocessing** (fitted on the training set only):
continuous → median imputation, IQR outlier clipping, log transform, robust scaling · binary → most-frequent imputation · ordinal (general health, age) → ordinal encoding · nominal → one-hot.
Engineered features: age midpoint, comorbidity count, lifestyle risk score, BMI category.

**Class imbalance:** class weights only (about 17:1), no resampling, so probabilities are not distorted twice.

**Supervised models** (tuned on the training set with stratified 5-fold CV, scoring PR-AUC): Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, a Stacking ensemble and a Keras MLP. Thresholds are fixed on the validation set and applied once to the test set.

**Unsupervised learning:** PCA, K-Means, GMM, hierarchical clustering, t-SNE, Isolation Forest and LOF anomaly detection, association rules (Apriori). The unsupervised features were tested as extra inputs and not added to the supervised models because they did not improve PR-AUC for both models.

## Results (held-out test set, n = 1,500)

Selected model: **Logistic Regression** (best validation PR-AUC), test ROC-AUC **0.871**, test PR-AUC **0.407**.

| Operating point | Threshold | Recall | Precision | Missed cases |
|-----------------|-----------|--------|-----------|--------------|
| Default | 0.50 | 0.759 | 0.217 | 20 of 83 |
| F1-optimal | 0.813 | 0.542 | 0.405 | 38 of 83 |
| Min recall ≥ 80% (chosen for screening) | 0.360 | 0.831 | 0.168 | 14 of 83 |

Test-set comparison of all models:

| Model | Test PR-AUC | Test ROC-AUC |
|-------|-------------|--------------|
| Neural Network (MLP) | 0.441 | 0.861 |
| XGBoost | 0.415 | 0.872 |
| CatBoost | 0.414 | 0.872 |
| Stacking Ensemble | 0.410 | 0.873 |
| Logistic Regression | 0.407 | 0.871 |
| Random Forest | 0.401 | 0.871 |
| LightGBM | 0.397 | 0.871 |

For screening, missing a true case is costlier than a false alarm, so the dashboard uses the recall-oriented threshold by default and lets you move it.

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Gemini assistant (optional)

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

or set the `GEMINI_API_KEY` environment variable. **Never commit your key.** Terminal chat: `python health_assistant.py`.

### Re-run the notebook

```bash
pip install -r requirements-notebook.txt
jupyter notebook HeartAttackRisk.ipynb
```

The notebook writes its artifacts to `artifacts/`; the app picks them up automatically.

## Limitations

- Heart-attack history is self-reported and the sample is small (555 positive cases), so results are associations, not causes.
- Age mixes into most of the gaps seen in the EDA.
- The risk score is a toy score for learning purposes, not a diagnosis.
