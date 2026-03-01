# DPE Challenge — Skeleton Adaptation Guide

This document explains the changes made to the Codabench template skeleton to adapt it for our **DPE Energy Consumption Prediction Challenge**.

---

## 1. Challenge Overview

- **Problem type**: Regression
- **Target variable**: `conso_5_usages_par_m2_ef` — the final energy consumption per square meter (kWh/m²/year)
- **Data format**: `.parquet` files (instead of the original `.csv`)
- **Current data**: The `train.parquet` and `test.parquet` currently in the repo are **synthetic / placeholder data** generated for pipeline testing purposes. They will be replaced with the real curated ADEME DPE dataset later.

---

## 2. Anti-Leakage Rules

The following columns are **always stripped** from the features before they reach any model. This is enforced at two levels (in `setup_data.py` when building the splits, and in `ingestion.py` at runtime as a safety net):

| Dropped Column | Reason |
|---|---|
| `conso_5_usages_par_m2_ef` | Target variable |
| `etiquette_dpe` | Directly derived from the target (DPE grade) |
| `etiquette_ges` | GHG emission grade |
| `cout_chauffage` | Heating cost (leaks energy info) |
| `cout_total_5_usages` | Total energy cost |
| `emission_ges_5_usages_par_m2` | GHG emissions per m² |

These constants are centralized in [`ingestion_program/bench_utils/__init__.py`](ingestion_program/bench_utils/__init__.py).

---

## 3. Baseline Model

The baseline submission in [`solution/submission.py`](solution/submission.py) uses a **scikit-learn `Pipeline`** with:

1. **`OrdinalEncoder`** — to handle categorical string columns (e.g., `type_batiment`, `qualite_isolation_enveloppe`)
2. **`RandomForestRegressor`** — with `n_estimators=100` and `random_state=42`

Participants must provide a `submission.py` file that exposes a **`get_model()`** function returning any scikit-learn-compatible model (i.e., with `.fit(X, y)` and `.predict(X)` methods). This can be a single estimator or a full `Pipeline`.

---

## 4. Scoring Metrics

Submissions are evaluated using **two regression metrics**:

| Metric | Formula | Role |
|---|---|---|
| **RMSE** (Root Mean Squared Error) | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | **Primary** — used for leaderboard ranking (lower is better) |
| **MAE** (Mean Absolute Error) | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | **Secondary** — displayed for reference, not used for ranking |

Both metrics are computed independently on the **public test set** and the **private test set**. The leaderboard ranks participants by their **RMSE on the public test set** during the development phase. The private test RMSE is used for the final ranking (hidden until the competition ends).

> **Note**: RMSE and MAE are **not combined** into a single score. RMSE alone determines the ranking. MAE is simply shown as an additional indicator.

The scoring logic is in [`scoring_program/scoring.py`](scoring_program/scoring.py).

---

## 5. Starting Kit Notebook

The file [`template_starting_kit.ipynb`](template_starting_kit.ipynb) is the participant-facing notebook. It contains:

1. **Introduction** — Challenge description, data source, task, and anti-leakage rules
2. **Exploratory Data Analysis** — Load data, inspect features, visualize target distribution and feature correlations
3. **Evaluation** — RMSE and MAE formulas, public/private split strategy
4. **Submission Format** — How `get_model()` works, baseline `Pipeline` code
5. **Local Testing Pipeline** — Full train → predict → score flow that participants can run locally before submitting

To run the notebook, make sure the `dev_phase/` directory exists (run `python tools/setup_data.py` first) and use the **`datacamp`** conda environment.

---

## 6. Pipeline Flow Summary

```
train.parquet + test.parquet
        │
        ▼
  setup_data.py          →  dev_phase/input_data/   (features as .parquet)
                          →  dev_phase/reference_data/ (labels as .csv)
        │
        ▼
  ingestion.py            →  Loads train data, drops leaked columns
                          →  Calls get_model().fit(X_train, y_train)
                          →  Predicts on test + private_test
                          →  Saves predictions + timing metadata
        │
        ▼
  scoring.py              →  Loads predictions + ground truth
                          →  Computes RMSE + MAE
                          →  Outputs scores.json
```

---

## 7. Modified Files Summary

| File | Change |
|---|---|
| `ingestion_program/bench_utils/__init__.py` | `TARGET_COLUMN` and `LEAK_COLUMNS` constants |
| `ingestion_program/ingestion.py` | Parquet loading + `clean_features()` anti-leakage guard |
| `scoring_program/scoring.py` | RMSE + MAE metrics (replaced accuracy) |
| `solution/submission.py` | Pipeline with OrdinalEncoder + RandomForestRegressor |
| `tools/setup_data.py` | Loads parquet, extracts target, drops leakage, splits test/private |
| `competition.yaml` | DPE title, RMSE/MAE leaderboard columns |
| `requirements.txt` | Added `pyarrow` |
| `template_starting_kit.ipynb` | Full notebook with EDA, evaluation, submission, local pipeline |
