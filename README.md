# Imbalanced Product Feedback Classifier

COMP9417 group project: multiclass classification of customer product feedback into **28 departments** using **300 NLP features**, with strong class imbalance (minority classes as small as ~6 samples).

This repository is organised as a small portfolio project: it includes the full training code, core course dataset, and scripts to reproduce the main experiments (baseline, resampling, ensembles).

## Approach

- **Baseline**: class-weighted multinomial logistic regression
- **Imbalance handling**: ADASYN, Borderline-SMOTE, random over/under-sampling, cost-sensitive XGBoost
- **Models**: Logistic Regression, SVM, LightGBM, XGBoost
- **Ensembles**: soft `VotingClassifier` and `StackingClassifier`

## Project structure

```
├── src/
│   ├── baseline_logistic_regression.py   # LR baseline + plots
│   ├── ensemble_voting.py                # 4-model soft voting ensemble
│   └── stacking_models.py                # DT/LR/SVM/XGB + stacking
├── data/                                 # Place CSVs here (see data/README.md)
├── notebooks/original/                   # Archived Jupyter notebooks
├── outputs/                              # Generated plots & metrics (gitignored)
└── requirements.txt
```

## Data

The repository includes the **COMP9417 project dataset CSVs** under `data/`:

- `X_train.csv`, `y_train.csv` – original training features/labels from the course
- `X_train_cleaned.csv`, `y_train_cleaned.csv` – cleaned training set used for final models
- `X_test_1.csv`, `X_test_2.csv`, `y_test_2_reduced.csv` – held-out test features and partial labels
- `eval_upsample_costsensitive.csv` – summary of resampling + cost-sensitive experiments
- `test_predictions.csv`, `test_pred_proba.csv` – saved predictions/probabilities for analysis

These files are used directly by the scripts in `src/` and are required to fully reproduce the course project results.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the project root:

```bash
# Baseline logistic regression (~1 min)
python src/baseline_logistic_regression.py

# Stacking pipeline with SMOTE upsampling (~10–30 min depending on hardware)
python src/stacking_models.py

# Full voting ensemble — slow; resamples inside each fold (~30+ min)
python src/ensemble_voting.py
```

Baseline plots are written to `outputs/baseline/`.

## Reproducibility notes

- All scripts assume the CSV files listed in the **Data** section are present under `data/`.
- Random seeds are fixed (`random_state=42`) to make splits and training runs repeatable.
- No external services are required; everything runs locally with `scikit-learn`, `imbalanced-learn`, `LightGBM`, and `XGBoost`.

## Course context

UNSW COMP9417 (T1 2025) — *Terrific Group* product feedback classification project.

## License

Academic coursework — please follow UNSW course policies if reusing the dataset or report materials.
