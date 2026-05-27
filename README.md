# Imbalanced Product Feedback Classifier

COMP9417 group project: multiclass classification of customer product feedback into **28 departments** using **300 NLP features**, with strong class imbalance (minority classes as small as ~6 samples).

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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `X_train_cleaned.csv` and `y_train_cleaned.csv` into `data/` (see [data/README.md](data/README.md)).

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

Plots are written to `outputs/baseline/` when running the baseline script.

## Course context

UNSW COMP9417 (T1 2025) — *Terrific Group* product feedback classification project.

## License

Academic coursework — check with course staff before redistributing data or report materials.
