#!/usr/bin/env python3
"""
Soft-voting ensemble: LR, SVM, LightGBM, XGBoost with ADASYN / Borderline-SMOTE pipelines.
"""

import argparse
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from paths import DATA_DIR

RANDOM_SEED = 42


class CostSensitiveXGBClassifier(XGBClassifier):
    """XGBoost with balanced sample weights per fit."""

    def fit(self, X, y, **kwargs):
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


def build_ensemble(num_classes: int) -> VotingClassifier:
    pipe_lr = ImbPipeline(
        steps=[
            ("adasyn", ADASYN(random_state=RANDOM_SEED, n_neighbors=3)),
            (
                "lr",
                LogisticRegression(
                    C=0.215,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    pipe_svm = ImbPipeline(
        steps=[
            ("bsmote", BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=3)),
            (
                "svm",
                SVC(
                    C=100,
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    pipe_lgbm = ImbPipeline(
        steps=[
            ("adasyn", ADASYN(random_state=RANDOM_SEED, n_neighbors=3)),
            (
                "lgbm",
                LGBMClassifier(
                    learning_rate=0.05,
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    pipe_xgb = ImbPipeline(
        steps=[
            ("bsmote", BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=3)),
            (
                "xgb",
                CostSensitiveXGBClassifier(
                    learning_rate=0.1,
                    n_estimators=200,
                    objective="multi:softprob",
                    num_class=num_classes,
                    eval_metric="mlogloss",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    return VotingClassifier(
        estimators=[
            ("lr", pipe_lr),
            ("svm", pipe_svm),
            ("lgbm", pipe_lgbm),
            ("xgb", pipe_xgb),
        ],
        voting="soft",
    )


def main():
    parser = argparse.ArgumentParser(description="Ensemble voting classifier")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    x_path = args.data_dir / "X_train_cleaned.csv"
    y_path = args.data_dir / "y_train_cleaned.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Expected {x_path} and {y_path}. See data/README.md for setup."
        )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)["label"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    num_classes = len(y_train.unique())
    print(f"Number of classes: {num_classes}")

    ensemble = build_ensemble(num_classes)
    print("Training ensemble (this may take several minutes)...")
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_val)
    print(f"Ensemble validation accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_val, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
