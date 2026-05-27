#!/usr/bin/env python3
"""
Individual models (DT, LR, SVM, XGBoost) with resampling and stacking ensemble.
Consolidated from the group stacking / grid-search notebook.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from paths import DATA_DIR

RANDOM_SEED = 42


def downsample(X, y, cap):
    def custom_sampling_strategy(cls_counts):
        if not isinstance(cls_counts, dict):
            cls_counts = dict(Counter(cls_counts))
        return {cls: min(count, cap) for cls, count in cls_counts.items()}

    undersampler = RandomUnderSampler(
        sampling_strategy=custom_sampling_strategy, random_state=RANDOM_SEED
    )
    return undersampler.fit_resample(X, y)


def upsample(X, y, target_count, use_smote=False):
    def smote_strategy(cls_counts):
        cls_counts = dict(Counter(cls_counts))
        return {
            cls: target_count
            for cls, count in cls_counts.items()
            if count < target_count
        }

    if not use_smote:
        oversampler = RandomOverSampler(random_state=RANDOM_SEED)
        return oversampler.fit_resample(X, y)

    smote = SMOTE(
        sampling_strategy=smote_strategy, k_neighbors=2, random_state=RANDOM_SEED
    )
    return smote.fit_resample(X, y)


def dt_train(X, y, class_weight=None):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_grid = {
        "criterion": ["entropy", "gini"],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    model = DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight=class_weight)
    grid = GridSearchCV(model, param_grid, cv=kf, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y)
    return grid, grid.best_estimator_


def lr_train(X, y, class_weight=None):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_grid = {"C": np.logspace(-2, 2, 10)}
    model = LogisticRegression(
        random_state=RANDOM_SEED, max_iter=1000, class_weight=class_weight
    )
    grid = GridSearchCV(model, param_grid, cv=kf, scoring="accuracy", n_jobs=4)
    grid.fit(X, y)
    return grid, grid.best_estimator_


def svm_train(X, y, class_weight=None):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    c_values = np.logspace(-2, 2, 5)
    param_grid = [
        {"kernel": ["linear"], "C": c_values},
        {"kernel": ["rbf"], "C": c_values, "gamma": ["scale", "auto"]},
    ]
    model = SVC(random_state=RANDOM_SEED, class_weight=class_weight)
    grid = GridSearchCV(model, param_grid, cv=kf, scoring="accuracy", n_jobs=4)
    grid.fit(X, y)
    return grid, grid.best_estimator_


def xgb_train(X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.2],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
        "gamma": [0, 0.1],
    }
    model = xgb.XGBClassifier(
        random_state=RANDOM_SEED, objective="multi:softprob", eval_metric="mlogloss"
    )
    grid = GridSearchCV(model, param_grid, cv=kf, scoring="accuracy", n_jobs=4)
    grid.fit(X, y)
    return grid, grid.best_estimator_


def stacking_train(models, final_estimator, X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    stacking_model = StackingClassifier(
        estimators=models, final_estimator=final_estimator, cv=kf
    )
    stacking_model.fit(X, y)
    return stacking_model


def model_eval(model, X, y):
    preds = model.predict(X)
    print(classification_report(y, preds, zero_division=0))
    return (
        accuracy_score(y, preds),
        precision_score(y, preds, average="weighted", zero_division=0),
        recall_score(y, preds, average="weighted", zero_division=0),
        f1_score(y, preds, average="weighted", zero_division=0),
    )


def train_eval(X_train, y_train, X_test, y_test, results_path, class_weight=None):
    rows = []

    _, dt_best = dt_train(X_train, y_train, class_weight)
    rows.append(["Decision Tree", *model_eval(dt_best, X_test, y_test)])

    lr_grid, lr_best = lr_train(X_train, y_train, class_weight)
    rows.append(["Logistic Regression", *model_eval(lr_grid, X_test, y_test)])

    svm_grid, svm_best = svm_train(X_train, y_train, class_weight)
    rows.append(["SVM", *model_eval(svm_grid, X_test, y_test)])

    xgb_grid, xgb_best = xgb_train(X_train, y_train)
    rows.append(["XGBoost", *model_eval(xgb_grid, X_test, y_test)])

    stacking_model = stacking_train(
        [("dt", dt_best), ("svm", svm_best), ("xgb", xgb_best)],
        lr_best,
        X_train,
        y_train,
    )
    rows.append(["Stacking", *model_eval(stacking_model, X_test, y_test)])

    df = pd.DataFrame(
        rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Stacking models with resampling")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stacking_eval.csv"),
    )
    parser.add_argument("--target-per-class", type=int, default=300)
    args = parser.parse_args()

    x_path = args.data_dir / "X_train_cleaned.csv"
    y_path = args.data_dir / "y_train_cleaned.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Expected {x_path} and {y_path}. See data/README.md for setup."
        )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).values.ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("Original class distribution:", Counter(y_train))
    X_train, y_train = upsample(
        X_train, y_train, target_count=args.target_per_class, use_smote=True
    )
    print("Resampled class distribution:", dict(Counter(y_train)))
    print(f"Training samples after resampling: {X_train.shape[0]}")

    train_eval(
        X_train,
        y_train,
        X_test,
        y_test,
        args.output,
        class_weight="balanced",
    )


if __name__ == "__main__":
    main()
