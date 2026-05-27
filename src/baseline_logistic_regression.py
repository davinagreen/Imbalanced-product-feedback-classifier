#!/usr/bin/env python3
"""
Logistic regression baseline for product feedback classification (28 classes, 300 NLP features).
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from paths import DATA_DIR, data_path

RANDOM_SEED = 42


def load_data(x_path, y_path):
    print(f"Loading data from {x_path} and {y_path}...")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)["label"]
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def analyze_data(X, y):
    print("\nDataset Analysis:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    class_counts = Counter(y)
    num_classes = len(class_counts)
    print(f"Number of classes: {num_classes}")

    majority_class = max(class_counts.values())
    minority_class = min(class_counts.values())
    imbalance_ratio = majority_class / minority_class

    print(
        f"Samples in the majority class: {majority_class} "
        f"({majority_class / len(y) * 100:.2f}%)"
    )
    print(f"Samples in the minority class: {minority_class}")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 classes by frequency:")
    for class_label, count in sorted_classes[:10]:
        print(f"Class {class_label}: {count} ({count / len(y) * 100:.2f}%)")

    print("\nFeature distribution summary:")
    print(X.describe().T[["mean", "std", "min", "max"]].head())
    print(f"Number of missing values: {X.isnull().sum().sum()}")

    return class_counts


def plot_class_distribution(class_counts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_counts)
    plt.bar(labels, counts)
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = output_dir / "class_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"Class distribution plot saved to {out}")


def train_logistic_regression(X_train, y_train, X_val, y_val, class_counts):
    print("\nTraining logistic regression model...")
    total_samples = sum(class_counts.values())
    class_weights = {
        label: total_samples / (len(class_counts) * count)
        for label, count in class_counts.items()
    }

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=0.215,
        max_iter=1000,
        class_weight=class_weights,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy:.4f}")
    return model


def evaluate_model(model, X_val, y_val, output_dir):
    print("\nDetailed model evaluation:")
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    print(classification_report(y_val, y_pred))

    active_classes = sorted(set(y_val))
    active_indices = [
        i
        for i, cls in enumerate(sorted(set(y_val) | set(y_pred)))
        if cls in active_classes
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    cm_subset = cm[np.ix_(active_indices, active_indices)]
    sns.heatmap(
        cm_subset,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=active_classes,
        yticklabels=active_classes,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out = output_dir / "confusion_matrix.png"
    plt.savefig(out)
    plt.close()
    print(f"Confusion matrix saved to {out}")


def analyze_feature_importance(model, n_features, output_dir):
    coef = model.coef_
    importances = np.abs(coef).mean(axis=0)
    feature_importance = pd.DataFrame(
        {
            "Feature": [f"feature_{i}" for i in range(n_features)],
            "Importance": importances,
        }
    )
    sorted_features = feature_importance.sort_values("Importance", ascending=False)
    print("\nTop 10 most important features:")
    print(sorted_features.head(10))

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    top_features = sorted_features.head(10)
    sns.barplot(x="Importance", y="Feature", data=top_features)
    plt.title("Top 10 Most Important Features")
    plt.tight_layout()
    out = output_dir / "feature_importance.png"
    plt.savefig(out)
    plt.close()
    print(f"Feature importance plot saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Logistic regression baseline")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()

    x_path = args.data_dir / "X_train_cleaned.csv"
    y_path = args.data_dir / "y_train_cleaned.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Expected {x_path} and {y_path}. See data/README.md for setup."
        )

    np.random.seed(RANDOM_SEED)
    print("===== Product Feedback Classification — Logistic Regression Baseline =====")

    X, y = load_data(x_path, y_path)
    class_counts = analyze_data(X, y)
    plot_class_distribution(class_counts, args.output_dir)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")

    model = train_logistic_regression(X_train, y_train, X_val, y_val, class_counts)

    train_loss = log_loss(y_train, model.predict_proba(X_train))
    val_loss = log_loss(y_val, model.predict_proba(X_val))
    print(f"Training Log Loss:   {train_loss:.4f}")
    print(f"Validation Log Loss: {val_loss:.4f}")

    evaluate_model(model, X_val, y_val, args.output_dir)
    analyze_feature_importance(model, X.shape[1], args.output_dir)
    print("\nModel training and evaluation complete.")


if __name__ == "__main__":
    main()
