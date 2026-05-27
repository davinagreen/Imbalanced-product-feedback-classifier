# Data files

Training data is **not** committed to Git (CSV files are large). Place the following files in this directory:

| File | Description |
|------|-------------|
| `X_train_cleaned.csv` | Training features (300 NLP features) |
| `y_train_cleaned.csv` | Training labels (`label` column, 28 classes) |

Optional (for test-set inference experiments):

| File | Description |
|------|-------------|
| `X_train.csv`, `y_train.csv` | Raw training set from course release |
| `X_test_1.csv`, `X_test_2.csv` | Held-out test features |
| `y_test_2_reduced.csv` | Partial test labels |

If you have the course zip `Group Project - Data-20250405.zip` in the project root, unzip it here and use your cleaned copies (`X_train_cleaned.csv`, `y_train_cleaned.csv`) produced during preprocessing.
