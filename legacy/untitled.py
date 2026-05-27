import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import VotingClassifier

# 1. Cost-Sensitive Wrapper for XGBoost
class CostSensitiveXGBClassifier(XGBClassifier):
    def fit(self, X, y, **kwargs):
        sample_weight = compute_sample_weight(class_weight='balanced', y=y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)

# 2. Logistic Regression Tuning Function
# def lr_train(X, y, cw=None):
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     C_values = np.logspace(-2, 2, 10)
#     grid = GridSearchCV(
#         LogisticRegression(random_state=42, max_iter=1000, class_weight=cw),
#         {'C': C_values},
#         cv=kf,
#         scoring='accuracy',
#         n_jobs=4,
#         verbose=1
#     )
#     grid.fit(X, y)
#     print(f"\nBest LR CV accuracy: {grid.best_score_:.4f}")
#     print("Best C:", grid.best_params_['C'])
#     return grid.best_estimator_

# 3. Load & Split Data
X = pd.read_csv('X_train_cleaned.csv')
y = pd.read_csv('y_train_cleaned.csv')['label']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
num_classes = len(np.unique(y_train))
print("Number of classes:", num_classes)

# 4. Define Oversampling + Model Pipelines

# 4a. ADASYN + Logistic Regression
pipe_lr = ImbPipeline(steps=[
    ('adasyn', ADASYN(random_state=42, n_neighbors=3)),
    ('lr', LogisticRegression(C=0.215,solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=42))
])

# 4b. BorderlineSMOTE + SVM
pipe_svm = ImbPipeline(steps=[
    ('bsmote', BorderlineSMOTE(random_state=42, k_neighbors=3)),
    ('svm', SVC(C=100,class_weight='balanced', probability=True, random_state=42))
])

# 4c. ADASYN + LightGBM
pipe_lgbm = ImbPipeline(steps=[
    ('adasyn', ADASYN(random_state=42, n_neighbors=3)),
    ('lgbm', LGBMClassifier(learning_rate=0.05,n_estimators=300,class_weight='balanced',random_state=42))
])

# 4d. BorderlineSMOTE + Cost-Sensitive XGBoost
pipe_xgb = ImbPipeline(steps=[
    ('bsmote', BorderlineSMOTE(random_state=42, k_neighbors=3)),
    ('xgb', CostSensitiveXGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        objective='multi:softprob',
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    ))
])

# # 5. Tune Logistic Regression (not in pipeline)
# best_lr = lr_train(X_train, y_train, cw='balanced')

# 6. Ensemble Voting (replace RF with SVM)
ensemble = VotingClassifier(
    estimators=[
        ('lr_best',pipe_lr),
        ('svm', pipe_svm),
        ('lgbm', pipe_lgbm),
        ('xgb', pipe_xgb)
    ],
    voting='soft'
)

# 7. Train Ensemble
ensemble.fit(X_train, y_train)

# 8. Evaluate on Validation Set
y_pred = ensemble.predict(X_val)
print("Ensemble Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Ensemble Classification Report:")
print(classification_report(y_val, y_pred, zero_division=0))
