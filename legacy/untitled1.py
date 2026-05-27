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


def weighted_log_loss_formula(y_true, y_pred_proba, eps=1e-15):
    y_true = np.asarray(y_true)
    N, C = y_pred_proba.shape

   
    class_counts = np.bincount(y_true, minlength=C)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts


    p = y_pred_proba[np.arange(N), y_true]
    p = np.clip(p, eps, 1 - eps)  

    # log‐loss
    losses = - class_weights[y_true] * np.log(p)
    return losses.sum() / N
#---------------------------------------------------------------------------------
#Ensemble+ADASYN and BorderlineSMOTE mixture
#---------------------------------------------------------------------------------
#  ADASYN + Logistic Regression
def Ensemble_mix:
    pipe_lr = ImbPipeline(steps=[
        ('adasyn', ADASYN(random_state=42, n_neighbors=3)),
        ('lr', LogisticRegression(C=0.215,solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=42))
    ])
    
    #  BorderlineSMOTE + SVM
    pipe_svm = ImbPipeline(steps=[
        ('bsmote', BorderlineSMOTE(random_state=42, k_neighbors=3)),
        ('svm', SVC(C=100,class_weight='balanced', probability=True, random_state=42))
    ])
    
    #  ADASYN + LightGBM
    pipe_lgbm = ImbPipeline(steps=[
        ('adasyn', ADASYN(random_state=42, n_neighbors=3)),
        ('lgbm', LGBMClassifier(learning_rate=0.05,n_estimators=300,class_weight='balanced',random_state=42))
    ])
    
    #  BorderlineSMOTE + Cost-Sensitive XGBoost
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

def Ensemble_ada:
    #  ADASYN + Logistic Regression
    pipe_lr = ImbPipeline(steps=[
      ('adasyn', ADASYN(random_state=42, n_neighbors=3)),
        ('lr', LogisticRegression(C=0.215,solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=42))
    ])
    
    # ADASYN + SVM
    pipe_svm = ImbPipeline(steps=[
       ('adasyn', ADASYN(random_state=42, n_neighbors=3)),
        ('svm', SVC(C=200,class_weight='balanced', probability=True, random_state=42))
    ])
    
    #  ADASYN + LightGBM
    pipe_lgbm = ImbPipeline(steps=[
        ('adasyn', ADASYN(random_state=42, n_neighbors=3)),
        ('lgbm', LGBMClassifier(learning_rate=0.05,n_estimators=300,class_weight='balanced',random_state=42))
    ])
    
    #  ADASYN + Cost-Sensitive XGBoost
    pipe_xgb = ImbPipeline(steps=[
         ('adasyn', ADASYN(random_state=42, n_neighbors=3)),
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

def ensemble_bor:
    pipe_lr = ImbPipeline(steps=[
       ('bsmote', BorderlineSMOTE(random_state=42, k_neighbors=3)),
        ('lr', LogisticRegression(C=0.215,solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=42))
    ])
    
    # 4b. BorderlineSMOTE + SVM
    pipe_svm = ImbPipeline(steps=[
        ('bsmote', BorderlineSMOTE(random_state=42, k_neighbors=3)),
        ('svm', SVC(C=200,class_weight='balanced', probability=True, random_state=42))
    ])
    
    # 4c. ADASYN + LightGBM
    pipe_lgbm = ImbPipeline(steps=[
         ('bsmote', BorderlineSMOTE(random_state=42, k_neighbors=3)),
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


if__main()_:
    ensemble = VotingClassifier(
        estimators=[
            ('lr_best',pipe_lr),
            ('svm', pipe_svm),
            ('lgbm', pipe_lgbm),
            ('xgb', pipe_xgb)
        ],
        voting='soft'
    )

    # 2. Load & Split Data
    X = pd.read_csv('X_train_cleaned.csv')
    y = pd.read_csv('y_train_cleaned.csv')['label']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    num_classes = len(np.unique(y_train))
    print("Number of classes:", num_classes)



    ensemble.fit(X_train, y_train)

    # 8. Evaluate on Validation Set
    y_pred = ensemble.predict(X_val)
    print("Ensemble Validation Accuracy:", accuracy_score(y_val, y_pred))
    print("Ensemble Classification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))
    
 



 
    y_train_proba = ensemble.predict_proba(X_train)
    y_val_proba   = ensemble.predict_proba(X_val)
    
    train_wll = weighted_log_loss_formula(y_train, y_train_proba)
    val_wll   = weighted_log_loss_formula(y_val,   y_val_proba)
    
    print(f"Training weighted log‐loss:   {train_wll:.4f}")
    print(f"Validation weighted log‐loss: {val_wll:.4f}")
    

# Compute the confusion matrix

    y_true = y_val
    y_pred = y_pred
    cm = confusion_matrix(y_true, y_pred)
    
    
    # Plot
    plt.figure(figsize=(12,10))
    sns.set(font_scale=1.2)  # for label size
    ax = sns.heatmap(
        cm,
        annot=True,            # write the data in each cell
        fmt='d',               # integer format
        cmap='Blues',          # or any other colormap
        cbar_kws={'label': 'Count'},
        xticklabels=classes,
        yticklabels=classes
    )

    # Decorations
    ax.set_title('Confusion Matrix', fontsize=16, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    plt.yticks(rotation=0)     # keep y‐tick labels horizontal
    plt.xticks(rotation=90)    # rotate x‐tick labels if they’re long
    plt.tight_layout()
    plt.show()

