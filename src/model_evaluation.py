# src/model_evaluation.py

import pandas as pd
import os
import joblib

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ==============================
# CONFIG
# ==============================
DATA_PATH = "data/processed/startups_featured.csv"
MODEL_DIR = "models"
TARGET_COLUMN = "success"
RANDOM_STATE = 42
N_SPLITS = 5
def get_scale_pos_weight(y):
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos



# ==============================
# LOAD DATA
# ==============================
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

# ==============================
# BALANCE DATA
# ==============================

# ==============================
# XGBOOST TUNING
# ==============================
def tune_xgboost(X, y):
    print("🔧 Tuning XGBoost...")

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
    }
    scale_pos_weight = get_scale_pos_weight(y)

    model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE
    )

    


    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=20,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search.fit(X, y)
    print("✅ Best XGBoost ROC-AUC:", search.best_score_)
    return search.best_estimator_

# ==============================
# CATBOOST TUNING
# ==============================
def tune_catboost(X, y):
    print("🔧 Tuning CatBoost...")

    param_grid = {
        "iterations": [200, 300, 400],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1]
    }

    model = CatBoostClassifier(
        loss_function="Logloss",
        verbose=False,
        random_seed=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=15,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=N_SPLITS),
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search.fit(X, y)
    print("✅ Best CatBoost ROC-AUC:", search.best_score_)
    return search.best_estimator_

# ==============================
# SAVE MODEL
# ==============================
def save_model(model, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(model, path)
    print(f"💾 Saved model: {path}")

# ==============================
# MAIN PIPELINE
# ==============================
def main():
    X, y = load_data()

    best_xgb = tune_xgboost(X, y)
    best_cat = tune_catboost(X, y)



    save_model(best_xgb, "xgboost_tuned.pkl")
    save_model(best_cat, "catboost_tuned.pkl")

    print("\n✅ HYPERPARAMETER TUNING COMPLETED")
    print("🔥 Models are cross-validated & optimized")

if __name__ == "__main__":
    main()

