# src/model_training.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ==============================
# CONFIG
# ==============================
DATA_PATH = "data/processed/startups_featured.csv"
MODEL_DIR = "models"
TARGET_COLUMN = "success"
RANDOM_STATE = 42

# ==============================
# LOAD DATA
# ==============================
def load_data(path):
    print("📥 Loading feature-engineered dataset...")
    df = pd.read_csv(path)
    print(f"✅ Dataset shape: {df.shape}")
    return df

# ==============================
# TRAIN-TEST SPLIT
# ==============================
def split_data(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

# ==============================
# HANDLE CLASS IMBALANCE
# ==============================
def balance_data(X_train, y_train):
    print("⚖️ Handling class imbalance using SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# ==============================
# TRAIN XGBOOST
# ==============================
def train_xgboost(X_train, y_train):
    print("🚀 Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    return model

# ==============================
# TRAIN CATBOOST
# ==============================
def train_catboost(X_train, y_train):
    print("🐱 Training CatBoost model...")

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        random_seed=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    return model

# ==============================
# EVALUATION
# ==============================
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n📊 Evaluation for {model_name}")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================
def save_model(model, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    print(f"💾 Model saved at: {path}")

# ==============================
# MAIN PIPELINE
# ==============================
def main():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    xgb_model = train_xgboost(X_train_bal, y_train_bal)
    cat_model = train_catboost(X_train_bal, y_train_bal)

    evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    evaluate_model(cat_model, X_test, y_test, "CatBoost")

    save_model(xgb_model, "xgboost_model.pkl")
    save_model(cat_model, "catboost_model.pkl")

    print("\n✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("🔥 Ready for ensemble or deployment")

if __name__ == "__main__":
    main()

