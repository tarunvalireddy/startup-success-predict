# src/inference.py

import numpy as np
import pandas as pd
import joblib
import os

# ==============================
# CONFIG
# ==============================
MODEL_DIR = "models"
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_tuned.pkl")
CAT_PATH = os.path.join(MODEL_DIR, "catboost_tuned.pkl")
STACK_PATH = os.path.join(MODEL_DIR, "stacked_ensemble.pkl")
FEATURE_TEMPLATE_PATH = "data/processed/startups_featured.csv"
TARGET_COLUMN = "success"

# ==============================
# LOAD MODELS
# ==============================
def load_models():
    xgb = joblib.load(XGB_PATH)
    cat = joblib.load(CAT_PATH)
    stack = joblib.load(STACK_PATH)
    return xgb, cat, stack

# ==============================
# LOAD FEATURE TEMPLATE
# ==============================
def load_feature_columns():
    df = pd.read_csv(FEATURE_TEMPLATE_PATH)
    return df.drop(columns=[TARGET_COLUMN]).columns.tolist()

# ==============================
# BUILD INPUT VECTOR
# ==============================
def build_input(feature_columns, user_input):
    df = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    for key, value in user_input.items():
        if key in df.columns:
            df.at[0, key] = float(value)

    return df

# ==============================
# PREDICT FUNCTION
# ==============================
def predict_startup_success(user_input):
    xgb, cat, stack = load_models()
    feature_columns = load_feature_columns()

    input_df = build_input(feature_columns, user_input)

    # Base model probabilities
    xgb_prob = xgb.predict_proba(input_df)[:, 1]
    cat_prob = cat.predict_proba(input_df)[:, 1]

    # Meta features (ONLY 2 FEATURES)
    meta_features = np.column_stack((xgb_prob, cat_prob))

    final_prob = stack.predict_proba(meta_features)[0][1]
    prediction = 1 if final_prob >= 0.5 else 0

    return prediction, final_prob

# ==============================
# CLI DEMO
# ==============================
def main():
    print("\n🚀 Startup Success Prediction System\n")

    user_input = {
        "is_CA": 1,
        "is_NY": 0,
        "is_MA": 0,
        "is_TX": 0,
        "is_otherstate": 0,
        "is_software": 1,
        "is_web": 1,
        "is_mobile": 0,
        "is_enterprise": 0,
        "is_advertising": 0,
        "is_gamesvideo": 0,
        "is_ecommerce": 0,
        "is_biotech": 0,
        "is_consulting": 0,
        "is_othercategory": 0,
        "latitude": 37.77,
        "longitude": -122.41
    }

    prediction, probability = predict_startup_success(user_input)

    print("📊 Prediction Result")
    print("-------------------")
    print("Success Probability:", round(probability * 100, 2), "%")

    if prediction == 1:
        print("✅ Prediction: STARTUP IS LIKELY TO SUCCEED")
    else:
        print("❌ Prediction: STARTUP IS LIKELY TO FAIL")

if __name__ == "__main__":
    main()

