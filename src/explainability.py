# src/explainability.py

import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
MODEL_DIR = "models"
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_tuned.pkl")
DATA_PATH = "data/processed/startups_featured.csv"
TARGET_COLUMN = "success"
OUTPUT_DIR = "outputs"

# ==============================
# LOAD DATA & MODEL
# ==============================
def load_data_and_model():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    model = joblib.load(XGB_PATH)
    return X, model

# ==============================
# GLOBAL EXPLANATION
# ==============================
def global_explanation(X, model):
    print("📊 Generating GLOBAL feature importance...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    global_path = os.path.join(OUTPUT_DIR, "shap_global_feature_importance.png")
    plt.savefig(global_path, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"💾 Saved global SHAP plot to: {global_path}")

# ==============================
# LOCAL EXPLANATION
# ==============================
def local_explanation(X, model, index=0):
    print(f"🔍 Explaining prediction for startup index: {index}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X.iloc[index],
        matplotlib=True,
        show=False
    )

    local_path = os.path.join(OUTPUT_DIR, f"shap_local_explanation_{index}.png")
    plt.savefig(local_path, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"💾 Saved local SHAP plot to: {local_path}")

# ==============================
# MAIN
# ==============================
def main():
    X, model = load_data_and_model()

    global_explanation(X, model)
    local_explanation(X, model, index=0)

    print("\n✅ Explainable AI (XAI) completed successfully")

if __name__ == "__main__":
    main()

