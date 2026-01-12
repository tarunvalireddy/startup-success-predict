# src/explainability.py

import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONFIG
# ==============================
MODEL_DIR = "models"
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_tuned.pkl")
DATA_PATH = "data/processed/startups_featured.csv"
TARGET_COLUMN = "success"
OUTPUT_DIR = "outputs"
TOP_N_GLOBAL = 10
TOP_N_LOCAL = 5

# ==============================
# LOAD DATA & MODEL
# ==============================
def load_data_and_model():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    model = joblib.load(XGB_PATH)
    return X, model

# ==============================
# GLOBAL EXPLANATION (CLEAN)
# ==============================
def global_explanation(X, model):
    print("📊 Generating CLEAN global feature importance...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": mean_shap
    }).sort_values(by="importance", ascending=False).head(TOP_N_GLOBAL)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.barh(
        feature_importance["feature"][::-1],
        feature_importance["importance"][::-1]
    )
    plt.title("Top Features Influencing Startup Success (Global)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "shap_global_top_features.png")
    plt.savefig(path, dpi=300)
    plt.show()

    print(f"💾 Saved global explanation to: {path}")

# ==============================
# LOCAL EXPLANATION (CLEAN)
# ==============================
def local_explanation(X, model, index=0):
    print(f"🔍 Generating CLEAN local explanation for startup {index}...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    local_shap = pd.DataFrame({
        "feature": X.columns,
        "shap_value": shap_values[index]
    })

    local_shap["abs"] = local_shap["shap_value"].abs()
    top_local = local_shap.sort_values("abs", ascending=False).head(TOP_N_LOCAL)

    plt.figure(figsize=(8, 4))
    colors = ["green" if v > 0 else "red" for v in top_local["shap_value"]]

    plt.barh(
        top_local["feature"][::-1],
        top_local["shap_value"][::-1],
        color=colors[::-1]
    )

    plt.title("Top Local Factors for This Startup")
    plt.xlabel("SHAP Contribution")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"shap_local_top_features_{index}.png")
    plt.savefig(path, dpi=300)
    plt.show()

    print(f"💾 Saved local explanation to: {path}")

# ==============================
# MAIN
# ==============================
def main():
    X, model = load_data_and_model()

    global_explanation(X, model)
    local_explanation(X, model, index=0)

    print("\n✅ Clean, stakeholder-friendly XAI generated")

if __name__ == "__main__":
    main()

