# src/feature_engineering.py

import pandas as pd
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
INPUT_DATA_PATH = "data/processed/startups_processed.csv"
OUTPUT_DATA_PATH = "data/processed/startups_featured.csv"
TARGET_COLUMN = "success"

# ==============================
# LOAD DATA
# ==============================
def load_data(path):
    print("📥 Loading processed dataset...")
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape}")
    return df

# ==============================
# FEATURE ENGINEERING
# ==============================
def engineer_features(df):
    print("🧠 Performing leakage-safe feature engineering...")

    # Columns that cause data leakage (post-outcome information)
    leakage_columns = [
        "Unnamed: 0", "id", "object_id", "name", "labels",
        "founded_at", "first_funding_at", "last_funding_at",
        "age_first_funding_year", "age_last_funding_year",
        "age_first_milestone_year", "age_last_milestone_year",
        "relationships", "milestones",
        "funding_rounds", "funding_total_usd", "funding_per_round",
        "avg_participants",
        "has_roundA", "has_roundB", "has_roundC", "has_roundD",
        "has_VC", "has_angel", "is_top500"
    ]

    # Drop leakage columns if present
    df = df.drop(columns=[c for c in leakage_columns if c in df.columns])

    print(f"❌ Dropped {len(leakage_columns)} leakage-prone columns")

    return df

# ==============================
# FEATURE SELECTION
# ==============================
def remove_low_variance(df):
    print("🧹 Removing low-variance features...")
    variance = df.var()
    low_variance_cols = variance[variance < 0.01].index.tolist()

    if TARGET_COLUMN in low_variance_cols:
        low_variance_cols.remove(TARGET_COLUMN)

    df.drop(columns=low_variance_cols, inplace=True)
    print(f"❌ Removed {len(low_variance_cols)} low-variance features")

    return df

# ==============================
# SAVE DATA
# ==============================
def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Feature-engineered data saved at: {path}")

# ==============================
# MAIN PIPELINE
# ==============================
def main():
    df = load_data(INPUT_DATA_PATH)
    df = engineer_features(df)
    df = remove_low_variance(df)
    save_data(df, OUTPUT_DATA_PATH)

    print("\n✅ FEATURE ENGINEERING COMPLETED")
    print("🚀 Dataset ready for model training")

if __name__ == "__main__":
    main()

