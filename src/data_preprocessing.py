# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# ==============================
# CONFIG
# ==============================
RAW_DATA_PATH = "data/raw/startups.csv"
PROCESSED_DATA_PATH = "data/processed/startups_processed.csv"
TARGET_COLUMN = "success"

# ==============================
# LOAD DATA
# ==============================
def load_data(path):
    print("📥 Loading dataset...")
    df = pd.read_csv(path)
    print("✅ Dataset loaded successfully\n")
    return df

# ==============================
# BASIC EDA
# ==============================
def basic_eda(df):
    print("📊 BASIC DATA INSPECTION")
    print("-" * 40)
    print(df.head(), "\n")
    print(df.info(), "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")
    print("Class distribution:\n", df["status"].value_counts(), "\n")

# ==============================
# LABEL CREATION
# ==============================
def create_target(df):
    print("🎯 Creating target column...")
    df[TARGET_COLUMN] = df["status"].apply(
        lambda x: 1 if x in ["acquired", "operating"] else 0
    )
    df.drop(columns=["status"], inplace=True)
    return df

# ==============================
# DATA CLEANING
# ==============================
def clean_data(df):
    print("🧹 Cleaning data...")

    # Drop columns with too many missing values
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Fill numerical missing values with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# ==============================
# ENCODING
# ==============================
def encode_features(df):
    print("🔢 Encoding categorical features...")
    encoder = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    return df

# ==============================
# SCALING
# ==============================
def scale_features(df):
    print("⚖️ Scaling numerical features...")
    scaler = StandardScaler()

    feature_cols = df.drop(columns=[TARGET_COLUMN]).columns
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df

# ==============================
# SAVE PROCESSED DATA
# ==============================
def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Processed data saved at: {path}")

# ==============================
# MAIN PIPELINE
# ==============================
def main():
    df = load_data(RAW_DATA_PATH)
    basic_eda(df)
    df = create_target(df)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    save_data(df, PROCESSED_DATA_PATH)

    print("\n✅ DATA PREPROCESSING COMPLETED SUCCESSFULLY")
    print("🚀 Ready for model training")

if __name__ == "__main__":
    main()

