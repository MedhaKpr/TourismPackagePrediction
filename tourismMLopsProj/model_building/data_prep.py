import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_and_split_data(data_path, output_dir):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # =========================================================================
    # STEP 1: Load the full raw dataset (tourism.csv)
    # =========================================================================
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")

    # =========================================================================
    # STEP 2: Clean the entire dataset BEFORE splitting
    # =========================================================================

    # A. Drop duplicate rows
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates_count} duplicate rows.")

    # B. Drop unique identifier column
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        print("Dropped 'CustomerID' column.")

    # C. Standardize inconsistent categorical values in the entire dataset
    if 'Gender' in df.columns:
        # Clean whitespaces and normalize variations of 'Fe male' / 'Fe Male' / 'Female'
        df['Gender'] = df['Gender'].astype(str).str.strip()
        df['Gender'] = df['Gender'].replace({
            'Fe male': 'Female',
            'Fe Male': 'Female',
            'female': 'Female',
            'male': 'Male'
        })
        print("Standardized 'Gender' categories (Fe male/Fe Male -> Female).")

    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].astype(str).str.strip()
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})
        print("Standardized 'MaritalStatus' categories (Unmarried -> Single).")

    # D. Handle Missing Values on the entire dataset
    # Numerical columns: impute with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ProdTaken' in num_cols:
        num_cols.remove('ProdTaken')

    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical columns: impute with mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    print("Completed handling all missing values across the full dataset.")
    print(f"Dataset shape after cleaning: {df.shape}")

    # =========================================================================
    # STEP 3: Split the already cleaned dataset into Train and Test
    # =========================================================================
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)
    
    # Reconstruct training and testing sets
    train_df = pd.concat([Xtrain, ytrain], axis=1)
    test_df = pd.concat([Xtest, ytest], axis=1)

    # Save the cleaned splits to CSV files locally
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nCleaned splits generated and saved successfully!")
    print(f"Saved Training Set: {train_path} {train_df.shape}")
    print(f"Saved Testing Set: {test_path} {test_df.shape}")

if __name__ == "__main__":
    clean_and_split_data("tourismMLopsProj/data/tourism.csv", "tourismMLopsProj/data")
