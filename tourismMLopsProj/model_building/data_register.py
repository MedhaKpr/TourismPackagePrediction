import pandas as pd
import os

def register_dataset(file_path):
    expected_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'Occupation',
        'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus',
        'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation',
        'MonthlyIncome', 'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups',
        'DurationOfPitch'
    ]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}.")

    # Validate columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following expected columns are missing: {missing_cols}")
    else:
        print("All expected columns are present. Validation passed!")

    # Dataset Summary
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\nTarget Variable ('ProdTaken') distribution:")
    print(df['ProdTaken'].value_counts(dropna=False))
    return df

# Run registration and validation
dataset_path = "tourismMLopsProj/data/tourism.csv"
try:
    # Using a sample or default fallback if tourism.csv was not uploaded yet
    if not os.path.exists(dataset_path):
        print("'tourism.csv' is missing...")
       

    df = register_dataset(dataset_path)
except Exception as e:
    print("Error during dataset registration:", e)
