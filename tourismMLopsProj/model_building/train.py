import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

def run_training():
    # 1. Path definitions
    train_path = "tourismMLopsProj/data/train.csv"
    test_path = "tourismMLopsProj/data/test.csv"
    model_save_dir = "tourismMLopsProj/model_building"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train or Test CSV files not found. Ensure data_prep has run successfully.")

    # 2. Load splits
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Unnamed: 0 indices sometimes carry over
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])

    X_train = train_df.drop(columns=['ProdTaken'])
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(columns=['ProdTaken'])
    y_test = test_df['ProdTaken']

    # 3. Preprocessing Setup
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 4. Define pipeline with XGBoost model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    # 5. Define Grid Parameters for Tuning
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2]
    }

    # 6. Set up MLflow Tracking experiment run
    # (Using active tracking URI configured in the notebook)
    mlflow.set_experiment("MLOps_CICD_experiment")

    with mlflow.start_run(run_name="XGBoost_Tuning_Run"):
        print("Initiating GridSearchCV hyperparameter optimization...")
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Best params
        best_params = grid_search.best_params_
        print(f"\nBest Hyperparameters Found: {best_params}")

        # Log hyperparameters to MLflow
        for param_name, param_val in best_params.items():
            mlflow.log_param(param_name, param_val)

        # Evaluate on Test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }

        print("\n--- Model Test Metrics ---")
        for metric_name, val in metrics.items():
            print(f"{metric_name.capitalize()}: {val:.4f}")
            mlflow.log_metric(metric_name, val)

        # Log Model structure to MLflow
        mlflow.sklearn.log_model(best_model, "model")
        print("Best model registered & logged in MLflow.")

        # Save model locally for Repo tracking/Deployment
        model_output_path = os.path.join(model_save_dir, "best_model.pkl")
        joblib.dump(best_model, model_output_path)
        print(f"Model saved locally at: {model_output_path}")

if __name__ == "__main__":
    run_training()
