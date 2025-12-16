# src/train.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


RANDOM_STATE = 42

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_data_customers.csv"


def load_model_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the customer level dataset with the proxy target is_high_risk.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model data not found at {path}. Run Task 4 pipeline first "
            "to generate model_data_customers.csv."
        )
    df = pd.read_csv(path)
    if "is_high_risk" not in df.columns:
        raise ValueError("is_high_risk column not found in model dataset.")
    return df


def train_test_split_data(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Drops CustomerId, keeps is_high_risk as target.
    """
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, F1 and ROC AUC on train and test sets.
    """
    # Train predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
        "train_roc_auc": roc_auc_score(y_train, y_train_proba),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_test_proba),
    }
    return metrics


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Logistic Regression with GridSearchCV.
    """
    base_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",  # works well for small grids and l1/l2
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    return best_model, best_params


def train_gradient_boosting(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[GradientBoostingClassifier, Dict[str, float]]:
    """
    Gradient Boosting with RandomizedSearchCV.
    """
    base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    return best_model, best_params


def run_experiment(
    model_name: str,
    train_fn,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    registered_model_name: str,
):
    """
    Wraps training and evaluation in an MLflow run.
    """
    with mlflow.start_run(run_name=model_name):
        # Train model with tuning
        model, best_params = train_fn(X_train, y_train)

        # Log hyperparameters
        mlflow.log_params(best_params)

        # Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        mlflow.log_metrics(metrics)

        # Log model as an artifact and register it
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        print(f"{model_name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return model, metrics


def main():
    # Set up MLflow experiment (stored locally in ./mlruns by default)
    mlflow.set_experiment("credit_risk_is_high_risk")

    # 1. Load data
    df = load_model_data()

    # 2. Train test split
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # 3. Run Logistic Regression experiment
    print("Training Logistic Regression...")
    run_experiment(
        model_name="logistic_regression",
        train_fn=train_logistic_regression,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        registered_model_name="credit_risk_log_reg",
    )

    # 4. Run Gradient Boosting experiment
    print("\nTraining Gradient Boosting...")
    run_experiment(
        model_name="gradient_boosting",
        train_fn=train_gradient_boosting,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        registered_model_name="credit_risk_gb",
    )


if __name__ == "__main__":
    main()
