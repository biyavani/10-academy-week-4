from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xverse.transformer import WOE
from sklearn.cluster import KMeans   


class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Aggregate transaction Amount per CustomerId.

    Creates:
      - total_amount: sum of Amount per customer
      - avg_amount: mean Amount per customer
      - txn_count: number of transactions per customer
      - std_amount: standard deviation of Amount per customer
    """
    def __init__(
        self,
        customer_id_col: str = "CustomerId",
        amount_col: str = "Amount",
    ):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X: pd.DataFrame, y=None):
        if self.customer_id_col not in X.columns:
            raise ValueError(f"Column {self.customer_id_col!r} not in input data")
        if self.amount_col not in X.columns:
            raise ValueError(f"Column {self.amount_col!r} not in input data")

        # Group by customer and compute aggregate features
        agg = (
            X.groupby(self.customer_id_col)[self.amount_col]
            .agg(
                total_amount="sum",
                avg_amount="mean",
                txn_count="count",
                std_amount="std",
            )
            .reset_index()
        )
        # Std can be NaN for customers with a single transaction
        agg["std_amount"] = agg["std_amount"].fillna(0.0)
        self._agg_features_ = agg
        return self

    def transform(self, X: pd.DataFrame):
        if not hasattr(self, "_agg_features_"):
            raise RuntimeError("The transformer has not been fitted yet.")
        X = X.copy()
        agg = self._agg_features_

        # Left join so each transaction row gets customer level features
        X = X.merge(agg, on=self.customer_id_col, how="left", validate="m:1")
        return X
    


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day, month, year from TransactionStartTime.
    """
    def __init__(self, datetime_col: str = "TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X: pd.DataFrame, y=None):
        if self.datetime_col not in X.columns:
            raise ValueError(f"Column {self.datetime_col!r} not in input data")
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce", utc=True)
        X["txn_hour"] = dt.dt.hour
        X["txn_day"] = dt.dt.day
        X["txn_month"] = dt.dt.month
        X["txn_year"] = dt.dt.year
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops a list of columns if they exist.
    """
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        drop_cols = [c for c in self.columns if c in X.columns]
        return X.drop(columns=drop_cols)
    
class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Thin wrapper around xverse.transformer.WOE so it can be used like a normal
    scikit learn transformer.

    This is optional and used mainly for inspecting Weight of Evidence
    and Information Value.
    """
    def __init__(
        self,
        feature_names: Union[str, List[str]] = "all",
        exclude_features: Optional[List[str]] = None,
    ):
        self.feature_names = feature_names
        self.exclude_features = exclude_features

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        if WOE is None:
            raise ImportError(
                "xverse is not installed. Install it with 'pip install xverse' "
                "before using WOETransformer."
            )
        self._woe = WOE(
            feature_names=self.feature_names,
            exclude_features=self.exclude_features,
        )
        self._woe.fit(X, y)
        # Save WoE and IV tables for later inspection
        self.iv_df_ = self._woe.iv_df.copy()
        self.woe_df_ = self._woe.woe_df.copy()
        return self

    def transform(self, X: pd.DataFrame):
        return self._woe.transform(X)


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build the full preprocessing pipeline for Task 3.

    Steps:
      1. CustomerAggregateFeatures - aggregate Amount per CustomerId
      2. DateTimeFeatures - extract hour/day/month/year
      3. ColumnDropper - drop ID like columns
      4. ColumnTransformer - impute missing, scale numerics, one hot encode categoricals
    """

    # Numerical columns that should be scaled
    numeric_features = [
        "Amount",
        "Value",
        "PricingStrategy",
        "total_amount",
        "avg_amount",
        "txn_count",
        "std_amount",
        "txn_hour",
        "txn_day",
        "txn_month",
        "txn_year",
    ]

    # Categorical columns to encode
    categorical_features = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
    ]

    # Numerical pipeline: impute then standardize
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute then one hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine numeric and categorical transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Columns that are IDs or constants, not useful as features
    id_columns = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "CurrencyCode",
        "CountryCode",
        "TransactionStartTime",
    ]

    pipeline = Pipeline(
        steps=[
            ("aggregate", CustomerAggregateFeatures()),
            ("datetime", DateTimeFeatures()),
            ("drop", ColumnDropper(columns=id_columns)),
            ("preprocess", preprocessor),
        ]
    )
    return pipeline


# Default data paths, relative to project root
PROJECT_ROOT = Path(".").resolve()
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "processed_features.csv"


def load_raw_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the raw Xente dataset from CSV.
    """
    path = Path(path) if path is not None else DEFAULT_RAW_PATH
    return pd.read_csv(path)


def run_data_processing(
    raw_data_path: Optional[Union[str, Path]] = None,
    processed_data_path: Optional[Union[str, Path]] = None,
    target_col: str = "FraudResult",
):
    df = load_raw_data(raw_data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not in data")

    # Separate features and target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Keep CustomerId for later merge with RFM labels
    if "CustomerId" not in X.columns:
        raise ValueError("CustomerId column is missing from the data")
    customer_ids = X["CustomerId"].copy()

    pipeline = build_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X, y)

    # If the output is a sparse matrix, convert it to a dense numpy array
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Recover feature names from the ColumnTransformer
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # Add CustomerId and original FraudResult back as columns
    processed_df["CustomerId"] = customer_ids.values
    processed_df[target_col] = y.values

    out_path = (
        Path(processed_data_path)
        if processed_data_path is not None
        else DEFAULT_PROCESSED_PATH
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(out_path, index=False)

    return pipeline, processed_df

def compute_rfm_features(
    df: pd.DataFrame,
    customer_id_col: str = "CustomerId",
    datetime_col: str = "TransactionStartTime",
    monetary_col: str = "Value",
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Compute RFM (Recency, Frequency, Monetary) per customer.

    Recency: days since last transaction (higher means less recent).
    Frequency: number of transactions.
    Monetary: total transaction value.
    """
    if customer_id_col not in df.columns:
        raise ValueError(f"{customer_id_col!r} not in dataframe")
    if datetime_col not in df.columns:
        raise ValueError(f"{datetime_col!r} not in dataframe")
    if monetary_col not in df.columns:
        raise ValueError(f"{monetary_col!r} not in dataframe")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")

    # Snapshot date: one day after the last transaction in the data
    snapshot_date = df[datetime_col].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby(customer_id_col)
        .agg(
            recency=(datetime_col, lambda x: (snapshot_date - x.max()).days),
            frequency=(customer_id_col, "size"),
            monetary=(monetary_col, "sum"),
        )
        .reset_index()
    )

    return rfm, snapshot_date



def compute_woe_iv(
    df: pd.DataFrame,
    target_col: str = "FraudResult",
    feature_names: Union[str, List[str]] = "all",
    exclude_features: Optional[List[str]] = None,
) -> Tuple[WOETransformer, pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper that runs a WOE transformation on the given dataframe
    and returns:
      - the fitted WOETransformer
      - the WoE table
      - the IV table
    """
    if WOE is None:
        raise ImportError(
            "xverse is not installed. Install it with 'pip install xverse' before calling compute_woe_iv."
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not in dataframe")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    woe_tr = WOETransformer(
        feature_names=feature_names,
        exclude_features=exclude_features,
    )
    woe_tr.fit(X, y)
    # Transform is not strictly needed if you only want IV, but we keep it for completeness
    _ = woe_tr.transform(X)

    return woe_tr, woe_tr.woe_df_, woe_tr.iv_df_


if __name__ == "__main__":
    print("Running data processing...")
    try:
        pipe, processed = run_data_processing()
        print("Processed data shape:", processed.shape)
        print(f"Saved processed features to: {DEFAULT_PROCESSED_PATH}")
    except FileNotFoundError:
        print("Raw data not found. Place data.csv in data/raw before running this script.")
