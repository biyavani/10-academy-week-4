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
