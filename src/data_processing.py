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