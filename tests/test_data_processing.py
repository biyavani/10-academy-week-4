# tests/test_data_processing.py

import pandas as pd

from src.data_processing import CustomerAggregateFeatures, DateTimeFeatures


def test_customer_aggregate_features_creates_expected_columns():
    data = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "Amount": [100.0, 200.0, 300.0],
        }
    )

    transformer = CustomerAggregateFeatures(
        customer_id_col="CustomerId",
        amount_col="Amount",
    )
    transformer.fit(data)
    transformed = transformer.transform(data)

    for col in ["total_amount", "avg_amount", "txn_count", "std_amount"]:
        assert col in transformed.columns

    # Check that aggregates for C1 are correct
    c1_row = transformed[transformed["CustomerId"] == "C1"].iloc[0]
    assert c1_row["total_amount"] == 300.0
    assert c1_row["txn_count"] == 2


def test_datetime_features_extracts_parts():
    data = pd.DataFrame(
        {
            "TransactionStartTime": ["2018-11-15T02:18:49Z"],
        }
    )

    transformer = DateTimeFeatures(datetime_col="TransactionStartTime")
    transformed = transformer.fit_transform(data)

    for col in ["txn_hour", "txn_day", "txn_month", "txn_year"]:
        assert col in transformed.columns

    assert transformed.loc[0, "txn_year"] == 2018
    assert transformed.loc[0, "txn_month"] == 11
    assert transformed.loc[0, "txn_day"] == 15
