import numpy as np
import pandas as pd
import pytest
from fraud_detection.features import (
    ALL_MODEL_FEATURES,
    build_preprocessing_pipeline,
    compute_batch_features,
    extract_features_single,
)


def test_extract_features_single_keys():
    tx = {
        "transaction_id": "TXN_TEST_01",
        "user_id": "USR_0001",
        "amount": 100.0,
        "timestamp": "2026-10-24T12:00:00+00:00",
        "merchant": "Apple Store",
        "merchant_category": "Electronics",
        "location": "New York, USA",
        "device": "iOS App (iPhone 15)",
        "payment_method": "Credit Card",
        "channel": "Online",
    }
    profile = {
        "user_id": "USR_0001",
        "home_location": "New York, USA",
        "primary_device": "iOS App (iPhone 15)",
        "avg_amount": 50.0,
        "std_amount": 15.0,
    }
    activity = {
        "tx_count_1h": 1,
        "tx_count_24h": 3,
        "amount_sum_1h": 40.0,
    }

    feats = extract_features_single(tx, user_profile=profile, recent_activity=activity)

    # Check all model features exist
    for f in ALL_MODEL_FEATURES:
        assert f in feats, f"Missing feature: {f}"

    # Check logical relationships
    assert feats["amount"] == 100.0
    assert feats["amount_to_user_avg_ratio"] == 2.0  # 100 / 50
    assert feats["is_new_device"] == 0  # Matches primary_device
    assert feats["is_new_location"] == 0  # Matches home_location
    assert feats["user_tx_velocity_1h"] == 1
    assert feats["hour_of_day"] == 12
    assert feats["is_night_hours"] == 0


def test_extract_features_mismatch_flags():
    tx = {
        "amount": 600.0,
        "timestamp": "2026-10-24T03:00:00+00:00",
        "merchant": "Cartier",
        "merchant_category": "Luxury & Jewelry",
        "location": "Lagos, NG",
        "device": "Tor Browser / Linux",
        "payment_method": "Credit Card",
    }
    profile = {
        "home_location": "London, UK",
        "primary_device": "Chrome / macOS Sonoma",
        "avg_amount": 50.0,
        "std_amount": 10.0,
    }
    feats = extract_features_single(tx, user_profile=profile)

    assert feats["is_new_location"] == 1  # Lagos != London
    assert feats["is_new_device"] == 1  # Tor != Chrome
    assert feats["is_night_hours"] == 1  # 3 AM is night
    assert feats["amount_to_user_avg_ratio"] == 12.0  # 600 / 50


def test_preprocessing_pipeline_fit_transform():
    pipe = build_preprocessing_pipeline()
    sample_data = pd.DataFrame([
        {
            "amount": 45.0,
            "log_amount": np.log1p(45.0),
            "amount_to_user_avg_ratio": 1.0,
            "amount_z_score": 0.0,
            "user_tx_velocity_1h": 0,
            "user_tx_velocity_24h": 1,
            "user_amount_sum_1h": 0.0,
            "is_new_device": 0,
            "is_new_location": 0,
            "hour_of_day": 14,
            "is_night_hours": 0,
            "day_of_week": 2,
            "is_weekend": 0,
            "merchant_category": "Grocery",
            "payment_method": "Debit Card",
            "channel": "In-Store",
        },
        {
            "amount": 900.0,
            "log_amount": np.log1p(900.0),
            "amount_to_user_avg_ratio": 9.0,
            "amount_z_score": 6.5,
            "user_tx_velocity_1h": 4,
            "user_tx_velocity_24h": 8,
            "user_amount_sum_1h": 850.0,
            "is_new_device": 1,
            "is_new_location": 1,
            "hour_of_day": 2,
            "is_night_hours": 1,
            "day_of_week": 5,
            "is_weekend": 1,
            "merchant_category": "Luxury & Jewelry",
            "payment_method": "Credit Card",
            "channel": "Online",
        },
    ])

    X_transformed = pipe.fit_transform(sample_data)
    assert X_transformed.shape[0] == 2
    assert X_transformed.shape[1] >= len(ALL_MODEL_FEATURES)
