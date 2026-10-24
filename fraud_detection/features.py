"""
Feature Engineering and Preprocessing Pipeline for Fraud Detection.

Guarantees ZERO data leakage:
1. Operational feature extraction uses only past/current transaction context and user baseline.
2. Ground-truth fraud labels and simulation metadata are strictly excluded.
3. Preprocessing transformers (scalers, encoders) fit strictly on training splits.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Explicit list of operational features fed into ML models
NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "amount_to_user_avg_ratio",
    "amount_z_score",
    "user_tx_velocity_1h",
    "user_tx_velocity_24h",
    "user_amount_sum_1h",
    "is_new_device",
    "is_new_location",
    "hour_of_day",
    "is_night_hours",
    "day_of_week",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "payment_method",
    "channel",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def extract_features_single(
    tx: Dict[str, Any],
    user_profile: Optional[Dict[str, Any]] = None,
    recent_activity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extracts engineered features for a single real-time transaction.
    Uses user profile and recent historical rolling activity.
    """
    amount = float(tx["amount"])
    log_amount = float(np.log1p(max(0.0, amount)))

    # User profile reference baseline
    avg_amount = float(user_profile.get("avg_amount", 50.0)) if user_profile else 50.0
    std_amount = float(user_profile.get("std_amount", 20.0)) if user_profile else 20.0
    home_loc = user_profile.get("home_location", "") if user_profile else ""
    primary_dev = user_profile.get("primary_device", "") if user_profile else ""

    ratio = amount / max(1.0, avg_amount)
    z_score = (amount - avg_amount) / max(1.0, std_amount)

    # Activity stats
    activity = recent_activity or {}
    v1h = int(activity.get("tx_count_1h", 0))
    v24h = int(activity.get("tx_count_24h", 0))
    s1h = float(activity.get("amount_sum_1h", 0.0))

    # Device & Location mismatch flags
    current_dev = str(tx.get("device", ""))
    current_loc = str(tx.get("location", ""))

    is_new_dev = 1 if (primary_dev and current_dev != primary_dev) else 0
    is_new_loc = 1 if (home_loc and current_loc != home_loc) else 0

    # Temporal features
    ts_str = str(tx["timestamp"])
    try:
        # Parse ISO timestamp
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)

    hour = dt.hour
    is_night = 1 if hour in (0, 1, 2, 3, 4, 5) else 0
    dow = dt.weekday()
    is_wknd = 1 if dow in (5, 6) else 0

    return {
        "amount": amount,
        "log_amount": log_amount,
        "amount_to_user_avg_ratio": float(ratio),
        "amount_z_score": float(z_score),
        "user_tx_velocity_1h": v1h,
        "user_tx_velocity_24h": v24h,
        "user_amount_sum_1h": s1h,
        "is_new_device": is_new_dev,
        "is_new_location": is_new_loc,
        "hour_of_day": hour,
        "is_night_hours": is_night,
        "day_of_week": dow,
        "is_weekend": is_wknd,
        "merchant_category": str(tx.get("merchant_category", "General")),
        "payment_method": str(tx.get("payment_method", "Credit Card")),
        "channel": str(tx.get("channel", "Online")),
    }


def compute_batch_features(
    df: pd.DataFrame,
    user_profiles: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Computes features across an offline batch dataset chronologically.
    Employs temporal rolling windows per user to avoid any future data leakage.
    """
    df = df.copy()
    user_map = {u["user_id"]: u for u in user_profiles}

    # Ensure chronological order
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["dt"] = pd.to_datetime(df["timestamp"])
    else:
        df["dt"] = df["timestamp"]

    df = df.sort_values("dt").reset_index(drop=True)

    # User baseline ratios
    df["user_avg"] = df["user_id"].map(lambda u: user_map.get(u, {}).get("avg_amount", 50.0))
    df["user_std"] = df["user_id"].map(lambda u: user_map.get(u, {}).get("std_amount", 20.0))
    df["user_home"] = df["user_id"].map(lambda u: user_map.get(u, {}).get("home_location", ""))
    df["user_dev"] = df["user_id"].map(lambda u: user_map.get(u, {}).get("primary_device", ""))

    df["amount"] = df["amount"].astype(float)
    df["log_amount"] = np.log1p(np.maximum(0.0, df["amount"]))
    df["amount_to_user_avg_ratio"] = df["amount"] / np.maximum(1.0, df["user_avg"])
    df["amount_z_score"] = (df["amount"] - df["user_avg"]) / np.maximum(1.0, df["user_std"])

    df["is_new_device"] = (df["device"] != df["user_dev"]).astype(int)
    df["is_new_location"] = (df["location"] != df["user_home"]).astype(int)

    # Temporal features
    df["hour_of_day"] = df["dt"].dt.hour
    df["is_night_hours"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Chronological rolling velocity per user (strictly past window, zero future leakage)
    df = df.sort_values(["user_id", "dt"]).reset_index(names="orig_sort_idx")
    times = (df["dt"].astype("int64") // 10**9).values
    amounts = df["amount"].values.astype(float)
    user_ids = df["user_id"].values
    n = len(df)

    v1h = np.zeros(n, dtype=int)
    v24h = np.zeros(n, dtype=int)
    s1h = np.zeros(n, dtype=float)

    user_change_idx = np.concatenate(
        ([0], np.where(user_ids[:-1] != user_ids[1:])[0] + 1, [n])
    )

    for i in range(len(user_change_idx) - 1):
        start = user_change_idx[i]
        end = user_change_idx[i + 1]
        u_times = times[start:end]
        u_amts = amounts[start:end]
        m = end - start

        # Strictly past window (closed='left')
        idx_1h = np.searchsorted(u_times, u_times - 3600, side="left")
        idx_24h = np.searchsorted(u_times, u_times - 86400, side="left")
        curr = np.arange(m)

        v1h[start:end] = np.maximum(0, curr - idx_1h)
        v24h[start:end] = np.maximum(0, curr - idx_24h)

        cum_amt = np.concatenate(([0.0], np.cumsum(u_amts)))
        s1h[start:end] = cum_amt[curr] - cum_amt[idx_1h]

    df["user_tx_velocity_1h"] = v1h
    df["user_tx_velocity_24h"] = v24h
    df["user_amount_sum_1h"] = s1h

    # Restore chronological order
    df = df.sort_values("orig_sort_idx").reset_index(drop=True)

    # Clean intermediate columns
    drop_cols = ["user_avg", "user_std", "user_home", "user_dev", "orig_sort_idx", "dt"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Constructs the Scikit-learn ColumnTransformer for model training and inference.
    Handles numerical standardization and categorical one-hot encoding with ignore unknown.
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
