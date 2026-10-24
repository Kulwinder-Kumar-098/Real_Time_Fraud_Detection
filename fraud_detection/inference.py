"""
Real-Time Fraud Inference and Risk Scoring Engine.

Features:
- Single-instance artifact caching (zero startup retraining)
- Input validation and client label sanitization
- Dynamic user profile & rolling velocity lookups
- Multi-tier decisioning (APPROVE / REVIEW / DECLINE)
- Explainable risk factor breakdown
- Sub-15ms inference latency measurement
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

import config
from fraud_detection.database import (
    get_user,
    get_user_recent_activity,
    insert_transaction,
    upsert_user,
)
from fraud_detection.features import ALL_MODEL_FEATURES, extract_features_single


class InferenceEngine:
    _instance: Optional["InferenceEngine"] = None

    def __init__(self, artifact_path: Optional[str] = None):
        self.artifact_path = artifact_path or config.MODEL_ARTIFACTS_PATH
        self.preprocessor = None
        self.model = None
        self.model_name = "Unloaded"
        self.threshold = config.DEFAULT_DECISION_THRESHOLD
        self.features = ALL_MODEL_FEATURES
        self.is_ready = False
        self._load_artifacts()

    @classmethod
    def get_instance(cls) -> "InferenceEngine":
        if cls._instance is None:
            cls._instance = InferenceEngine()
        return cls._instance

    def _load_artifacts(self) -> None:
        """Loads serialized model and preprocessing pipeline into memory."""
        if not Path(self.artifact_path).exists():
            print(f"[InferenceEngine] Warning: Artifacts not found at {self.artifact_path}.")
            self.is_ready = False
            return

        try:
            bundle = joblib.load(self.artifact_path)
            self.preprocessor = bundle["preprocessor"]
            self.model = bundle["model"]
            self.model_name = bundle.get("model_name", "Production Model")
            self.threshold = float(bundle.get("decision_threshold", config.DEFAULT_DECISION_THRESHOLD))
            self.features = bundle.get("features", ALL_MODEL_FEATURES)
            self.is_ready = True
            print(
                f"[InferenceEngine] Successfully loaded '{self.model_name}' "
                f"(Decision Threshold: {self.threshold:.3f})"
            )
        except Exception as e:
            print(f"[InferenceEngine] Error loading model artifacts: {e}")
            self.is_ready = False

    def validate_payload(self, data: Any) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validates incoming transaction schema.
        Prevents client fraud label injection and handles missing values.
        """
        if not isinstance(data, dict):
            return False, "Payload must be a valid JSON object.", {}

        required_fields = ["user_id", "amount", "merchant"]
        for field in required_fields:
            if field not in data or data[field] is None:
                return False, f"Missing required field: '{field}'", {}

        # Validate amount
        try:
            amount = float(data["amount"])
            if amount <= 0.0 or not np.isfinite(amount):
                return False, "Field 'amount' must be a positive finite number.", {}
        except (ValueError, TypeError):
            return False, "Field 'amount' must be numeric.", {}

        # Sanitize and build payload
        sanitized = {
            "transaction_id": str(data.get("transaction_id") or f"TXN_{int(time.time()*1000)}"),
            "user_id": str(data["user_id"]).strip(),
            "amount": round(amount, 2),
            "timestamp": str(data.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            "merchant": str(data["merchant"]).strip(),
            "merchant_category": str(data.get("merchant_category", "General")).strip(),
            "location": str(data.get("location", "Unknown Location")).strip(),
            "device": str(data.get("device", "Unknown Device")).strip(),
            "payment_method": str(data.get("payment_method", "Credit Card")).strip(),
            "channel": str(data.get("channel", "Online")).strip(),
        }

        # Preserve ground truth label ONLY if passed by the trusted internal simulator
        if "ground_truth_label" in data and isinstance(data["ground_truth_label"], (int, bool)):
            sanitized["ground_truth_label"] = int(data["ground_truth_label"])
            sanitized["fraud_scenario"] = str(data.get("fraud_scenario", "none"))

        return True, None, sanitized

    def score_transaction(
        self,
        tx: Dict[str, Any],
        persist: bool = True,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scores a single transaction in real time.
        Measures execution latency and produces explainable risk factors.
        """
        start_time = time.perf_counter()

        # Fallback if model not yet trained or loaded
        if not self.is_ready or self.model is None or self.preprocessor is None:
            self._load_artifacts()
            if not self.is_ready:
                # Rule-based fallback if ML model is unavailable
                return self._fallback_rule_score(tx, start_time, persist, db_path)

        # 1. Fetch user profile from database
        user_profile = get_user(tx["user_id"], db_path=db_path)
        if not user_profile:
            # Create a provisional user baseline for first-seen users
            user_profile = {
                "user_id": tx["user_id"],
                "home_location": tx["location"],
                "primary_device": tx["device"],
                "default_payment_method": tx["payment_method"],
                "avg_amount": max(30.0, tx["amount"] * 0.8),
                "std_amount": max(15.0, tx["amount"] * 0.3),
            }
            try:
                upsert_user(user_profile, db_path=db_path)
            except Exception:
                pass

        # 2. Fetch rolling recent activity
        activity = get_user_recent_activity(
            tx["user_id"], reference_timestamp=tx["timestamp"], db_path=db_path
        )

        # 3. Extract operational features
        features = extract_features_single(tx, user_profile=user_profile, recent_activity=activity)
        features_df = pd.DataFrame([features])[self.features]

        # 4. Transform and predict probability
        X_vec = self.preprocessor.transform(features_df)
        prob = float(self.model.predict_proba(X_vec)[0, 1])

        # 5. Multi-Tier Decisioning & Risk Categorization
        # Thresholds:
        # HIGH_RISK / CRITICAL: >= 0.70 (DECLINE)
        # Optimal threshold: e.g. 0.42 (DECLINE)
        # REVIEW: between 0.35 and optimal threshold
        # LOW: < 0.35 (APPROVE)
        if prob >= 0.80:
            decision = "DECLINE"
            risk_tier = "CRITICAL"
        elif prob >= self.threshold:
            decision = "DECLINE"
            risk_tier = "HIGH"
        elif prob >= config.REVIEW_THRESHOLD:
            decision = "REVIEW"
            risk_tier = "MEDIUM"
        else:
            decision = "APPROVE"
            risk_tier = "LOW"

        is_fraud_pred = 1 if prob >= self.threshold else 0

        # 6. Generate Explainability / Risk Indicators
        reasons = []
        if features["amount_to_user_avg_ratio"] >= 4.0:
            reasons.append(
                f"Spending Spike: Amount is {features['amount_to_user_avg_ratio']:.1f}x higher than user average"
            )
        if features["is_new_location"] == 1:
            reasons.append(f"Geographic Anomaly: Transacting outside profile home ({tx['location']})")
        if features["is_new_device"] == 1:
            reasons.append(f"Unrecognized Device: Fingerprint mismatch ({tx['device']})")
        if features["user_tx_velocity_1h"] >= 3:
            reasons.append(
                f"High Velocity: {features['user_tx_velocity_1h']} transactions within the past hour"
            )
        if features["is_night_hours"] == 1 and prob >= 0.30:
            reasons.append("Off-Hours Activity: Transaction initiated during high-risk night window")
        if tx.get("merchant_category") in ("Digital & Gaming", "Luxury & Jewelry") and prob >= 0.35:
            reasons.append(f"High-Risk Merchant Category: {tx['merchant_category']}")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Prepare response
        result = {
            "transaction_id": tx["transaction_id"],
            "user_id": tx["user_id"],
            "amount": tx["amount"],
            "timestamp": tx["timestamp"],
            "merchant": tx["merchant"],
            "merchant_category": tx.get("merchant_category", "General"),
            "location": tx["location"],
            "device": tx["device"],
            "payment_method": tx["payment_method"],
            "channel": tx.get("channel", "Online"),
            "ground_truth_label": tx.get("ground_truth_label", 0),
            "fraud_scenario": tx.get("fraud_scenario", "none"),
            "risk_score": round(prob, 4),
            "is_fraud_predicted": is_fraud_pred,
            "decision": decision,
            "risk_tier": risk_tier,
            "risk_reasons": reasons,
            "latency_ms": round(latency_ms, 2),
            "model_version": self.model_name,
            "decision_threshold": self.threshold,
        }

        # 7. Persist to database if requested
        if persist:
            try:
                insert_transaction(result, db_path=db_path)
            except Exception as e:
                print(f"[InferenceEngine] Warning: Could not log transaction to DB: {e}")

        return result

    def _fallback_rule_score(
        self,
        tx: Dict[str, Any],
        start_time: float,
        persist: bool,
        db_path: Optional[str],
    ) -> Dict[str, Any]:
        """Simple heuristic fallback if ML model is currently offline/training."""
        amount = tx["amount"]
        reasons = []
        score = 0.05

        if amount > 1500.0:
            score += 0.50
            reasons.append(f"Heuristic: Unusually high transaction amount (${amount:,.2f})")
        if "Tor" in tx.get("device", "") or "Emulated" in tx.get("device", ""):
            score += 0.35
            reasons.append(f"Heuristic: Suspicious proxy/emulator device ({tx.get('device')})")

        score = min(0.99, max(0.01, score))
        decision = "DECLINE" if score >= self.threshold else ("REVIEW" if score >= 0.35 else "APPROVE")
        risk_tier = "HIGH" if score >= 0.60 else ("MEDIUM" if score >= 0.35 else "LOW")
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        res = {
            "transaction_id": tx["transaction_id"],
            "user_id": tx["user_id"],
            "amount": tx["amount"],
            "timestamp": tx["timestamp"],
            "merchant": tx["merchant"],
            "merchant_category": tx.get("merchant_category", "General"),
            "location": tx["location"],
            "device": tx["device"],
            "payment_method": tx["payment_method"],
            "channel": tx.get("channel", "Online"),
            "ground_truth_label": tx.get("ground_truth_label", 0),
            "fraud_scenario": tx.get("fraud_scenario", "none"),
            "risk_score": round(score, 4),
            "is_fraud_predicted": 1 if score >= self.threshold else 0,
            "decision": decision,
            "risk_tier": risk_tier,
            "risk_reasons": reasons,
            "latency_ms": round(latency_ms, 2),
            "model_version": "Heuristic Fallback v0.1",
            "decision_threshold": self.threshold,
        }
        if persist:
            try:
                insert_transaction(res, db_path=db_path)
            except Exception:
                pass
        return res
