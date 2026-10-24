import pytest
from fraud_detection.inference import InferenceEngine


@pytest.fixture
def engine():
    return InferenceEngine.get_instance()


def test_inference_scoring_structure(engine):
    tx = {
        "transaction_id": "TXN_TEST_INF_01",
        "user_id": "USR_0001",
        "amount": 55.0,
        "timestamp": "2026-10-24T14:30:00+00:00",
        "merchant": "Whole Foods Market",
        "merchant_category": "Grocery",
        "location": "New York, USA",
        "device": "iOS App (iPhone 15)",
        "payment_method": "Credit Card",
        "channel": "In-Store",
    }

    res = engine.score_transaction(tx, persist=False)

    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["decision"] in ("APPROVE", "REVIEW", "DECLINE")
    assert res["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert isinstance(res["risk_reasons"], list)
    assert res["latency_ms"] >= 0.0
    assert res["transaction_id"] == "TXN_TEST_INF_01"


def test_inference_high_risk_decline(engine):
    # Suspicious payload: massive amount, unknown foreign location, proxy device
    tx = {
        "transaction_id": "TXN_TEST_INF_FRAUD",
        "user_id": "USR_0001",
        "amount": 9500.0,
        "timestamp": "2026-10-24T03:15:00+00:00",
        "merchant": "Crypto Exchange Portal",
        "merchant_category": "Digital & Gaming",
        "location": "Lagos, NG",
        "device": "Tor Browser / Linux",
        "payment_method": "Credit Card",
        "channel": "Online",
    }

    res = engine.score_transaction(tx, persist=False)

    # Should flag as high risk / decline
    assert res["decision"] in ("DECLINE", "REVIEW")
    assert len(res["risk_reasons"]) > 0
