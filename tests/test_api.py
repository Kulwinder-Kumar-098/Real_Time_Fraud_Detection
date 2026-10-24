import json
import pytest
from fraud_detection.app import create_app


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_healthcheck_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data
    assert "uptime_seconds" in data
    assert "model_status" in data


def test_api_predict_valid(client):
    payload = {
        "user_id": "USR_0005",
        "amount": 75.25,
        "merchant": "FreshMart Supermarket",
        "merchant_category": "Grocery",
        "location": "New York, USA",
        "device": "iOS App (iPhone 15)",
        "payment_method": "Credit Card",
    }
    res = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "risk_score" in data
    assert "decision" in data
    assert data["decision"] in ("APPROVE", "REVIEW", "DECLINE")
    assert "latency_ms" in data


def test_api_predict_invalid_missing_fields(client):
    payload = {"merchant": "FreshMart"}
    res = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_api_transactions_pagination(client):
    res = client.get("/api/transactions?limit=10&offset=0")
    assert res.status_code == 200
    data = res.get_json()
    assert "transactions" in data
    assert "total" in data
    assert isinstance(data["transactions"], list)


def test_api_metrics(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "kpis" in data
    assert "timeline" in data
    assert "category_risk" in data
    assert "risk_distribution" in data


def test_html_views(client):
    # Dashboard view
    res = client.get("/")
    assert res.status_code == 200
    assert b"VERITAS RISK" in res.data

    # Transactions ledger view
    res = client.get("/transactions")
    assert res.status_code == 200
    assert b"Transaction Ledger" in res.data

    # Models view
    res = client.get("/models")
    assert res.status_code == 200

    # Monitoring view
    res = client.get("/monitoring")
    assert res.status_code == 200
