import pytest
from fraud_detection.inference import InferenceEngine


@pytest.fixture
def engine():
    return InferenceEngine.get_instance()


def test_validation_valid_payload(engine):
    payload = {
        "user_id": "USR_0001",
        "amount": 142.50,
        "merchant": "Apple Store Fifth Ave",
        "merchant_category": "Electronics",
        "location": "New York, USA",
        "device": "iOS App (iPhone 15)",
        "payment_method": "Credit Card",
    }
    is_valid, err_msg, sanitized = engine.validate_payload(payload)
    assert is_valid is True
    assert err_msg is None
    assert sanitized["amount"] == 142.50
    assert sanitized["user_id"] == "USR_0001"
    assert "transaction_id" in sanitized


def test_validation_missing_amount(engine):
    payload = {
        "user_id": "USR_0001",
        "merchant": "Apple Store",
    }
    is_valid, err_msg, sanitized = engine.validate_payload(payload)
    assert is_valid is False
    assert "amount" in err_msg.lower()


def test_validation_negative_or_zero_amount(engine):
    payload = {
        "user_id": "USR_0001",
        "amount": -50.00,
        "merchant": "FreshMart",
    }
    is_valid, err_msg, sanitized = engine.validate_payload(payload)
    assert is_valid is False
    assert "positive" in err_msg.lower()


def test_validation_non_numeric_amount(engine):
    payload = {
        "user_id": "USR_0001",
        "amount": "not-a-number",
        "merchant": "FreshMart",
    }
    is_valid, err_msg, sanitized = engine.validate_payload(payload)
    assert is_valid is False
    assert "numeric" in err_msg.lower()


def test_validation_non_dict_payload(engine):
    is_valid, err_msg, sanitized = engine.validate_payload("invalid string")
    assert is_valid is False
    assert "json object" in err_msg.lower()
