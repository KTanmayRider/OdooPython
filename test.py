# test_fraud_api.py
import pytest
from fastapi.testclient import TestClient
from main import app, DETECT_FRAUD_ENDPOINT   # adjust import path if the app file is not main.py

client = TestClient(app)


@pytest.mark.parametrize(
    "txn, expect_fraud, expect_rules, expect_iso, expect_logit",
    [
        # 1. Legitimate domestic purchase
        ({"amount": 100, "country": "US"}, False, [], False, False),

        # 2. High-amount domestic purchase
        ({"amount": 20_000, "country": "US"}, True, ["HighAmount"], None, None),

        # 3. Black-listed country, small amount
        ({"amount": 50, "country": "IR"}, True, ["BlacklistedCountry"], None, None),

        # 4. Foreign low amount (Logistic fraud, no rules)
        ({"amount": 200, "country": "GB"}, True, [], None, True),

        # 5. Foreign high amount (>10k) – rule + Logistic
        ({"amount": 12_000, "country": "GB"}, True, ["HighAmount"], None, True),

        # 6. Black-listed country with high amount
        ({"amount": 15_000, "country": "KP"}, True, ["HighAmount", "BlacklistedCountry"], None, True),

        # 7. Negative amount (Isolation anomaly)
        ({"amount": -50, "country": "US"}, True, [], True, False),

        # 8. Missing amount field, domestic
        ({"country": "US"}, False, [], False, False),

        # 9. Missing country, high amount
        ({"amount": 11_000}, True, ["HighAmount"], None, None),

        # 10. Malformed amount type
        ({"amount": "not-a-number", "country": "US"}, False, [], False, False),
    ],
)
def test_detect_fraud_scenarios(txn, expect_fraud, expect_rules, expect_iso, expect_logit):
    """
    Parametrized end-to-end tests that hit the /detect_fraud endpoint
    and assert on the key aspects returned by the API.
    """
    response = client.post(DETECT_FRAUD_ENDPOINT, json=txn)
    assert response.status_code == 200

    data = response.json()

    # Core fraud flag
    assert data["fraud_detected"] is expect_fraud

    # Rules
    if expect_rules:
        for rule in expect_rules:
            assert rule in data["triggered_rules"]
    else:
        assert data["triggered_rules"] == []

    # Isolation Forest (allow None = don't-care in this scenario)
    if expect_iso is not None:
        assert data["isolation_anomaly"] is expect_iso

    # Logistic Regression fraud prediction
    if expect_logit is not None:
        assert data["fraud_prediction"] is expect_logit


def test_mask_pan_additional_cases():
    """
    Extra edge-case coverage for the mask_pan utility.
    """
    from main import mask_pan  # adjust if mask_pan lives elsewhere

    # Very long PAN
    assert mask_pan("9" * 19) == "*" * 15 + "9999"
    # Empty string
    assert mask_pan("") is None
    # Non-numeric string with spaces
    assert mask_pan("  4242abcd  ") == "*****abcd"
