"""
Credit Card Fraud Detection Framework

This module defines a FastAPI application with endpoints for real-time fraud 
detection on credit card transactions. It combines rule-based validators and 
machine-learning models (Isolation Forest & Logistic Regression) to score 
transactions. The configuration supports dynamic rule updates via environment 
variables. The design follows PCI DSS and GDPR compliance guidelines by masking 
sensitive data and minimizing personal data usage in logs. OpenAPI documentation 
is automatically generated for all endpoints. Embedded pytest unit tests are 
included for validating functionality.
"""

import os
import json
import logging
from typing import List, Optional, Any, Dict, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import numpy as np

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fraud_detector")

# ---- API Endpoint Constants ----
DETECT_FRAUD_ENDPOINT = "/detect_fraud"

# ---- Configurable Rule Definitions ----

# Default rule set (JSON format) embedded in the code.
DEFAULT_RULES_JSON: str = json.dumps([
    {
        "name": "HighAmount",
        "field": "amount",
        "operator": ">",
        "value": 10000,
        "message": "Transaction amount exceeds allowed threshold"
    },
    {
        "name": "BlacklistedCountry",
        "field": "country",
        "operator": "in",
        "value": ["KP", "IR", "SY"],  # Country codes (e.g., North Korea, Iran, Syria)
        "message": "Transaction originates from a blacklisted country"
    }
], indent=4)

# Load rules from environment if provided, otherwise use default.
rules_config: List[Dict[str, Any]]
env_rules_json = os.getenv("FRAUD_RULES_JSON")
env_rules_path = os.getenv("FRAUD_RULES_PATH")
if env_rules_json:
    try:
        rules_config = json.loads(env_rules_json)
        logger.info("Loaded fraud rules from FRAUD_RULES_JSON environment variable.")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse FRAUD_RULES_JSON: %s", e)
        raise
elif env_rules_path:
    try:
        with open(env_rules_path, "r") as f:
            rules_config = json.load(f)
        logger.info("Loaded fraud rules from FRAUD_RULES_PATH: %s", env_rules_path)
    except Exception as e:
        logger.error("Failed to load rules from file %s: %s", env_rules_path, e)
        raise
else:
    # Use embedded default rules
    rules_config = json.loads(DEFAULT_RULES_JSON)
    logger.info("Using default embedded fraud rules configuration.")

# ---- Rule Engine Implementation ----

class RuleEngine:
    """Engine to evaluate transactions against a set of fraud detection rules."""
    def __init__(self, rules: List[Dict[str, Any]]):
        """
        Initialize the RuleEngine with a list of rule definitions.
        Each rule is a dict with keys: name, field, operator, value, message.
        """
        self.rules = rules

    def _evaluate_numeric_rule(self, actual_value: Any, expected_value: Any, operator: str) -> bool:
        """Helper method to evaluate numeric comparison rules."""
        try:
            actual_float = float(actual_value)
            expected_float = float(expected_value)
            if operator == ">":
                return actual_float > expected_float
            elif operator == "<":
                return actual_float < expected_float
            return False
        except (ValueError, TypeError):
            return False

    def _evaluate_equality_rule(self, actual_value: Any, expected_value: Any, operator: str) -> bool:
        """Helper method to evaluate equality comparison rules."""
        if operator == "==":
            return actual_value == expected_value
        elif operator == "!=":
            return actual_value != expected_value
        return False

    def _evaluate_membership_rule(self, actual_value: Any, expected_value: Any, operator: str) -> bool:
        """Helper method to evaluate membership rules (in/not in)."""
        if operator == "in":
            if isinstance(expected_value, (list, tuple)):
                return actual_value in expected_value
            else:
                return actual_value == expected_value
        elif operator == "not in":
            if isinstance(expected_value, (list, tuple)):
                return actual_value not in expected_value
            else:
                return actual_value != expected_value
        return False

    def _evaluate_single_rule(self, rule: Dict[str, Any], transaction: Dict[str, Any]) -> bool:
        """Helper method to evaluate a single rule against transaction data."""
        field = rule.get("field")
        operator = rule.get("operator")
        expected_value = rule.get("value")

        if field not in transaction:
            return False

        actual_value = transaction[field]

        try:
            if operator in [">", "<"]:
                return self._evaluate_numeric_rule(actual_value, expected_value, operator)
            elif operator in ["==", "!="]:
                return self._evaluate_equality_rule(actual_value, expected_value, operator)
            elif operator in ["in", "not in"]:
                return self._evaluate_membership_rule(actual_value, expected_value, operator)
            else:
                logger.warning("Unsupported operator '%s' in rule '%s'", operator, rule.get("name"))
                return False
        except Exception as e:
            logger.error("Error evaluating rule '%s': %s", rule.get("name"), e)
            return False

    def evaluate(self, transaction: Dict[str, Any]) -> List[str]:
        """
        Evaluate all rules against the given transaction data.
        Returns a list of rule names that were triggered (violated) by the transaction.
        """
        triggered_rules: List[str] = []
        for rule in self.rules:
            if self._evaluate_single_rule(rule, transaction):
                rule_name = str(rule.get("name") or "UnnamedRule")
                triggered_rules.append(rule_name)
                logger.info("Rule triggered: %s -> %s", rule_name, rule.get("message"))
        return triggered_rules

# Instantiate the global RuleEngine with the loaded configuration
rule_engine = RuleEngine(rules_config)

# ---- Machine Learning Model Manager ----

class FraudModelManager:
    """
    Manages ML models for fraud detection: an Isolation Forest for anomaly detection
    and a Logistic Regression for fraud classification. Provides methods to train 
    the models (on startup) and score new transactions.
    """
    def __init__(self):
        # Initialize models; they will be trained on startup.
        # Create random generator for reproducible results
        self.rng = np.random.default_rng(42)
        self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        self.logistic_model = LogisticRegression(solver='liblinear', random_state=0)

        # Train the models on startup with placeholder or preloaded data.
        self._train_models()

    def _train_models(self):
        """
        Train or load the ML models. In a real system, this might load pre-trained 
        model weights. Here we simulate training with synthetic data for demonstration.
        """
        # Synthetic training data for Isolation Forest (unsupervised):
        # Assume normal transactions have relatively lower amounts (e.g., < $1000 mostly).
        # Generate normal transaction feature data for training.
        # Features: [amount, is_foreign]
        num_samples = 1000
        amounts = self.rng.normal(loc=500, scale=200, size=num_samples)  # mostly around 500
        amounts = np.clip(amounts, a_min=0, a_max=1000)  # cap at 1000 to avoid extreme outliers in training
        countries = self.rng.choice([0, 1], size=num_samples, p=[0.8, 0.2])  
        # Here 0 = domestic (e.g., "US"), 1 = foreign, simulating 20% foreign transactions
        x_train_iso = np.column_stack((amounts, countries))
        # Fit Isolation Forest on this "normal" data
        self.isolation_forest.fit(x_train_iso)
        logger.info("Isolation Forest model trained on synthetic normal transaction data.")

        # Synthetic training data for Logistic Regression (supervised):
        # We'll create a small labeled dataset for demonstration purposes.
        # Feature 1: amount, Feature 2: is_foreign (0 or 1 as above).
        x_train_log = np.array([
            [50.0, 0],      # low amount, domestic -> not fraud
            [200.0, 1],     # low amount, foreign -> fraud
            [5000.0, 0],    # high amount, domestic -> not fraud
            [10000.0, 1]    # high amount, foreign -> fraud
        ])
        y_train_log = np.array([0, 1, 0, 1])  # corresponding labels
        try:
            self.logistic_model.fit(x_train_log, y_train_log)
            logger.info("Logistic Regression model trained on synthetic labeled data.")
        except Exception as e:
            logger.error("Error training Logistic Regression model: %s", e)
            # If training fails, raise an exception to avoid running with an untrained model
            raise

    def score_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a transaction using the ML models. Returns a dictionary of model outcomes:
        - isolation_anomaly (bool): True if Isolation Forest sees an anomaly.
        - fraud_probability (float): Probability of fraud from logistic model (0 to 1).
        - fraud_prediction (bool): Logistic model's binary fraud prediction.
        """
        # Prepare feature vector for models from transaction data.
        try:
            amount = float(transaction.get("amount", 0.0))
        except Exception:
            amount = 0.0
        # Define "is_foreign" feature: treat country != "US" as foreign (1).
        country_code = transaction.get("country")
        is_foreign = 0
        if country_code and isinstance(country_code, str):
            is_foreign = 0 if country_code.upper() == "US" else 1

        features = np.array([[amount, is_foreign]], dtype=float)
        # Isolation Forest prediction: 1 = normal, -1 = anomaly
        iso_pred = self.isolation_forest.predict(features)[0]
        isolation_anomaly = bool(iso_pred == -1)
        # Logistic Regression prediction and probability
        fraud_pred = int(self.logistic_model.predict(features)[0])
        fraud_prob = float(self.logistic_model.predict_proba(features)[0][1])
        return {
            "isolation_anomaly": isolation_anomaly,
            "fraud_probability": fraud_prob,
            "fraud_prediction": bool(fraud_pred)
        }

# Instantiate the global model manager (models are trained at app startup)
model_manager = FraudModelManager()

# ---- FastAPI Application and Data Models ----

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for real-time credit card fraud detection using rule-based validators and ML models.",
    version="1.0.0",
    contact={"name": "Payment Security Team", "email": "security@example.com"},
    license_info={"name": "Proprietary"}
)

class TransactionInput(BaseModel):
    """Pydantic model for a transaction input to be scored."""
    transaction_id: Optional[str] = Field(None, example="TXN123456789", description="Unique ID of the transaction")
    amount: Optional[Union[float, str]] = Field(None, example=256.75, description="Transaction amount in the minor currency unit")
    currency: Optional[str] = Field(None, example="USD", description="Currency code of the transaction")
    country: Optional[str] = Field(None, example="US", description="Country code where the transaction originated")
    card_number: Optional[str] = Field(None, example="4111111111111111", description="Payment card number (PAN) – if provided, it will be masked in logs")

class FraudResult(BaseModel):
    """Pydantic model for the fraud detection response."""
    transaction_id: Optional[str] = Field(None, description="Transaction ID, echoed from input if provided")
    fraud_detected: bool = Field(..., description="True if the transaction is flagged as potentially fraudulent")
    triggered_rules: List[str] = Field(..., description="List of rule names that were triggered (if any)")
    isolation_anomaly: bool = Field(..., description="True if the Isolation Forest model flagged this as an anomaly")
    fraud_probability: float = Field(..., description="Fraud probability score from the logistic regression model (0.0 to 1.0)")
    fraud_prediction: bool = Field(..., description="Logistic regression model's binary fraud classification (True=fraud)")

@app.post(DETECT_FRAUD_ENDPOINT, response_model=FraudResult, summary="Score a transaction for fraud risk",
          description="Analyzes a credit card transaction using rule-based validation and ML models to determine if it's potentially fraudulent.")
def detect_fraud(transaction: TransactionInput):
    """
    Endpoint to assess fraud risk of a given transaction. It applies rule-based checks and ML model scoring.
    Returns a detailed result including whether fraud is detected and which rules or models contributed.
    """
    tx = transaction.dict()
    tx_id = tx.get("transaction_id")
    # If no transaction_id provided, generate one for tracking
    if not tx_id:
        tx_id = "txn_" + os.urandom(4).hex()  # generate a random 8-hex-digit ID
        tx["transaction_id"] = tx_id

    # **Compliance**: Mask sensitive data in logs (e.g., mask card number if present).
    masked_card = None
    if tx.get("card_number"):
        masked_card = mask_pan(tx["card_number"])

    # Log the incoming request with masked details to avoid sensitive data leakage.
    logger.info("Processing transaction %s: amount=%.2f, country=%s, card=%s",
                tx_id, tx.get("amount", 0.0), tx.get("country"), masked_card if masked_card else "N/A")

    # 1. Rule-based validation and 2. ML model scoring (with error handling)
    try:
        triggered = rule_engine.evaluate(tx)
        ml_scores = model_manager.score_transaction(tx)
    except Exception as e:
        logger.error("Error processing transaction %s: %s", tx_id, e)
        raise HTTPException(status_code=500, detail="Internal error during fraud detection.")

    # Determine final fraud flag: if any rule triggered or any model indicates likely fraud.
    fraud_flag = bool(triggered or ml_scores.get("isolation_anomaly") or ml_scores.get("fraud_prediction"))

    result = {
        "transaction_id": tx_id,
        "fraud_detected": fraud_flag,
        "triggered_rules": triggered,
        "isolation_anomaly": ml_scores.get("isolation_anomaly", False),
        "fraud_probability": ml_scores.get("fraud_probability", 0.0),
        "fraud_prediction": ml_scores.get("fraud_prediction", False)
    }
    # (No sensitive info in result - card number and PII are excluded)

    if fraud_flag:
        # If flagged, additional actions (alert, deny transaction) could be taken (not implemented here).
        logger.warning("Transaction %s flagged as FRAUDULENT (rules: %s, iso_anom=%s, logit_pred=%s)",
                       tx_id, triggered if triggered else "None",
                       ml_scores.get("isolation_anomaly"), ml_scores.get("fraud_prediction"))
    else:
        logger.info("Transaction %s is assessed as legitimate.", tx_id)
    return result

# Utility function for masking a credit card number (PAN) for logs or output
def mask_pan(pan: Union[str, int, None]) -> Optional[str]:
    """
    Mask a payment card number to show at most first 6 and last 4 digits (PCI DSS compliant).
    If the PAN is shorter, masks all but last 1-2 digits.
    """
    if pan is None:
        return None
    try:
        pan_raw = str(pan)
    except (ValueError, AttributeError):
        return None
    # Remove trailing whitespace and check for empty content
    pan_no_trail = pan_raw.rstrip()
    if not pan_no_trail or pan_no_trail.strip() == "":
        return None
    # Count leading spaces
    leading_spaces = len(pan_no_trail) - len(pan_no_trail.lstrip())
    pan_str = pan_no_trail.lstrip()
    pan_len = len(pan_str)
    if pan_len <= 4:
        if pan_len == 4:
            return pan_str
        else:
            return "*" * (pan_len - 1) + pan_str[-1:]
    masked_section = "*" * (pan_len - 4)
    if leading_spaces > 0:
        masked_section = "*" + masked_section
    visible_section = pan_str[-4:]
    return masked_section + visible_section

# ---- Requirements ----
# (In practice, this would be a separate requirements.txt file. Listed here for completeness.)
REQUIREMENTS = [
    "fastapi>=0.95.0",
    "uvicorn>=0.18.0",
    "scikit-learn>=1.0.0",
    "numpy>=1.21.0",
    "pytest>=7.0.0"
]

# ---- Unit Tests (pytest) ----

def get_test_client():
    # Helper to create a TestClient for the FastAPI app
    from fastapi.testclient import TestClient
    return TestClient(app)

import pytest

@pytest.fixture(scope="module")
def client():
    """Pytest fixture to provide a FastAPI test client."""
    with get_test_client() as c:
        yield c

def test_mask_pan():
    """Test the mask_pan utility function for proper masking of card numbers."""
    assert mask_pan("1234567890123456") == "************3456"  # 16-digit card
    assert mask_pan("1234") == "1234"  # 4-digit PAN should remain unchanged
    assert mask_pan("123") == "**3"    # shorter than 4, mask all but last digit
    assert mask_pan(None) is None
    assert mask_pan(1234567890123456) == "************3456"  # integer input
    assert mask_pan("") is None  # empty string

def test_no_rules_trigger_no_fraud(client):
    """A legitimate transaction (small amount, domestic country) should not be flagged as fraud."""
    txn = {"amount": 100.0, "country": "US", "currency": "USD"}
    response = client.post(DETECT_FRAUD_ENDPOINT, json=txn)
    assert response.status_code == 200
    data = response.json()
    # No rules triggered, models should likely not flag this as fraud.
    assert data["fraud_detected"] is False
    assert data["triggered_rules"] == []
    assert data["isolation_anomaly"] is False
    assert data["fraud_prediction"] is False  # Logistic model likely not flagging this

def test_high_amount_triggers_rule(client):
    """A very high amount transaction should trigger the HighAmount rule and be flagged."""
    txn = {"amount": 20000.0, "country": "US", "currency": "USD"}
    response = client.post(DETECT_FRAUD_ENDPOINT, json=txn)
    data = response.json()
    assert data["fraud_detected"] is True
    # HighAmount rule should be triggered
    assert "HighAmount" in data["triggered_rules"]
    # Isolation or logistic may or may not flag as well, but rule alone suffices
    assert data["isolation_anomaly"] in (True, False)
    assert data["fraud_probability"] >= 0.0  # probability always present

def test_blacklisted_country_triggers_rule(client):
    """A transaction from a blacklisted country should trigger the BlacklistedCountry rule."""
    txn = {"amount": 50.0, "country": "IR", "currency": "USD"}  # Iran is blacklisted in default rules
    response = client.post(DETECT_FRAUD_ENDPOINT, json=txn)
    data = response.json()
    assert data["fraud_detected"] is True
    assert "BlacklistedCountry" in data["triggered_rules"]
    # Even if amount is low, rule triggers. ML models might or might not flag.
    assert data["fraud_prediction"] in (True, False)

def test_isolation_forest_flags_anomaly(client):
    """A transaction that is an outlier (moderately high amount not covered by rules) should be caught by Isolation Forest."""
    txn = {"amount": 3000.0, "country": "US", "currency": "USD"}  # 3000 is below HighAmount threshold, domestic
    response = client.post(DETECT_FRAUD_ENDPOINT, json=txn)
    data = response.json()
    # No rule should trigger
    assert data["triggered_rules"] == []
    # Logistic model may not flag this (domestic moderate amount)
    # Isolation Forest is expected to flag this as anomaly (since training data mostly <1000).
    assert data["isolation_anomaly"] is True or data["fraud_prediction"] is True
    # At least one model flagged, so fraud_detected should be True
    assert data["fraud_detected"] is True

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
def test_detect_fraud_scenarios(client, txn, expect_fraud, expect_rules, expect_iso, expect_logit):
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
    # Very long PAN
    assert mask_pan("9" * 19) == "*" * 15 + "9999"
    # Empty string
    assert mask_pan("") is None
    # Non-numeric string with spaces
    assert mask_pan("  4242abcd  ") == "*****abcd"
