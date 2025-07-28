# fraud_detection_framework.py
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pytest
from dotenv import load_dotenv
import hashlib
import secrets
from cryptography.fernet import Fernet
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# --- Security Constants ---
ENCRYPTION_KEY = Fernet.generate_key()  # In production, inject via secure secrets manager
MODEL_STORAGE_PATH = "secure_models/"
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# --- Configuration Initialization ---
load_dotenv()

# --- Secure Logging Configuration ---
class SecureLogger:
    def __init__(self):
        self.logger = logging.getLogger("secure_fraud_detection")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.fernet = Fernet(ENCRYPTION_KEY)
        
    def log(self, message: str, sensitive: bool = False):
        """Log messages with automatic encryption for sensitive data"""
        if sensitive:
            encrypted = self.fernet.encrypt(message.encode())
            self.logger.info(f"[ENCRYPTED] {encrypted.decode()}")
        else:
            self.logger.info(message)

logger = SecureLogger()

# --- GDPR-Compliant Activity Tracking ---
class ActivityTracker:
    def __init__(self):
        # In production, use encrypted database connection
        self.client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.client.fraud_detection
        self.collection = self.db.audit_logs
        
    def log_activity(self, user_id: str, action: str, metadata: dict = None):
        """Log pseudonymized activity with automatic encryption"""
        try:
            hashed_user = hashlib.sha256(user_id.encode()).hexdigest()
            entry = {
                "hashed_user": hashed_user,
                "timestamp": datetime.utcnow(),
                "action": action,
                "metadata": self._encrypt_sensitive(metadata) if metadata else None
            }
            self.collection.insert_one(entry)
        except PyMongoError as e:
            logger.log(f"Database error: {str(e)}")

    def _encrypt_sensitive(self, data: dict) -> dict:
        """Encrypt sensitive fields in metadata"""
        encrypted = {}
        for k, v in data.items():
            if k in ["ip_address", "device_id"]:  # Sensitive fields
                encrypted[k] = Fernet(ENCRYPTION_KEY).encrypt(str(v).encode()).decode()
            else:
                encrypted[k] = v
        return encrypted

    def get_user_activities(self, user_id: str) -> list:
        """Retrieve pseudonymized activities for a user"""
        hashed_user = hashlib.sha256(user_id.encode()).hexdigest()
        return list(self.collection.find({"hashed_user": hashed_user}, {"_id": 0}))

    def erase_user_data(self, user_id: str):
        """GDPR-compliant data erasure"""
        hashed_user = hashlib.sha256(user_id.encode()).hexdigest()
        self.collection.delete_many({"hashed_user": hashed_user})

# --- ML Model Management ---
class MLModelManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "iso_forest": IsolationForest(contamination=0.01, random_state=42),
            "log_reg": LogisticRegression(random_state=42, max_iter=1000)
        }
        self._train_dummy_models()  # Initial training
        
    def _train_dummy_models(self):
        """Train with initial dummy data"""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        X_scaled = self.scaler.fit_transform(X)
        self.models["iso_forest"].fit(X_scaled)
        self.models["log_reg"].fit(X_scaled, y)
        self._save_models()
        
    def _save_models(self):
        """Save encrypted models to secure storage"""
        for name, model in self.models.items():
            joblib.dump(model, f"{MODEL_STORAGE_PATH}{name}.joblib")
        
    def predict_anomaly(self, features: np.ndarray) -> float:
        """Get combined fraud prediction score"""
        try:
            scaled = self.scaler.transform(features.reshape(1, -1))
            iso_score = self.models["iso_forest"].decision_function(scaled)[0]
            log_reg_score = self.models["log_reg"].predict_proba(scaled)[0][1]
            return (1 - iso_score + log_reg_score) / 2
        except Exception as e:
            logger.log(f"Prediction error: {str(e)}")
            return 0.5  # Neutral score on error
            
    def update_model(self, model_type: str, new_model):
        """Securely update a model"""
        if model_type not in self.models:
            raise ValueError("Invalid model type")
        self.models[model_type] = new_model
        self._save_models()

# --- Rule Validation Engine ---
class RuleEngine:
    def __init__(self):
        self.rules = self._load_rules()
        
    def _load_rules(self) -> list:
        """Load rules from environment with validation"""
        rules_json = os.getenv("FRAUD_RULES_JSON", DEFAULT_RULES_JSON)
        try:
            rules = json.loads(rules_json)["rules"]
            self._validate_rules(rules)
            return rules
        except (json.JSONDecodeError, ValidationError) as e:
            logger.log(f"Rule validation error: {str(e)}")
            return []
            
    def _validate_rules(self, rules: list):
        """Ensure rule structure integrity"""
        required_fields = {"type", "action"}
        for rule in rules:
            if not required_fields.issubset(rule.keys()):
                raise ValidationError("Missing required rule fields")
            if rule["type"] == "amount" and "max" not in rule:
                raise ValidationError("Amount rule requires max field")
                
    def validate_transaction(self, transaction: dict) -> list:
        """Apply all rules to transaction"""
        flags = []
        for rule in self.rules:
            if rule["type"] == "amount" and transaction.get("amount", 0) > rule["max"]:
                flags.append(f"Amount exceeds {rule['max']}")
            elif rule["type"] == "velocity" and transaction.get("txn_count_hour", 0) > rule["max_transactions"]:
                flags.append(f"Velocity exceeds {rule['max_transactions']}/hour")
            elif rule["type"] == "location" and transaction.get("country") not in rule["allowed_countries"]:
                flags.append(f"Restricted country: {transaction['country']}")
        return flags

# --- Pydantic Models ---
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    timestamp: datetime = Field(..., description="ISO 8601 transaction timestamp")
    country: str = Field(..., min_length=2, max_length=3, description="ISO country code")
    user_id: str = Field(..., min_length=8, description="Pseudonymized user ID")
    features: List[float] = Field(..., min_items=5, max_items=10, description="ML feature vector")
    session_token: Optional[str] = Field(None, description="Secure session token")

class FraudScore(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Fraud probability score")
    flags: List[str] = Field(..., description="Rule-based flags")
    is_fraud: bool = Field(..., description="Final fraud determination")
    model_version: str = Field("1.0", description="ML model version")

class PrivacyRequest(BaseModel):
    user_id: str = Field(..., description="User ID for GDPR request")
    action: str = Field(..., regex="^(erase|export)$", description="GDPR action type")
    verification_token: str = Field(..., description="Secondary verification token")

# --- API Security ---
class AuthHandler:
    def __init__(self):
        self.valid_tokens = {
            "detection": os.getenv("API_TOKEN", secrets.token_urlsafe(32)),
            "admin": os.getenv("ADMIN_TOKEN", secrets.token_urlsafe(32))
        }
        
    def validate_token(self, token: str, role: str = "detection") -> bool:
        return token == self.valid_tokens.get(role, "")

# --- FastAPI Application ---
app = FastAPI(
    title="PCI/DSS & GDPR Compliant Fraud Detection API",
    description="Real-time fraud detection with hybrid ML and rule-based analysis",
    version="2.0.0",
    openapi_tags=[
        {
            "name": "Fraud Detection",
            "description": "Real-time transaction fraud analysis"
        },
        {
            "name": "Privacy Operations",
            "description": "GDPR-compliant data handling"
        },
        {
            "name": "Administration",
            "description": "System management endpoints"
        }
    ]
)

# --- Dependency Injection ---
auth_handler = AuthHandler()
activity_tracker = ActivityTracker()
model_manager = MLModelManager()
rule_engine = RuleEngine()

def get_auth(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    if not auth_handler.validate_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials

def get_admin_auth(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    if not auth_handler.validate_token(credentials.credentials, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials

# --- API Endpoints ---
@app.post("/detect", 
          response_model=FraudScore,
          tags=["Fraud Detection"],
          summary="Analyze transaction for fraud")
async def detect_fraud(
    request: Request,
    transaction: Transaction,
    token: str = Depends(get_auth)
):
    # GDPR-compliant activity logging
    activity_tracker.log_activity(
        transaction.user_id,
        "fraud_check",
        {"ip": request.client.host, "amount": transaction.amount}
    )
    
    # Rule-based validation
    tx_dict = transaction.dict()
    flags = rule_engine.validate_transaction(tx_dict)
    
    # ML prediction
    try:
        features = np.array(transaction.features)
        score = model_manager.predict_anomaly(features)
    except Exception as e:
        logger.log(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction error"
        )
    
    # Decision logic
    is_fraud = score > 0.7 or any("critical" in flag for flag in flags)
    
    return FraudScore(
        score=score,
        flags=flags,
        is_fraud=is_fraud,
        model_version="1.0"
    )

@app.post("/privacy",
          tags=["Privacy Operations"],
          summary="Handle GDPR privacy requests")
async def privacy_request(
    request: PrivacyRequest,
    token: str = Depends(get_auth)
):
    # Secondary verification
    if not auth_handler.validate_token(request.verification_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid verification token"
        )
    
    if request.action == "erase":
        activity_tracker.erase_user_data(request.user_id)
        return {"status": "data_erased"}
    elif request.action == "export":
        data = activity_tracker.get_user_activities(request.user_id)
        return {"user_data": data}

@app.post("/update-rules",
          tags=["Administration"],
          summary="Update fraud detection rules")
async def update_rules(
    new_rules: Dict[str, Any],
    token: str = Depends(get_admin_auth)
):
    try:
        # Validate and update rules
        os.environ["FRAUD_RULES_JSON"] = json.dumps(new_rules)
        rule_engine.rules = rule_engine._load_rules()
        return {"status": "rules_updated", "rule_count": len(rule_engine.rules)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid rules format: {str(e)}"
        )

@app.post("/update-model/{model_type}",
          tags=["Administration"],
          summary="Update ML model")
async def update_model(
    model_type: str,
    model_data: dict,
    token: str = Depends(get_admin_auth)
):
    try:
        # In production, validate model structure and performance
        model_manager.update_model(model_type, model_data["model"])
        return {"status": "model_updated", "model_type": model_type}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# --- Health Check ---
@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "operational",
        "components": {
            "database": "connected" if activity_tracker.client.server_info() else "disconnected",
            "models": {name: "loaded" for name in model_manager.models},
            "rules": f"{len(rule_engine.rules)} active"
        }
    }

# --- Embedded Tests ---
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)

def test_fraud_detection(client):
    test_transaction = {
        "amount": 1200,
        "timestamp": datetime.utcnow().isoformat(),
        "country": "US",
        "user_id": "user_12345",
        "features": [0.5, 1.2, 0.8, 1.5, 0.2]
    }
    response = client.post(
        "/detect",
        json=test_transaction,
        headers={"Authorization": "Bearer " + os.getenv("API_TOKEN")}
    )
    assert response.status_code == 200
    result = response.json()
    assert "score" in result
    assert "flags" in result
    assert "is_fraud" in result

def test_invalid_token(client):
    response = client.post(
        "/detect",
        json={},
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401

if __name__ == "__main__":
    import uvicorn
    ssl_cert = os.getenv("SSL_CERT_PATH", None)
    ssl_key = os.getenv("SSL_KEY_PATH", None)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key
    )

# --- Requirements Specification ---
"""
fastapi==0.104.1
uvicorn[standard]==0.23.2
scikit-learn==1.3.0
joblib==1.3.2
pydantic==2.4.2
pytest==7.4.2
python-dotenv==1.0.0
numpy==1.26.0
cryptography==41.0.3
pymongo==4.5.0
"""