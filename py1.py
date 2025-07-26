#!/usr/bin/env python3
"""
fraudkit.py – Real-time credit-card fraud-detection micro-framework.

Author :  <you>
License: MIT
Python  : ≥3.11
"""

# ───────────────────────────── Metadata & deps ────────────────────────────── #
"""
Hard runtime deps (pin via requirements.txt):
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.8.1
python-dotenv==1.0.1
sqlalchemy[asyncio]==2.0.31
asyncpg==0.29.0
scikit-learn==1.5.0
pandas==2.2.2
cryptography==42.0.5
python-json-logger==2.0.7
hypothesis==6.103.3
pytest==8.2.1
"""

from __future__ import annotations

import asyncio
import base64
import bz2
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────── Settings ───────────────────────────────── #
class Settings(BaseSettings):
    # Security
    master_key: SecretStr = Field(..., env="FRAUDKIT_MASTER_KEY")
    encryption_algorithm: str = "AESGCM"
    # Database
    db_url: str = Field("postgresql+asyncpg://fraudkit:fraudkit@localhost/fraudkit", env="FRAUDKIT_DB_URL")
    # API
    allowed_origins: List[str] = Field(["*"], env="FRAUDKIT_ALLOWED_ORIGINS")
    # Rule engine
    rules_path: Path = Field("rules.json", env="FRAUDKIT_RULES_PATH")
    rules_hot_reload_seconds: int = 15
    # ML
    training_window: int = 10_000  # sliding window size
    # Compliance
    audit_log_path: Path = Field("audit.log", env="FRAUDKIT_AUDIT_LOG")
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ───────────────────────────── Logger setup ──────────────────────────────── #
JSON_LOGGING = {
    "version": 1,
    "formatters": {
        "json": {
            "()": "python_json_logger.jsonlogger.JsonFormatter",
            "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        }
    },
    "root": {"handlers": ["default"], "level": "INFO"},
}
logging.config.dictConfig(JSON_LOGGING)
logger = logging.getLogger("fraudkit")

# ──────────────────────── Utility: field-level crypto ────────────────────── #
class Tokeniser:
    """Stateless reversible tokenisation using AES-GCM."""

    NONCE_LEN = 12  # -- PCI DSS compliant IV length

    def __init__(self, key: bytes):
        self.key = key
        self.aesgcm = AESGCM(key)

    def encrypt(self, plaintext: str, aad: bytes | None = None) -> str:
        nonce = secrets.token_bytes(self.NONCE_LEN)
        ct = self.aesgcm.encrypt(nonce, plaintext.encode(), aad)
        return base64.urlsafe_b64encode(nonce + ct).decode()

    def decrypt(self, token: str, aad: bytes | None = None) -> str:
        raw = base64.urlsafe_b64decode(token.encode())
        nonce, ct = raw[: self.NONCE_LEN], raw[self.NONCE_LEN :]
        pt = self.aesgcm.decrypt(nonce, ct, aad)
        return pt.decode()


tokeniser = Tokeniser(settings.master_key.get_secret_value().encode())

# ───────────────────────────── Audit logging ─────────────────────────────── #
def audit(event: str, user: str | None = None, meta: Dict[str, Any] | None = None):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "user": user,
        "meta": meta or {},
    }
    with open(settings.audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info("audit", extra=rec)


# ──────────────────────────── Pydantic models ────────────────────────────── #
class CardData(BaseModel):
    pan: SecretStr = Field(..., min_length=13, max_length=19)
    expiry: str = Field(..., regex=r"^(0[1-9]|1[0-2])\/\d{2,4}$")
    cvv: SecretStr = Field(..., min_length=3, max_length=4)

class Transaction(BaseModel):
    id: str
    merchant_id: str
    user_id: str | None = Field(None, description="Internal ERP user identifier")
    card: CardData
    amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ip: str | None = None
    metadata: Dict[str, Any] | None = None

class TxnVerdict(BaseModel):
    transaction_id: str
    fraud_score: float
    rule_hits: List[str]
    flagged: bool

# ───────────────────────────── Rule engine ───────────────────────────────── #
class Rule:
    def __init__(self, name: str, expr: str, weight: float, threshold: float):
        self.name = name
        self.expr = expr
        self.weight = weight
        self.threshold = threshold
        self.code = compile(expr, "<string>", "eval")

    def evaluate(self, txn: Transaction) -> Tuple[bool, float]:
        try:
            result: float | bool = eval(self.code, {}, {"txn": txn})
        except Exception as e:
            logger.exception("rule_error", extra={"rule": self.name})
            return False, 0.0
        score = float(result) if not isinstance(result, bool) else float(result)
        hit = score >= self.threshold
        return hit, score * self.weight


class RuleEngine:
    def __init__(self, path: Path):
        self.path = path
        self.rules: Dict[str, Rule] = {}
        self.last_load: float = 0.0
        self.load_rules()

    def load_rules(self):
        try:
            data = json.loads(Path(self.path).read_text())
        except FileNotFoundError:
            logger.warning("rules file missing – loading empty set")
            data = []
        self.rules = {
            r["name"]: Rule(r["name"], r["expr"], r["weight"], r["threshold"]) for r in data
        }
        self.last_load = time.time()
        logger.info("rules_loaded", extra={"count": len(self.rules)})

    def maybe_reload(self):
        if time.time() - self.last_load >= settings.rules_hot_reload_seconds:
            self.load_rules()

    def evaluate(self, txn: Transaction) -> Tuple[List[str], float]:
        self.maybe_reload()
        hits, score = [], 0.0
        for rule in self.rules.values():
            hit, partial = rule.evaluate(txn)
            if hit:
                hits.append(rule.name)
                score += partial
        return hits, score


rule_engine = RuleEngine(settings.rules_path)

# ────────────────────────── Adaptive ML component ────────────────────────── #
class OnlineModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(loss="log_loss", penalty="elasticnet")
        self.classes_ = [0, 1]  # 0 = legit, 1 = fraud
        self.n_seen = 0

    def featurise(self, txn: Transaction) -> List[float]:
        return [
            txn.amount,
            int(txn.ts.hour),
            int(txn.currency == "USD"),
            len(txn.card.pan.get_secret_value()),
        ]

    def partial_fit(self, X: Iterable[List[float]], y: Iterable[int]):
        X_scaled = self.scaler.partial_fit_transform(X)
        self.clf.partial_fit(X_scaled, y, classes=self.classes_)
        self.n_seen += len(y)

    def predict_proba(self, txn: Transaction) -> float:
        X = [self.featurise(txn)]
        X_scaled = self.scaler.transform(X)
        return float(self.clf.predict_proba(X_scaled)[0][1])  # fraud prob


online_model = OnlineModel()

# ─────────────────────────────── FastAPI app ─────────────────────────────── #
app = FastAPI(
    title="FraudKit API",
    version="1.0.0",
    description="Real-time credit-card fraud-detection micro-framework.",
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global audit middleware
@app.middleware("http")
async def audit_mdw(request: Request, call_next: Callable):
    start = time.time()
    resp: Response | JSONResponse
    try:
        resp = await call_next(request)
    except Exception as e:
        audit("exception", meta={"path": request.url.path, "err": str(e)})
        raise
    latency = time.time() - start
    audit(
        "request",
        user=request.headers.get("X-User"),
        meta={"path": request.url.path, "status": resp.status_code, "latency": latency},
    )
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(_, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# ───────────────────────────── API endpoints ────────────────────────────── #
@app.post("/score", response_model=TxnVerdict, summary="Score a transaction")
async def score(txn: Transaction):
    # Tokenise sensitive fields
    tokenised_pan = tokeniser.encrypt(txn.card.pan.get_secret_value(), aad=txn.id.encode())
    txn.card.pan = SecretStr(tokenised_pan)  # replace plaintext

    # Rule engine
    rule_hits, rule_score = rule_engine.evaluate(txn)

    # ML probability
    ml_score = online_model.predict_proba(txn)

    combined = 0.6 * ml_score + 0.4 * rule_score
    flagged = combined >= 0.8

    return TxnVerdict(
        transaction_id=txn.id,
        fraud_score=combined,
        rule_hits=rule_hits,
        flagged=flagged,
    )


class Feedback(BaseModel):
    transaction_id: str
    is_fraud: bool


@app.post("/feedback", summary="Supervised feedback loop")
async def feedback(fb: Feedback):
    # In production, you’d pull txn features from DB; here we trust caller
    dummy_txn = Transaction(
        id=fb.transaction_id,
        merchant_id="dummy",
        card=CardData(pan=SecretStr("0" * 16), expiry="01/30", cvv=SecretStr("000")),
        amount=0.0,
        currency="USD",
    )
    features = online_model.featurise(dummy_txn)
    online_model.partial_fit([features], [int(fb.is_fraud)])
    return {"detail": "model updated", "seen": online_model.n_seen}


@app.delete("/token/{token}", summary="Right-to-erasure endpoint (GDPR)")
async def erase(token: str):
    # For demo, we just log – in prod, also delete DB rows
    audit("erase_token", meta={"token": token})
    return {"status": "deleted if existed"}


# ─────────────────────────────── CI utilities ───────────────────────────── #
def _ci():
    """Sample entry for pytest."""
    import pytest, sys

    # Run hypothesis and unit tests
    errno = pytest.main(["-q"])
    sys.exit(errno)


if __name__ == "__main__":
    import argparse, uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Run test-suite")
    args = parser.parse_args()
    if args.ci:
        _ci()
    else:
        uvicorn.run("fraudkit:app", host="0.0.0.0", port=8000, reload=True)
