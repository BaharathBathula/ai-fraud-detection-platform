from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from app.models.predict import predict_fraud, predict_batch
from app.services.scoring_service import log_prediction

app = FastAPI(title="AI Fraud Detection Platform")


class TransactionInput(BaseModel):
    amount: float
    transaction_hour: int
    account_age_days: int
    num_prev_transactions: int
    is_foreign: int
    is_high_risk_country: int
    has_chargeback_history: int


@app.get("/")
def root():
    return {"message": "AI Fraud Detection Platform API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: TransactionInput):
    payload = transaction.dict()
    result = predict_fraud(payload)
    log_prediction(payload, result)
    return result


@app.post("/predict/batch")
def batch_predict(transactions: List[TransactionInput]):
    payloads = [t.dict() for t in transactions]
    results = predict_batch(payloads)

    for payload, result in zip(payloads, results):
        log_prediction(payload, result)

    return {
        "total_records": len(results),
        "results": results,
    }