from fastapi import FastAPI
from pydantic import BaseModel
from app.models.predict import predict_fraud

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


@app.post("/predict")
def predict(transaction: TransactionInput):
    result = predict_fraud(transaction.dict())
    return result