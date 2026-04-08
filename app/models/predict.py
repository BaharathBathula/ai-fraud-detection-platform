import joblib
import pandas as pd

FEATURES = [
    "amount",
    "transaction_hour",
    "account_age_days",
    "num_prev_transactions",
    "is_foreign",
    "is_high_risk_country",
    "has_chargeback_history",
]

model = joblib.load("model.joblib")


def predict_fraud(payload: dict):
    df = pd.DataFrame([payload])[FEATURES]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0].max()
    return {"fraud_prediction": int(pred), "confidence": float(prob)}