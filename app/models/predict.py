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
    prob = float(model.predict_proba(df)[0].max())

    risk_level = "low"
    if prob >= 0.85:
        risk_level = "high"
    elif prob >= 0.65:
        risk_level = "medium"

    return {
        "fraud_prediction": int(pred),
        "confidence": round(prob, 4),
        "risk_level": risk_level,
    }


def predict_batch(records: list):
    df = pd.DataFrame(records)[FEATURES]
    preds = model.predict(df)
    probs = model.predict_proba(df).max(axis=1)

    results = []
    for record, pred, prob in zip(records, preds, probs):
        risk_level = "low"
        if prob >= 0.85:
            risk_level = "high"
        elif prob >= 0.65:
            risk_level = "medium"

        results.append(
            {
                "input": record,
                "fraud_prediction": int(pred),
                "confidence": round(float(prob), 4),
                "risk_level": risk_level,
            }
        )

    return results