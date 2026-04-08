import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


FEATURES = [
    "amount",
    "transaction_hour",
    "account_age_days",
    "num_prev_transactions",
    "is_foreign",
    "is_high_risk_country",
    "has_chargeback_history",
]


def train():
    df = pd.read_csv("app/data/sample_transactions.csv")

    X = df[FEATURES]
    y = df["fraud_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, preds))

    joblib.dump(model, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    train()