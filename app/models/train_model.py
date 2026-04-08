import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from app.utils.metrics import calculate_metrics


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

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    best_model_name = None
    best_model = None
    best_f1 = -1
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = calculate_metrics(y_test, preds)
        results[model_name] = metrics

        print(f"\n=== {model_name.upper()} ===")
        print(json.dumps(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "confusion_matrix": metrics["confusion_matrix"],
            },
            indent=2
        ))
        print("\nClassification Report:\n")
        print(metrics["classification_report"])

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_model = model

    joblib.dump(best_model, "model.joblib")

    summary = {
        "best_model": best_model_name,
        "best_f1_score": best_f1,
        "all_results": results,
    }

    with open("app/data/model_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest model: {best_model_name}")
    print("Saved model.joblib")
    print("Saved app/data/model_metrics.json")


if __name__ == "__main__":
    train()