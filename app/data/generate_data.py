import random
import pandas as pd


def generate_transactions(n: int = 50000) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        amount = round(random.uniform(5, 5000), 2)
        transaction_hour = random.randint(0, 23)
        account_age_days = random.randint(1, 4000)
        num_prev_transactions = random.randint(0, 500)
        is_foreign = random.randint(0, 1)
        is_high_risk_country = random.randint(0, 1)
        has_chargeback_history = random.randint(0, 1)

        amount_risk = 1 if amount > 2500 else 0
        hour_risk = 1 if transaction_hour in [0, 1, 2, 3, 4] else 0
        account_risk = 1 if account_age_days < 90 else 0
        velocity_risk = 1 if num_prev_transactions < 5 else 0

        fraud_score_seed = (
            amount_risk
            + hour_risk
            + account_risk
            + velocity_risk
            + is_foreign
            + is_high_risk_country
            + has_chargeback_history
        )

        fraud_label = 1 if fraud_score_seed >= 3 else 0

        rows.append(
            {
                "amount": amount,
                "transaction_hour": transaction_hour,
                "account_age_days": account_age_days,
                "num_prev_transactions": num_prev_transactions,
                "is_foreign": is_foreign,
                "is_high_risk_country": is_high_risk_country,
                "has_chargeback_history": has_chargeback_history,
                "fraud_label": fraud_label,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_transactions(50000)
    df.to_csv("app/data/sample_transactions.csv", index=False)
    print(f"Generated app/data/sample_transactions.csv with {len(df)} rows")