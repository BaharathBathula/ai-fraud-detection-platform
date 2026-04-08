import random
import pandas as pd


def generate_transactions(n: int = 5000) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        amount = round(random.uniform(5, 5000), 2)
        transaction_hour = random.randint(0, 23)
        account_age_days = random.randint(1, 4000)
        num_prev_transactions = random.randint(0, 500)
        is_foreign = random.randint(0, 1)
        is_high_risk_country = random.randint(0, 1)
        has_chargeback_history = random.randint(0, 1)

        fraud_score_seed = (
            (amount > 2000)
            + (transaction_hour in [0, 1, 2, 3, 4])
            + is_foreign
            + is_high_risk_country
            + has_chargeback_history
            + (num_prev_transactions < 3)
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
    df = generate_transactions(5000)
    df.to_csv("app/data/sample_transactions.csv", index=False)
    print("Generated app/data/sample_transactions.csv")
    