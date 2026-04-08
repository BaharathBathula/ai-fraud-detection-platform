import json
import os
from datetime import datetime

LOG_FILE = "app/data/prediction_logs.jsonl"


def log_prediction(payload: dict, result: dict):
    os.makedirs("app/data", exist_ok=True)

    log_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": payload,
        "result": result,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_record) + "\n")