import random
from datetime import datetime
import uuid
import time

import pandas as pd

HEADERS = "user_id,amount,timestamp,transaction_id,merchant,payment_method,device,location\n"
USER_IDS = [f"User{i}" for i in range(1, 101)]
MERCHANTS = [
  "Dmart",
  "Punjab stores",
  "ANAND GROCERY STORE",
  "Aggarwal Super Market",
  "Sham Super Store",
]
PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"]
DEVICES = ["Android Mobile", "iOS Mobile", "Desktop", "POS Terminal"]
LOCATIONS = [
    "Industrial Area, Mohali",
    "Sector 71, Mohali",
    "Sector 79, Mohali",
    "Sector 70, Mohali",
    "Sector 91, Mohali",
]


def create_user_profiles():
    profiles = {}
    for user_id in USER_IDS:
        profiles[user_id] = {
            "location": random.choice(LOCATIONS),
            "device": random.choice(DEVICES),
            "payment_method": random.choice(PAYMENT_METHODS),
            "min_spending_range": random.randint(1, 1000),
            "max_spending_range": random.randint(1000, 20000),
        }

    return pd.DataFrame(profiles).T.reset_index(names="user_id")


def generate_transaction(user_df):
    user = user_df.sample(1).iloc[0]
    return {
        "user_id": user["user_id"],
        "amount": random.randint(user["min_spending_range"], user["max_spending_range"]),
        "timestamp": datetime.now(),
        "transaction_id": uuid.uuid4(),
        "merchant": random.choice(MERCHANTS),
        "payment_method": user["payment_method"],
        "device": user["device"],
        "location": user["location"],
    }


def generate_transactions(user_df, count=10, delay=1):
    transactions = []
    for _ in range(count):
        transactions.append(generate_transaction(user_df))
        time.sleep(delay)

    return pd.DataFrame(transactions)


def save_transactions(transactions, filename="transaction.csv"):
    transactions.to_csv(filename, index=False)


def main():
    user_df = create_user_profiles()
    transactions = generate_transactions(user_df)
    save_transactions(transactions)


if __name__ == "__main__":
    main()

