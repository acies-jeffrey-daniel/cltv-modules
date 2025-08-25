import pandas as pd
from pandas.tseries.offsets import DateOffset

# Example df
df = pd.read_csv("ADS_Only_Transactions_for_Subscription.csv", parse_dates=["plan_start_date"])

observation_date = pd.to_datetime("2025-05-31")

# Compute expiry date depending on plan term
df["expiry_date"] = df.apply(
    lambda row: row["plan_start_date"] + DateOffset(years=1) if row["plan_term"] == "Annual"
                else row["plan_start_date"] + DateOffset(months=1),
    axis=1
)

# Flag churned (1) vs active (0)
df["churn_flag"] = (df["expiry_date"] < observation_date).astype(int)

df.to_csv('ADS_Only_Transactions_for_Subscription_Churnflag.csv')