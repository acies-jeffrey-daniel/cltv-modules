# Swap the prices
import pandas as pd
df = pd.read_csv('ADS_Only_Transactions_for_Subscription_Churnflag.csv')
df["plan_price"] = df["plan_price"].replace({12.99: 119.99, 119.99: 12.99})
df.to_csv('ADS_Trans_Clean.csv')
