import pandas as pd
import openpyxl
# Load Excel file (first sheet by default)
df = pd.read_excel("ADS_Trans_Clean.xlsx")

# Save as CSV
df.to_csv("ADS_Trans_CSV_Clean.csv", index=False)
