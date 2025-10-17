import pandas as pd
import requests
import io

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSKsxrQBqPOQdF_KOsD3ub81wynRnXX6Pw0BxRsDikzXZgEQOoEsTS0ILD7xnNNeBTOKd7xFCFmtsqM/pub?output=csv"

# Fetch the sheet
resp = requests.get(SHEET_CSV_URL)
resp.raise_for_status()

df = pd.read_csv(io.StringIO(resp.text))

# Quick overview
print("Total rows in CSV:", len(df))
print("Columns:", df.columns.tolist())

# Check for missing dates
missing_dates = df[df['date'].isna()]
print("Rows with missing dates:", len(missing_dates))
print(missing_dates)

# Check for NaNs in key numeric columns
for col in ['gmv', 'users', 'marketing_cost']:
    missing = df[df[col].isna()]
    print(f"Rows with missing {col}:", len(missing))

# Preview last few rows to see if end of sheet is loaded
print(df.tail(20))
