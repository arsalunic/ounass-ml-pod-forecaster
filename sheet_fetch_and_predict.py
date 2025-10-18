# sheet_fetch_and_predict.py

import requests
import io
import pandas as pd
import json
import re
from dateutil import parser

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSKsxrQBqPOQdF_KOsD3ub81wynRnXX6Pw0BxRsDikzXZgEQOoEsTS0ILD7xnNNeBTOKd7xFCFmtsqM/pub?output=csv"
API_URL = "http://127.0.0.1:8080/predict"

# --------------------------
# 1. Fetch the CSV
# --------------------------
resp = requests.get(SHEET_CSV_URL)
resp.raise_for_status()
df = pd.read_csv(io.StringIO(resp.text))
print(f"\n CSV fetched successfully. Raw rows: {len(df)}")

# --------------------------
# 2. Robust date parsing
# --------------------------


def smart_parse_date(x):
    """Handle dates like '11/12/2024', strip quotes, and force day-first parsing."""
    s = str(x).strip().replace("'", "").replace('"', "")
    try:
        if re.match(r"\d{1,2}/\d{1,2}/\d{4}", s):
            return parser.parse(s, dayfirst=True)
        return parser.parse(s, dayfirst=True)
    except Exception:
        return pd.NaT


df['date'] = df['date'].apply(smart_parse_date)
df = df.dropna(subset=['date'])
df = df.sort_values('date').reset_index(drop=True)

print(" Dates parsed successfully.")
print("Date range:", df['date'].min(), "→", df['date'].max())
print("Total rows after date cleanup:", len(df))
print("Unique months found:", sorted(df['date'].dt.month.unique()))

# --------------------------
# 3. Clean numeric columns
# --------------------------
for col in ['gmv', 'users', 'marketing_cost']:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace("'", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['gmv', 'users', 'marketing_cost'])
print(" Cleaned numeric columns. Remaining rows:", len(df))

# --------------------------
# 4. Filter July–Dec 2024
# --------------------------
start_date = pd.Timestamp('2024-07-01')
end_date = pd.Timestamp('2024-12-31')

budget_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
if budget_df.empty:
    print(" No budget rows found to predict. Exiting.")
    exit()

print(
    f" Budget rows to send: {len(budget_df)} ({budget_df['date'].min()} → {budget_df['date'].max()})")

# --------------------------
# 5. Prepare for API
# --------------------------
rows = budget_df[['date', 'gmv', 'users', 'marketing_cost']].copy()
rows['date'] = rows['date'].dt.strftime('%Y-%m-%d')
rows = rows.to_dict(orient='records')

# --------------------------
# 6. POST request
# --------------------------
try:
    resp = requests.post(API_URL, json={'rows': rows})
    resp.raise_for_status()
    try:
        predictions = resp.json()
        print("\n Predictions received:")
        print(json.dumps(predictions, indent=2))
    except json.JSONDecodeError:
        print(" Flask did not return valid JSON. Response text:")
        print(resp.text)
except requests.exceptions.RequestException as e:
    print(" Error sending request to Flask API:")
    print(e)
