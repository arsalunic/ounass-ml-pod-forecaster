# predict_local.py

import math
import pickle
import numpy as np
import pandas as pd
from features import make_features

# CONFIG for July to Dec
BUDGET_START_DATE = pd.Timestamp('2024-07-01')
BUDGET_END_DATE = pd.Timestamp('2024-12-31')

# Loading trained models
with open('models/pipe_fe.pkl', 'rb') as f:
    fe_model = pickle.load(f)
with open('models/pipe_be.pkl', 'rb') as f:
    be_model = pickle.load(f)

# Fetching cleaned historical data:
# Make sure the cleaned_google_sheet.csv is available in the directory
df_clean = pd.read_csv("cleaned_google_sheet.csv")


# Checking for rows with null dates before they were dropped
raw_df = df_clean.copy()
bad_dates = raw_df[~pd.to_datetime(
    raw_df['date'], dayfirst=True, errors='coerce').notna()]
print(f"\n Rows dropped due to bad dates: {len(bad_dates)}")
if not bad_dates.empty:
    print(bad_dates[['date']].head(10))

# Ensuring date column is proper datetime
df_clean['date'] = pd.to_datetime(
    df_clean['date'], errors='coerce', infer_datetime_format=True)

# Dropping any rows where date parsing failed
df_clean = df_clean.dropna(subset=['date'])

# Sorting chronologically
df_clean = df_clean.sort_values('date').reset_index(drop=True)

print("#-------- DEBUG-------#")
print("\n After cleaning:")
print("Total valid rows:", len(df_clean))
print("Date range (cleaned):",
      df_clean['date'].min(), ":", df_clean['date'].max())
print("Total budget rows in CSV:", df_clean[(df_clean['date'] >= BUDGET_START_DATE) &
                                            (df_clean['date'] <= BUDGET_END_DATE)].shape[0])

# Filtering budget rows
df_budget = df_clean[(df_clean['date'] >= BUDGET_START_DATE) &
                     (df_clean['date'] <= BUDGET_END_DATE)].copy()

if df_budget.empty:
    print("No budget rows found in cleaned data. Exiting.")
    exit()

df_budget['is_budget'] = True
df_hist = df_clean[df_clean['date'] < BUDGET_START_DATE].copy()
df_hist['is_budget'] = False

# Combining historical + budget
combined_df = pd.concat([df_hist, df_budget], ignore_index=True)
combined_df = combined_df.sort_values('date').reset_index(drop=True)

# Feature engineering
X_combined = make_features(combined_df)
X_combined = X_combined.reset_index(drop=True)

# Masking for budget rows
budget_mask = combined_df['is_budget'].values

# Predictions
fe_pred_all = fe_model.predict(X_combined)
be_pred_all = be_model.predict(X_combined)

# Budget-only predictions
fe_pred_budget = fe_pred_all[budget_mask]
be_pred_budget = be_pred_all[budget_mask]

# Round and enforce min 1
fe_pred_budget = [max(1, int(math.ceil(x))) for x in fe_pred_budget]
be_pred_budget = [
    max(1, int(
        math.ceil(x + np.random.choice([0, -1, 1, 2], p=[0.5, 0.2, 0.2, 0.1]))))
    for x in be_pred_budget
]

# Output
df_budget_out = df_budget.copy()
df_budget_out['fe_pods'] = fe_pred_budget
df_budget_out['be_pods'] = be_pred_budget

# Ensuring chronological order
df_budget_out = df_budget_out.sort_values('date').reset_index(drop=True)

# Printing output
print(df_budget_out[['date', 'fe_pods', 'be_pods']])

# Saving to CSV
df_budget_out[['date', 'gmv', 'users', 'marketing_cost', 'fe_pods',
               'be_pods']].to_csv('predicted_budget_pods.csv', index=False)
print("\nPredictions saved to 'predicted_budget_pods.csv'")
