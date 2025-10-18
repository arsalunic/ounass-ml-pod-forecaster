# app.py

import math
import pickle
import numpy as np
import pandas as pd
from features import make_features
from fetch_data import fetch_sheet
from flask import Flask, request, jsonify

app = Flask(__name__)

# Loading trained models
with open('models/pipe_fe.pkl', 'rb') as f:
    fe_model = pickle.load(f)
with open('models/pipe_be.pkl', 'rb') as f:
    be_model = pickle.load(f)


# /predict API
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'ok'})  # health check

    # Reading INPUT BUDGET data
    input_budget = pd.DataFrame(request.get_json(force=True).get('rows', []))
    if input_budget.empty:
        return jsonify({'error': 'provide rows as list of objects'}), 400

    # Cleaning the input
    input_budget['date'] = pd.to_datetime(input_budget['date'].astype(
        str).str.strip(), dayfirst=True, errors='coerce')
    for col in ['gmv', 'users', 'marketing_cost']:
        input_budget[col] = pd.to_numeric(input_budget[col].astype(
            str).str.replace(',', '', regex=False), errors='coerce')
    input_budget = input_budget.dropna(
        subset=['date', 'gmv', 'users', 'marketing_cost'])

    # Creating the full date range
    pred_start = pd.Timestamp('2024-07-01')
    pred_end = pd.Timestamp('2024-12-31')
    all_dates = pd.date_range(pred_start, pred_end, freq='D')
    df_budget = pd.DataFrame({'date': all_dates})

    # Merging input budget onto full date range
    df_budget = df_budget.merge(input_budget, on='date', how='left')

    # Filling missing numeric values (forward fill, then 0)
    for col in ['gmv', 'users', 'marketing_cost']:
        if col in df_budget:
            df_budget[col] = df_budget[col].fillna(method='ffill').fillna(0)

    df_budget['is_budget'] = True

    # Historical Data
    df_hist = fetch_sheet()
    df_hist = df_hist.dropna(subset=['fe_pods', 'be_pods'])
    df_hist['date'] = pd.to_datetime(
        df_hist['date'], dayfirst=True, errors='coerce')
    df_hist = df_hist.dropna(subset=['date'])
    df_hist['is_budget'] = False

    # COMBINING FOR FEATURE ENGINEERING
    combined_df = pd.concat([df_hist, df_budget], ignore_index=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)

    # --- FEATURE ENGINEERING ---
    X_combined = make_features(combined_df)
    combined_df = combined_df.reset_index(drop=True)
    X_combined = X_combined.reset_index(drop=True)

    # --- PREDICTIONS ---
    fe_pred_all = fe_model.predict(X_combined)
    be_pred_all = be_model.predict(X_combined)

    # Only keep budget rows
    budget_mask = combined_df['is_budget'].values
    fe_pred_budget = fe_pred_all[budget_mask]
    be_pred_budget = be_pred_all[budget_mask]

    # Post-process predictions
    fe_pred_budget = [max(1, int(math.ceil(x))) for x in fe_pred_budget]
    be_pred_budget = [
        max(1, int(
            math.ceil(x + np.random.choice([0, -1, 1, 2], p=[0.5, 0.2, 0.2, 0.1]))))
        for x in be_pred_budget
    ]

    # Build response
    df_budget_sorted = df_budget.sort_values('date').reset_index(drop=True)
    out = []
    for i, row in df_budget_sorted.iterrows():
        out.append({
            'date': str(row['date'].date()),
            'fe_pods': fe_pred_budget[i],
            'be_pods': be_pred_budget[i]
        })

    return jsonify({'predictions': out})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
