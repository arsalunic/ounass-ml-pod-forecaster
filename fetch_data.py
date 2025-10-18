# fetch_data.py

import io
import requests
import pandas as pd

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSKsxrQBqPOQdF_KOsD3ub81wynRnXX6Pw0BxRsDikzXZgEQOoEsTS0ILD7xnNNeBTOKd7xFCFmtsqM/pub?output=csv"


def fetch_sheet():
    # 1. Fetching the data from Google sheet
    resp = requests.get(SHEET_CSV_URL)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    # 2. Converting Data Column
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # 3. Cleaning NUMERIC COLUMNS
    for col in ['gmv', 'users', 'marketing_cost']:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .astype(float)
        )

    # 4. Constants for POD Estimation
    FE_USERS_PER_POD = 3800    # 1 frontend pod per 3800 users
    BE_GMV_PER_POD = 2_100_000  # 1 backend pod per $2.1M GMV

    # 5. Filling pods fir Historical data only
    historical_mask = df['date'] < pd.to_datetime("2024-06-01")

    df.loc[historical_mask, 'fe_pods'] = df.loc[historical_mask, 'fe_pods'].fillna(
        (df.loc[historical_mask, 'users'] /
         FE_USERS_PER_POD).apply(lambda x: max(1, round(x)))
    ).astype(int)

    df.loc[historical_mask, 'be_pods'] = df.loc[historical_mask, 'be_pods'].fillna(
        (df.loc[historical_mask, 'gmv'] /
         BE_GMV_PER_POD).apply(lambda x: max(1, round(x)))
    ).astype(int)

    # 6. Saving CLEANED DATA to SCV fotr investigation
    df.to_csv("cleaned_google_sheet.csv", index=False)

    return df


if __name__ == "__main__":
    df = fetch_sheet()
    print(df.tail(20))
