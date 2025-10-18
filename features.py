# features.py
def make_features(df):
    
    df = df.sort_values('date').reset_index(drop=True).copy()

    # Date-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month

    # Rolling features (ensure no leakage across unrelated segments)
    df['gmv_7d_avg'] = df['gmv'].rolling(7, min_periods=1).mean()
    df['users_7d_avg'] = df['users'].rolling(7, min_periods=1).mean()
    df['marketing_7d_avg'] = df['marketing_cost'].rolling(7, min_periods=1).mean()

    feature_cols = [
        'gmv', 'users', 'marketing_cost',
        'day_of_week', 'day_of_month', 'is_weekend', 'month',
        'gmv_7d_avg', 'users_7d_avg', 'marketing_7d_avg'
    ]

    return df[feature_cols]

