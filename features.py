# features.py

def make_features(df):

    df = df.sort_values('date').reset_index(drop=True).copy()

    # Date-based features

    """
        day_of_week (0–6): 
            Captures weekly patterns — traffic often differs on weekends vs weekdays and specific weekdays (e.g., Monday calm, weekend spikes).
        day_of_month: 
            Can help capture month-end behaviors (e.g., payroll, salary day, billing cycles).
        is_weekend: 
            A binary that compresses weekend signal; simpler for some models.
        month: 
            Coarse seasonality (Jul–Dec budget period; month captures monthly seasonality).
    """

    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month

    # Rolling features (this ensures no leakage across unrelated segments)

    """
        gmv:
            Core business volume; BE pods often correlate with GMV-driven backend load (order processing, payments).
        users:
            Front-end load correlates with concurrent user counts and sessions (affects FE pods).
        marketing_cost:
            Proxy for expected campaign-driven traffic (spikes from ad spend).
        gmv_7d_avg, users_7d_avg, marketing_7d_avg: 
            smooth short-term noise and capture recent trends. Rolling means reduce volatility and help the model generalize (e.g., last-week baseline).
    """

    df['gmv_7d_avg'] = df['gmv'].rolling(7, min_periods=1).mean()
    df['users_7d_avg'] = df['users'].rolling(7, min_periods=1).mean()
    df['marketing_7d_avg'] = df['marketing_cost'].rolling(
        7, min_periods=1).mean()

    feature_cols = [
        'gmv', 'users', 'marketing_cost',
        'day_of_week', 'day_of_month', 'is_weekend', 'month',
        'gmv_7d_avg', 'users_7d_avg', 'marketing_7d_avg'
    ]

    return df[feature_cols]
