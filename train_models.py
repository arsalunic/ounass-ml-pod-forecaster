# train_models.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from fetch_data import fetch_sheet
from features import make_features

# 1. Fetch data
df = fetch_sheet()

# 2. Fill missing pod values with 0 (or drop rows without pods for training)
train_df = df.dropna(subset=['fe_pods', 'be_pods']).copy()

# 3. Add extra features for BE variance
train_df['month'] = train_df['date'].dt.month
train_df['gmv_7d_avg'] = train_df['gmv'].rolling(7, min_periods=1).mean()

# 4. Prepare features
X = make_features(train_df)

y_fe = train_df['fe_pods']
y_be = train_df['be_pods']

# 5. Train models with higher variance settings
fe_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

be_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

fe_model.fit(X, y_fe)
be_model.fit(X, y_be)

# 6. Save models
with open('models/pipe_fe.pkl', 'wb') as f:
    pickle.dump(fe_model, f)

with open('models/pipe_be.pkl', 'wb') as f:
    pickle.dump(be_model, f)

print("Models trained and saved successfully!")
