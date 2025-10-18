# train_models.py

import pickle
import pandas as pd
from fetch_data import fetch_sheet
from features import make_features
from sklearn.ensemble import RandomForestRegressor

# Fetching the data
df = fetch_sheet()

# Filling missing pod values with 0 (or drop rows without pods for training)
train_df = df.dropna(subset=['fe_pods', 'be_pods']).copy()

# Add extra features for BE variance
train_df['month'] = train_df['date'].dt.month
train_df['gmv_7d_avg'] = train_df['gmv'].rolling(7, min_periods=1).mean()

# Preparing features
X = make_features(train_df)

y_fe = train_df['fe_pods']
y_be = train_df['be_pods']

# Training models with higher variance settings
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

# Saving models to the respective pickle files
with open('models/pipe_fe.pkl', 'wb') as f:
    pickle.dump(fe_model, f)

with open('models/pipe_be.pkl', 'wb') as f:
    pickle.dump(be_model, f)

print("Models trained and saved successfully!")
