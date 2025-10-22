# train_models.py

"""Tradeoff: 
    pickle is simple but not ideal for model versioning; 
    in production I'd use model registry or joblib + metadata.
"""
import pickle
import pandas as pd
from fetch_data import fetch_sheet
from features import make_features
from sklearn.ensemble import RandomForestRegressor

# Fetching the data
df = fetch_sheet()

"""
    Filling missing pod values with 0 (or drop rows without pods for training)
    This is important as supervised learning needs labels. We’re training on historical 
    days where pod counts are known (either observed or imputed historically).
"""
train_df = df.dropna(subset=['fe_pods', 'be_pods']).copy()

# Add month and rolling gmv mean features for BE variance
train_df['month'] = train_df['date'].dt.month
train_df['gmv_7d_avg'] = train_df['gmv'].rolling(7, min_periods=1).mean()

# Preparing features
X = make_features(train_df)

y_fe = train_df['fe_pods']
y_be = train_df['be_pods']

"""
    Training models with higher variance settings
    
    Instantiate two RandomForest regressors with 500 trees
    n_estimators=500:
        More trees stabilize predictions (lower variance), but increases training and inference time.
    max_depth=None:
        Trees can grow fully; with enough trees regularization is through averaging. May risk overfitting if training data is small.
    min_samples_leaf=1:
        Allows small leaves; again risks overfitting on small noisy datasets.
    random_state=42:
        Reproducibility.
"""
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
