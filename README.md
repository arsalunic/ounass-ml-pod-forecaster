# Ounass ML Pods Forecaster

**Goal:** Predict how many frontend (FE) and backend (BE) Kubernetes pods are needed each day based on business activity, things like GMV, user traffic, and marketing spend.  
The idea is to help DevOps plan capacity _before_ spikes hit production.

---

## What this project does (Quick Summary)

This repo connects to a Google Sheet (with both historical and budgeted data), cleans it up, builds features, trains two Random Forest models to predict FE and BE pods, and exposes an API endpoint `/predict` that returns daily pod forecasts for upcoming budget periods (July–Dec 2024).

You can run it end-to-end:  
`fetch` -> `clean` -> `train` -> `serve` -> `predict`.

---

## File-by-file breakdown

### **1. fetch_data.py**

**Purpose:**  
Pull data from a Google Sheet (exported as CSV), clean up the messy bits (dates, commas in numbers), and fill in missing pods for _historical_ rows so we can train the model.

**Inputs:**  
Google Sheets CSV (via `SHEET_CSV_URL`)

**Outputs:**

- A cleaned DataFrame returned to Python
- `cleaned_google_sheet.csv` saved locally
    

**What it does:**

- Parses dates robustly with `pd.to_datetime(..., dayfirst=True)`
- Cleans numbers (removes commas and quotes)
- Fills FE and BE pods **only** for rows _before June 2024_ using constants:
    - `FE_USERS_PER_POD = 3800`
    - `BE_GMV_PER_POD = 2_100_000`
    
    That means we don’t pre-fill pods for future (budgeted) rows - those are what the model will predict later.
    

**Why this matters:**  
If we fill in pods for budget days, the model wouldn’t have anything to learn from - it’d just memorize our constants. So we keep budget rows blank for prediction time.

---

### **2. features.py**

**Purpose:**  
Transform the cleaned data into something machine-learning-friendly.

**Inputs:**  
DataFrame with columns like `date`, `gmv`, `users`, `marketing_cost`, and optional `fe_pods`, `be_pods`.

**Outputs:**  
A new DataFrame with added feature columns — same number of rows, same order.

**Features generated:**

- `day_of_week`, `day_of_month`, `month`, `is_weekend`
- 7-day rolling averages for GMV, users, marketing cost
- The raw features themselves (no dropping!)
    

**Important design choice:**  
`make_features()` must **not** drop or reorder rows — the alignment with the original dataset is critical.  
If you drop rows during rolling averages, downstream predictions and merges will break.

---

### **3. train_models.py**

**Purpose:**  
Train two separate regressors — one for FE pods, one for BE pods.

**How it works:**

- Fetches the cleaned sheet data
- Filters for historical rows (where pod numbers exist)
- Generates features via `make_features()`
- Trains:
    `RandomForestRegressor(n_estimators=500, random_state=42)`
    
- Saves:
    - `models/pipe_fe.pkl`
    - `models/pipe_be.pkl`
        
**Why Random Forest?**

- Works great on small-to-medium tabular data
- Handles nonlinearities well
- Easy to interpret (feature importance)
- Stable and low-maintenance for early-stage models
    

**Alternatives we considered:**

- XGBoost or LightGBM (for larger data / faster training)
- Prophet or SARIMA (if we find strong time dependencies)
- Hybrid ensemble (future direction)

---

### **4. app.py**

**Purpose:**  
Expose a lightweight Flask API so DevOps (or any client) can hit `/predict` and get daily FE/BE pod numbers for a given budget period.

**How it works:**

- Accepts JSON input like:
    - `{   "rows": [     {"date": "2024-07-01", "gmv": 10162599, "users": 72673, "marketing_cost": 135196},     {"date": "2024-07-02", "gmv": 8738607, "users": 84784, "marketing_cost": 178427}   ] }`
- Cleans and merges these budget rows with historical data
- Builds features using `make_features()`
- Runs predictions using the trained models
- Post-processes results:
    - Ceil pod values (no decimals)
    - Adds tiny random variation to BE predictions (to reflect uncertainty)
- Returns:
    - `{   "predictions": [     {"date": "2024-07-01", "fe_pods": 14, "be_pods": 7},     ...   ] }`
    

**Gotchas we handled:**

- Row alignment: make sure features don’t shift rows.
- Only predict on budget rows (via `is_budget` flag).
- Random BE variance is intentional — gives a bit of uncertainty buffer for capacity planning.

---

### **5. sheet_fetch_and_predict.py**

**Purpose:**  
Utility script that does everything in one go:

- Fetch the Google Sheet CSV
- Parse and clean it
- Filter July–Dec rows
- POST them to the running Flask `/predict` endpoint
- Print the JSON output neatly

It’s the easiest way to test the full pipeline locally.

---

### **6. predict_local.py**

**Purpose:**  
Offline runner.  
Loads `cleaned_google_sheet.csv`, trains/predicts locally without Flask, and writes `predicted_budget_pods.csv`.  
Handy for debugging or quick experiments.

---

### **7. requirements.txt**

Lists all dependencies:

- idna==3.11
- rsa==4.9.1
- six==1.17.0
- Flask==2.3.3
- click==8.3.0
- pytz==2025.2
- scipy==1.16.2
- Jinja2==3.1.6
- joblib==1.5.2
- pandas==2.2.2
- pyasn1==0.6.1
- numpy==1.26.4
- xgboost==2.0.3
- urllib3==2.5.0
- blinker==1.9.0
- cycler==0.12.1
- pillow==12.0.0
- tzdata==2025.2
- seaborn==0.13.2
- Werkzeug==3.1.3
- gspread==5.12.0
- oauthlib==3.3.1
- packaging==25.0
- pyparsing==3.2.5
- protobuf==6.33.0
- httplib2==0.31.0
- gunicorn==21.2.0
- contourpy==1.3.3
- cachetools==6.2.1
- requests==2.32.5
- fonttools==4.60.1
- kiwisolver==1.4.9
- MarkupSafe==3.0.3
- matplotlib==3.8.4
- uritemplate==4.2.0
- certifi==2025.10.5
- proto-plus==1.26.1
- itsdangerous==2.2.0
- google-auth==2.41.1
- scikit-learn==1.4.2
- python-dotenv==1.0.1
- threadpoolctl==3.6.0
- pyasn1_modules==0.4.2
- google-api-core==2.26.0
- requests-oauthlib==2.0.0
- charset-normalizer==3.4.4
- google-auth-httplib2==0.2.0
- google-auth-oauthlib==1.2.2
- python-dateutil==2.9.0.post0
- googleapis-common-protos==1.70.0
- google-api-python-client==2.127.0


Python 3.10+ recommended. Use a virtualenv to isolate installs.

---

## Model Calibration (June Constants)

Before training and forecasting, I performed a quick calibration step using actual June data to make sure our simple rule-based pod estimation formulas were realistic.

### Original Formulas

I started with:

FE_pods = users / FE_USERS_PER_POD  

BE_pods = GMV / BE_GMV_PER_POD

  

These formulas assume I can estimate the required number of frontend and backend pods based on user count and GMV, respectively.

### Adjusting Constants from June Data

I computed new constants so the formulas match real June pod usage on average.

Frontend constant

FE_USERS_PER_POD = mean(users / FE_pods)


Backend constant

BE_GMV_PER_POD = mean(GMV / BE_pods)


### June Data Snapshot (June 16–30)

|   |   |   |   |   |
|---|---|---|---|---|
|Date|GMV|Users|FE Pods|BE Pods|
|16/06|19,364,132|74,540|20|9|
|17/06|18,558,648|73,152|20|9|
|…|…|…|…|…|
|30/06|19,191,123|86,558|20|9|

From these values:

- users / fe_pods averaged ≈ 3,800 users per FE pod
- GMV / be_pods averaged ≈ 2,100,000 GMV per BE pod  


### Updated Constants

FE_USERS_PER_POD = 3800       # ~1 frontend pod per 3.8 K users
BE_GMV_PER_POD   = 2_100_000  # ~1 backend pod per $2.1 M GMV

**



---


## How to run locally

### 1. Clone and setup

`git clone <repo-url> cd <repo> python -m venv venv source venv/bin/activate pip install -r requirements.txt`

### 2. Train models

`python train_models.py`

### 3. Start the API

`python app.py # running on http://127.0.0.1:8080`

### 4. Test predictions

`python sheet_fetch_and_predict.py`

or directly vis an `API`:

`curl -X POST http://127.0.0.1:8080/predict \ 
-H "Content-Type: application/json" \ 
-d '{"rows":[{"date":"2024-07-01","gmv":10162599,"users":72673,"marketing_cost":135196}]}'`

---

## Common Issues & Fixes

| Problem                     | Likely Cause                   | Fix                                                           |
| --------------------------- | ------------------------------ | ------------------------------------------------------------- |
| Missing days or weird dates | CSV formatting or quotes       | Use smart parsing (we strip quotes and use `dateutil.parser`) |
| --------------------------- | ------------------------------ | ------------------------------------------------------------- |
| Mismatch in row counts      | `make_features()` dropped rows | Never drop rows — preserve order                              |
| --------------------------- | ------------------------------ | ------------------------------------------------------------- |
| Float pods (e.g. 2.0)       | Model outputs floats           | Use `math.ceil()` or cast to int                              |
| --------------------------- | ------------------------------ | ------------------------------------------------------------- |
| Random BE pods              | Small noise added on purpose   | Remove `np.random.choice()` if you want deterministic output  |
| --------------------------- | ------------------------------ | ------------------------------------------------------------- |


---

## 💡 Why this approach works

We treat this as a **tabular regression** problem:  
Pods are determined by _business demand metrics_, not just time, so Random Forest works well here.  
If later we find stronger time-seasonality patterns, we can hybridize with a time-series model.

---

## Next Steps (Future Improvements)

- Add model versioning + metrics
- Dockerize the API
- Add authentication for `/predict`
- Integrate CI/CD for retraining
- Introduce uncertainty bounds (e.g. 90th percentile)
- Monitoring: detect drift in GMV/traffic vs training data
- Optionally replace RF with LightGBM for faster inference
    

---
