# Veritas Risk Operations: Real-Time Fraud Detection System

A production-minded, end-to-end **Real-Time Fraud Detection System** and risk analytics platform built with Python, Scikit-learn, XGBoost, SQLite, and Flask.

Designed with an internal banking risk operations interface: dark navy/slate navigation, neutral high-density data tables, restrained teal actions, muted alert states, and explainable ML scoring.

> **DISCLAIMER:** All transaction streams, user profiles, merchants, and ground-truth fraud labels generated in this repository are **purely synthetic simulations** designed for machine learning experimentation and operational demonstration. They do **not** represent real banking, credit card, or customer financial records.

---

## 1. System Architecture & Pipeline Workflow

```
[Synthetic Transaction Generator]
               │ (Realistic User Profiles & Controlled Fraud Anomalies)
               ▼
   [Data Validation & Cleaning]
               │ (Schema Validation, Type Enforcing, Label Sanitization)
               ▼
     [Feature Engineering]
               │ (Rolling Velocity 1h/24h, User Spending Ratio, Geo/Device Flags)
               ▼
[Machine Learning Model Pipeline]
               │ (Fitted ColumnTransformer + Logistic Regression / Random Forest / XGBoost)
               ▼
[Cost-Sensitive Threshold Tuning]
               │ (Minimizes Asymmetric Financial Losses: False Negatives vs False Positives)
               ▼
   [Flask Inference REST API]  ◄───►  [SQLite3 Database (WAL Mode)]
               │ (Sub-15ms Latency, Zero Startup Retraining)
               ▼
[Real-Time Risk Dashboard (SSE)]
  ├── KPI Metrics & Live Transaction Feed
  ├── Hourly Volume & Risk Distribution Charts
  ├── Deep Forensic Transaction Inspection
  └── Continuous Simulation Start/Stop Controls
```

---

## 2. Key Capabilities

- **Zero Target & Temporal Leakage:** Preprocessing transformations fit strictly on training splits. Operational feature extraction uses strictly past rolling windows (`closed="left"`). Inference features are strictly decoupled from synthetic labels.
- **Cost-Sensitive Threshold Optimization:** Replaces the arbitrary 0.50 threshold with an economic cost objective:
  $$\text{Loss} = (\text{FN} \times C_{\text{FN}}) + (\text{FP} \times C_{\text{FP}})$$
  where $C_{\text{FN}} = \$350$ (stolen funds, chargeback fees) and $C_{\text{FP}} = \$20$ (customer friction, investigation).
- **Sub-15ms Inference Latency:** Pre-cached singleton model instance in memory, indexed SQLite queries, and streamlined vectorization.
- **Continuous Transaction Stream Simulation:** Background multi-threaded generator emits realistic transactions with controlled fraud anomalies (spending spikes, velocity bursts, impossible geographic travel, proxy devices).
- **Zero Startup Retraining:** Loads pre-trained serialized model bundles (`models/model_artifacts.joblib`) on application startup.

---

## 3. Directory Structure

```
Real_Time_Fraud_Detection/
├── config.py                     # Central configuration & hyperparameters
├── wsgi.py                       # WSGI entrypoint (Gunicorn / Waitress)
├── requirements.txt              # Production & test dependencies
├── .env.example                  # Environment configuration template
├── .gitignore                    # Version control excludes
├── README.md                     # Documentation & setup guide
├── fraud_detection/
│   ├── __init__.py
│   ├── database.py               # SQLite schema, queries, KPIs, and indexing
│   ├── generator.py              # Synthetic transaction & user baseline generator
│   ├── features.py               # Leak-free feature extraction & ColumnTransformer
│   ├── train.py                  # ML model training, benchmarking & threshold tuner
│   ├── evaluate.py               # PR-AUC, ROC-AUC, cost curves, confusion matrices
│   ├── inference.py              # Real-time scoring engine & explainability rules
│   ├── simulator.py              # Background streaming simulation & SSE broadcast
│   ├── app.py                    # Flask application factory, views & REST APIs
│   ├── templates/                # Banking operations HTML templates
│   │   ├── base.html             # Shell with slate sidebar & status bar
│   │   ├── dashboard.html        # Live transaction table, KPIs & charts
│   │   ├── transactions.html     # Historical transaction explorer & filters
│   │   ├── transaction_detail.html # Deep-dive forensic risk analysis
│   │   ├── models.html           # ML model comparisons & ROC/PR curves
│   │   ├── monitoring.html       # System latency & simulated drift telemetry
│   │   └── 404.html              # Clean 404 error template
│   └── static/
│       ├── css/style.css         # Professional banking UI stylesheet
│       └── js/dashboard.js       # Live SSE streaming, charts, and modal controls
├── models/                       # Serialized model bundles & comparison metadata
│   ├── model_artifacts.joblib
│   └── model_metadata.json
├── data/                         # SQLite database storage
│   └── fraud_detection.db
└── tests/                        # Comprehensive unit & integration test suite
    ├── test_validation.py
    ├── test_features.py
    ├── test_inference.py
    └── test_api.py
```

---

## 4. Quickstart & Installation

### Step 1: Clone and Set Up Virtual Environment

```bash
# Using standard Python
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or using uv (recommended):
uv sync
```

### Step 2: Environment Configuration

Copy the sample environment file:
```bash
cp .env.example .env
```

### Step 3: Run the Training Pipeline

Generate the synthetic baseline dataset, train the candidate models (Logistic Regression, Random Forest, XGBoost), tune the cost-optimal decision threshold, and persist the winning model:

```bash
python -m fraud_detection.train
```

### Step 4: Run the Test Suite

Verify validation, features, scoring, and Flask endpoints:
```bash
pytest
```

### Step 5: Start the Application

**Development Server:**
```bash
python -m fraud_detection.app
```

**Production WSGI Server (Windows & Cross-Platform via Waitress):**
```bash
python wsgi.py
```

**Production WSGI Server (Linux / Docker via Gunicorn):**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

Open your browser at: **`http://localhost:5000`**

---

## 5. REST API Reference

### Health Check
`GET /health`
```json
{
  "status": "healthy",
  "uptime_seconds": 128.4,
  "model_status": {
    "is_loaded": true,
    "model_name": "XGBoost Classifier",
    "threshold": 0.42
  },
  "database_connected": true,
  "version": "1.0.0"
}
```

### Real-Time Transaction Prediction
`POST /api/predict`

**Request Body:**
```json
{
  "user_id": "USR_0042",
  "amount": 1250.00,
  "merchant": "Apple Store Fifth Ave",
  "merchant_category": "Electronics",
  "location": "Lagos, NG",
  "device": "Tor Browser / Linux",
  "payment_method": "Credit Card",
  "channel": "Online"
}
```

**Response Body (200 OK):**
```json
{
  "transaction_id": "TXN_837492817291",
  "user_id": "USR_0042",
  "amount": 1250.00,
  "timestamp": "2026-10-24T12:45:00+00:00",
  "risk_score": 0.8842,
  "risk_tier": "CRITICAL",
  "decision": "DECLINE",
  "is_fraud_predicted": 1,
  "risk_reasons": [
    "Spending Spike: Amount is 12.5x higher than user average",
    "Geographic Anomaly: Transacting outside profile home (Lagos, NG)",
    "Unrecognized Device: Fingerprint mismatch (Tor Browser / Linux)",
    "High-Risk Merchant Category: Electronics"
  ],
  "latency_ms": 7.8,
  "model_version": "XGBoost Classifier",
  "decision_threshold": 0.42
}
```

### Simulation Controls
- `POST /api/simulation/start` — `{ "interval_sec": 1.0, "fraud_rate": 0.06 }`
- `POST /api/simulation/stop`
- `GET /api/simulation/status`
- `GET /api/stream` — Server-Sent Events (SSE) feed

---

## 6. Cloud Deployment Guide (Render & Railway)

The repository includes turnkey deployment configurations:
- [`Procfile`](file:///c:/Users/hp/OneDrive/Desktop/New%20folder/Procfile): Multi-threaded Gunicorn worker entry point.
- [`build.sh`](file:///c:/Users/hp/OneDrive/Desktop/New%20folder/build.sh): Zero-downtime installation, pre-trained model verification, and DB init.
- [`render.yaml`](file:///c:/Users/hp/OneDrive/Desktop/New%20folder/render.yaml): Render Blueprint with persistent disk specification.
- [`railway.json`](file:///c:/Users/hp/OneDrive/Desktop/New%20folder/railway.json): Railway Nixpacks build and deploy settings.

---

### Option A: Deploy to Render (Recommended)

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy Veritas Real-Time Fraud Detection System"
   git push origin main
   ```
2. **Create Web Service on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com) and click **New + > Web Service**.
   - Select your GitHub repository: `Kulwinder-Kumar-098/Real_Time_Fraud_Detection`.
   - Configure settings:
     - **Name**: `veritas-fraud-detection`
     - **Runtime**: `Python`
     - **Build Command**: `bash build.sh`
     - **Start Command**: `gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT wsgi:app`
     - **Health Check Path**: `/health`
   - Under **Advanced > Environment Variables**, add:
     - `PYTHON_VERSION`: `3.11.9`
     - `SECRET_KEY`: (Click Generate)
     - `DEFAULT_DECISION_THRESHOLD`: `0.94`
   - Under **Disks** (optional, for persistent SQLite across redeploys):
     - Mount Path: `/app/data` (1 GB)
   - Click **Deploy Web Service**.

---

### Option B: Deploy to Railway

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy Veritas Real-Time Fraud Detection System"
   git push origin main
   ```
2. **Deploy on Railway**:
   - Go to [railway.app](https://railway.app) and click **New Project > Deploy from GitHub repo**.
   - Select `Kulwinder-Kumar-098/Real_Time_Fraud_Detection`.
   - Railway will automatically detect [`railway.json`](file:///c:/Users/hp/OneDrive/Desktop/New%20folder/railway.json) and [`Procfile`](file:///c:/Users/hp/OneDrive/Desktop/New%20folder/Procfile).
   - In Settings > Networking, click **Generate Domain** to get your public HTTPS URL.
   - Your live service will be running at `https://<your-railway-app>.up.railway.app`!

---

## 7. Limitations & Real-World Considerations

- **Synthetic vs Production Data:** Real card transaction networks exhibit complex concepts such as card-present EMV cryptograms, 3DS authentication challenges, and merchant chargeback feedback lag (often 30–90 days).
- **Drift Detection:** In live banking systems with delayed ground truth, statistical drift must be monitored via unsupervised covariates (Population Stability Index / Wasserstein distance on input features) rather than instant label comparisons.
