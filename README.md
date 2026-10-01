# 🚀 Predictive Revenue Engine & Anomaly Detection API

An end-to-end Machine Learning pipeline designed for SaaS and E-commerce environments. This system ingests daily web traffic and conversion metrics, detects anomalous behavior in real-time (e.g., bot attacks, gateway crashes), and forecasts expected revenue using time-series feature engineering.

## 🎯 Strategic Value Proposition

This project is built to demonstrate production-ready ML architecture, addressing three core business verticals:

1. **Revenue Operations & Funnel Analytics (Vonage approach)**
   - **Challenge:** Unpredictable revenue streams and conversion rate volatility.
   - **Solution:** Implemented a **LightGBM** forecasting model leveraging time-series lags and rolling windows to predict daily revenue. The architecture understands weekly business seasonality and acts as a compass for sales and marketing budget allocation.

2. **High-Volume Traffic & Infrastructure Security (Wikimedia approach)**
   - **Challenge:** Distinguishing organic viral traffic from malicious bot attacks or system outages.
   - **Solution:** Deployed an **Isolation Forest** unsupervised model to monitor the multidimensional relationship between traffic volume and conversion rates. It successfully flags severe anomalies (DDoS patterns, checkout crashes) without requiring prior labeled data, acting as an automated guardrail for traffic integrity.

3. **End-to-End MLOps Leadership (Global Executive approach)**
   - **Challenge:** Bridging the gap between Jupyter Notebooks and business consumption.
   - **Solution:** Designed a strictly modular, OOP-oriented Python codebase. The models are serialized and served through a **FastAPI** REST interface with strict Pydantic validation, ready for containerization (Docker) and dashboard integration (Power BI/Tableau).

## 🛠️ Architecture & Tech Stack

- **Data Engineering:** Python, Pandas, NumPy (Synthetic generation, seasonal trend injection).
- **Feature Engineering:** Time-based lags (`lag_7`), rolling windows (moving averages/std dev) to capture business inertia and prevent data leakage.
- **Machine Learning:** 
  - `scikit-learn` (Isolation Forest for multivariate anomaly detection).
  - `LightGBM` (Gradient Boosting for robust, high-performance time-series regression).
- **Serving & MLOps:** `FastAPI`, `Uvicorn`, `Pydantic` (Strict typing and interactive Swagger UI documentation).

## 🚀 How to Run Locally

1. **Clone the repository and set up the environment:**
   ```bash
   git clone [https://github.com/tu-usuario/revenue-anomaly-engine.git](https://github.com/tu-usuario/revenue-anomaly-engine.git)
   cd revenue-anomaly-engine
   pip install -r requirements.txt


2. **Deploy the API:**
    ```bash
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    Navigate to http://localhost:8000/docs to interact with the API via the Swagger UI.
   

## 📊 API Endpoints
**POST /predict:** Ingests daily operational metrics and returns a highly accurate revenue forecast (measured by MAPE).

**POST /detect:** Evaluates current traffic-to-conversion relationships to flag severe infrastructural or security anomalies.

**GET /:** Health check and model versioning.

## 🧠 Model Evaluation & Insights
The Isolation Forest successfully captures severe injected anomalies (bot traffic spikes with zero conversion, payment gateway downtimes) maintaining a high precision rate against standard operational variance.

The LightGBM model achieved a Mean Absolute Percentage Error (MAPE) of ~11% on the hold-out test set, primarily driven by 7-day conversion lags and recent traffic volume, proving highly reliable for short-term financial planning.
