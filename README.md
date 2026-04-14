<h1 align="center">📉 Customer Churn Prediction</h1>
<h3 align="center">End-to-End Machine Learning System</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/DVC-3.50-945DD6?logo=dvc&logoColor=white" />
  <img src="https://img.shields.io/badge/DagsHub-Tracking-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p align="center">
  A production-ready, end-to-end machine learning pipeline that predicts whether a telecom customer will churn — from raw data all the way to a live REST API.
</p>

---

## 📋 Table of Contents

- [🔍 Project Overview](#-project-overview)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [🔄 Pipeline Stages](#-pipeline-stages)
- [📊 Models & Metrics](#-models--metrics)
- [🌐 API Reference](#-api-reference)
- [🧪 Experiment Tracking](#-experiment-tracking)
- [📦 Data Versioning with DVC](#-data-versioning-with-dvc)
- [🤝 Contributing](#-contributing)

---

## 🔍 Project Overview

Customer churn — when a customer stops using a service — is one of the most critical challenges in the telecom industry. This project builds a **fully automated ML pipeline** that:

- 📥 Ingests and cleans the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- 🛠️ Engineers meaningful features (tenure groups, charge ratios, encoding & scaling)
- 🤖 Trains and compares **3 classifiers**: Logistic Regression, LightGBM, and XGBoost
- 🎯 Tunes hyperparameters with **Optuna**
- 📈 Evaluates models and registers the best one to **MLflow Model Registry**
- ⚡ Serves real-time predictions via a **FastAPI** REST API

---

## 🏗️ Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────────┐
│  Feature Engineering │  ← Cleaning, Encoding, Scaling, Train/Test Split
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Model Training     │  ← Logistic Regression | LightGBM | XGBoost
│   (MLflow tracked)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Hyperparameter      │  ← Optuna (50 trials)
│  Tuning              │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Evaluation &        │  ← Best model → MLflow Model Registry → Production
│  Registration        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  FastAPI REST API    │  ← /predict  |  /predict_batch
└─────────────────────┘
```

---

## 📁 Project Structure

```
Customer-chunk-prediction-end-to-end-ml-system/
│
├── src/
│   ├── data_ingestion.py       # Load & inspect raw data
│   ├── feature_engineering.py  # Clean, encode, scale, split & save
│   ├── train.py                # Train 3 models with MLflow logging
│   ├── tune.py                 # Hyperparameter tuning with Optuna
│   ├── evaluate.py             # Evaluate best model & register to MLflow
│   └── app.py                  # FastAPI prediction API
│
├── data/
│   ├── raw/                    # Original Telco CSV (DVC tracked)
│   └── processed/              # Train/test splits (DVC tracked)
│
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory Data Analysis
│
├── evaluation_results/         # Metrics JSON & confusion matrix plots
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC pipeline lock file
├── params.yaml                 # Model hyperparameters
├── best_lgbm_params.yaml       # Best LightGBM params from tuning
├── environment.yml             # Conda environment
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

---

## ⚙️ Tech Stack

| Category               | Tools & Libraries                                    |
|------------------------|------------------------------------------------------|
| **Language**           | Python 3.12                                          |
| **ML / Modeling**      | scikit-learn, XGBoost, LightGBM                      |
| **Hyperparameter Tuning** | Optuna                                            |
| **Experiment Tracking**| MLflow, DagsHub                                      |
| **Data Versioning**    | DVC (with S3/HTTP remote)                            |
| **API Serving**        | FastAPI, Uvicorn                                     |
| **Data Processing**    | Pandas, NumPy                                        |
| **Visualization**      | Matplotlib, Seaborn                                  |
| **Config / Env**       | python-dotenv, PyYAML                                |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shovo896/Customer-chunk-prediction-end-to-end-ml-system.git
cd Customer-chunk-prediction-end-to-end-ml-system
```

### 2. Create & Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate churn-env
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in your DagsHub credentials
```

| Variable               | Description                              |
|------------------------|------------------------------------------|
| `DAGSHUB_USERNAME`     | Your DagsHub username                    |
| `DAGSHUB_REPO`         | Repository name on DagsHub              |
| `DAGSHUB_USER_TOKEN`   | DagsHub personal access token            |
| `MLFLOW_TRACKING_URI`  | Auto-set via DagsHub MLflow integration  |

### 4. Pull Data with DVC

```bash
dvc pull
```

### 5. Run the Full Pipeline

```bash
dvc repro
```

> This runs all stages: feature engineering → training → evaluation.

---

## 🔄 Pipeline Stages

The DVC pipeline (`dvc.yaml`) defines three reproducible stages:

### Stage 1 — Feature Engineering

```bash
python src/feature_engineering.py
```

- Loads raw Telco CSV and handles missing values in `TotalCharges`
- Engineers `tenure_group` (binned) and `charge_ratio` features
- Encodes binary/categorical columns and applies `StandardScaler`
- Splits data (80/20) and saves to `data/processed/`

### Stage 2 — Model Training

```bash
python src/train.py
```

- Trains **Logistic Regression**, **LightGBM**, and **XGBoost**
- Logs parameters, metrics, and model artifacts to **MLflow via DagsHub**
- Tracks: `accuracy`, `f1_score`, `roc_auc`

### Stage 3 — Evaluation & Registration

```bash
python src/evaluate.py
```

- Fetches the best run by `roc_auc` from MLflow
- Generates a **Confusion Matrix** plot saved to `evaluation_results/`
- Saves metrics as JSON
- Registers the best model to **MLflow Model Registry** and transitions it to `Production`

### (Optional) Hyperparameter Tuning

```bash
python src/tune.py
```

- Uses **Optuna** to run 50 trials for XGBoost
- Each trial is logged to MLflow as a nested run
- Best params are saved to `best_lgbm_params.yaml`

---

## 📊 Models & Metrics

Three classifiers are trained and compared:

| Model                 | Key Hyperparameters                        |
|-----------------------|--------------------------------------------|
| Logistic Regression   | `C=1.0`, `max_iter=100`                    |
| XGBoost               | `n_estimators=100`, `learning_rate=0.1`    |
| LightGBM              | `n_estimators=100`, `learning_rate=0.1`    |

**Evaluation Metrics:**

- ✅ **Accuracy** — overall correctness
- ✅ **F1 Score** — balance of precision & recall (important for imbalanced churn data)
- ✅ **ROC-AUC** — model's ability to separate churners from non-churners

The model with the highest `roc_auc` is automatically selected and promoted to Production.

---

## 🌐 API Reference

Start the API server:

```bash
python src/app.py
```

The server starts on `http://localhost:8001`. Interactive docs at `http://localhost:8001/docs`.

---

### `GET /`

Returns a welcome message.

```json
{
  "message": "Welcome to Customer Churn Prediction API",
  "docs": "/docs"
}
```

---

### `GET /health`

Health check endpoint.

```json
{ "status": "ok" }
```

---

### `POST /predict`

Predicts churn for a single customer.

**Request Body:**

```json
{
  "tenure": 12,
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0,
  "SeniorCitizen": 0,
  "Partner": 1,
  "gender": 1,
  "Dependents": 0,
  "PhoneService": 1,
  "PaperlessBilling": 1,
  "charge_ratio": 0.083
}
```

**Response:**

```json
{
  "churn_probability": 0.7321,
  "churn_prediction": 1,
  "risk_level": "High",
  "message": "Prediction successful"
}
```

| `risk_level` | Probability Range |
|--------------|-------------------|
| `Low`        | ≤ 0.4             |
| `Medium`     | 0.4 – 0.7         |
| `High`       | > 0.7             |

---

### `POST /predict_batch`

Predicts churn for multiple customers at once.

**Request Body:** Array of customer objects (same schema as `/predict`).

**Response:**

```json
{
  "predictions": [
    { "churn_probability": 0.72, "churn_prediction": 1, "risk_level": "High" },
    { "churn_probability": 0.21, "churn_prediction": 0, "risk_level": "Low" }
  ],
  "message": "Batch prediction successful"
}
```

---

## 🧪 Experiment Tracking

All experiments are tracked with **MLflow** synced to **DagsHub**.

👉 View live experiments: [DagsHub MLflow Dashboard](https://dagshub.com/shovo896/Customer-chunk-prediction-end-to-end-ml-system.mlflow)

Each training run logs:
- 📌 Model parameters
- 📈 Metrics (accuracy, F1, ROC-AUC)
- 🗃️ Model artifact (serialized sklearn-compatible model)

The best model is registered in MLflow Model Registry under `CustomerChurnModel` and automatically promoted to the **Production** stage.

---

## 📦 Data Versioning with DVC

This project uses **DVC** to version datasets and pipeline outputs.

```bash
# Pull latest data from remote
dvc pull

# Run the full reproducible pipeline
dvc repro

# Check pipeline DAG
dvc dag
```

Pipeline stages and their dependencies are defined in `dvc.yaml`. Parameters are stored in `params.yaml` and are tracked as part of the pipeline.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/shovo896">shovo896</a>
</p>
