# FinSentiment AI

<p align="center">

<img src="https://img.shields.io/badge/FinBERT-NLP-2563EB?style=for-the-badge" alt="FinBERT" />
<img src="https://img.shields.io/badge/XGBoost-Time--Series-16A34A?style=for-the-badge" alt="XGBoost" />
<img src="https://img.shields.io/badge/LSTM-Model%20Comparison-7C3AED?style=for-the-badge" alt="LSTM" />
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/Next.js-Frontend-000000?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js" />
<img src="https://img.shields.io/badge/Docker-Containerization-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</p>

<p align="center">

<img src="https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge" alt="Status" />
<img src="https://img.shields.io/badge/Market%20Data-yfinance-0F766E?style=for-the-badge" alt="Market Data" />
<img src="https://img.shields.io/badge/Risk%20Detection-Model%20Disagreement-DC2626?style=for-the-badge" alt="Risk Detection" />

</p>


> **Hybrid NLP + Time-Series system for financial market analysis.**


> Combines FinBERT sentiment from financial news with XGBoost price
> forecasting to predict market direction --- and flags risk when the
> two models disagree.

------------------------------------------------------------------------

## Overview

FinSentiment AI is a production-style machine learning system that
analyzes financial markets from **two independent angles**:

1.  **Sentiment Analysis (NLP)** --- Uses a fine-tuned **FinBERT** model to classify financial news as *positive*, *negative*, or *neutral* from an investor's perspective.
2.  **Price Trend Forecasting (Time-Series)** --- Uses **XGBoost** (and an LSTM comparison model) trained on historical OHLCV data fetched via `yfinance` to predict the next-day price direction.

The real value: when **sentiment says bullish but the price model says
bearish** (or vice versa), the system raises a **risk flag** --- that
disagreement may indicate elevated uncertainty or volatility and
deserves further investigation.

**Backend:** FastAPI · **Frontend:** Next.js · **Deployment:**
Dockerized

------------------------------------------------------------------------

## Architecture

``` text
┌─────────────────────────────────────────────────────┐
│ Next.js 14 (App Router) + Tailwind + Recharts       │ ← Frontend
└──────────────────────┬──────────────────────────────┘
                       │ REST (JSON)
┌──────────────────────▼──────────────────────────────┐
│ FastAPI + Pydantic + Uvicorn                        │ ← Backend
│ Loads FinBERT + XGBoost, returns combined verdict   │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
       ┌──────────────┐  ┌──────────────┐
       │   FinBERT    │  │   XGBoost    │
       │ (sentiment)  │  │ (price trend)│
       └──────────────┘  └──────────────┘
              │                 │
              └────────┬────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ Docker Compose — one command runs everything        │
└─────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

## Folder Structure

``` text
finsentiment-ai/
├── backend/
│   ├── src/
│   │   ├── config.py                 # Paths, hyperparameters, tickers
│   │   ├── data/
│   │   │   ├── sentiment_loader.py   # Financial PhraseBank loader
│   │   │   └── price_fetcher.py      # yfinance wrapper
│   │   ├── models/
│   │   │   ├── sentiment_model.py    # FinBERT fine-tuning + inference
│   │   │   └── timeseries_model.py   # XGBoost / LSTM training + inference
│   │   ├── api/
│   │   │   ├── main.py               # FastAPI app entrypoint
│   │   │   ├── routes/
│   │   │   │   ├── predict.py        # /predict endpoint
│   │   │   │   └── health.py         # /health endpoint
│   │   │   ├── schemas.py            # Pydantic request/response models
│   │   │   └── dependencies.py       # Model loading, DI
│   │   └── utils/
│   │       └── disagreement.py       # Disagreement/risk flag logic
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_models.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                         # Next.js app
│   ├── app/
│   ├── components/
│   ├── lib/
│   │   └── api.ts                    # API client
│   ├── Dockerfile
│   └── package.json
├── data/
│   ├── raw/                          # Downloaded datasets
│   └── processed/                    # Train/val/test splits
├── models/
│   ├── sentiment/                    # Fine-tuned FinBERT checkpoints
│   └── timeseries/                   # Saved XGBoost / LSTM models
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| NLP Model | **FinBERT** (`ProsusAI/finbert`) | Domain-specific transformer for financial text |
| NLP Dataset | **Financial PhraseBank** | Human-labeled, investor-aligned sentiment data |
| Time-Series Data | **yfinance** | Historical OHLCV data; no API key required |
| Time-Series Model | **XGBoost** (+ LSTM for comparison) | Effective gradient boosting for engineered tabular features |
| Backend | **FastAPI + Pydantic** | Typed API contracts, validation, and automatic Swagger docs |
| Frontend | **Next.js 14 + Tailwind + Recharts** | Modern dashboard and interactive charts |
| Containerization | **Docker + Docker Compose** | Reproducible, one-command local setup |
| Testing | **pytest + httpx** | Unit and API integration tests |

------------------------------------------------------------------------

## Getting Started

### Prerequisites

-   **Python 3.10+**
-   **Node.js 18+** and **npm**
-   **Docker** and **Docker Compose** (recommended)
-   **Git**

### 1. Clone the repository

``` bash
git clone https://github.com/Harshpreet-Singh/finsentiment-ai.git
cd finsentiment-ai
```

### 2. Environment variables

Copy the example environment file:

``` bash
cp .env.example .env
```

### 3A. Quick Start --- Docker (recommended)

Run the entire stack (backend + frontend) with one command:

``` bash
docker compose up --build
```

Once running:

-   **Frontend:** http://localhost:3000
-   **Backend API:** http://localhost:8000
-   **API Docs (Swagger):** http://localhost:8000/docs

To stop:

``` bash
docker compose down
```

### 3B. Manual Setup --- Backend only

``` bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Backend: http://localhost:8000\
Swagger UI: http://localhost:8000/docs

### 3C. Manual Setup --- Frontend only

``` bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:3000

> The frontend expects the backend at `http://localhost:8000`. Update
> `NEXT_PUBLIC_API_URL` in `.env` if you change this.

------------------------------------------------------------------------

## API Endpoints (planned)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health and model status |
| `POST` | `/predict` | Accepts a ticker and news headlines; returns the combined verdict |
| `GET` | `/prices/{ticker}` | Historical OHLCV data for charting |

Example `/predict` request:

``` json
{
  "ticker": "AAPL",
  "news": [
    "Apple reports stronger-than-expected quarterly earnings."
  ]
}
```

Example `/predict` response:

``` json
{
  "ticker": "AAPL",
  "sentiment": {
    "label": "positive",
    "confidence": 0.91
  },
  "price_trend": {
    "direction": "up",
    "confidence": 0.68
  },
  "combined_verdict": "BULLISH",
  "disagreement": false,
  "risk_flag": null
}
```

------------------------------------------------------------------------

## Running Tests

``` bash
cd backend
pytest tests/ -v
```

------------------------------------------------------------------------

## Roadmap

-   [x] **Phase 0:** Project setup, architecture, and Docker scaffold
-   [ ] **Phase 1:** FinBERT fine-tuning on Financial PhraseBank
-   [ ] **Phase 2:** XGBoost / LSTM time-series model on yfinance data
-   [ ] **Phase 3:** FastAPI backend loading both models + disagreement
    logic
-   [ ] **Phase 4:** Next.js dashboard with charts and verdict UI
-   [ ] **Phase 5:** Deployment (Vercel + Railway) + CI/CD

------------------------------------------------------------------------

## Concepts Covered

-   Transformer fine-tuning with Hugging Face `transformers`
-   Financial domain sentiment classification
-   Feature engineering for OHLCV time-series data
-   Model disagreement and risk-flag logic
-   Production REST API design with FastAPI + Pydantic
-   Containerized full-stack deployment with Docker
-   React / Next.js frontend consuming ML APIs

------------------------------------------------------------------------

## License

This project is intended for **educational and portfolio purposes only**.

Market predictions generated by this system are **not financial advice**.

------------------------------------------------------------------------

## Author

**Harshpreet Singh**

-   GitHub:
    [`Harshpreet-Singh`](https://github.com/Harshpreet-Singh/)
-   LinkedIn:
    [`Harshpreet Singh`](https://www.linkedin.com/in/harshpreet-singh-989419225/)
