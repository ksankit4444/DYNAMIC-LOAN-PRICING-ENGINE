# Core Engine Source Code

This repository follows a modular "Layered Architecture" where each directory represents a distinct phase of the lending lifecycle.

## Architectural Layers

### 1. `features/` — Data Engineering
Responsible for transforming raw Kaggle CSVs into high-signal numeric features.
* **`build_features.py`**: The main pipeline. Handles missing values, clips outliers, and engineers domain features like DTI Ratio, Credit Utilization, and Income Stability scores.

### 2. `models/` — The Pricing Engine
* **Layer 1: Risk Modelling**: Training the Triple Threat ensemble (XGB + LGBM + LR).
* **Layer 2: Expected Loss (EL)**: Calculating the loss floor ($PD \times LGD \times EAD$).
* **Layer 3: Price Elasticity**: Modeling the relationship between interest rates and customer conversion.
* **Layer 4: Optimization Engine**: Using `scipy.optimize` to find the profit-maximizing rate.
* **Layer 5: Decision Engine**: The final logic that orchestrates all layers to produce a recommendation.

### 3. `agents/` — Agentic Intelligence
Integrates LLMs into the underwriting process.
* **`underwriting_agent.py`**: Uses LangGraph to synthesize model scores with market intelligence.
* **`market_scraper.py`**: Automated background worker that retrieves current market rates to update competitive bands.

### 4. `data/` — Persistence Layer
* **`init_db.py`**: Sets up the PostgreSQL schema for storing application history and drift logs.
* **`seed_portfolio.py`**: Populates the system with initial data for local testing and demonstration.

## Python Environment
All code is compatible with Python 3.9+. It is recommended to run operations from the project root using a virtual environment.
