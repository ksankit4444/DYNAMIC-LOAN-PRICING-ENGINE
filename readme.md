# Dynamic Loan Pricing Engine

A production-grade, end-to-end framework for credit risk assessment and interest rate optimization. Designed for modern fintechs, this project bridges the gap between binary probability-of-default (PD) classification and real-world commercial loan pricing.

---

## 🚀 The Strategic Advantage
Unlike standard data science projects that stop at "will they default?", the **Dynamic Loan Pricing Engine** answers the critical commercial question: *"What is the optimal interest rate to offer this specific borrower?"*

It balances three competing forces:
1. **Risk Mitigation**: Ensuring every loan covers its Expected Loss (EL).
2. **Competitive Conversion**: Modeling price elasticity to ensure the quote is attractive.
3. **Profit Maximization**: Finding the "efficient frontier" where margin and volume are optimized.

---

## 🏗️ Technical Architecture
The system is built on a **Five-Layer Decision Waterfall**:

| Layer | Name | Technical Core | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | **Risk Modeling** | Hybrid Ensemble (XGBoost + Logistic) | Estimates the raw Probability of Default (PD). |
| **2** | **Expected Loss (EL)** | $PD \times LGD \times EAD$ Framework | Sets the minimum interest rate floor to cover risk. |
| **3** | **Price Elasticity** | Logistic Conversion Curve | Models how acceptance probability drops as rates rise. |
| **4** | **Optimization** | Scipy Bounded Minimization | Finds the rate that maximizes E[Profit] per applicant. |
| **5** | **Agentic Support** | LangGraph + Gemini 2.0 | Conversational AI for explainability and market Intel. |

---

## 🛠️ Key Features

### 1. Hybrid Ensemble Underwriting
The engine uses a combination of **XGBoost** for maximum predictive precision and **Logistic Regression (with WoE Encoding)** for regulatory auditability. The weights are dynamically managed based on the compliance-vs-performance tradeoff.

### 2. Market-Aware Optimization
The `PricingOptimizer` doesn't just look inward. It integrates:
- **Market Benchmarking**: Real-time competitor rate ranges (via async scraper).
- **Fairness Constraints**: Built-in logic to prevent predatory pricing (max rate spreads).
- **Regulatory Guardrails**: Automated capping at legal interest rate ceilings.

### 3. Agentic Underwriting Intelligence
Integrated **LangGraph** agent serves as a "Co-pilot" for credit officers:
- **Natural Language Explanations**: Converts complex SHAP values into readable risk narratives.
- **Live Market Refresh**: Can trigger on-demand market scrapes to verify if a quoted rate is still competitive.
- **Audit Logging**: Seamlessly writes results to a PostgreSQL backend for monitoring.

### 4. Enterprise Monitoring (PSI & Drift)
Equipped with a **Grafana-powered Monitoring Stack** that tracks **Population Stability Index (PSI)**. The system alerts you when incoming applicant distributions drift from training data, signaling the need for model recalibration.

---

## 🚦 Getting Started

### Prerequisites
- **Python**: 3.9+ 
- **Database**: PostgreSQL (for application logging and drift tracking)
- **API Keys**: Google Gemini (for Agentic Layer) and Tavily (optional, for Market Search)

### Installation
1. Clone the repository and navigate to the directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment:
   ```bash
   cp .env.example .env
   # Add your DB credentials and API keys to .env
   ```

### Running the Engine
- **Interactive Dashboard**:
  ```bash
  streamlit run streamlit/app.py
  ```
- **Automated Underwriting Agent**:
  ```bash
  python -m src.agents.underwriting_agent
  ```
- **Monitoring Stack**:
  ```bash
  docker-compose up -d
  ```

---

## 📊 Dataset Requirement
This project is built to work with the **Home Credit Default Risk** dataset from Kaggle. 
To run the full pipeline, place the following files in `data/raw/`:
- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`

