# Project Configuration

This directory contains the central brain of the Dynamic Loan Pricing Engine. The configuration system handles everything from database connections to complex business logic constraints.

## Central File: `config.yaml`

The configuration is divided into thematic sections:

### 1. Core Infrastructure
* **Database**: Settings for PostgreSQL connectivity (host, port, user, password).
* **Paths**: Directories for raw/processed data and model artifacts.

### 2. Machine Learning Pipeline
* **Feature Engineering**: Parameters like `min_iv_threshold` (Information Value) to filter predictive features and train/test split ratios.
* **Model Hyperparameters**: Specific settings for:
  * `xgboost`: Performance-driven risk model.
  * `logistic_regression`: Stability-driven scorecard model.
  * `ensemble`: Weights for blending models, including Optuna optimization settings.

### 3. Business & Risk Logic
* **LGD (Loss Given Default)**: Percentage of loss expected if a borrower defaults.
* **Cost of Capital**: The internal cost to the lender for providing the funds.
* **Regulatory Ceiling**: Maximum interest rate allowed by law (e.g., 36%).
* **Fairness Constraints**: Limits on the "spread" or difference between rates offered to low-risk vs. high-risk borrowers to prevent exploitative pricing.

### 4. Agentic Interaction
* **Tavily/Gemini**: LLM and search tool settings for the agentic underwriting layer.
* **Scraper**: Schedules and templates for automated market rate intelligence.

## Security Note
Sensitive values like API keys should **never** be added to this file. They are loaded at runtime from the `.env` file located in the project root.
