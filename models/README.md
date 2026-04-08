# Model Registry & Artifacts

This directory serves as the "Model Registry" for the engine, containing all serialzed model objects and their associated validation metadata.

## Included Artifacts

### 1. Risk Estimators (Layer 1)
* `xgboost_model.joblib`: High-precision gradient boosting model.
* `lightgbm_model.joblib`: High-efficiency gradient boosting for complex interactions.
* `logistic_model.joblib`: Compliance-driven scorecard model for auditability.

### 2. Feature Encoders
* `woe_encoder.joblib`: Stores the Weight of Evidence (WoE) mappings and Information Value (IV) for all categorical variables.
* `feature_columns.joblib`: A static list of feature names used to ensure consistent schema between training and inference.

### 3. Explainability
* `shap_explainer.joblib`: A TreeExplainer used to generate "Why was I given this rate?" explanations for applicants.

### 4. Configuration & Metadata
* `ensemble_config.json`: Defines the blending weights (e.g., 60% XGBoost, 40% Logistic) used by the decision engine.
* `training_metrics.json`: A snapshot of performance at train-time, including AUC-ROC per segment, KS-Statistic, and Gini coefficient.

## Operational Note
During production inference (Streamlit or API), the engine loads these objects using `joblib.load()`. Any update to these files immediately affects the live recommendations. Always run the test suite in `tests/` after manual model updates.
