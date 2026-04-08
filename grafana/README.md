# Portfolio Monitoring & Drift Detection (Grafana)

Visualizing model performance and population stability is critical for any production lending system. This directory provides a complete monitoring stack using Grafana.

## Dashboard Architecture

### 1. Portfolio Stability (PSI)
Tracking **Population Stability Index** to detect "drift" in incoming applicant profiles. If the distribution of features (like Income or DTI) changes significantly from the training data, the engine triggers an alert.

### 2. Risk Metrics
Real-time tracking of:
* **Ensemble AUC**: The discriminatory power of the current risk models.
* **KS Statistic**: Separation between defaulters and non-defaulters.
* **Expected Default Rate**: Portfolio-wide risk exposure.

### 3. Profitability Analytics
* **Net Margin**: Expected profit after EL (Expected Loss) and Cost of Capital.
* **Conversion Rate**: Percentage of applicants accepting the quoted rates.

## Provisioning Details
* `provisioning/datasources/`: Automatically connects Grafana to the project's PostgreSQL database.
* `provisioning/dashboards/`: Auto-loads the JSON dashboard definitions on startup.

## How to View
The monitoring stack is containerized. To launch:
1. Run `docker-compose up -d`.
2. Browse to `http://localhost:3000`.
3. Login with `admin / admin`.
4. Navigate to the "Loan Engine Dashboard" in the Dashboards section.
