# Interactive Command Center (Streamlit)

The Streamlit application serves as the user-facing interface for the Dynamic Loan Pricing Engine, designed for both Underwriters and Portfolio Managers.

## Application Modules

### 1. The Underwriting Dashboard
Designed for individual applicant assessment.
* **Input Panel**: Allows underwriters to enter borrower data (Income, DTI, Employment status).
* **Live Optimizer**: As you change parameters, the optimizer recalculates the recommended rate instantly.
* **Explainability Unit**: Displays a SHAP waterfall chart explaining which applicant features (e.g., high credit utilization) drove the final rate recommendation.

### 2. Portfolio Health View
Designed for C-suite and risk management teams.
* **Efficient Frontier**: Visualizes the tradeoff between Portfolio Net Margin and Default Rate Appetite.
* **PSI Alerts**: Highlights any population drift that might compromise future model accuracy.

### 3. Scenario Analysis ("What-If" Engine)
Strategic tool for long-term planning.
* **Competitor Impact**: Simulate what happens to our conversion rate if a major competitor drops their rates by 50 basis points.
* **Economic Stress Test**: Increase the Cost of Capital globally and see how many borrowers are "priced out" of the portfolio.

## Tech Stack Note
The app uses `streamlit` for the UI, `plotly` for interactive risk charts, and `matplotlib` for SHAP visualization. It communicates directly with the optimization engine in `src/models/`.

## Deployment
For local development:
```bash
streamlit run github/streamlit/app.py
```
For production, the app is containerized as the `frontend` service in `docker-compose.yml`.
