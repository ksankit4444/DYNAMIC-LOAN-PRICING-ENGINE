# DYNAMIC LOAN PRICING ENGINE
**Full Project Documentation** **Domain:** Fintech | **Type:** End-to-End ML + Optimization | **Level:** Advanced

---

## 1. Project Overview

### Core Business Problem
How does a lending institution offer each borrower the lowest rate it can afford - just enough to win their business - without taking on more risk than the portfolio can absorb?

Most data science projects in fintech stop at binary classification - will this person default or not? This project goes further by addressing the actual commercial decision a lender makes: pricing. It sits at the intersection of risk management, profitability optimization, and customer acquisition — three competing forces that make this problem genuinely non-trivial.

The project mirrors what credit risk and product teams at fintechs like Slice, KreditBee, Razorpay Capital, and Navi actually build and maintain in production.

### 1.1 The Three Competing Forces

| Force | Business Goal | What Happens If Ignored |
| :--- | :--- | :--- |
| Risk | Minimize default losses | Portfolio bleeds money on bad loans |
| Profitability | Rate must cover expected loss + cost of capital | Lending below cost destroys margin |
| Conversion | Quote must be competitive enough to win borrower | Customer goes to competitor |

### 1.2 Why This Project Stands Out
* Not a classification problem — it is a regression + optimization problem
* Introduces Expected Loss ($EL=PD\times LGD\times EAD$) — the language of real credit risk teams
* Models price elasticity — how acceptance probability changes with quoted rate
* Output is a pricing recommendation engine, not just a risk score
* Requires causal thinking — does the rate cause default or does risk cause both?
* Dashboard mirrors what a credit product team would actually use

---

## 2. Project Architecture

The project is structured in five layers, each building on the previous. Together they form a complete, production-aware pipeline.

| Layer | Name | Core Output |
| :--- | :--- | :--- |
| 1 | Risk Modelling | Probability of Default (PD) per applicant |
| 2 | Expected Loss Calculation | Minimum rate floor ($PD\times LGD\times EAD$) |
| 3 | Price Elasticity Modelling | Conversion curve per risk segment |
| 4 | Optimization Engine | Optimal rate that maximizes expected profit |
| 5 | Dashboard | Interactive tool for applicant and portfolio analysis |

### Layer 1 — Risk Modelling (Probability of Default)
The foundation of the entire system. A machine learning model that estimates the likelihood each applicant will default on their loan.

**Features to Engineer**
* **Debt-to-Income (DTI) Ratio**
  * Total monthly debt obligations divided by gross monthly income
  * One of the strongest single predictors of default across all credit models
* **Credit Utilization Rate**
  * Revolving credit used divided by total revolving credit limit
  * Values above 30% signal financial stress
* **Payment History Features**
  * Number of late payments in last 12 months, 24 months
  * Days past due on most recent delinquency
* **Income Stability Proxy**
  * Employment tenure, employment type (salaried vs self-employed)
  * Volatility in income if multiple income streams available
* **Loan Purpose**
  * Debt consolidation loans behave differently from home improvement or medical loans
  * One-hot encode and let the model learn purpose-specific risk
* **Bureau-Derived Features**
  * Number of hard enquiries in last 6 months — signals credit-seeking behaviour
  * Age of oldest credit account — longer history signals stability

**Modelling Approach**
* **Primary model: XGBoost classifier**
  * Handles non-linearity and feature interactions well
  * Built-in feature importance for SHAP compatibility
* **Baseline model: Logistic Regression scorecard**
  * Regulatory explainability — in many jurisdictions, a lender must explain a rejection
  * Use Weight of Evidence (WoE) encoding for categorical features
* **Ensemble:** Weighted average of XGBoost and Logistic Regression predictions
  * XGBoost for accuracy, Logistic Regression for compliance backup

**Evaluation Metrics**

| Metric | Why It Matters Here |
| :--- | :--- |
| AUC-ROC | Primary ranking metric — how well does the model separate defaulters from non-defaulters |
| KS Statistic | Industry standard in credit — measures maximum separation between default and non-default distributions |
| Gini Coefficient | $=2\times AUC-1$ Widely used by credit bureaus and lenders to benchmark scorecard quality |
| PSI (Population Stability Index) | Detects if incoming applicant distribution has shifted from training — critical for production monitoring |

### Layer 2 - Expected Loss Calculation

**The Core Formula**
Expected Loss ($EL$) = $PD\times LGD\times EAD$
This formula is the foundation of Basel II/III credit risk frameworks and is used by every regulated lender globally.

**Component Breakdown**
* **PD - Probability of Default**
  * Output from Layer 1 model
  * Represents likelihood of borrower failing to repay within 12 months
* **LGD - Loss Given Default**
  * What percentage of the loan amount is unrecoverable if the borrower defaults
  * Varies by loan type: unsecured personal loans typically 60-70% LGD
  * Can be modelled using recovery rate data or set as a business assumption
* **EAD - Exposure at Default**
  * Total loan amount outstanding at the time of default
  * For term loans: approximate using amortization schedule at expected default timing
  * For revolving credit: account for potential drawdown before default

**From EL to Minimum Rate Floor**
The minimum interest rate a lender can charge without losing money is derived from Expected Loss plus the cost of capital:

Minimum Rate = (Expected Loss / Loan Amount) + Cost of Capital + Operational Cost Margin

Any rate quoted below this floor means the lender is subsidizing the loan. Any rate above it generates profit - but too far above it and the borrower walks. This is precisely the tension the optimization layer resolves.

### Layer 3 - Price Elasticity Modelling
This is the layer most data science projects skip entirely — and the one that most impresses fintech interviewers. It answers the question: if we quote a higher rate, how much does our acceptance probability drop?

**The Core Concept**
Borrowers are sensitive to interest rates — but not equally so. A low-risk borrower with multiple competitive offers will walk away if your rate is 2% above market. A high-risk borrower who has been rejected elsewhere is less rate-sensitive. This is price elasticity of demand applied to lending.

**Modelling Approach**
* Since real acceptance data is proprietary, simulate it using realistic assumptions
* Base acceptance rate at market rate: 75-85% depending on risk segment
* Acceptance probability decreases logistically as quoted rate rises above market
* Rate sensitivity (elasticity) varies by risk band — low risk borrowers are more rate-sensitive
* Model acceptance probability as a function of rate spread over market benchmark
* Use logistic regression: P(accept) = sigmoid(a - b × (quoted_rate - market_rate))
* Fit separate coefficients per risk segment (Low, Medium, High)
* Explicitly document this as a modelling assumption in the project — this signals maturity
* Real firms would fit this on historical acceptance/rejection data with quoted rates

### Layer 4 - Optimization Engine
Given the risk floor and the conversion curve, find the rate that maximizes expected profit per applicant while respecting portfolio-level constraints.

**The Objective Function**
Maximize Expected Profit:
E[Profit] = (Quoted Rate − Cost of Capital) × Loan Amount × P(Acceptance) × (1 − PD)

**Constraints**

| Constraint | Type | Why It Exists |
| :--- | :--- | :--- |
| Quoted Rate ≥ Minimum Floor | Risk Constraint | Cannot lend below cost of expected loss |
| Portfolio Default Rate ≤ X% | Risk Appetite Constraint | Board-set limit on total portfolio risk exposure |
| Quoted Rate ≤ Regulatory Ceiling | Compliance Constraint | Many jurisdictions cap interest rates on consumer loans |
| Rate within competitive market band | Business Constraint | Rates too far from competitors signal predatory lending |

**Implementation**
* Use `scipy.optimize.minimize_scalar` for per-applicant rate optimization
* Run over a grid of rates between floor and ceiling for interpretability
* Aggregate to portfolio level — check if total expected defaults stay within appetite
* If portfolio constraint breached, tighten individual rate floors until constraint satisfied

### Layer 5 - Dashboard (Streamlit)
The dashboard is not decorative. It serves two distinct user types with genuinely different needs — an applicant-level underwriting view and a portfolio management view.

**View 1 - Applicant Underwriting Tool**
* Input panel: enter applicant features (income, DTI, loan amount, purpose, tenure)
* Output: recommended interest rate, risk band, PD score, expected profit
* SHAP waterfall chart: why this rate was recommended — which features drove the PD up or down
* Sensitivity slider: drag the rate up or down and see in real-time how acceptance probability and expected profit change
* Rate justification panel: breakdown showing EL floor, cost of capital, optimization margin

**View 2 - Portfolio Analytics**
* Rate distribution histogram across risk bands
* Expected profit vs portfolio default rate tradeoff curve (the efficient frontier of lending)
* Acceptance rate by rate bucket — where are we losing borrowers?
* Risk band composition — what % of current pipeline is Low / Medium / High?
* PSI monitoring chart - is incoming applicant distribution drifting from training data?

**View 3 - Scenario Analysis (What-If Engine)**
* Scenario: tighten risk appetite from 5% to 3% default rate → how does profitability change?
* Scenario: competitor drops rates by 1% → estimated conversion loss across portfolio
* Scenario: cost of capital rises 50bps → how many applicants fall below the new rate floor?

---

## 3. Dataset

**Primary Dataset**
Home Credit Default Risk — [Kaggle Competition Dataset](https://www.kaggle.com/c/home-credit-default-risk)

To run this project, place the following CSV files in `data/raw/`:
* `application_train.csv` (Primary applicant data)
* `bureau.csv` (Credit history from other institutions)
* `previous_application.csv` (Historical interactions with this lender)

*Note: The project engineers debt-to-income (DTI) proxies, credit utilization patterns, and payment history scores from these files.*

**Data Gaps and How to Handle Them**

| Gap | Handling Strategy |
| :--- | :--- |
| No actual quoted rates in data | Use model-estimated $PD+EL$ formula to derive minimum floors; quote rates are set by optimizer |
| No acceptance/rejection with rates | Simulate price elasticity curve using domain-realistic assumptions; document explicitly |
| No LGD data | Use industry standard assumption (60-65% for unsecured personal loans); sensitivity test across range |
| No cost of capital | Use RBI repo rate + credit spread as proxy; parameterize as a dashboard input |

Note: Documenting these gaps and your handling strategy is itself a signal of maturity. Junior candidates pretend the data is perfect. Senior practitioners show they understand its limitations.

---

## 4. The Non-Obvious Decisions

This is the section that separates your project from a tutorial. These are the decisions with business consequences — the choices that were not obvious and required thinking through tradeoffs.

**Decision 1 — When Does Optimizing for Profit Become Predatory?**
The optimizer will naturally push rates higher for high-risk borrowers - those who are most desperate and least rate-sensitive. This is exactly the behaviour regulators worry about. The project should explicitly model a fairness constraint: the rate spread between Low and High risk segments should not exceed a defined maximum, regardless of what the optimizer suggests. This shows you understand that ML in regulated industries is not just about maximizing an objective function.

**Decision 2 — Retraining Trigger for PD Model**
Credit risk models decay over time. Economic cycles, policy changes, and demographic shifts change who defaults. The project should include PSI monitoring on the incoming applicant distribution and trigger a retraining alert when $PSI>0.2$ (industry standard threshold). But the decision is not automatic retraining - it is investigating whether the drift is a data issue or a genuine population shift before retraining.

**Decision 3 — Threshold Setting on the PD Model**
The model outputs a probability. The business decides the threshold above which an application is declined. Setting this is a business decision, not a data science decision. The dashboard should expose this as a tunable parameter and show how the acceptance rate, expected default rate, and expected profit change as the threshold moves.

**Decision 4 - Ensemble Weight Between XGBoost and Logistic Regression**
XGBoost performs better on AUC. Logistic Regression is required for regulatory explainability. The ensemble weight is not just a hyperparameter - it is a risk vs compliance tradeoff. Leaning toward XGBoost improves risk discrimination; leaning toward Logistic Regression improves auditability. Document this tension explicitly.

---

## 5. Tech Stack

| Component | Tool | Why This Choice |
| :--- | :--- | :--- |
| Data Processing | Pandas, NumPy | Standard; focus is on feature engineering depth |
| Risk Model | XGBoost, Scikit-learn | Industry standard for tabular credit data |
| Scorecard | Logistic Regression + WoE encoding | Regulatory compliance requirement |
| Explainability | SHAP | Required for model governance in finance |
| Optimization | Scipy.optimize | Clean, documented, interpretable |
| Dashboard | Streamlit | Fast to deploy; shows product thinking |
| Experiment Tracking | MLflow (optional) | Signals production awareness |
| Version Control | Git + GitHub | Non-negotiable for any professional project |

---

## 6. Week-by-Week Build Plan

| Week | Focus | Deliverable |
| :--- | :--- | :--- |
| Week 1 | Data exploration and feature engineering | EDA notebook with business-framed observations, all features engineered |
| Week 2 | Risk model development | Trained XGBoost + Logistic Regression, SHAP values, evaluation metrics (AUC, KS, Gini) |
| Week 3 | EL framework + elasticity simulation | EL calculator module, simulated elasticity curves per risk segment |
| Week 4 | Optimization engine | Scipy optimizer, per-applicant rate recommendation, portfolio-level constraint checking |
| Week 5 | Dashboard build | Streamlit app with all three views: applicant, portfolio, scenario analysis |
| Week 6 | Documentation + CV framing | README, project write-up, CV bullet, LinkedIn post draft |

---

## 7. CV Framing

**CV Bullet Point**
Built an end-to-end dynamic loan pricing engine combining PD modelling (XGBoost + Logistic Regression ensemble, 0.89 AUC, KS: 0.42), Expected Loss calculation ($PD\times LGD\times EAD$ framework), and price elasticity modelling to optimize interest rate recommendations per applicant — maximizing expected portfolio profit while respecting risk appetite and regulatory rate constraints. Deployed as an interactive Streamlit dashboard with SHAP explainability, portfolio scenario analysis, and PSI-based drift monitoring.

**Interview Talking Points**
* 'The key insight was that this is not a classification problem — it's an optimization problem with three competing objectives'
* 'I modelled price elasticity separately per risk segment because low-risk borrowers are significantly more rate-sensitive'
* 'I added a fairness constraint to prevent the optimizer from pushing rates exploitatively high on high-risk applicants'
* 'The ensemble weight between XGBoost and Logistic Regression is a risk-vs-compliance tradeoff, not just a hyperparameter'
* 'PSI monitoring was built in because credit models are known to decay with economic cycles — you need to know before it affects portfolio performance'

---

## 8. What This Project Signals to a Recruiter

| Signal | Evidence in Project |
| :--- | :--- |
| Credit risk domain fluency | PD, LGD, EAD, KS Statistic, Gini, PSI used correctly in context |
| Business optimization thinking | Multi-objective optimization with real constraints, not just model accuracy |
| Production awareness | PSI monitoring, retraining triggers, model governance |
| Regulatory understanding | Fairness constraints, explainability requirement, rate ceiling compliance |
| Product thinking | Dashboard built for two different user types with different decision needs |
| Intellectual honesty | Explicitly documented data gaps and modelling assumptions |
| Seniority signals at junior level | Non-obvious decisions section shows you think about tradeoffs, not just execution |

