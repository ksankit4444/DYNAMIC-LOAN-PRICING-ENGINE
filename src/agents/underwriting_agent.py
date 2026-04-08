"""
underwriting_agent.py — LangGraph Underwriting + Market Intelligence Agent
=====
A conversational credit analyst agent powered by LangGraph + Gemini.

The agent has access to the full loan pricing pipeline as tools:
  - predict_pd:          Run ensemble PD model
  - calculate_el:        Compute Expected Loss (PD × LGD × EAD)
  - optimize_rate:       Find optimal rate via scipy optimizer
  - explain_prediction:  Generate natural-language SHAP explanation
  - get_market_benchmark: Get current competitive rates from DB
  - refresh_market_data: On-demand Tavily scrape (if benchmarks expired)
  - log_application:     Write scored application to PostgreSQL

Usage:
    # Interactive mode
    python src/agents/underwriting_agent.py

    # Programmatic (from Streamlit/FastAPI)
    from src.agents.underwriting_agent import create_agent, run_agent
    agent = create_agent()
    result = run_agent(agent, "Score a 30-year-old salaried applicant...")
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(ROOT_DIR, '.env'))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_db_session():
    """Create a DB session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = load_config()
    db = config['database']
    url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    return Session(), engine


# ──────────────────────────────────────────────
# Load Model Artifacts (lazy, cached)
# ──────────────────────────────────────────────
_model_cache = {}

def _load_models():
    """Load all V2 model artifacts into cache (once)."""
    if _model_cache:
        return _model_cache

    config = load_config()
    models_path = os.path.join(ROOT_DIR, config['paths']['models'], 'v2')

    try:
        _model_cache['xgb_model'] = joblib.load(os.path.join(models_path, 'xgboost_model.joblib'))
        _model_cache['lr_model'] = joblib.load(os.path.join(models_path, 'logistic_model.joblib'))
        _model_cache['lr_scaler'] = joblib.load(os.path.join(models_path, 'logistic_scaler.joblib'))
        _model_cache['shap_explainer'] = joblib.load(os.path.join(models_path, 'shap_explainer.joblib'))
        _model_cache['xgb_feats'] = joblib.load(os.path.join(models_path, 'xgb_feature_columns.joblib'))
        _model_cache['lr_feats'] = joblib.load(os.path.join(models_path, 'lr_feature_columns.joblib'))
        with open(os.path.join(models_path, 'ensemble_config.json'), 'r') as f:
            _model_cache['ensemble_config'] = json.load(f)
    except Exception as e:
        print(f"⚠️  Model artifact not found: {e}")
        _model_cache['xgb_model'] = None

    return _model_cache


# ══════════════════════════════════════════════
# TOOL DEFINITIONS
# ══════════════════════════════════════════════

@tool
def predict_pd(
    amt_income_total: float,
    amt_credit: float,
    amt_annuity: float,
    days_birth: float,
    days_employed: float,
    ext_source_1: float = 0.5,
    ext_source_2: float = 0.5,
    ext_source_3: float = 0.5,
) -> str:
    """
    Predict Probability of Default for a loan applicant.
    Returns PD scores from XGBoost, Logistic Regression, and the ensemble.

    Args:
        amt_income_total: Annual gross income (e.g., 600000)
        amt_credit: Loan credit amount (e.g., 1500000)
        amt_annuity: Loan annuity / monthly payment (e.g., 35000)
        days_birth: Age as negative days from application (e.g., -10950 for ~30 years)
        days_employed: Employment duration as negative days (e.g., -2000 for ~5.5 years)
        ext_source_1: External source score 1 (0-1, default 0.5)
        ext_source_2: External source score 2 (0-1, default 0.5)
        ext_source_3: External source score 3 (0-1, default 0.5)
    """
    models = _load_models()

    if models.get('xgb_model') is None:
        return json.dumps({
            'error': 'Models not trained yet. Run train_risk_model.py first.',
            'status': 'model_not_found'
        })

    # Engineer features from inputs
    dti_ratio = amt_annuity / max(amt_income_total, 1)
    credit_income_ratio = amt_credit / max(amt_income_total, 1)
    annuity_credit_ratio = amt_annuity / max(amt_credit, 1)
    age_years = abs(days_birth) / 365.25
    employment_years = abs(days_employed) / 365.25 if days_employed < 0 else 0
    employed_flag = 1 if days_employed != 365243 else 0
    employment_ratio = employment_years / max(age_years, 1)
    income_stability_score = 0.5 * employment_ratio + 0.3 * employed_flag + 0.2 * min(employment_years, 30) / 30
    # Engineer features from inputs
    dti_ratio = amt_annuity / max(amt_income_total, 1)
    credit_income_ratio = amt_credit / max(amt_income_total, 1)
    annuity_credit_ratio = amt_annuity / max(amt_credit, 1)
    age_years = abs(days_birth) / 365.25
    employment_years = abs(days_employed) / 365.25 if days_employed < 0 else 0
    employed_flag = 1 if days_employed != 365243 else 0
    employment_ratio = employment_years / max(age_years, 1)
    income_stability_score = 0.5 * employment_ratio + 0.3 * employed_flag + 0.2 * min(employment_years, 30) / 30
    ext_source_mean = np.mean([ext_source_1, ext_source_2, ext_source_3])

    # Build original feature dictionary
    features = {
        'AMT_INCOME_TOTAL': amt_income_total,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'EXT_SOURCE_1': ext_source_1,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'ext_source_mean': ext_source_mean,
        'ext_source_prod': ext_source_1 * ext_source_2 * ext_source_3,
        'dti_ratio': dti_ratio,
        'credit_income_ratio': credit_income_ratio,
        'annuity_credit_ratio': annuity_credit_ratio,
        'goods_credit_ratio': amt_credit / max(amt_credit, 1), # Approx
        'age_years': age_years,
        'employment_years': employment_years,
        'employed_flag': employed_flag,
        'employment_ratio': employment_ratio,
        'income_stability_score': income_stability_score,
        'prev_avg_annuity': amt_annuity,
    }

    xgb_feats = models['xgb_feats']
    lr_feats = models['lr_feats']
    
    xgb_dict = {col: features.get(col, 0.0) for col in xgb_feats}
    lr_dict = {col: features.get(col, 0.0) for col in lr_feats}

    X_xgb = pd.DataFrame([xgb_dict])[xgb_feats]
    X_lr = pd.DataFrame([lr_dict])[lr_feats]

    # Predict
    pd_xgb = float(models['xgb_model'].predict_proba(X_xgb)[:, 1][0])
    X_lr_scaled = models['lr_scaler'].transform(X_lr)
    pd_lr = float(models['lr_model'].predict_proba(X_lr_scaled)[:, 1][0])

    weights = models['ensemble_config']['optimal_weights']
    w_xgb = weights.get('xgboost', 0.6)
    w_lr = weights.get('logistic', 0.4)
    pd_ensemble = w_xgb * pd_xgb + w_lr * pd_lr

    # Risk band
    if pd_ensemble < 0.08:
        risk_band = 'Low'
    elif pd_ensemble < 0.20:
        risk_band = 'Medium'
    else:
        risk_band = 'High'

    result = {
        'pd_xgboost': round(pd_xgb, 6),
        'pd_logistic': round(pd_lr, 6),
        'pd_ensemble': round(pd_ensemble, 6),
        'risk_band': risk_band,
        'ensemble_weights': {'xgboost': w_xgb, 'logistic': w_lr},
        'key_features': {
            'dti_ratio': round(dti_ratio, 4),
            'income_stability': round(income_stability_score, 4),
            'ext_source_mean': round(ext_source_mean, 4),
            'age_years': round(age_years, 1),
        }
    }

    return json.dumps(result, indent=2)


@tool
def calculate_expected_loss(
    pd_score: float,
    loan_amount: float,
    lgd: float = 0.65,
) -> str:
    """
    Calculate Expected Loss and minimum rate floor.
    EL = PD × LGD × EAD. Minimum Rate = EL/Loan + Cost of Capital + Margin.

    Args:
        pd_score: Probability of Default (0-1)
        loan_amount: Total loan amount (EAD proxy)
        lgd: Loss Given Default (default 0.65 for unsecured personal loans)
    """
    config = load_config()
    business = config['business']

    ead = loan_amount  # For term loans, EAD ≈ loan amount
    expected_loss = pd_score * lgd * ead
    el_rate = expected_loss / max(loan_amount, 1)

    cost_of_capital = business.get('cost_of_capital', 0.085)
    operational_margin = business.get('operational_margin', 0.015)
    min_rate_floor = el_rate + cost_of_capital + operational_margin

    result = {
        'expected_loss': round(expected_loss, 2),
        'expected_loss_rate': round(el_rate, 6),
        'cost_of_capital': cost_of_capital,
        'operational_margin': operational_margin,
        'min_rate_floor': round(min_rate_floor, 6),
        'min_rate_floor_pct': f"{min_rate_floor*100:.2f}%",
        'breakdown': {
            'PD': round(pd_score, 6),
            'LGD': lgd,
            'EAD': loan_amount,
            'EL_component': f"{el_rate*100:.2f}%",
            'CoC_component': f"{cost_of_capital*100:.2f}%",
            'Margin_component': f"{operational_margin*100:.2f}%",
        }
    }

    return json.dumps(result, indent=2)


@tool
def get_market_benchmark(risk_band: str = "Medium") -> str:
    """
    Get current market benchmark rate for a risk segment.
    Queries PostgreSQL for the latest active benchmark (fast, ~2ms).
    Falls back to config.yaml if no benchmark exists.

    Args:
        risk_band: Risk segment — 'Low', 'Medium', or 'High'
    """
    from src.agents.market_scraper import get_current_benchmark

    config = load_config()
    benchmark = get_current_benchmark(config)

    rate_map = {
        'Low': benchmark.get('low_risk_rate', 0.085),
        'Medium': benchmark.get('medium_risk_rate', 0.11),
        'High': benchmark.get('high_risk_rate', 0.145),
    }

    band_width = config.get('market_benchmarks', {}).get('competitive_band_width', 0.02)
    segment_rate = rate_map.get(risk_band, rate_map['Medium'])

    result = {
        'risk_band': risk_band,
        'market_benchmark_rate': round(segment_rate, 4),
        'market_benchmark_pct': f"{segment_rate*100:.2f}%",
        'competitive_band': {
            'lower': round(segment_rate - band_width, 4),
            'upper': round(segment_rate + band_width, 4),
            'lower_pct': f"{(segment_rate - band_width)*100:.2f}%",
            'upper_pct': f"{(segment_rate + band_width)*100:.2f}%",
        },
        'all_segments': {
            'Low': f"{rate_map.get('Low', 0)*100:.2f}%",
            'Medium': f"{rate_map.get('Medium', 0)*100:.2f}%",
            'High': f"{rate_map.get('High', 0)*100:.2f}%",
        },
        'source': benchmark.get('source', 'unknown'),
        'fetched_at': benchmark.get('fetched_at'),
        'expired': benchmark.get('expired', False),
    }

    return json.dumps(result, indent=2)


@tool
def refresh_market_data() -> str:
    """
    Trigger an on-demand market rate refresh using Tavily search + Gemini.
    Use this when the user asks 'are we competitive?' or when benchmarks are expired.
    This runs the full scrape → extract → store pipeline.
    """
    from src.agents.market_scraper import run_scraper

    rates = run_scraper(fetch_type='on_demand')

    if rates:
        return json.dumps({
            'status': 'success',
            'message': 'Market benchmarks refreshed successfully',
            'rates': {
                'Low': f"{rates['low_risk_rate']*100:.2f}%",
                'Medium': f"{rates['medium_risk_rate']*100:.2f}%",
                'High': f"{rates['high_risk_rate']*100:.2f}%",
            }
        }, indent=2)
    else:
        return json.dumps({
            'status': 'failed',
            'message': 'Could not refresh market data. Using cached/fallback rates.'
        })


@tool
def explain_prediction(
    amt_income_total: float,
    amt_credit: float,
    amt_annuity: float,
    days_birth: float,
    days_employed: float,
) -> str:
    """
    Generate a SHAP-based explanation of a PD prediction.
    Returns top risk-increasing and risk-decreasing feature contributions.

    Args:
        amt_income_total: Annual gross income
        amt_credit: Loan credit amount
        amt_annuity: Loan annuity amount
        days_birth: Age as negative days
        days_employed: Employment duration as negative days
    """
    models = _load_models()

    if models.get('shap_explainer') is None:
        return json.dumps({'error': 'SHAP explainer not found. Train models first.'})

    # Build features (same as predict_pd)
    xgb_feats = models['xgb_feats']
    features = {col: 0.0 for col in xgb_feats}

    dti_ratio = amt_annuity / max(amt_income_total, 1)
    age_years = abs(days_birth) / 365.25
    employment_years = abs(days_employed) / 365.25 if days_employed < 0 else 0
    employed_flag = 1 if days_employed != 365243 else 0
    employment_ratio = employment_years / max(age_years, 1)
    income_stability_score = 0.5 * employment_ratio + 0.3 * employed_flag + 0.2 * min(employment_years, 30) / 30

    mapping = {
        'dti_ratio': dti_ratio,
        'credit_income_ratio': amt_credit / max(amt_income_total, 1),
        'annuity_credit_ratio': amt_annuity / max(amt_credit, 1),
        'goods_credit_ratio': amt_credit / max(amt_credit, 1),
        'age_years': age_years,
        'employment_years': employment_years,
        'employed_flag': employed_flag,
        'employment_ratio': employment_ratio,
        'income_stability_score': income_stability_score,
        'AMT_INCOME_TOTAL': amt_income_total,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.5,
        'EXT_SOURCE_3': 0.5,
        'ext_source_mean': 0.5,
        'ext_source_prod': 0.125,
        'prev_avg_annuity': amt_annuity,
    }

    for key, val in mapping.items():
        if key in features:
            features[key] = val

    X = pd.DataFrame([features])[xgb_feats]

    # SHAP values
    shap_values = models['shap_explainer'].shap_values(X)[0]
    base_value = float(models['shap_explainer'].expected_value)

    # Top contributors
    contributions = sorted(
        zip(xgb_feats, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    risk_drivers = [
        {'feature': f, 'shap_value': round(float(v), 4), 'direction': 'increases risk' if v > 0 else 'decreases risk'}
        for f, v in contributions[:10]
    ]

    result = {
        'base_value': round(base_value, 4),
        'top_risk_drivers': risk_drivers,
        'explanation': _format_shap_narrative(risk_drivers),
    }

    return json.dumps(result, indent=2)


def _format_shap_narrative(risk_drivers):
    """Convert SHAP values into a readable narrative."""
    increasing = [d for d in risk_drivers if d['direction'] == 'increases risk']
    decreasing = [d for d in risk_drivers if d['direction'] == 'decreases risk']

    parts = []
    if increasing:
        factors = ', '.join([f"{d['feature']} (+{abs(d['shap_value']):.3f})" for d in increasing[:3]])
        parts.append(f"Key risk-increasing factors: {factors}")
    if decreasing:
        factors = ', '.join([f"{d['feature']} (-{abs(d['shap_value']):.3f})" for d in decreasing[:3]])
        parts.append(f"Risk-reducing factors: {factors}")

    return '. '.join(parts) + '.'


@tool
def log_application(
    pd_ensemble: float,
    risk_band: str,
    amt_income_total: float,
    amt_credit: float,
    quoted_rate: float = None,
    expected_loss: float = None,
    decision: str = "REVIEW",
) -> str:
    """
    Log a scored application to the PostgreSQL database.
    This creates a record that Grafana can query for portfolio monitoring.

    Args:
        pd_ensemble: Ensemble PD score
        risk_band: Low / Medium / High
        amt_income_total: Applicant income
        amt_credit: Loan amount
        quoted_rate: Recommended rate (optional, Phase 2)
        expected_loss: Expected loss amount (optional, Phase 2)
        decision: APPROVE / DECLINE / REVIEW
    """
    from sqlalchemy import text

    session, engine = get_db_session()

    try:
        session.execute(
            text("""
                INSERT INTO loan_applications
                (amt_income_total, amt_credit, pd_ensemble, risk_band,
                 quoted_rate, expected_loss, decision, model_version, created_at)
                VALUES (:income, :credit, :pd, :band, :rate, :el, :decision, :version, :ts)
            """),
            {
                'income': amt_income_total,
                'credit': amt_credit,
                'pd': pd_ensemble,
                'band': risk_band,
                'rate': quoted_rate,
                'el': expected_loss,
                'decision': decision,
                'version': 'v1.0-phase1',
                'ts': datetime.utcnow(),
            }
        )
        session.commit()

        return json.dumps({
            'status': 'success',
            'message': f'Application logged. Decision: {decision}, PD: {pd_ensemble:.4f}, Band: {risk_band}'
        })
    except Exception as e:
        session.rollback()
        return json.dumps({'status': 'error', 'message': str(e)})
    finally:
        session.close()
        engine.dispose()


# ══════════════════════════════════════════════
# LANGGRAPH AGENT DEFINITION
# ══════════════════════════════════════════════

# Agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# All tools the agent can use
TOOLS = [
    get_market_benchmark,
    refresh_market_data,
    log_application,
]

SYSTEM_PROMPT = """You are a Senior Credit Analyst AI assistant for a Dynamic Loan Pricing Engine.

Your role is to help loan officers and credit product managers:
1. Explain predictions, SHAP feature contributions, and risk drivers.
2. Explain Expected Loss and minimum rate floors.
3. Check market competitiveness of recommended rates.
4. Log scored applications to the portfolio database.

WORKFLOW for scoring an applicant (CRITICAL INSTRUCTION):
DO NOT attempt to score the applicant or run computational pipelines yourself. 
If the user asks you to score an applicant based on data, respond by instructing them to:
1. Enter the applicant's details using the sliders on the left-hand sidebar.
2. Click the 'Score Applicant' button directly in the UI.
3. Tell them that once they click the button, all the complex SHAP metrics and Probability of Default results will seamlessly sync into your system context.

When the user asks you to explain the current metrics (or if metrics are present in your 'SYSTEM CONTEXT LAYER'):
Read the hidden context injected into your prompt and explain the SHAP values, risk band, and Expected Loss breakdowns logically and professionally in plain language.
Flag any concerns (like high DTI or risky external scores) explicitly.

When the user asks about market rates or competitiveness, use get_market_benchmark.
If benchmarks are expired, offer to optionally refresh_market_data.
"""

def create_agent():
    """Build and compile the LangGraph agent."""
    config = load_config()
    llm_config = config.get('agent', {}).get('llm', {})
    langgraph_config = config.get('agent', {}).get('langgraph', {})

    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key or google_key == 'your_gemini_api_key_here':
        raise ValueError(
            "GOOGLE_API_KEY not set. Add your Gemini API key to .env file.\n"
            "Get one from: https://aistudio.google.com/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model=llm_config.get('model', 'gemini-2.5-flash'),
        google_api_key=google_key,
        temperature=llm_config.get('temperature', 0.1),
        max_output_tokens=llm_config.get('max_tokens', 2048),
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    # Define nodes
    def agent_node(state: AgentState):
        """The LLM reasoning node \u2014 decides which tool to call or responds."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}

    def should_continue(state: AgentState) -> Literal["tools", END]:
        """Route to tools if the LLM made tool calls, otherwise end."""
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    max_recursion = langgraph_config.get('recursion_limit', 15)
    agent = graph.compile()

    return agent, max_recursion


def run_agent(agent, user_message: str, max_recursion: int = 15) -> str:
    """Run the agent with a user message. Returns the final response."""
    result = agent.invoke(
        {'messages': [HumanMessage(content=user_message)]},
        config={'recursion_limit': max_recursion},
    )

    # Get the last AI message
    for msg in reversed(result['messages']):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            # Handle Gemini SDK returning a list of dict blocks instead of a string
            if isinstance(content, list):
                texts = [blk.get('text', '') for blk in content if isinstance(blk, dict) and 'text' in blk]
                return "\n".join(texts)
            return str(content)

    return "Agent did not produce a response."


# ──────────────────────────────────────────────
# Interactive CLI
# ──────────────────────────────────────────────
def interactive_mode():
    """Run the agent in interactive chat mode."""
    print("=" * 60)
    print("  DYNAMIC LOAN PRICING ENGINE — Underwriting Agent")
    print("  Powered by Gemini + LangGraph")
    print("=" * 60)
    print("\nType your query (or 'quit' to exit).\n")
    print("Example queries:")
    print('  "Score a 30-year-old salaried applicant, 6L income, wants 15L loan"')
    print('  "What are current market rates for medium risk borrowers?"')
    print('  "Are our rates competitive for high-risk applicants?"')
    print()

    agent, max_recursion = create_agent()

    while True:
        try:
            user_input = input("🧑 You: ").strip()
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\n👋 Goodbye!")
                break
            if not user_input:
                continue

            print("\n🤖 Agent: ", end="", flush=True)
            response = run_agent(agent, user_input, max_recursion)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == '__main__':
    interactive_mode()
