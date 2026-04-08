"""
app.py — Dynamic Loan Pricing Engine Dashboard (Streamlit)
=====
View 1: Applicant Underwriting Tool

Runs ALL scoring locally (zero API calls for core pipeline):
  Layer 1 → V2 Ensemble PD (XGBoost + LogReg)
  Layer 2 → Expected Loss (Basel II/III)
  Layer 3 → Price Elasticity (Optuna-tuned)
  Layer 4 → Scipy Rate Optimization

LangGraph Agent chat is ON-DEMAND only (Gemini API: 5 RPM / 20 RPD).

Usage:
    streamlit run streamlit/app.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import yaml
from dotenv import load_dotenv

# ── Project Path Setup ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

# Load environment variables from .env explicitly
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, '.env'), override=True)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.models.expected_loss import ExpectedLossCalculator
from src.models.price_elasticity import PriceElasticityModel
from src.models.optimization_engine import PricingOptimizer

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Dynamic Loan Pricing Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px; 
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }
    .metric-sublabel {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* ── Risk Band Badges ── */
    .badge-low {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white; padding: 6px 18px; border-radius: 20px;
        font-weight: 600; font-size: 0.9rem; display: inline-block;
    }
    .badge-medium {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white; padding: 6px 18px; border-radius: 20px;
        font-weight: 600; font-size: 0.9rem; display: inline-block;
    }
    .badge-high {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white; padding: 6px 18px; border-radius: 20px;
        font-weight: 600; font-size: 0.9rem; display: inline-block;
    }
    
    /* ── Rate Breakdown ── */
    .rate-breakdown {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        font-family: 'Inter', monospace;
    }
    .rate-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #1e293b;
    }
    .rate-row:last-child { border-bottom: none; }
    .rate-label { color: #94a3b8; }
    .rate-value { color: #e2e8f0; font-weight: 600; }
    .rate-total {
        color: #818cf8 !important;
        font-size: 1.1rem;
        font-weight: 700 !important;
    }
    
    /* ── Constraint Badges ── */
    .constraint-pass {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 4px 12px; border-radius: 8px;
        font-size: 0.8rem; font-weight: 500;
    }
    .constraint-fail {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
        padding: 4px 12px; border-radius: 8px;
        font-size: 0.8rem; font-weight: 500;
    }
    
    /* ── Section Headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
    
    /* ── Chat ── */
    .rate-limit-badge {
        background: rgba(251, 191, 36, 0.15);
        border: 1px solid rgba(251, 191, 36, 0.3);
        color: #fbbf24;
        padding: 6px 14px; border-radius: 8px;
        font-size: 0.8rem; font-weight: 500;
        text-align: center;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODEL LOADING (Cached)
# ══════════════════════════════════════════════
@st.cache_resource
def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_v2_models():
    """Load all V2 model artifacts once."""
    config = load_config()
    models_path = os.path.join(ROOT_DIR, config['paths']['models'], 'v2')

    artifacts = {
        'xgb_model':    joblib.load(os.path.join(models_path, 'xgboost_model.joblib')),
        'lgb_model':    joblib.load(os.path.join(models_path, 'lightgbm_model.joblib')),
        'lr_model':     joblib.load(os.path.join(models_path, 'logistic_model.joblib')),
        'lr_scaler':    joblib.load(os.path.join(models_path, 'logistic_scaler.joblib')),
        'xgb_feats':    joblib.load(os.path.join(models_path, 'xgb_feature_columns.joblib')),
        'lgb_feats':    joblib.load(os.path.join(models_path, 'lgb_feature_columns.joblib')),
        'lr_feats':     joblib.load(os.path.join(models_path, 'lr_feature_columns.joblib')),
        'shap_explainer': joblib.load(os.path.join(models_path, 'shap_explainer.joblib')),
    }

    with open(os.path.join(models_path, 'ensemble_config.json'), 'r') as f:
        artifacts['ensemble_config'] = json.load(f)

    with open(os.path.join(models_path, 'training_metrics.json'), 'r') as f:
        artifacts['training_metrics'] = json.load(f)

    return artifacts


@st.cache_resource
def load_optimizer():
    """Load the pricing optimizer (includes EL + Elasticity models)."""
    return PricingOptimizer(load_config())


# ══════════════════════════════════════════════
# SCORING ENGINE (100% Local — Zero API Calls)
# ══════════════════════════════════════════════
def engineer_features(income, credit, annuity, days_birth, days_employed,
                      ext1, ext2, ext3):
    """Engineer the same features as build_features.py."""
    age_years = abs(days_birth) / 365.25
    emp_years = abs(days_employed) / 365.25 if days_employed < 0 else 0
    employed_flag = 1 if days_employed != 365243 else 0
    emp_ratio = emp_years / max(age_years, 1)

    features = {
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': credit,
        'AMT_ANNUITY': annuity,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'EXT_SOURCE_1': ext1,
        'EXT_SOURCE_2': ext2,
        'EXT_SOURCE_3': ext3,
        'ext_source_mean': np.mean([ext1, ext2, ext3]),
        'ext_source_prod': ext1 * ext2 * ext3,
        'dti_ratio': annuity / max(income, 1),
        'credit_income_ratio': credit / max(income, 1),
        'annuity_credit_ratio': annuity / max(credit, 1),
        'goods_credit_ratio': credit / max(credit, 1),  # Assume goods ≈ credit
        'age_years': age_years,
        'employment_years': emp_years,
        'employed_flag': employed_flag,
        'employment_ratio': emp_ratio,
        'income_stability_score': 0.5 * emp_ratio + 0.3 * employed_flag + 0.2 * min(emp_years, 30) / 30,
        'prev_avg_annuity': annuity,  # Proxy: first application
    }
    return features


def predict_pd_v2(features, models, w_xgb_override=None, w_lgb_override=None, w_lr_override=None):
    """Run V2 ensemble with optional weight overrides."""
    xgb_feats = models['xgb_feats']
    lgb_feats = models['lgb_feats']
    lr_feats = models['lr_feats']
    
    # Build feature vectors
    xgb_dict = {col: features.get(col, 0.0) for col in xgb_feats}
    lgb_dict = {col: features.get(col, 0.0) for col in lgb_feats}
    lr_dict = {col: features.get(col, 0.0) for col in lr_feats}

    X_xgb = pd.DataFrame([xgb_dict])[xgb_feats]
    X_lgb = pd.DataFrame([lgb_dict])[lgb_feats]
    X_lr = pd.DataFrame([lr_dict])[lr_feats]

    # Predict
    pd_xgb = float(models['xgb_model'].predict_proba(X_xgb)[:, 1][0])
    pd_lgb = float(models['lgb_model'].predict_proba(X_lgb)[:, 1][0])
    X_lr_scaled = models['lr_scaler'].transform(X_lr)
    pd_lr = float(models['lr_model'].predict_proba(X_lr_scaled)[:, 1][0])

    # Weights
    cfg_weights = models['ensemble_config']['optimal_weights']
    w_xgb = w_xgb_override if w_xgb_override is not None else cfg_weights.get('xgboost', 0.4)
    w_lgb = w_lgb_override if w_lgb_override is not None else cfg_weights.get('lightgbm', 0.3)
    w_lr = w_lr_override if w_lr_override is not None else cfg_weights.get('logistic', 0.3)

    pd_ensemble = w_xgb * pd_xgb + w_lgb * pd_lgb + w_lr * pd_lr

    # Risk band
    if pd_ensemble < 0.08:
        risk_band = 'Low'
    elif pd_ensemble < 0.20:
        risk_band = 'Medium'
    else:
        risk_band = 'High'

    return {
        'pd_xgb': pd_xgb,
        'pd_lgb': pd_lgb,
        'pd_lr': pd_lr,
        'pd_ensemble': pd_ensemble,
        'risk_band': risk_band,
        'w_xgb': w_xgb,
        'w_lgb': w_lgb,
        'w_lr': w_lr,
    }


def get_shap_explanation(features, models):
    """Get SHAP values for the XGBoost model."""
    xgb_feats = models['xgb_feats']
    xgb_dict = {col: features.get(col, 0.0) for col in xgb_feats}
    X_xgb = pd.DataFrame([xgb_dict])[xgb_feats]

    shap_values = models['shap_explainer'].shap_values(X_xgb)[0]
    base_value = float(models['shap_explainer'].expected_value)

    contributions = sorted(
        zip(xgb_feats, shap_values, X_xgb.iloc[0].values),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return contributions[:10], base_value


# ══════════════════════════════════════════════
# RATE LIMITER (for LangGraph Agent)
# ══════════════════════════════════════════════
def check_rate_limit():
    """Check if we can make an API call (5 RPM, 20 RPD)."""
    now = time.time()
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = []

    # Clean old entries
    st.session_state.api_calls = [t for t in st.session_state.api_calls if now - t < 86400]

    # Check limits
    calls_last_minute = sum(1 for t in st.session_state.api_calls if now - t < 60)
    calls_today = len(st.session_state.api_calls)

    return {
        'can_call': calls_last_minute < 5 and calls_today < 20,
        'rpm_used': calls_last_minute,
        'rpd_used': calls_today,
        'rpm_limit': 5,
        'rpd_limit': 20,
    }


def record_api_call():
    """Record an API call timestamp."""
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = []
    st.session_state.api_calls.append(time.time())


# ══════════════════════════════════════════════
# PLOTLY CHART BUILDERS
# ══════════════════════════════════════════════
CHART_TEMPLATE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#e2e8f0'),
    margin=dict(l=40, r=20, t=40, b=40),
)


def build_profit_curve(grid_analysis, optimal_rate, risk_band):
    """Build the Expected Profit vs Quoted Rate curve."""
    df = pd.DataFrame(grid_analysis)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['quoted_rate'] * 100,
        y=df['expected_profit'],
        mode='lines',
        line=dict(color='#818cf8', width=3),
        fill='tozeroy',
        fillcolor='rgba(129, 140, 248, 0.1)',
        name='E[Profit]',
        hovertemplate='Rate: %{x:.2f}%<br>E[Profit]: ₹%{y:,.0f}<extra></extra>',
    ))

    # Optimal rate marker
    opt_profit = df.loc[(df['quoted_rate'] - optimal_rate).abs().idxmin(), 'expected_profit']
    fig.add_trace(go.Scatter(
        x=[optimal_rate * 100],
        y=[opt_profit],
        mode='markers+text',
        marker=dict(color='#f59e0b', size=14, symbol='star',
                    line=dict(color='white', width=2)),
        text=[f'  Optimal: {optimal_rate*100:.2f}%'],
        textposition='top right',
        textfont=dict(color='#f59e0b', size=12, family='Inter'),
        name='Optimal',
        showlegend=False,
    ))

    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text=f'Expected Profit Curve — {risk_band} Risk',
                   font=dict(size=16)),
        xaxis_title='Quoted Rate (%)',
        yaxis_title='Expected Profit (₹)',
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
        height=350,
    )
    return fig


def build_acceptance_curve(optimizer, risk_band, optimal_rate, rate_floor, market_rate):
    """Build acceptance probability vs rate curve."""
    elasticity = optimizer.elasticity_model
    curve = elasticity.generate_curve(risk_band, n_points=80)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve['quoted_rate'] * 100,
        y=curve['acceptance_probability'] * 100,
        mode='lines',
        line=dict(color='#34d399', width=3),
        fill='tozeroy',
        fillcolor='rgba(52, 211, 153, 0.08)',
        name='P(Accept)',
        hovertemplate='Rate: %{x:.2f}%<br>P(Accept): %{y:.1f}%<extra></extra>',
    ))

    # Vertical lines for key rates
    for label, val, color, dash in [
        ('Floor', rate_floor, '#ef4444', 'dash'),
        ('Market', market_rate, '#94a3b8', 'dot'),
        ('Optimal', optimal_rate, '#f59e0b', 'solid'),
    ]:
        fig.add_vline(x=val * 100, line=dict(color=color, width=2, dash=dash))
        fig.add_annotation(
            x=val * 100, y=95,
            text=f'{label}<br>{val*100:.1f}%',
            showarrow=False,
            font=dict(color=color, size=10),
            bgcolor='rgba(15,23,42,0.8)',
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text=f'Acceptance Probability — {risk_band} Risk', font=dict(size=16)),
        xaxis_title='Quoted Rate (%)',
        yaxis_title='P(Accept) %',
        xaxis=dict(gridcolor='rgba(52,211,153,0.1)'),
        yaxis=dict(gridcolor='rgba(52,211,153,0.1)', range=[0, 100]),
        height=350,
    )
    return fig


def build_shap_chart(contributions, base_value):
    """Horizontal bar chart of SHAP feature contributions."""
    feats = [c[0] for c in reversed(contributions[:8])]
    vals = [c[1] for c in reversed(contributions[:8])]
    colors = ['#ef4444' if v > 0 else '#34d399' for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats,
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))

    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text='SHAP Risk Drivers (XGBoost)', font=dict(size=16)),
        xaxis_title='SHAP Value (→ increases risk)',
        height=350,
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)', zeroline=True,
                   zerolinecolor='rgba(148,163,184,0.3)'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
    )
    return fig


def build_el_sensitivity_chart(el_calc, pd_val, loan_amount):
    """LGD Sensitivity waterfall."""
    sens = el_calc.sensitivity_analysis(pd_val, loan_amount)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sens['lgd'] * 100,
        y=sens['min_rate_floor'] * 100,
        mode='lines+markers',
        line=dict(color='#f472b6', width=3),
        marker=dict(size=8, color='#f472b6'),
        name='Min Rate Floor',
        hovertemplate='LGD: %{x}%<br>Floor: %{y:.2f}%<extra></extra>',
    ))

    fig.add_hline(y=36, line=dict(color='#ef4444', dash='dash', width=1),
                  annotation_text='Regulatory Ceiling (36%)')

    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text='Rate Floor Sensitivity to LGD', font=dict(size=16)),
        xaxis_title='LGD Assumption (%)',
        yaxis_title='Min Rate Floor (%)',
        xaxis=dict(gridcolor='rgba(244,114,182,0.1)'),
        yaxis=dict(gridcolor='rgba(244,114,182,0.1)'),
        height=300,
    )
    return fig


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏦 Loan Pricing Engine")
        st.markdown("##### V2 — Optuna-Tuned Ensemble")
        st.divider()

        # ── Applicant Details ──
        st.markdown("### 📋 Applicant Details")

        income = st.number_input("Annual Income (₹)", min_value=100000,
                                 max_value=50000000, value=600000, step=50000,
                                 help="Gross annual income")
        credit = st.number_input("Loan Amount (₹)", min_value=50000,
                                 max_value=50000000, value=1500000, step=100000,
                                 help="Total credit/loan amount requested")
        annuity = st.number_input("Monthly Payment (₹)", min_value=1000,
                                  max_value=5000000, value=35000, step=1000,
                                  help="Loan annuity / EMI")

        st.markdown("### 👤 Applicant Profile")
        age = st.slider("Age (years)", 21, 65, 30)
        emp_years = st.slider("Employment (years)", 0, 40, 5)

        st.markdown("### 📊 Bureau Scores")
        ext1 = st.slider("External Score 1", 0.0, 1.0, 0.5, 0.01)
        ext2 = st.slider("External Score 2", 0.0, 1.0, 0.5, 0.01)
        ext3 = st.slider("External Score 3", 0.0, 1.0, 0.5, 0.01)

        st.divider()

        # ── Compliance Override ──
        st.markdown("### ⚖️ Compliance Override")
        models = load_v2_models()
        cfg_w = models['ensemble_config']['optimal_weights']
        
        # Dirichlet-style sum-to-1 weighting
        w_xgb = st.slider("XGBoost Weight", 0.0, 1.0, float(cfg_w.get('xgboost', 0.4)), 0.01)
        w_lgb = st.slider("LightGBM Weight", 0.0, 1.0 - w_xgb, float(cfg_w.get('lightgbm', 0.3)), 0.01)
        w_lr = round(1.0 - w_xgb - w_lgb, 2)
        
        st.caption(f"XGB: {w_xgb:.0%} | LGB: {w_lgb:.0%} | LR: {w_lr:.0%}")

        st.divider()

        # ── Elasticity Stress Multiplier ──
        st.markdown("### 📈 Sensitivity Override")
        sensitivity_mult = st.slider(
            "Market Sensitivity",
            0.5, 2.0, 1.0, 0.1,
            help="1.0 = normal. >1 = borrowers more price-sensitive. <1 = less sensitive"
        )

        st.divider()
        score_btn = st.button("🚀 Score Applicant", type="primary", use_container_width=True)

        return {
            'income': income,
            'credit': credit,
            'annuity': annuity,
            'days_birth': -int(age * 365.25),
            'days_employed': -int(emp_years * 365.25),
            'ext1': ext1, 'ext2': ext2, 'ext3': ext3,
            'w_xgb': w_xgb,
            'sensitivity_mult': sensitivity_mult,
            'score_btn': score_btn,
        }


# ══════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════
def main():
    inputs = render_sidebar()
    models = load_v2_models()
    optimizer = load_optimizer()
    config = load_config()
    el_calc = ExpectedLossCalculator(config)

    # ── Header ──
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h1 style="background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.4rem; font-weight: 700; margin-bottom: 4px;">
            Dynamic Loan Pricing Engine
        </h1>
        <p style="color: #64748b; font-size: 1rem;">
            Risk Assessment · Expected Loss · Rate Optimization · SHAP Explainability
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Score on button press ──
    if inputs['score_btn']:
        with st.spinner("Running scoring pipeline..."):
            # Engineer features
            features = engineer_features(
                inputs['income'], inputs['credit'], inputs['annuity'],
                inputs['days_birth'], inputs['days_employed'],
                inputs['ext1'], inputs['ext2'], inputs['ext3']
            )

            # Layer 1: PD
            pd_result = predict_pd_v2(
                features, models,
                w_xgb_override=inputs['w_xgb'],
                w_lgb_override=inputs['w_lgb'],
                w_lr_override=inputs['w_lr']
            )

            # Layer 2: Expected Loss
            el_result = el_calc.calculate(pd_result['pd_ensemble'], inputs['credit'])

            # Layer 4: Optimization
            opt_result = optimizer.optimize_rate(
                pd_result['pd_ensemble'], inputs['credit'], pd_result['risk_band']
            )

            # SHAP
            shap_contribs, base_val = get_shap_explanation(features, models)

            # Store in session
            st.session_state['results'] = {
                'pd': pd_result,
                'el': el_result,
                'opt': opt_result,
                'shap': (shap_contribs, base_val),
                'features': features,
                'inputs': inputs,
            }

    # ── Display Results ──
    if 'results' in st.session_state:
        res = st.session_state['results']
        pd_r = res['pd']
        el_r = res['el']
        opt_r = res['opt']
        shap_data = res['shap']
        inp = res['inputs']

        # ── Row 1: Key Metrics ──
        c1, c2, c3, c4, c5 = st.columns(5)

        band = pd_r['risk_band']
        badge_class = f"badge-{band.lower()}"

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Band</div>
                <div style="margin:12px 0"><span class="{badge_class}">{band}</span></div>
                <div class="metric-sublabel">PD Ensemble</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Probability of Default</div>
                <div class="metric-value">{pd_r['pd_ensemble']*100:.2f}%</div>
                <div class="metric-sublabel">XGB: {pd_r['pd_xgb']*100:.1f}% | LGB: {pd_r['pd_lgb']*100:.1f}% | LR: {pd_r['pd_lr']*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Loss</div>
                <div class="metric-value">{el_r['formatted']['expected_loss']}</div>
                <div class="metric-sublabel">{el_r['formatted']['el_rate']} of loan</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Optimal Rate</div>
                <div class="metric-value">{opt_r['optimal_rate_pct']}</div>
                <div class="metric-sublabel">Floor: {opt_r['rate_floor_pct']}</div>
            </div>""", unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">P(Accept)</div>
                <div class="metric-value">{opt_r['acceptance_pct']}</div>
                <div class="metric-sublabel">E[Profit]: ₹{opt_r['expected_profit']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Charts ──
        col_left, col_right = st.columns(2)

        with col_left:
            fig_profit = build_profit_curve(
                opt_r['grid_analysis'], opt_r['optimal_rate'], pd_r['risk_band']
            )
            st.plotly_chart(fig_profit, use_container_width=True)

        with col_right:
            fig_accept = build_acceptance_curve(
                optimizer, pd_r['risk_band'],
                opt_r['optimal_rate'], opt_r['rate_floor'],
                opt_r['market_benchmark']
            )
            st.plotly_chart(fig_accept, use_container_width=True)

        # ── Row 3: SHAP + Rate Breakdown ──
        col_shap, col_breakdown = st.columns([3, 2])

        with col_shap:
            fig_shap = build_shap_chart(*shap_data)
            st.plotly_chart(fig_shap, use_container_width=True)

        with col_breakdown:
            st.markdown('<div class="section-header">📐 Rate Build-Up</div>',
                        unsafe_allow_html=True)

            breakdown = el_r['rate_breakdown']
            items = [
                ("Expected Loss", breakdown['el_component'], False),
                ("Cost of Capital", breakdown['cost_of_capital'], False),
                ("Operational Margin", breakdown['operational_margin'], False),
                ("Rate Floor", breakdown['total_floor'], True),
                ("Market Benchmark", opt_r['market_benchmark'], False),
                ("✅ Optimal Rate", opt_r['optimal_rate'], True),
            ]

            html = '<div class="rate-breakdown">'
            for label, val, is_total in items:
                cls = 'rate-total' if is_total else 'rate-value'
                html += f"""<div class="rate-row">
                    <span class="rate-label">{label}</span>
                    <span class="{cls}">{val*100:.2f}%</span>
                </div>"""
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

            # Constraint checks
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔒 Constraints</div>',
                        unsafe_allow_html=True)

            constraints = opt_r['constraints']
            for label, ok in [
                ("Rate ≥ Floor", constraints['rate_floor_respected']),
                ("Rate ≤ Ceiling (36%)", constraints['regulatory_ceiling_respected']),
                ("Within Competitive Band", constraints['within_competitive_band']),
            ]:
                cls = 'constraint-pass' if ok else 'constraint-fail'
                ico = '✅' if ok else '⚠️'
                st.markdown(f'<span class="{cls}">{ico} {label}</span>&nbsp;',
                            unsafe_allow_html=True)

        # ── Row 4: LGD Sensitivity ──
        with st.expander("📊 LGD Sensitivity Analysis"):
            fig_lgd = build_el_sensitivity_chart(
                el_calc, pd_r['pd_ensemble'], inp['credit']
            )
            st.plotly_chart(fig_lgd, use_container_width=True)

        # ── Row 5: Ensemble Weights Comparison ──
        with st.expander("⚖️ Ensemble Weight Details"):
            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                st.metric("XGBoost Weight", f"{pd_r['w_xgb']:.1%}")
                st.metric("XGBoost PD", f"{pd_r['pd_xgb']*100:.4f}%")
            with w_col2:
                st.metric("LightGBM Weight", f"{pd_r['w_lgb']:.1%}")
                st.metric("LightGBM PD", f"{pd_r['pd_lgb']*100:.4f}%")
            with w_col3:
                st.metric("LogReg Weight", f"{pd_r['w_lr']:.1%}")
                st.metric("LogReg PD", f"{pd_r['pd_lr']*100:.4f}%")

            metrics = models.get('training_metrics', {})
            if metrics:
                st.markdown("**V2 Training Metrics:**")
                st.json(metrics)

    else:
        # Landing state
        st.markdown("""
        <div style="text-align:center; padding:60px 0; color:#64748b;">
            <p style="font-size:3rem; margin-bottom:8px;">👈</p>
            <p style="font-size:1.2rem;">Configure applicant details in the sidebar</p>
            <p style="font-size:1rem;">then press <b>🚀 Score Applicant</b></p>
        </div>
        """, unsafe_allow_html=True)

    # ── Agent Chat (Bottom Section) ──
    st.divider()
    st.markdown("### 🤖 Credit Analyst Agent")
    st.caption("Powered by Gemini + LangGraph • Rate Limited: 5 RPM / 20 RPD")

    rate_status = check_rate_limit()
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown(
            f'<div class="rate-limit-badge">'
            f'Per-Minute: {rate_status["rpm_used"]}/{rate_status["rpm_limit"]}'
            f'</div>', unsafe_allow_html=True
        )
    with r_col2:
        st.markdown(
            f'<div class="rate-limit-badge">'
            f'Per-Day: {rate_status["rpd_used"]}/{rate_status["rpd_limit"]}'
            f'</div>', unsafe_allow_html=True
        )

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Chat input
    if prompt := st.chat_input("Ask the credit analyst agent..."):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if not rate_status['can_call']:
            err_msg = "⚠️ Rate limit reached. Please wait before sending another message."
            st.session_state.chat_history.append({'role': 'assistant', 'content': err_msg})
            with st.chat_message("assistant"):
                st.warning(err_msg)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking..."):
                    try:
                        from src.agents.underwriting_agent import create_agent, run_agent
                        record_api_call()
                        
                        # Inject the Streamlit UI context invisibly into the agent's prompt
                        context_str = "SYSTEM CONTEXT LAYER (Invisible to user):\n"
                        context_str += "IMPORTANT INSTRUCTION: If the user explicitly provides applicant details in this chat message, prioritize their chat inputs! Only use the following Streamlit Sidebar Inputs if they don't specify otherwise:\n"
                        context_str += "\n".join([f"- {k}: {v}" for k, v in inputs.items() if k != 'score_btn'])
                        
                        if 'results' in st.session_state:
                            res = st.session_state['results']
                            context_str += "\n\nTHE APPLICANT HAS ALREADY BEEN SCORED ON THE DASHBOARD. Here are the precise metrics currently displayed on the user's screen. Use these to explain the results if asked:\n"
                            
                            # Condense results into a readable summary for the LLM
                            summary = {
                                'Risk_Band': res['pd']['risk_band'],
                                'Probability_of_Default': f"{res['pd']['pd_ensemble']*100:.2f}%",
                                'Expected_Loss_Amount': res['el']['formatted']['expected_loss'],
                                'Loss_Given_Default_Assumption': "65% (Standard Unsecured Loan Proxy)",
                                'Optimal_Quoted_Rate': res['opt']['optimal_rate_pct'],
                                'Min_Rate_Floor': res['opt']['rate_floor_pct'],
                                'Key_Risk_Drivers_SHAP': {str(feat): round(float(val), 4) for feat, val, _ in res['shap'][0][:5]},
                                'Ensemble_Model_Weights': {
                                    'XGBoost': f"{res['pd']['w_xgb']*100:.1f}%",
                                    'Logistic_Regression': f"{res['pd']['w_lr']*100:.1f}%"
                                }
                            }
                            context_str += json.dumps(summary, indent=2)
                        
                        enhanced_prompt = f"{context_str}\n\n=====\nUSER MESSAGE:\n{prompt}"
                        
                        agent, max_rec = create_agent()
                        response = run_agent(agent, enhanced_prompt, max_rec)
                        st.markdown(response)
                        st.session_state.chat_history.append(
                            {'role': 'assistant', 'content': response}
                        )
                    except Exception as e:
                        err = f"❌ Agent error: {str(e)}"
                        st.error(err)
                        st.session_state.chat_history.append(
                            {'role': 'assistant', 'content': err}
                        )


if __name__ == '__main__':
    main()
