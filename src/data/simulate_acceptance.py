"""
simulate_acceptance.py — Synthetic Market History Generator (Layer 3)
=====
Uses V2 Risk Model artifacts (models/v2/) to generate synthetic
"Sales History" — borrower acceptance decisions based on offered rates.

Hidden Logic (what the Elasticity Tuner must "discover"):
  1. Rate Spread: Core driver — higher spread → lower acceptance
  2. Risk Band: Low-risk borrowers are most price-sensitive (more alternatives)
  3. Loan Amount: Larger loans → more comparison shopping → higher sensitivity
  4. Income Level: Higher income → slightly more choice → more sensitivity
  5. Noise: Human behavior is inherently noisy

Output: data/processed/acceptance_history.csv

Usage:
    python src/data/simulate_acceptance.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import yaml
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)


def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def load_v2_artifacts(models_v2_path):
    """Load all V2 model artifacts."""
    xgb_model  = joblib.load(os.path.join(models_v2_path, 'xgboost_model.joblib'))
    lr_model   = joblib.load(os.path.join(models_v2_path, 'logistic_model.joblib'))
    lr_scaler  = joblib.load(os.path.join(models_v2_path, 'logistic_scaler.joblib'))
    xgb_feats  = joblib.load(os.path.join(models_v2_path, 'xgb_feature_columns.joblib'))
    lr_feats   = joblib.load(os.path.join(models_v2_path, 'lr_feature_columns.joblib'))

    with open(os.path.join(models_v2_path, 'ensemble_config.json'), 'r') as f:
        ensemble_config = json.load(f)

    w_xgb = ensemble_config['optimal_weights']['xgboost']
    w_lr  = ensemble_config['optimal_weights']['logistic']

    return xgb_model, lr_model, lr_scaler, xgb_feats, lr_feats, w_xgb, w_lr


def compute_pd_ensemble(df, xgb_model, lr_model, lr_scaler, xgb_feats, lr_feats, w_xgb, w_lr):
    """Generate PD ensemble scores from V2 models."""
    # XGB on raw features
    X_xgb = df[[c for c in xgb_feats if c in df.columns]]
    pd_xgb = xgb_model.predict_proba(X_xgb)[:, 1]

    # LR on scaled all-features
    X_lr = df[[c for c in lr_feats if c in df.columns]]
    X_lr_scaled = lr_scaler.transform(X_lr)
    pd_lr = lr_model.predict_proba(X_lr_scaled)[:, 1]

    return (w_xgb * pd_xgb) + (w_lr * pd_lr)


def assign_risk_band(pd_score):
    """Consistent risk banding with the rest of the pipeline."""
    if pd_score < 0.08:
        return 'Low'
    elif pd_score < 0.20:
        return 'Medium'
    else:
        return 'High'


def simulate_market_offer(risk_band, market_rates, rng):
    """
    Simulate a loan offer to a borrower.
    Offered rate = Market Rate + random spread (-3% to +7%)
    This simulates a mix of competitive and profitable offers.
    """
    mkt = market_rates[risk_band]
    spread = rng.uniform(-0.03, 0.07)
    offered = np.clip(mkt + spread, 0.04, 0.36)
    return mkt, offered


def hidden_acceptance_logic(rate_spread, risk_band, amt_credit, amt_income, rng):
    """
    THE HIDDEN TRUTH — the signal the Elasticity Tuner must recover.

    Formula: logit = a_base - b_final × rate_spread
    where b_final = b_base × loan_size_factor × income_factor

    This is intentionally complex and multi-dimensional so that the
    Optuna-tuned LogReg has a real, non-trivial task to solve.
    """
    # Base parameters per risk band
    band_params = {
        'Low':    {'a': 1.80, 'b': 15.0},   # 85% at market, very sensitive
        'Medium': {'a': 1.30, 'b': 8.0},    # 79% at market, moderately sensitive
        'High':   {'a': 0.95, 'b': 4.0},    # 72% at market, least sensitive
    }
    params = band_params.get(risk_band, band_params['Medium'])

    # Loan size interaction: larger loans → borrowers compare more (+0.5 per 1M)
    loan_millions = np.clip(amt_credit / 1_000_000, 0.1, 5.0)
    loan_factor = 1.0 + (loan_millions - 0.5) * 0.4

    # Income interaction: higher income → slightly more rate-sensitive (+20% per log unit)
    income_log_factor = np.log1p(amt_income / 100_000) / 10.0
    income_factor = 1.0 + income_log_factor * 0.2

    # Final sensitivity
    b_final = params['b'] * loan_factor * income_factor

    # Compute acceptance probability
    logit = params['a'] - b_final * rate_spread
    p_accept = float(_sigmoid(logit))

    # Human noise — real-world acceptance is imperfect
    noise = rng.normal(0, 0.05)
    p_accept = float(np.clip(p_accept + noise, 0.01, 0.99))

    return 1 if rng.random() < p_accept else 0, p_accept


def run_simulation():
    """Main simulation runner."""
    config = load_config()
    rng = np.random.default_rng(seed=42)  # Reproducible

    print("=" * 60)
    print("  DYNAMIC LOAN PRICING — ACCEPTANCE SIMULATOR (V2)")
    print("  Hidden Logic: Spread × Risk × Loan Size × Income × Noise")
    print("=" * 60)

    # ── Load V2 Artifacts ──
    models_v2_path = os.path.join(ROOT_DIR, config['paths']['models'], 'v2')
    if not os.path.exists(models_v2_path):
        print(f"❌ V2 models not found at: {models_v2_path}")
        print("   Please run train_risk_model_v2.py first.")
        return

    print("\n📦 Loading V2 Risk Model artifacts...")
    (xgb_model, lr_model, lr_scaler,
     xgb_feats, lr_feats, w_xgb, w_lr) = load_v2_artifacts(models_v2_path)
    print(f"   Ensemble weights: XGB={w_xgb:.2%}, LR={w_lr:.2%}")

    # ── Load Test Data ──
    # Use test set — it's held-out and not seen by the risk model during training
    processed_path = os.path.join(ROOT_DIR, config['paths']['processed_data'])
    print("\n📂 Loading processed test data...")
    df = pd.read_csv(os.path.join(processed_path, 'features_test.csv'))
    print(f"   Rows: {len(df):,} | Columns: {df.shape[1]}")

    # Keep original scale columns for simulation logic
    amt_credit_col  = 'AMT_CREDIT'  if 'AMT_CREDIT'  in df.columns else None
    amt_income_col  = 'AMT_INCOME_TOTAL' if 'AMT_INCOME_TOTAL' in df.columns else None

    if not amt_credit_col or not amt_income_col:
        print("❌ AMT_CREDIT or AMT_INCOME_TOTAL not found. Check feature engineering.")
        return

    # ── Compute PD Ensemble from V2 Models ──
    print("\n🧠 Computing PD ensemble scores (V2)...")
    df['pd_ensemble'] = compute_pd_ensemble(
        df, xgb_model, lr_model, lr_scaler, xgb_feats, lr_feats, w_xgb, w_lr
    )
    df['risk_band'] = df['pd_ensemble'].apply(assign_risk_band)

    print(f"   PD Stats: mean={df['pd_ensemble'].mean():.4f}, "
          f"median={df['pd_ensemble'].median():.4f}")
    print(f"   Risk Band Distribution:")
    band_counts = df['risk_band'].value_counts()
    for band in ['Low', 'Medium', 'High']:
        count = band_counts.get(band, 0)
        pct = count / len(df) * 100
        print(f"     {band:6s}: {count:6,} ({pct:.1f}%)")

    # ── Market Rate Setup ──
    benchmarks = config['market_benchmarks']
    base_rate = benchmarks['base_rate']
    premiums = benchmarks['risk_premium']
    market_rates = {
        'Low':    base_rate + premiums['Low'],
        'Medium': base_rate + premiums['Medium'],
        'High':   base_rate + premiums['High'],
    }
    print(f"\n📊 Market Rates: Low={market_rates['Low']:.1%}, "
          f"Medium={market_rates['Medium']:.1%}, High={market_rates['High']:.1%}")

    # ── Simulate Offers and Acceptance Decisions ──
    print("\n🎲 Simulating market offers and acceptance decisions...")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Simulating"):
        risk_band   = row['risk_band']
        amt_credit  = row[amt_credit_col]
        amt_income  = row[amt_income_col]
        pd_score    = row['pd_ensemble']

        mkt_rate, offered_rate = simulate_market_offer(risk_band, market_rates, rng)
        rate_spread = offered_rate - mkt_rate

        accepted, p_true = hidden_acceptance_logic(
            rate_spread, risk_band, amt_credit, amt_income, rng
        )

        records.append({
            'pd_ensemble':      pd_score,
            'risk_band':        risk_band,
            'market_rate':      mkt_rate,
            'offered_rate':     offered_rate,
            'rate_spread':      rate_spread,
            'AMT_CREDIT':       amt_credit,
            'AMT_INCOME_TOTAL': amt_income,
            'p_accept_true':    p_true,   # Ground truth (for validation only)
            'ACCEPTED':         accepted,
        })

    sim_df = pd.DataFrame(records)

    # ── Summary Statistics ──
    print(f"\n📈 Simulation Summary:")
    print(f"   Total offers:      {len(sim_df):,}")
    print(f"   Overall acceptance: {sim_df['ACCEPTED'].mean():.1%}")
    print(f"\n   Acceptance by Risk Band:")
    for band in ['Low', 'Medium', 'High']:
        mask = sim_df['risk_band'] == band
        if mask.sum() > 0:
            acc_rate = sim_df.loc[mask, 'ACCEPTED'].mean()
            avg_spread = sim_df.loc[mask, 'rate_spread'].mean()
            print(f"     {band:6s}: {acc_rate:.1%} acceptance | avg spread: {avg_spread:+.2%}")

    print(f"\n   Rate Spread Statistics:")
    print(f"     Mean:  {sim_df['rate_spread'].mean():+.2%}")
    print(f"     Std:   {sim_df['rate_spread'].std():.2%}")
    print(f"     Min:   {sim_df['rate_spread'].min():+.2%}")
    print(f"     Max:   {sim_df['rate_spread'].max():+.2%}")

    # ── Save Output ──
    output_path = os.path.join(processed_path, 'acceptance_history.csv')
    sim_df.to_csv(output_path, index=False)
    print(f"\n✅ Acceptance history saved to: {output_path}")
    print(f"   Shape: {sim_df.shape}")
    print("=" * 60)


if __name__ == '__main__':
    run_simulation()
