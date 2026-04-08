"""
seed_portfolio.py — Seed Loan Applications for Grafana Dashboards
=====
Scores a batch of test applicants through the full V2 pipeline
and writes results to the PostgreSQL `loan_applications` table.

This gives Grafana real data to visualize in the Portfolio Analytics
and Scenario Analysis dashboards.

Usage:
    python src/data/seed_portfolio.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import yaml
from datetime import datetime, timedelta
from tqdm import tqdm

# Fix Windows cp1252 encoding crash with emoji/unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.models.expected_loss import ExpectedLossCalculator
from src.models.price_elasticity import PriceElasticityModel
from src.models.optimization_engine import PricingOptimizer


def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_db_session(config):
    db = config['database']
    url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    return Session(), engine


def load_v2_models(config):
    models_path = os.path.join(ROOT_DIR, config['paths']['models'], 'v2')
    return {
        'xgb_model':  joblib.load(os.path.join(models_path, 'xgboost_model.joblib')),
        'lr_model':   joblib.load(os.path.join(models_path, 'logistic_model.joblib')),
        'lr_scaler':  joblib.load(os.path.join(models_path, 'logistic_scaler.joblib')),
        'xgb_feats':  joblib.load(os.path.join(models_path, 'xgb_feature_columns.joblib')),
        'lr_feats':   joblib.load(os.path.join(models_path, 'lr_feature_columns.joblib')),
        'ensemble_config': json.load(open(os.path.join(models_path, 'ensemble_config.json'))),
    }


def predict_pd_v2(row_feats, models):
    """Score a single row using V2 ensemble."""
    xgb_feats = models['xgb_feats']
    lr_feats  = models['lr_feats']

    xgb_dict = {c: row_feats.get(c, 0.0) for c in xgb_feats}
    lr_dict  = {c: row_feats.get(c, 0.0) for c in lr_feats}

    X_xgb = pd.DataFrame([xgb_dict])[xgb_feats]
    X_lr  = pd.DataFrame([lr_dict])[lr_feats]

    pd_xgb = float(models['xgb_model'].predict_proba(X_xgb)[:, 1][0])
    pd_lr  = float(models['lr_model'].predict_proba(models['lr_scaler'].transform(X_lr))[:, 1][0])

    w = models['ensemble_config']['optimal_weights']
    pd_ens = w['xgboost'] * pd_xgb + w['logistic'] * pd_lr

    if pd_ens < 0.08:
        band = 'Low'
    elif pd_ens < 0.20:
        band = 'Medium'
    else:
        band = 'High'

    return pd_xgb, pd_lr, pd_ens, band


def run_seeding():
    config = load_config()
    rng = np.random.default_rng(seed=42)

    print("=" * 60)
    print("  PORTFOLIO SEEDER — Populating loan_applications for Grafana")
    print("=" * 60)

    # Load models + optimizer
    models = load_v2_models(config)
    optimizer = PricingOptimizer(config)
    el_calc  = ExpectedLossCalculator(config)
    w_xgb = models['ensemble_config']['optimal_weights']['xgboost']

    # Load test features
    processed_path = os.path.join(ROOT_DIR, config['paths']['processed_data'])
    df = pd.read_csv(os.path.join(processed_path, 'features_test.csv'))
    print(f"\n📂 Loaded {len(df):,} test records")

    # Sample a manageable subset (500 applications)
    n_sample = min(500, len(df))
    sample = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
    print(f"   Sampling {n_sample} for seeding")

    # Score each applicant
    print("\n🧠 Scoring applicants through full pipeline...")
    records = []
    decisions = {'APPROVE': 0, 'DECLINE': 0, 'REVIEW': 0}

    for idx, row in tqdm(sample.iterrows(), total=n_sample, desc="   Scoring"):
        feats = row.to_dict()

        try:
            pd_xgb, pd_lr, pd_ens, band = predict_pd_v2(feats, models)
        except Exception as e:
            print("SCO SKIP:", e)
            continue

        # EL + Optimization
        amt_credit = feats.get('AMT_CREDIT', 500000)
        amt_income = feats.get('AMT_INCOME_TOTAL', 300000)

        el = el_calc.calculate(pd_ens, amt_credit)

        try:
            opt = optimizer.optimize_rate(pd_ens, amt_credit, band)
        except Exception as e:
            print("OPT SKIP:", e)
            continue

        # Decision logic
        if pd_ens < 0.10:
            decision = 'APPROVE'
        elif pd_ens < 0.35:
            decision = 'REVIEW'
        else:
            decision = 'DECLINE'
        decisions[decision] += 1

        # Simulate spread of created_at over last 30 days
        days_ago = rng.integers(0, 30)
        hours_ago = rng.integers(0, 24)
        from datetime import timezone
        created = datetime.now(timezone.utc) - timedelta(days=int(days_ago), hours=int(hours_ago))

        records.append({
            'amt_income_total':       float(amt_income),
            'amt_credit':             float(amt_credit),
            'amt_annuity':            float(feats.get('AMT_ANNUITY', 25000)),
            'days_birth':             float(feats.get('DAYS_BIRTH', -10950)),
            'days_employed':          float(feats.get('DAYS_EMPLOYED', -2000)),
            'dti_ratio':              float(feats.get('dti_ratio', 0.1)),
            'income_stability_score': float(feats.get('income_stability_score', 0.5)),
            'risk_band':              band,
            'pd_xgboost':            pd_xgb,
            'pd_logistic':           pd_lr,
            'pd_ensemble':           pd_ens,
            'ensemble_weight_xgb':   w_xgb,
            'lgd':                   0.65,
            'ead':                   float(amt_credit),
            'expected_loss':         el['expected_loss'],
            'min_rate_floor':        el['min_rate_floor'],
            'quoted_rate':           float(opt['optimal_rate']),
            'expected_profit':       float(opt['expected_profit']),
            'acceptance_probability': float(opt['acceptance_probability']),
            'decision':               decision,
            'model_version':          'v2-optuna',
            'created_at':             created,
        })

    print(f"\\n📊 Scored {len(records)} applicants successfully")
    print(f"   Decisions: {decisions}")

    # Write to PostgreSQL
    session, engine = get_db_session(config)
    try:
        print("\\n💾 Writing to PostgreSQL `loan_applications`...")

        for rec in tqdm(records, desc="   Inserting"):
            try:
                session.execute(
                    text("""
                        INSERT INTO loan_applications
                        (amt_income_total, amt_credit, amt_annuity,
                         days_birth, days_employed, dti_ratio, income_stability_score,
                         risk_band, pd_xgboost, pd_logistic, pd_ensemble, ensemble_weight_xgb,
                         lgd, ead, expected_loss, min_rate_floor,
                         quoted_rate, expected_profit, acceptance_probability,
                         decision, model_version, created_at)
                        VALUES
                        (:amt_income_total, :amt_credit, :amt_annuity,
                         :days_birth, :days_employed, :dti_ratio, :income_stability_score,
                         :risk_band, :pd_xgboost, :pd_logistic, :pd_ensemble, :ensemble_weight_xgb,
                         :lgd, :ead, :expected_loss, :min_rate_floor,
                         :quoted_rate, :expected_profit, :acceptance_probability,
                         :decision, :model_version, :created_at)
                    """),
                    rec
                )
            except Exception as inner_e:
                import traceback
                print(f"INNER DB ERROR: {inner_e}")
                traceback.print_exc()

        session.commit()
        print(f"\\n✅ {len(records)} applications seeded successfully!")

        # Verify
        count = session.execute(text("SELECT COUNT(*) FROM loan_applications")).scalar()
        print(f"   Total rows in loan_applications: {count}")

    except Exception as e:
        session.rollback()
        import traceback
        print(f"\\n❌ Seeding failed: {e}")
        traceback.print_exc()
    finally:
        session.close()
        engine.dispose()

    # Also seed a market benchmark from config fallback
    print("\n📈 Seeding market benchmark from config...")
    session2, engine2 = get_db_session(config)
    try:
        benchmarks = config['market_benchmarks']
        base_rate = benchmarks['base_rate']
        premiums = benchmarks['risk_premium']

        session2.execute(
            text("""
                INSERT INTO market_benchmarks
                (low_risk_rate, medium_risk_rate, high_risk_rate,
                 source_query, fetch_type, is_active, confidence_score,
                 fetched_at, expires_at)
                VALUES
                (:low, :med, :high,
                 :query, :fetch_type, 1, 1.0,
                 :fetched, :expires)
            """),
            {
                'low':  base_rate + premiums['Low'],
                'med':  base_rate + premiums['Medium'],
                'high': base_rate + premiums['High'],
                'query': 'Config fallback — RBI repo rate + risk premiums',
                'fetch_type': 'seed',
                'fetched': datetime.utcnow(),
                'expires': datetime.utcnow() + timedelta(hours=24),
            }
        )
        session2.commit()
        print("   ✅ Market benchmark seeded.")
    except Exception as e:
        session2.rollback()
        print(f"   ⚠️ Benchmark seed skipped: {e}")
    finally:
        session2.close()
        engine2.dispose()

    print("\n" + "=" * 60)
    print("  SEEDING COMPLETE — Grafana dashboards are ready!")
    print("=" * 60)


if __name__ == '__main__':
    run_seeding()
