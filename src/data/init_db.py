"""
init_db.py — Database Schema Initialization
=====
Creates the PostgreSQL database and the loan_applications table.
Uses SQLAlchemy ORM with declarative base.

Usage:
    python src/data/init_db.py
"""

import sys
import os
import yaml
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Text, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import text

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ──────────────────────────────────────────────
# ORM Model
# ──────────────────────────────────────────────
Base = declarative_base()

class LoanApplication(Base):
    """
    Stores every scored loan application.
    Populated by the Streamlit app at inference time.
    Queried by Grafana for portfolio monitoring.
    """
    __tablename__ = 'loan_applications'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # --- Applicant Identifiers ---
    sk_id_curr = Column(Integer, index=True, comment='Home Credit application ID')

    # --- Raw Applicant Features ---
    amt_income_total = Column(Float, comment='Gross annual income')
    amt_credit = Column(Float, comment='Loan credit amount')
    amt_annuity = Column(Float, comment='Loan annuity amount')
    amt_goods_price = Column(Float, comment='Price of goods for which loan is given')
    name_contract_type = Column(String(50), comment='Cash loans or Revolving loans')
    name_income_type = Column(String(100), comment='Income type (Working, Commercial, etc.)')
    name_education_type = Column(String(100), comment='Education level')
    name_family_status = Column(String(50), comment='Family status')
    name_housing_type = Column(String(50), comment='Housing type')
    days_birth = Column(Float, comment='Age in days (negative from application)')
    days_employed = Column(Float, comment='Employment duration in days')
    cnt_children = Column(Integer, comment='Number of children')

    # --- Engineered Features ---
    dti_ratio = Column(Float, comment='Debt-to-Income ratio proxy')
    credit_utilization = Column(Float, comment='Credit utilization rate')
    payment_history_score = Column(Float, comment='Payment history risk score')
    income_stability_score = Column(Float, comment='Income/employment stability proxy')
    bureau_enquiry_count = Column(Float, comment='Number of bureau enquiries')
    credit_history_length_days = Column(Float, comment='Age of oldest credit account (days)')

    # --- Model Outputs ---
    risk_band = Column(String(20), comment='Low / Medium / High')
    pd_xgboost = Column(Float, comment='PD from XGBoost model')
    pd_logistic = Column(Float, comment='PD from Logistic Regression scorecard')
    pd_ensemble = Column(Float, comment='Weighted ensemble PD')
    ensemble_weight_xgb = Column(Float, comment='XGBoost weight used in ensemble')

    # --- Expected Loss & Pricing (Phase 2+) ---
    lgd = Column(Float, comment='Loss Given Default assumption used')
    ead = Column(Float, comment='Exposure at Default')
    expected_loss = Column(Float, comment='PD × LGD × EAD')
    min_rate_floor = Column(Float, comment='Minimum rate = EL/Loan + CoC + margin')
    quoted_rate = Column(Float, comment='Optimizer-recommended interest rate')
    expected_profit = Column(Float, comment='Expected profit at quoted rate')
    acceptance_probability = Column(Float, comment='Price elasticity P(accept)')

    # --- Decision ---
    decision = Column(String(20), comment='APPROVE / DECLINE / REVIEW')
    decline_reason = Column(Text, comment='Reason if declined')

    # --- Metadata ---
    model_version = Column(String(50), comment='Model version identifier')
    created_at = Column(DateTime, default=datetime.utcnow, comment='Record creation timestamp')

    def __repr__(self):
        return f"<LoanApplication(id={self.id}, sk_id={self.sk_id_curr}, pd={self.pd_ensemble:.4f})>"


class MarketBenchmark(Base):
    """
    Stores daily market benchmark rates scraped by the Market Intelligence Agent.
    
    Architecture:
      - A scheduled job (daily 9:00 AM) uses Tavily search + Gemini LLM
        to extract current personal loan rates from the Indian market.
      - LLM extracts structured rates per risk segment from search results.
      - Rates are stored here with source attribution and expiry.
      - The live FastAPI/Streamlit app queries this table for benchmarks
        (~2ms), never scraping in the hot path.
      - Grafana can visualize rate trends over time.
    """
    __tablename__ = 'market_benchmarks'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # --- Rates per risk segment ---
    low_risk_rate = Column(Float, nullable=False, comment='Market avg rate for low-risk (excellent credit)')
    medium_risk_rate = Column(Float, nullable=False, comment='Market avg rate for medium-risk (good credit)')
    high_risk_rate = Column(Float, nullable=False, comment='Market avg rate for high-risk (fair credit)')

    # --- Source & Provenance ---
    source_query = Column(Text, comment='Tavily search query used')
    source_urls = Column(Text, comment='Comma-separated URLs from Tavily results')
    raw_llm_response = Column(Text, comment='Full LLM extraction response for audit trail')
    extraction_model = Column(String(100), comment='LLM model used (e.g., gemini-2.0-flash)')

    # --- Metadata ---
    fetch_type = Column(String(20), default='scheduled', comment='scheduled / on_demand / manual')
    is_active = Column(Integer, default=1, comment='1=current benchmark, 0=superseded')
    confidence_score = Column(Float, comment='LLM self-assessed confidence in extraction')
    fetched_at = Column(DateTime, default=datetime.utcnow, comment='When the scrape ran')
    expires_at = Column(DateTime, comment='When this benchmark should be refreshed')

    def __repr__(self):
        return (f"<MarketBenchmark(id={self.id}, low={self.low_risk_rate:.3f}, "
                f"med={self.medium_risk_rate:.3f}, high={self.high_risk_rate:.3f}, "
                f"fetched={self.fetched_at})>")

# ──────────────────────────────────────────────
# Database Initialization
# ──────────────────────────────────────────────
def get_engine(config):
    """Build SQLAlchemy engine from config."""
    db = config['database']
    url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
    return create_engine(url, echo=db.get('echo', False))


def init_database():
    """Create database (if not exists) and all tables."""
    config = load_config()
    db = config['database']

    # First connect to default 'postgres' database to check/create our database
    admin_url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/postgres"
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")

    # Check if database exists, create if not
    target_url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
    with admin_engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
            {"dbname": db['name']}
        )
        if not result.fetchone():
            conn.execute(text(f"CREATE DATABASE {db['name']}"))
            print(f"✅ Created database: {db['name']}")
        else:
            print(f"ℹ️  Database '{db['name']}' already exists.")

    admin_engine.dispose()

    # Now connect to the target database and create tables
    engine = get_engine(config)
    Base.metadata.create_all(engine)

    # Verify
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"✅ Tables in '{db['name']}': {tables}")

    columns = inspector.get_columns('loan_applications')
    print(f"✅ 'loan_applications' has {len(columns)} columns:")
    for col in columns:
        print(f"   - {col['name']:30s} {str(col['type']):20s}")

    engine.dispose()
    print("\n🎯 Database initialization complete.")
    return True


if __name__ == '__main__':
    try:
        init_database()
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Is PostgreSQL running? Check pgAdmin4.")
        print("  2. Verify configs/config.yaml credentials.")
        print(f"  3. Default connection: postgres:password@localhost:5432")
        sys.exit(1)
