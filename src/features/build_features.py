"""
build_features.py — Feature Engineering Pipeline (Layer 1)
=====
Reads raw Kaggle Home Credit Default Risk CSVs and produces
a clean, engineered feature matrix ready for model training.

Engineered features:
  - DTI Ratio proxy (AMT_ANNUITY / AMT_INCOME_TOTAL)
  - Credit Utilization (from bureau data)
  - Payment History score (from bureau + previous apps)
  - Income Stability proxy (employment tenure + type)
  - Bureau-derived features (enquiry count, credit history length)
  - WoE encoding for categorical variables

Usage:
    python src/features/build_features.py

Output:
    data/processed/features_train.csv
    data/processed/features_test.csv
    models/woe_encoder.joblib
"""

import os
import sys
import warnings
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════
# SECTION 1: Data Loading
# ══════════════════════════════════════════════

def load_raw_data(config):
    """Load raw CSVs from data/raw/."""
    raw_path = os.path.join(ROOT_DIR, config['paths']['raw_data'])

    print("📂 Loading raw data...")

    app_train = pd.read_csv(os.path.join(raw_path, 'application_train.csv'))
    print(f"   application_train: {app_train.shape}")

    bureau = None
    bureau_path = os.path.join(raw_path, 'bureau.csv')
    if os.path.exists(bureau_path):
        bureau = pd.read_csv(bureau_path)
        print(f"   bureau: {bureau.shape}")
    else:
        print("   ⚠️  bureau.csv not found — bureau features will be skipped.")

    prev_app = None
    prev_path = os.path.join(raw_path, 'previous_application.csv')
    if os.path.exists(prev_path):
        prev_app = pd.read_csv(prev_path)
        print(f"   previous_application: {prev_app.shape}")
    else:
        print("   ⚠️  previous_application.csv not found — prev app features skipped.")

    return app_train, bureau, prev_app


# ══════════════════════════════════════════════
# SECTION 2: Feature Engineering
# ══════════════════════════════════════════════

def engineer_dti_ratio(df):
    """
    Debt-to-Income Ratio proxy.
    DTI = AMT_ANNUITY / AMT_INCOME_TOTAL
    One of the strongest single predictors of default.
    """
    print("🔧 Engineering DTI Ratio...")
    df['dti_ratio'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['dti_ratio'] = df['dti_ratio'].clip(0, 5)  # Cap extreme outliers
    df['dti_ratio'] = df['dti_ratio'].fillna(df['dti_ratio'].median())

    # Credit-to-income ratio (complementary)
    df['credit_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['credit_income_ratio'] = df['credit_income_ratio'].clip(0, 50)
    df['credit_income_ratio'] = df['credit_income_ratio'].fillna(df['credit_income_ratio'].median())

    # Annuity-to-credit ratio (payment burden relative to loan size)
    df['annuity_credit_ratio'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
    df['annuity_credit_ratio'] = df['annuity_credit_ratio'].fillna(df['annuity_credit_ratio'].median())

    return df


def engineer_credit_utilization(df, bureau):
    """
    Credit Utilization Rate from bureau data.
    Utilization = AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM
    Values above 30% signal financial stress.
    """
    print("🔧 Engineering Credit Utilization...")

    if bureau is None:
        df['credit_utilization'] = 0.0
        df['bureau_active_count'] = 0
        df['bureau_credit_sum'] = 0.0
        df['bureau_debt_sum'] = 0.0
        return df

    # Per-applicant bureau aggregations
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        bureau_active_count=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        bureau_closed_count=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
        bureau_credit_sum=('AMT_CREDIT_SUM', 'sum'),
        bureau_debt_sum=('AMT_CREDIT_SUM_DEBT', 'sum'),
        bureau_overdue_sum=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        bureau_credit_count=('SK_ID_BUREAU', 'count'),
        bureau_max_overdue=('CREDIT_DAY_OVERDUE', 'max'),
        bureau_avg_days_credit=('DAYS_CREDIT', 'mean'),
        bureau_oldest_credit=('DAYS_CREDIT', 'min'),   # Most negative = oldest
        bureau_latest_credit=('DAYS_CREDIT', 'max'),
    ).reset_index()

    # Credit utilization ratio
    bureau_agg['credit_utilization'] = (
        bureau_agg['bureau_debt_sum'] /
        bureau_agg['bureau_credit_sum'].replace(0, np.nan)
    ).fillna(0).clip(0, 5)

    # Credit history length (in years, from most negative DAYS_CREDIT)
    bureau_agg['credit_history_length_days'] = bureau_agg['bureau_oldest_credit'].abs()
    bureau_agg['credit_history_length_years'] = bureau_agg['credit_history_length_days'] / 365.25

    # Has overdue flag
    bureau_agg['has_bureau_overdue'] = (bureau_agg['bureau_overdue_sum'] > 0).astype(int)

    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')

    # Fill missing (no bureau record = no credit history)
    fill_zeros = ['credit_utilization', 'bureau_active_count', 'bureau_closed_count',
                  'bureau_credit_sum', 'bureau_debt_sum', 'bureau_overdue_sum',
                  'bureau_credit_count', 'bureau_max_overdue', 'has_bureau_overdue',
                  'credit_history_length_days', 'credit_history_length_years']
    for col in fill_zeros:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    fill_medians = ['bureau_avg_days_credit', 'bureau_oldest_credit', 'bureau_latest_credit']
    for col in fill_medians:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def engineer_payment_history(df, bureau):
    """
    Payment History features.
    Count of late/delinquent payments from bureau data.
    """
    print("🔧 Engineering Payment History...")

    if bureau is None:
        df['payment_history_score'] = 0.0
        df['late_payment_count'] = 0
        return df

    # Bureau CREDIT_DAY_OVERDUE > 0 signals past delinquency
    bureau_payment = bureau.groupby('SK_ID_CURR').agg(
        late_payment_count=('CREDIT_DAY_OVERDUE', lambda x: (x > 0).sum()),
        max_days_overdue=('CREDIT_DAY_OVERDUE', 'max'),
        mean_days_overdue=('CREDIT_DAY_OVERDUE', 'mean'),
        total_prolongation=('CNT_CREDIT_PROLONG', 'sum'),
    ).reset_index()

    # Composite payment history score (higher = worse)
    late_norm = late_payment_count_max = bureau_payment['late_payment_count'].max()
    if late_norm == 0:
        late_norm = 1
    bureau_payment['payment_history_score'] = (
        0.4 * (bureau_payment['late_payment_count'] / late_norm) +
        0.3 * (bureau_payment['max_days_overdue'].clip(0, 365) / 365) +
        0.2 * (bureau_payment['mean_days_overdue'].clip(0, 100) / 100) +
        0.1 * (bureau_payment['total_prolongation'].clip(0, 10) / 10)
    ).clip(0, 1)

    df = df.merge(
        bureau_payment[['SK_ID_CURR', 'payment_history_score', 'late_payment_count',
                        'max_days_overdue', 'mean_days_overdue']],
        on='SK_ID_CURR', how='left'
    )

    for col in ['payment_history_score', 'late_payment_count', 'max_days_overdue', 'mean_days_overdue']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def engineer_income_stability(df):
    """
    Income Stability proxy.
    Based on employment tenure relative to age, and employment type.
    """
    print("🔧 Engineering Income Stability...")

    # Age in years
    df['age_years'] = df['DAYS_BIRTH'].abs() / 365.25

    # Employment tenure in years
    df['employment_years'] = df['DAYS_EMPLOYED'].apply(
        lambda x: 0 if x > 0 else abs(x) / 365.25  # Positive DAYS_EMPLOYED = unemployed (365243 anomaly)
    )

    # Flag the DAYS_EMPLOYED anomaly (365243 = not employed / retired)
    df['employed_flag'] = (df['DAYS_EMPLOYED'] != 365243).astype(int)

    # Employment ratio (tenure / age)
    df['employment_ratio'] = df['employment_years'] / df['age_years'].replace(0, np.nan)
    df['employment_ratio'] = df['employment_ratio'].clip(0, 1).fillna(0)

    # Income stability score composite
    df['income_stability_score'] = (
        0.5 * df['employment_ratio'] +
        0.3 * df['employed_flag'] +
        0.2 * (df['employment_years'].clip(0, 30) / 30)
    ).clip(0, 1)

    # Income per family member
    family_size = df['CNT_FAM_MEMBERS'].replace(0, 1).fillna(1)
    df['income_per_family'] = df['AMT_INCOME_TOTAL'] / family_size

    return df


def engineer_previous_app_features(df, prev_app):
    """
    Features from previous applications.
    Captures historical interaction patterns with lender.
    """
    print("🔧 Engineering Previous Application Features...")

    if prev_app is None:
        df['prev_app_count'] = 0
        df['prev_approved_rate'] = 0.0
        df['prev_refused_count'] = 0
        return df

    prev_agg = prev_app.groupby('SK_ID_CURR').agg(
        prev_app_count=('SK_ID_PREV', 'count'),
        prev_approved_count=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
        prev_refused_count=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
        prev_cancelled_count=('NAME_CONTRACT_STATUS', lambda x: (x == 'Canceled').sum()),
        prev_avg_credit=('AMT_CREDIT', 'mean'),
        prev_max_credit=('AMT_CREDIT', 'max'),
        prev_avg_annuity=('AMT_ANNUITY', 'mean'),
    ).reset_index()

    prev_agg['prev_approved_rate'] = (
        prev_agg['prev_approved_count'] /
        prev_agg['prev_app_count'].replace(0, 1)
    )

    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')

    fill_cols = ['prev_app_count', 'prev_approved_count', 'prev_refused_count',
                 'prev_cancelled_count', 'prev_approved_rate']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in ['prev_avg_credit', 'prev_max_credit', 'prev_avg_annuity']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def engineer_domain_features(df):
    """
    Additional domain-specific features from application table.
    """
    print("🔧 Engineering Domain Features...")

    # External source scores (pre-computed risk scores in the dataset)
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Better external source features
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
    if ext_cols:
        df['ext_source_min'] = df[ext_cols].min(axis=1)
        df['ext_source_max'] = df[ext_cols].max(axis=1)
        df['ext_source_prod'] = df[ext_cols].prod(axis=1)
        df['ext_source_nan_count'] = df[ext_cols].isnull().sum(axis=1)
        # Drop raw mean/std as they are collinear noise
        if 'ext_source_mean' in df.columns: df.drop('ext_source_mean', axis=1, inplace=True)
        if 'ext_source_std' in df.columns: df.drop('ext_source_std', axis=1, inplace=True)

    # Documents provided count
    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
    if doc_cols:
        df['documents_provided_count'] = df[doc_cols].sum(axis=1)

    # Phone / email / work phone flags
    contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
    existing_contact = [c for c in contact_cols if c in df.columns]
    if existing_contact:
        df['contact_reachability'] = df[existing_contact].sum(axis=1)

    # Social circle defaults
    social_cols = ['DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']
    for col in social_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Goods price vs credit amount ratio
    if 'AMT_GOODS_PRICE' in df.columns:
        df['goods_credit_ratio'] = (
            df['AMT_GOODS_PRICE'] / df['AMT_CREDIT'].replace(0, np.nan)
        ).fillna(1.0)

    # Interactive Features for Precision
    df['credit_stress_product'] = df['dti_ratio'] * df['EXT_SOURCE_2'].fillna(0.5)
    df['age_stability_interaction'] = df['age_years'] * df['income_stability_score']
    df['annuity_income_interaction'] = df['AMT_ANNUITY'] / df['income_per_family'].replace(0, np.nan)
    df['annuity_income_interaction'] = df['annuity_income_interaction'].fillna(0)

    return df


# ══════════════════════════════════════════════
# SECTION 3: Weight of Evidence Encoding
# ══════════════════════════════════════════════

class WoEEncoder:
    """
    Weight of Evidence encoder for categorical features.

    WoE = ln(Distribution of Events / Distribution of Non-Events)
    IV  = Σ (Distribution of Events - Distribution of Non-Events) × WoE

    Features with IV < threshold are dropped (low predictive power).

    Industry standard in credit risk scorecard development.
    """

    def __init__(self, min_iv=0.02, regularization=0.5):
        self.min_iv = min_iv
        self.regularization = regularization  # Laplace smoothing for rare categories
        self.woe_maps = {}
        self.iv_values = {}
        self.selected_features = []

    def fit(self, df, target_col='TARGET', categorical_cols=None):
        """Compute WoE mappings for each categorical feature."""
        print("\n📊 Computing WoE Encodings...")

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        total_events = df[target_col].sum()
        total_non_events = len(df) - total_events

        for col in tqdm(categorical_cols, desc="   WoE fitting"):
            woe_map = {}
            iv = 0.0

            for cat in df[col].unique():
                mask = df[col] == cat
                events = df.loc[mask, target_col].sum() + self.regularization
                non_events = mask.sum() - df.loc[mask, target_col].sum() + self.regularization

                dist_events = events / (total_events + self.regularization * len(df[col].unique()))
                dist_non_events = non_events / (total_non_events + self.regularization * len(df[col].unique()))

                if dist_events > 0 and dist_non_events > 0:
                    woe = np.log(dist_non_events / dist_events)
                    iv += (dist_non_events - dist_events) * woe
                else:
                    woe = 0.0

                woe_map[cat] = woe

            self.woe_maps[col] = woe_map
            self.iv_values[col] = iv

        # Select features by IV threshold
        self.selected_features = [
            col for col, iv in self.iv_values.items()
            if iv >= self.min_iv
        ]

        print(f"\n   Information Value Summary:")
        for col, iv in sorted(self.iv_values.items(), key=lambda x: x[1], reverse=True):
            status = "✅ KEEP" if iv >= self.min_iv else "❌ DROP"
            print(f"   {status}  {col:40s}  IV = {iv:.4f}")

        print(f"\n   Kept {len(self.selected_features)}/{len(categorical_cols)} categorical features (IV ≥ {self.min_iv})")
        return self

    def transform(self, df):
        """Apply WoE transformation to categorical features."""
        df_transformed = df.copy()

        for col in self.selected_features:
            if col in df_transformed.columns:
                woe_map = self.woe_maps[col]
                # Map known categories, unseen categories get 0 (neutral WoE)
                df_transformed[f'{col}_woe'] = df_transformed[col].map(woe_map).fillna(0.0)

        return df_transformed

    def fit_transform(self, df, target_col='TARGET', categorical_cols=None):
        """Fit and transform in one step."""
        self.fit(df, target_col, categorical_cols)
        return self.transform(df)


# ══════════════════════════════════════════════
# SECTION 4: Feature Selection & Output
# ══════════════════════════════════════════════

def select_final_features(df, woe_encoder):
    """
    Select the final feature set for model training.
    Drops raw categoricals (replaced by WoE), IDs, and high-null columns.
    """
    print("\n🎯 Selecting final feature set...")

    # Columns to always drop
    drop_cols = ['SK_ID_CURR', 'TARGET', 'index']

    # DROP NOISY/COLLINEAR COLS (identified in QC)
    redundant_cols = [
        'FLAG_EMP_PHONE', 'employed_flag', 'FLOORSMAX_MEDI', 
        'FLOORSMAX_MODE', 'EXT_SOURCE_1_nan', 'EXT_SOURCE_2_nan', 
        'EXT_SOURCE_3_nan', 'DEF_60_CNT_SOCIAL_CIRCLE'
    ]
    drop_cols.extend(redundant_cols)

    # Drop raw categorical columns that have WoE equivalents
    raw_cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
    drop_cols.extend(raw_cats)

    # Drop high-null ratio columns (>70% missing before our fills)
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Keep only numeric columns
    numeric_features = []
    for col in feature_cols:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32', 'uint8']:
            numeric_features.append(col)

    print(f"   Final feature count: {len(numeric_features)}")

    return numeric_features


# ══════════════════════════════════════════════
# SECTION 5: Main Pipeline
# ══════════════════════════════════════════════

def run_pipeline():
    """Execute the full feature engineering pipeline."""
    config = load_config()

    print("=" * 60)
    print("  DYNAMIC LOAN PRICING ENGINE — Feature Engineering")
    print("=" * 60)

    # 1. Load data
    app_train, bureau, prev_app = load_raw_data(config)

    # 2. Engineer features
    df = app_train.copy()
    df = engineer_dti_ratio(df)
    df = engineer_credit_utilization(df, bureau)
    df = engineer_payment_history(df, bureau)
    df = engineer_income_stability(df)
    df = engineer_previous_app_features(df, prev_app)
    df = engineer_domain_features(df)

    print(f"\n📐 Shape after feature engineering: {df.shape}")

    # 3. WoE encoding for categoricals
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Remove ID-like columns from WoE
    categorical_cols = [c for c in categorical_cols if c not in ['SK_ID_CURR']]

    woe_encoder = WoEEncoder(min_iv=config['features']['min_iv_threshold'])
    df = woe_encoder.fit_transform(df, target_col='TARGET', categorical_cols=categorical_cols)

    # 4. Select final features
    feature_cols = select_final_features(df, woe_encoder)

    # 5. Handle remaining missing values
    print("\n🔧 Handling remaining missing values...")
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    null_counts = df[feature_cols].isnull().sum().sum()
    print(f"   Remaining nulls in features: {null_counts}")

    # 6. Train/test split (stratified on TARGET)
    print(f"\n📊 Splitting data (test_size={config['features']['test_size']})...")
    X = df[feature_cols]
    y = df['TARGET']

    # Also keep SK_ID_CURR for traceability
    ids = df['SK_ID_CURR']

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids,
        test_size=config['features']['test_size'],
        random_state=config['features']['random_state'],
        stratify=y
    )

    # Add target and ID back for saving
    train_df = X_train.copy()
    train_df['TARGET'] = y_train.values
    train_df['SK_ID_CURR'] = ids_train.values

    test_df = X_test.copy()
    test_df['TARGET'] = y_test.values
    test_df['SK_ID_CURR'] = ids_test.values

    # 7. Save outputs
    processed_path = os.path.join(ROOT_DIR, config['paths']['processed_data'])
    os.makedirs(processed_path, exist_ok=True)

    train_path = os.path.join(processed_path, 'features_train.csv')
    test_path = os.path.join(processed_path, 'features_test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n💾 Saved:")
    print(f"   Train: {train_path} ({train_df.shape})")
    print(f"   Test:  {test_path} ({test_df.shape})")

    # 8. Save WoE encoder
    models_path = os.path.join(ROOT_DIR, config['paths']['models'])
    os.makedirs(models_path, exist_ok=True)

    woe_path = os.path.join(models_path, 'woe_encoder.joblib')
    joblib.dump(woe_encoder, woe_path)
    print(f"   WoE Encoder: {woe_path}")

    # 9. Save feature list
    feature_list_path = os.path.join(models_path, 'feature_columns.joblib')
    joblib.dump(feature_cols, feature_list_path)
    print(f"   Feature list: {feature_list_path}")

    # 10. Summary
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Train samples:  {len(train_df)} (default rate: {y_train.mean():.4f})")
    print(f"  Test samples:   {len(test_df)} (default rate: {y_test.mean():.4f})")
    print(f"  WoE features:   {len(woe_encoder.selected_features)}")
    print(f"  IV-filtered:    {len(woe_encoder.iv_values) - len(woe_encoder.selected_features)} dropped")
    print("=" * 60)

    return train_df, test_df, woe_encoder, feature_cols


if __name__ == '__main__':
    run_pipeline()
