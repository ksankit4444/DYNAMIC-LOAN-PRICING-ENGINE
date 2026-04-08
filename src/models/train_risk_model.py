"""
train_risk_model.py — Risk Model Training Pipeline (Layer 1)
=====
Trains XGBoost + Logistic Regression ensemble for Probability of Default.
Implements multi-objective Bayesian optimization via Optuna for:
  - XGBoost hyperparameters
  - Logistic Regression regularization
  - Ensemble weighting (XGB vs LR)

Objectives: Maximize AUC-ROC and KS-Statistic.

Usage:
    python src/models/train_risk_model.py
"""

import os
import sys
import json
import warnings
import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════
# SECTION 1: Data Loading
# ══════════════════════════════════════════════

def load_processed_data(config):
    """Load processed feature CSVs."""
    processed_path = os.path.join(ROOT_DIR, config['paths']['processed_data'])

    train_df = pd.read_csv(os.path.join(processed_path, 'features_train.csv'))
    test_df = pd.read_csv(os.path.join(processed_path, 'features_test.csv'))

    feature_cols_path = os.path.join(ROOT_DIR, config['paths']['models'], 'feature_columns.joblib')
    if not os.path.exists(feature_cols_path):
         # If doesn't exist, use all columns except TARGET
         feature_cols = [c for c in train_df.columns if c != 'TARGET']
    else:
         feature_cols = joblib.load(feature_cols_path)

    # Use only features present in both
    feature_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df['TARGET']
    X_test = test_df[feature_cols]
    y_test = test_df['TARGET']

    print(f"📂 Loaded data:")
    print(f"   Train: {X_train.shape}, default rate: {y_train.mean():.4f}")
    print(f"   Test:  {X_test.shape}, default rate: {y_test.mean():.4f}")
    print(f"   Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


# ══════════════════════════════════════════════
# SECTION 2: Evaluation Metrics
# ══════════════════════════════════════════════

def compute_ks_statistic(y_true, y_prob):
    """Kolmogorov-Smirnov Statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return ks, ks_threshold


def evaluate_model(y_true, y_prob, model_name="Model"):
    """Compute all credit risk metrics."""
    auc = roc_auc_score(y_true, y_prob)
    ks, ks_thresh = compute_ks_statistic(y_true, y_prob)
    gini = 2 * auc - 1

    metrics = {
        'auc_roc': round(float(auc), 6),
        'ks_statistic': round(float(ks), 6),
        'ks_threshold': round(float(ks_thresh), 6),
        'gini_coefficient': round(float(gini), 6),
    }

    print(f"   📊 {model_name} Metrics: AUC={auc:.4f}, KS={ks:.4f}, Gini={gini:.4f}")
    return metrics


# ══════════════════════════════════════════════
# SECTION 3: Model Training Functions
# ══════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test, config, params=None):
    """Train XGBoost. Uses provided params or defaults from config."""
    if params is None:
        xgb_config = config['model']['xgboost']
    else:
        # Map Optuna param names to XGBoost names if necessary
        xgb_config = {
            'max_depth': params.get('xgb_max_depth', 6),
            'learning_rate': params.get('xgb_learning_rate', 0.05),
            'subsample': params.get('xgb_subsample', 0.8),
            'colsample_bytree': params.get('xgb_colsample', 0.8),
            'gamma': params.get('xgb_gamma', 0),
            'reg_lambda': params.get('xgb_lambda', 1.0),
            'reg_alpha': params.get('xgb_alpha', 0),
            'n_estimators': 1000,
            'early_stopping_rounds': 50
        }

    # Industry Standard scale_pos_weight (92/8 ratio)
    scale_pos_weight = 11.5
    
    model = xgb.XGBClassifier(
        n_estimators=xgb_config.get('n_estimators', 1000),
        max_depth=xgb_config.get('max_depth', 6),
        learning_rate=xgb_config.get('learning_rate', 0.05),
        subsample=xgb_config.get('subsample', 0.8),
        colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
        gamma=xgb_config.get('gamma', 0),
        reg_lambda=xgb_config.get('reg_lambda', 1.0),
        reg_alpha=xgb_config.get('reg_alpha', 0),
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=xgb_config.get('early_stopping_rounds', 50),
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_prob_test, "XGBoost")
    
    return model, y_prob_test, metrics


def train_lightgbm(X_train, y_train, X_test, y_test, config, params=None):
    """Train LightGBM. Uses provided params or defaults from config."""
    if params is None:
        lgb_config = config['model']['lightgbm']
    else:
        lgb_config = {
            'num_leaves': params.get('lgb_num_leaves', 31),
            'learning_rate': params.get('lgb_learning_rate', 0.05),
            'feature_fraction': params.get('lgb_feature_fraction', 0.8),
            'bagging_fraction': params.get('lgb_bagging_fraction', 0.8),
            'bagging_freq': 5,
            'max_depth': params.get('lgb_max_depth', -1),
            'n_estimators': 1000
        }

    # Industry Standard scale_pos_weight
    scale_pos_weight = 11.5
    
    model = lgb.LGBMClassifier(
        n_estimators=lgb_config.get('n_estimators', 1000),
        num_leaves=lgb_config.get('num_leaves', 31),
        learning_rate=lgb_config.get('learning_rate', 0.05),
        feature_fraction=lgb_config.get('feature_fraction', 0.8),
        bagging_fraction=lgb_config.get('bagging_fraction', 0.8),
        bagging_freq=lgb_config.get('bagging_freq', 5),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        importance_type='gain',
        verbosity=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_prob_test, "LightGBM")
    
    return model, y_prob_test, metrics


def train_logistic_regression(X_train, y_train, X_test, y_test, config, C=None):
    """Train Logistic Regression scorecard."""
    if C is None:
        C = config['model']['logistic_regression']['C']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        C=C, 
        class_weight='balanced', 
        solver='lbfgs', 
        max_iter=1000, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    metrics = evaluate_model(y_test, y_prob_test, "LogReg")
    
    return model, scaler, y_prob_test, metrics


# ══════════════════════════════════════════════
# SECTION 4: Optuna Tuning
# ══════════════════════════════════════════════

def tune_models_and_ensemble_optuna(X_train, y_train, config):
    """Multi-Objective Bayesian Optimization for full pipeline."""
    print("\n🚀 Starting Multi-Objective Optuna Tuning (100 Trials)...")
    
    X_tr_opt, X_val_opt, y_tr_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    scaler_opt = StandardScaler()
    X_tr_scaled = scaler_opt.fit_transform(X_tr_opt)
    X_val_scaled = scaler_opt.transform(X_val_opt)

    def objective(trial):
        xgb_params = {
            'xgb_max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'xgb_learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
            'xgb_subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'xgb_colsample': trial.suggest_float('xgb_colsample', 0.6, 1.0),
            'xgb_gamma': trial.suggest_float('xgb_gamma', 0, 5),
            'xgb_lambda': trial.suggest_float('xgb_lambda', 1e-3, 10.0, log=True),
            'xgb_alpha': trial.suggest_float('xgb_alpha', 1e-3, 10.0, log=True),
        }
        lgb_params = {
            'lgb_num_leaves': trial.suggest_int('lgb_num_leaves', 20, 150),
            'lgb_learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2, log=True),
            'lgb_feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
            'lgb_bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 1.0),
            'lgb_max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        }
        lr_c = trial.suggest_float('lr_c', 1e-4, 10.0, log=True)
        
        # Dirichlet-style sum-to-1 weighting
        w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
        w_lgb = trial.suggest_float('w_lgb', 0.0, 1.0 - w_xgb)
        w_lr = max(0, 1.0 - w_xgb - w_lgb)
        
        try:
            # Training shells for fast evaluation
            m_xgb, probs_xgb, _ = train_xgboost(X_tr_opt, y_tr_opt, X_val_opt, y_val_opt, config, params=xgb_params)
            m_lgb, probs_lgb, _ = train_lightgbm(X_tr_opt, y_tr_opt, X_val_opt, y_val_opt, config, params=lgb_params)
            m_lr, _, probs_lr, _ = train_logistic_regression(X_tr_opt, y_tr_opt, X_val_opt, y_val_opt, config, C=lr_c)
            
            probs_ensemble = (w_xgb * probs_xgb) + (w_lgb * probs_lgb) + (w_lr * probs_lr)
            
            auc = roc_auc_score(y_val_opt, probs_ensemble)
            ks, _ = compute_ks_statistic(y_val_opt, probs_ensemble)
            
            return auc, ks
        except Exception as e:
            return 0.0, 0.0

    opt_config = config['model']['ensemble'].get('optuna', {})
    n_trials = 100 
    timeout = 1800
    
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Selection
    auc_w = 0.5
    ks_w = 0.5
    
    best_trial = None
    best_score = -1
    results = []

    for t in study.trials:
        if t.values is None: continue
        score = (t.values[0] * auc_w) + (t.values[1] * ks_w)
        results.append({'trial': t.number, 'auc': t.values[0], 'ks': t.values[1], **t.params})
        if score > best_score:
            best_score = score
            best_trial = t

    results_df = pd.DataFrame(results)
    print(f"\n   🏆 Best Trial (#{best_trial.number}): AUC={best_trial.values[0]:.4f}, KS={best_trial.values[1]:.4f}")
    
    return best_trial.params, results_df


# ══════════════════════════════════════════════
# SECTION 5: Final Evaluation & Pipeline
# ══════════════════════════════════════════════

def create_shap_explainer(xgb_model, X_test, config):
    """SHAP Explainer."""
    print("\n🔬 Creating SHAP Explainer...")
    explainer = shap.TreeExplainer(xgb_model)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    shap_importance = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"   📊 Top feature: {shap_importance.iloc[0]['feature']}")
    return explainer, shap_importance


def assign_risk_bands(pd_scores):
    """Categorize into risk bands."""
    return pd.cut(pd_scores, bins=[-np.inf, 0.10, 0.30, np.inf], labels=['Low', 'Medium', 'High'])


def save_artifacts(xgb_model, lgb_model, lr_model, scaler, explainer, metrics, ensemble_info, config, feature_cols=None):
    """Save all artifacts."""
    models_path = os.path.join(ROOT_DIR, config['paths']['models'], 'v2')
    os.makedirs(models_path, exist_ok=True)
    joblib.dump(xgb_model, os.path.join(models_path, 'xgboost_model.joblib'))
    joblib.dump(lgb_model, os.path.join(models_path, 'lightgbm_model.joblib'))
    joblib.dump(lr_model, os.path.join(models_path, 'logistic_model.joblib'))
    joblib.dump(scaler, os.path.join(models_path, 'logistic_scaler.joblib'))
    joblib.dump(explainer, os.path.join(models_path, 'shap_explainer.joblib'))
    
    # [v2] Explicitly save feature columns for the QC script
    if feature_cols is not None:
        joblib.dump(feature_cols, os.path.join(models_path, 'xgb_feature_columns.joblib'))
        joblib.dump(feature_cols, os.path.join(models_path, 'lr_feature_columns.joblib'))
        joblib.dump(feature_cols, os.path.join(models_path, 'lgb_feature_columns.joblib'))
    
    with open(os.path.join(models_path, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(models_path, 'ensemble_config.json'), 'w') as f:
        json.dump(ensemble_info, f, indent=2, default=str)
    print(f"\n✅ All artifacts saved to: {models_path}")


def run_training_pipeline():
    """Main Orchestrator."""
    config = load_config()
    print("=" * 60 + "\n  DYNAMIC LOAN PRICING ENGINE — AI Pipeline\n" + "=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test, feature_cols = load_processed_data(config)

    # 2. Optuna Optimization
    best_params, weight_results = tune_models_and_ensemble_optuna(X_train, y_train, config)
    
    # 3. Final Training with optimal params
    print("\n🏗️ Training Champion Ensemble Models...")
    xgb_champion, pd_xgb_test, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, config, params=best_params)
    lgb_champion, pd_lgb_test, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test, config, params=best_params)
    lr_champion, scaler, pd_lr_test, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test, config, C=best_params['lr_c'])
    
    w_xgb = float(best_params['w_xgb'])
    w_lgb = float(best_params['w_lgb'])
    w_lr = max(0.0, 1.0 - w_xgb - w_lgb)
    
    pd_ensemble_test = (w_xgb * pd_xgb_test) + (w_lgb * pd_lgb_test) + (w_lr * pd_lr_test)
    ensemble_metrics = evaluate_model(y_test, pd_ensemble_test, "Ensemble")

    # 4. Impact Analysis (SHAP on XGBoost as representative)
    explainer, _ = create_shap_explainer(xgb_champion, X_test, config)
    
    # 5. Save Everything
    safe_params = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in best_params.items()}
    all_metrics = {
        'xgboost': xgb_metrics, 'lightgbm': lgb_metrics, 'logistic': lr_metrics, 
        'ensemble': ensemble_metrics, 'best_params': safe_params, 
        'data_summary': {'samples': len(y_train)}
    }
    ensemble_info = {
        'optimal_weights': {
            'xgboost': w_xgb, 'lightgbm': w_lgb, 'logistic': w_lr
        },
        'optimal_params': safe_params
    }
    save_artifacts(xgb_champion, lgb_champion, lr_champion, scaler, explainer, all_metrics, ensemble_info, config, feature_cols)

    print("\n" + "=" * 60 + "\n  PIPELINE COMPLETE\n" + "=" * 60)


if __name__ == '__main__':
    run_training_pipeline()
