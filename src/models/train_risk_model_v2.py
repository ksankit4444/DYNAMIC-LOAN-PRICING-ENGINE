"""
train_risk_model_v2.py — Risk Model Training Pipeline V2
=====
V2 Improvements over V1:
  1. SEPARATED FEATURE SETS: XGBoost gets raw numerics only (no WoE).
     LogReg gets scaled features (with WoE). This lets each model
     play to its strengths — XGBoost finds non-linear splits on raw
     continuous data, LogReg uses linearized WoE features for stability.
  2. min_child_weight added to Optuna search space.
  3. Removed unused scaler in Optuna objective.
  4. Saves artifacts to models/v2/ so V1 results are preserved.

Usage:
    python src/models/train_risk_model_v2.py
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════
# SECTION 1: Data Loading — SEPARATED FEATURE SETS
# ══════════════════════════════════════════════

def load_processed_data(config):
    """Load processed data and split into XGB-friendly and LR-friendly feature sets."""
    processed_path = os.path.join(ROOT_DIR, config['paths']['processed_data'])

    train_df = pd.read_csv(os.path.join(processed_path, 'features_train.csv'))
    test_df = pd.read_csv(os.path.join(processed_path, 'features_test.csv'))

    # All numeric feature columns
    all_features = [c for c in train_df.columns
                    if c not in ['SK_ID_CURR', 'TARGET', 'index']
                    and train_df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'uint8']
                    and c in test_df.columns]

    # SPLIT: XGBoost gets RAW numerics (no WoE columns)
    #        LogReg gets ALL features (including WoE)
    xgb_features = [c for c in all_features if not c.endswith('_woe')]
    lr_features  = all_features  # LogReg benefits from WoE linearization

    y_train = train_df['TARGET']
    y_test  = test_df['TARGET']

    X_train_xgb = train_df[xgb_features]
    X_test_xgb  = test_df[xgb_features]
    X_train_lr  = train_df[lr_features]
    X_test_lr   = test_df[lr_features]

    print(f"📂 V2 Data Loaded:")
    print(f"   Train: {len(y_train)} samples, default rate: {y_train.mean():.4f}")
    print(f"   Test:  {len(y_test)} samples, default rate: {y_test.mean():.4f}")
    print(f"   XGBoost features: {len(xgb_features)} (raw numerics, NO WoE)")
    print(f"   LogReg features:  {len(lr_features)} (all, including WoE)")
    print(f"   WoE features excluded from XGB: {len(lr_features) - len(xgb_features)}")

    return (X_train_xgb, X_test_xgb, X_train_lr, X_test_lr,
            y_train, y_test, xgb_features, lr_features, all_features)


# ══════════════════════════════════════════════
# SECTION 2: Evaluation Metrics
# ══════════════════════════════════════════════

def compute_ks_statistic(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return ks, ks_threshold


def evaluate_model(y_true, y_prob, model_name="Model"):
    auc = roc_auc_score(y_true, y_prob)
    ks, ks_thresh = compute_ks_statistic(y_true, y_prob)
    gini = 2 * auc - 1
    metrics = {
        'auc_roc': round(float(auc), 6),
        'ks_statistic': round(float(ks), 6),
        'ks_threshold': round(float(ks_thresh), 6),
        'gini_coefficient': round(float(gini), 6),
    }
    print(f"   📊 {model_name}: AUC={auc:.4f}, KS={ks:.4f}, Gini={gini:.4f}")
    return metrics


# ══════════════════════════════════════════════
# SECTION 3: Model Training (Separated Inputs)
# ══════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test, params):
    """Train XGBoost on RAW numeric features only."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=params.get('xgb_max_depth', 6),
        learning_rate=params.get('xgb_learning_rate', 0.05),
        subsample=params.get('xgb_subsample', 0.8),
        colsample_bytree=params.get('xgb_colsample', 0.8),
        min_child_weight=params.get('xgb_min_child_weight', 5),
        gamma=params.get('xgb_gamma', 0),
        reg_lambda=params.get('xgb_lambda', 1.0),
        reg_alpha=params.get('xgb_alpha', 0),
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    probs = model.predict_proba(X_test)[:, 1]
    return model, probs


def train_logreg(X_train, y_train, X_test, y_test, C):
    """Train LogReg on SCALED features (including WoE)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        C=C, class_weight='balanced', solver='lbfgs',
        max_iter=1000, random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    return model, scaler, probs


# ══════════════════════════════════════════════
# SECTION 4: Optuna V2 — Separated Feature Sets
# ══════════════════════════════════════════════

def tune_v2(X_train_xgb, X_train_lr, y_train, config):
    """
    Multi-Objective Optuna with SEPARATED feature sets.
    XGBoost trains on raw numerics. LogReg trains on all (incl. WoE).
    """
    print("\n🚀 V2 Optuna Tuning — Separated Feature Sets")
    print("   XGBoost: raw numerics only | LogReg: all features (incl. WoE)")

    # Stratified hold-out for Optuna
    Xtr_xgb, Xval_xgb, ytr, yval = train_test_split(
        X_train_xgb, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    # Use same split indices for LR features
    Xtr_lr = X_train_lr.iloc[Xtr_xgb.index]
    Xval_lr = X_train_lr.iloc[Xval_xgb.index]

    # Reset indices to avoid alignment issues
    Xtr_xgb = Xtr_xgb.reset_index(drop=True)
    Xval_xgb = Xval_xgb.reset_index(drop=True)
    Xtr_lr = Xtr_lr.reset_index(drop=True)
    Xval_lr = Xval_lr.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True)
    yval = yval.reset_index(drop=True)

    # Pre-scale LR data ONCE (not per trial — saves time)
    scaler = StandardScaler()
    Xtr_lr_scaled = scaler.fit_transform(Xtr_lr)
    Xval_lr_scaled = scaler.transform(Xval_lr)

    def objective(trial):
        # XGBoost params — now includes min_child_weight
        xgb_params = {
            'xgb_max_depth':         trial.suggest_int('xgb_max_depth', 3, 10),
            'xgb_learning_rate':     trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
            'xgb_subsample':         trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'xgb_colsample':         trial.suggest_float('xgb_colsample', 0.6, 1.0),
            'xgb_min_child_weight':  trial.suggest_int('xgb_min_child_weight', 1, 20),
            'xgb_gamma':             trial.suggest_float('xgb_gamma', 0, 5),
            'xgb_lambda':            trial.suggest_float('xgb_lambda', 1e-3, 10.0, log=True),
            'xgb_alpha':             trial.suggest_float('xgb_alpha', 1e-3, 10.0, log=True),
        }
        lr_c = trial.suggest_float('lr_c', 1e-4, 10.0, log=True)
        w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)

        try:
            # XGBoost on RAW numerics
            m_xgb, probs_xgb = train_xgboost(Xtr_xgb, ytr, Xval_xgb, yval, xgb_params)

            # LogReg on pre-scaled ALL features (including WoE)
            m_lr = LogisticRegression(
                C=lr_c, class_weight='balanced', solver='lbfgs',
                max_iter=1000, random_state=42, n_jobs=-1
            )
            m_lr.fit(Xtr_lr_scaled, ytr)
            probs_lr = m_lr.predict_proba(Xval_lr_scaled)[:, 1]

            # Blend
            probs_ens = (w_xgb * probs_xgb) + ((1 - w_xgb) * probs_lr)
            auc = roc_auc_score(yval, probs_ens)
            ks, _ = compute_ks_statistic(yval, probs_ens)
            return auc, ks
        except Exception as e:
            return 0.0, 0.0

    opt_config = config['model']['ensemble'].get('optuna', {})
    n_trials = opt_config.get('n_trials', 100)
    timeout = opt_config.get('timeout', 1800)

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=TPESampler(seed=42),
        study_name="v2_separated_features"
    )

    # Progress callback
    def log_progress(study, trial):
        if trial.number % 10 == 0 and trial.values:
            print(f"   Trial {trial.number:3d}: AUC={trial.values[0]:.4f}, KS={trial.values[1]:.4f}")

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[log_progress])

    # Select best by composite score
    auc_w = opt_config.get('auc_weight', 0.5)
    ks_w = opt_config.get('ks_weight', 0.5)

    best_trial = None
    best_score = -1
    results = []
    for t in study.trials:
        if t.values is None:
            continue
        score = (t.values[0] * auc_w) + (t.values[1] * ks_w)
        results.append({
            'trial': t.number, 'auc': t.values[0], 'ks': t.values[1],
            'composite': score, **t.params
        })
        if score > best_score:
            best_score = score
            best_trial = t

    results_df = pd.DataFrame(results)

    print(f"\n   🏆 V2 Best Trial #{best_trial.number}:")
    print(f"      AUC={best_trial.values[0]:.4f}, KS={best_trial.values[1]:.4f}")
    print(f"      w_xgb={best_trial.params['w_xgb']:.3f}")

    return best_trial.params, results_df, study


# ══════════════════════════════════════════════
# SECTION 5: Save & Summary
# ══════════════════════════════════════════════

def save_v2_artifacts(xgb_model, lr_model, scaler, explainer, metrics, ensemble_info,
                      config, xgb_features, lr_features, results_df):
    """Save all V2 artifacts to models/v2/."""
    models_path = os.path.join(ROOT_DIR, config['paths']['models'], 'v2')
    os.makedirs(models_path, exist_ok=True)

    joblib.dump(xgb_model, os.path.join(models_path, 'xgboost_model.joblib'))
    joblib.dump(lr_model, os.path.join(models_path, 'logistic_model.joblib'))
    joblib.dump(scaler, os.path.join(models_path, 'logistic_scaler.joblib'))
    joblib.dump(explainer, os.path.join(models_path, 'shap_explainer.joblib'))
    joblib.dump(xgb_features, os.path.join(models_path, 'xgb_feature_columns.joblib'))
    joblib.dump(lr_features, os.path.join(models_path, 'lr_feature_columns.joblib'))

    with open(os.path.join(models_path, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(models_path, 'ensemble_config.json'), 'w') as f:
        json.dump(ensemble_info, f, indent=2, default=str)

    results_df.to_csv(os.path.join(models_path, 'optuna_trial_results.csv'), index=False)

    print(f"\n✅ V2 artifacts saved to: {models_path}")


def run_v2_pipeline():
    """V2 Pipeline Orchestrator."""
    config = load_config()

    print("=" * 60)
    print("  DYNAMIC LOAN PRICING ENGINE — V2 TRAINING")
    print("  Key change: Separated feature sets (XGB raw / LR WoE)")
    print("=" * 60)

    # 1. Load with separated features
    (X_train_xgb, X_test_xgb, X_train_lr, X_test_lr,
     y_train, y_test, xgb_features, lr_features, all_features) = load_processed_data(config)

    # 2. Optuna V2
    best_params, results_df, study = tune_v2(X_train_xgb, X_train_lr, y_train, config)

    # 3. Final champion training with optimal params
    print("\n🏗️ Training V2 Champions...")

    xgb_champion, pd_xgb_test = train_xgboost(
        X_train_xgb, y_train, X_test_xgb, y_test, best_params
    )
    xgb_metrics = evaluate_model(y_test, pd_xgb_test, "V2 XGBoost (raw features)")

    lr_champion, scaler, pd_lr_test = train_logreg(
        X_train_lr, y_train, X_test_lr, y_test, best_params['lr_c']
    )
    lr_metrics = evaluate_model(y_test, pd_lr_test, "V2 LogReg (with WoE)")

    # 4. Ensemble
    w_xgb = best_params['w_xgb']
    pd_ensemble = (w_xgb * pd_xgb_test) + ((1 - w_xgb) * pd_lr_test)
    ensemble_metrics = evaluate_model(y_test, pd_ensemble, "V2 Ensemble")

    # 5. SHAP on raw XGB features
    explainer, shap_imp = create_shap_explainer(xgb_champion, X_test_xgb)

    # 6. Save
    safe_params = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in best_params.items()}

    all_metrics = {
        'version': 'V2 — Separated Feature Sets',
        'xgboost': xgb_metrics,
        'logistic': lr_metrics,
        'ensemble': ensemble_metrics,
        'best_params': safe_params,
        'data_summary': {
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'xgb_features': len(xgb_features),
            'lr_features': len(lr_features),
        }
    }
    ensemble_info = {
        'version': 'V2',
        'optimal_weights': {'xgboost': float(w_xgb), 'logistic': float(1 - w_xgb)},
        'optimal_params': safe_params,
        'feature_separation': {
            'xgb': 'Raw numerics only (no _woe columns)',
            'lr': 'All features including WoE-encoded',
        }
    }

    save_v2_artifacts(
        xgb_champion, lr_champion, scaler, explainer,
        all_metrics, ensemble_info, config,
        xgb_features, lr_features, results_df
    )

    # 7. Comparison Summary
    print("\n" + "=" * 60)
    print("  V2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  {'Metric':<25s} {'XGBoost':>10s} {'LogReg':>10s} {'Ensemble':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'AUC-ROC':<25s} {xgb_metrics['auc_roc']:>10.4f} {lr_metrics['auc_roc']:>10.4f} {ensemble_metrics['auc_roc']:>10.4f}")
    print(f"  {'KS Statistic':<25s} {xgb_metrics['ks_statistic']:>10.4f} {lr_metrics['ks_statistic']:>10.4f} {ensemble_metrics['ks_statistic']:>10.4f}")
    print(f"  {'Gini':<25s} {xgb_metrics['gini_coefficient']:>10.4f} {lr_metrics['gini_coefficient']:>10.4f} {ensemble_metrics['gini_coefficient']:>10.4f}")
    print(f"\n  Ensemble: {w_xgb:.0%} XGBoost + {1-w_xgb:.0%} LogReg")
    print(f"  XGB features: {len(xgb_features)} | LR features: {len(lr_features)}")
    print("=" * 60)
    print("\n  Compare with V1 results in models/training_metrics.json")
    print("  V2 results saved in models/v2/")
    print("=" * 60)


def create_shap_explainer(xgb_model, X_test):
    """SHAP on XGBoost (raw features only — more interpretable)."""
    print("\n🔬 SHAP Explainer (V2 — raw features)...")
    explainer = shap.TreeExplainer(xgb_model)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    shap_importance = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    print(f"   📊 Top 5 features (raw, no WoE noise):")
    for i, (_, row) in enumerate(shap_importance.head(5).iterrows()):
        print(f"      {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")

    return explainer, shap_importance


if __name__ == '__main__':
    run_v2_pipeline()
