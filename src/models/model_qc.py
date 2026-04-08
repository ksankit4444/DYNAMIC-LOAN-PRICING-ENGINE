import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, accuracy_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
from src.models.decision_engine import DecisionEngine

# ──────────────────────────────────────────────
# Configuration & Setup
# ──────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_v2_models(config):
    models_path = os.path.join(ROOT_DIR, config['paths']['models'], 'v2')
    artifacts = {
        'xgb_model':    joblib.load(os.path.join(models_path, 'xgboost_model.joblib')),
        'lgb_model':    joblib.load(os.path.join(models_path, 'lightgbm_model.joblib')),
        'lr_model':     joblib.load(os.path.join(models_path, 'logistic_model.joblib')),
        'lr_scaler':    joblib.load(os.path.join(models_path, 'logistic_scaler.joblib')),
        'xgb_feats':    joblib.load(os.path.join(models_path, 'xgb_feature_columns.joblib')),
        'lgb_feats':    joblib.load(os.path.join(models_path, 'lgb_feature_columns.joblib')),
        'lr_feats':     joblib.load(os.path.join(models_path, 'lr_feature_columns.joblib')),
    }
    with open(os.path.join(models_path, 'ensemble_config.json'), 'r') as f:
        artifacts['ensemble_config'] = json.load(f)
    return artifacts

# ──────────────────────────────────────────────
# Visualization Helpers
# ──────────────────────────────────────────────

def plot_correlation_heatmap(df, feature_cols, title, filename):
    """Generate a correlation heatmap for the specified features."""
    print(f"📊 Generating Heatmap: {filename}...")
    plt.figure(figsize=(16, 12))
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=len(feature_cols) <= 30, fmt=".2f", 
                cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(y_true, y_prob, threshold, title, filename):
    """Generate and plot a confusion matrix."""
    print(f"📊 Generating Confusion Matrix: {filename} (Threshold={threshold:.2f})...")
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f"{title}\n(Threshold: {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_performance_curves(y_true, y_prob, output_dir):
    """Plot ROC and Precision-Recall curves."""
    print(f"📊 Generating Performance Curves...")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

# ──────────────────────────────────────────────
# Main QC Pipeline
# ──────────────────────────────────────────────

def run_qc_audit():
    config = load_config()
    models = load_v2_models(config)
    
    # Load test data
    processed_path = os.path.join(ROOT_DIR, config['paths']['processed_data'])
    df_test = pd.read_csv(os.path.join(processed_path, 'features_test.csv'))
    
    X_test_xgb = df_test[models['xgb_feats']]
    X_test_lr = df_test[models['lr_feats']]
    X_test_lr_scaled = models['lr_scaler'].transform(X_test_lr)
    y_test = df_test['TARGET']
    
    print(f"📂 Loaded {len(df_test)} test samples with {len(models['xgb_feats'])} features.")

    # 1. Generate Ensemble Predictions
    pd_xgb = models['xgb_model'].predict_proba(X_test_xgb[models['xgb_feats']])[:, 1]
    pd_lgb = models['lgb_model'].predict_proba(X_test_xgb[models['lgb_feats']])[:, 1]
    pd_lr = models['lr_model'].predict_proba(X_test_lr_scaled)[:, 1]
    
    cfg_weights = models['ensemble_config']['optimal_weights']
    w_xgb = cfg_weights['xgboost']
    w_lgb = cfg_weights['lightgbm']
    w_lr = cfg_weights['logistic']
    y_prob = (w_xgb * pd_xgb) + (w_lgb * pd_lgb) + (w_lr * pd_lr)
    
    # 2. Threshold Analysis
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ks_stat = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    print(f"📈 KS-Statistic: {ks_stat:.4f} at Threshold: {ks_threshold:.4f}")

    # 3. Visualizations - Heatmaps
    output_dir = os.path.join(ROOT_DIR, 'models', 'v2', 'qc_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Top 50 features by XGB importance
    importances = pd.Series(models['xgb_model'].feature_importances_, index=models['xgb_feats'])
    top_50 = importances.sort_values(ascending=False).head(50).index.tolist()
    
    plot_correlation_heatmap(df_test, top_50, "Correlation Matrix: Top 50 Features", 
                             os.path.join(output_dir, 'corr_top_50.png'))
    
    plot_correlation_heatmap(df_test, models['xgb_feats'], "Correlation Matrix: All Features (150+)", 
                             os.path.join(output_dir, 'corr_all.png'))

    # 4. Visualizations - Performance Curves
    plot_performance_curves(y_test, y_prob, output_dir)

    # 5. Visualizations - Confusion Matrices
    plot_confusion_matrix(y_test, y_prob, 0.50, "Ensemble Confusion Matrix (Standard)", 
                          os.path.join(output_dir, 'cm_050.png'))
    
    plot_confusion_matrix(y_test, y_prob, ks_threshold, "Ensemble Confusion Matrix (Optimal KS)", 
                          os.path.join(output_dir, 'cm_ks.png'))

    # 5. Core Metrics
    thresholds_to_test = [0.5, ks_threshold]
    metric_results = {}
    
    for t in thresholds_to_test:
        y_pred = (y_prob >= t).astype(int)
        t_name = f"thresh_{t:.2f}"
        metric_results[t_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob)
        }
        print(f"\n📊 Metrics for {t_name}:")
        print(f"   Accuracy:  {metric_results[t_name]['accuracy']:.4f}")
        print(f"   Precision: {metric_results[t_name]['precision']:.4f} (Ability to reject the 'bad' correctly)")
        print(f"   Recall:    {metric_results[t_name]['recall']:.4f} (Ability to find 'bad' loans)")

    # 6. VIF Analysis (Top 30 features to keep it readable and fast)
    print("\n🧮 Calculating VIF for Multi-collinearity check...")
    vif_data = pd.DataFrame()
    top_30 = top_50[:30]
    vif_data["feature"] = top_30
    
    # Add constant for VIF
    X_vif = X_test_xgb[top_30].copy()
    X_vif['const'] = 1
    
    vif_values = []
    for i in range(len(top_30)):
        vif_values.append(variance_inflation_factor(X_vif.values, i))
    
    vif_data["VIF"] = vif_values
    vif_data = vif_data.sort_values("VIF", ascending=False)
    print(vif_data.head(15))

    # 7. Strategic Waterfall Audit (V6.2)
    print("\n🕵️ Running Strategic Waterfall Audit (V6.2)...")
    engine = DecisionEngine()
    
    tier_results = []
    # Vectorized check where possible, but for clarity let's use the engine's logic
    # We'll calculate ext_source_min for the whole test set
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df_test.columns]
    ext_min_series = df_test[ext_cols].min(axis=1)
    
    # Store decisions
    decisions = []
    for i in range(len(df_test)):
        pd_val = y_prob[i]
        esm_val = ext_min_series.iloc[i]
        tier, desc = engine.decide_tier(pd_val, esm_val)
        decisions.append({'tier': tier, 'pd': pd_val, 'esm': esm_val, 'actual': y_test.iloc[i]})
    
    df_decisions = pd.DataFrame(decisions)
    
    for tier in ['Elite', 'Bureau Prime', 'Engine Core', 'Watchlist']:
        mask = df_decisions['tier'] == tier
        count = mask.sum()
        pct = count / len(df_decisions)
        dr = df_decisions.loc[mask, 'actual'].mean() if count > 0 else 0
        
        tier_results.append({
            'tier': tier,
            'count': int(count),
            'population_pct': round(float(pct), 4),
            'default_rate': round(float(dr), 4)
        })
        print(f"   [{tier:13s}] Pop: {pct:6.1%} | Default Rate: {dr:7.2%}")

    # Save Results
    report_path = os.path.join(ROOT_DIR, 'models', 'v2', 'qc_metrics.json')
    with open(report_path, 'w') as f:
        json.dump({
            'metrics': metric_results,
            'ks_stats': {'ks_stat': float(ks_stat), 'ks_threshold': float(ks_threshold)},
            'waterfall_audit': tier_results,
            'vif': vif_data.to_dict(orient='records')
        }, f, indent=4)
    
    print(f"\n✅ QC Audit Complete. Results saved to {report_path}")

if __name__ == "__main__":
    run_qc_audit()
