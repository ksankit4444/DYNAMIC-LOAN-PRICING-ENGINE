import os
import numpy as np
import pandas as pd
import joblib
import json

class DecisionEngine:
    """
    Layer 1.5 - Layered Decision Engine (V6.2)
    -----
    Implements the 'Strategic Waterfall' logic:
    1. Elite (Golden Consensus)
    2. Bureau Prime (Low-Risk Prime)
    3. Engine Core (Standard ML Scoring)
    4. Watchlist (High Risk / Policy Reject)
    """

    def __init__(self, models_path='models/v2'):
        self.models_path = models_path
        self.artifacts = self._load_artifacts()
        
    def _load_artifacts(self):
        """Load models, weights, and scalers."""
        try:
            return {
                'xgb_model': joblib.load(os.path.join(self.models_path, 'xgboost_model.joblib')),
                'lgb_model': joblib.load(os.path.join(self.models_path, 'lightgbm_model.joblib')),
                'lr_model': joblib.load(os.path.join(self.models_path, 'logistic_model.joblib')),
                'lr_scaler': joblib.load(os.path.join(self.models_path, 'logistic_scaler.joblib')),
                'ensemble_config': json.load(open(os.path.join(self.models_path, 'ensemble_config.json'), 'r')),
                'xgb_feats': joblib.load(os.path.join(self.models_path, 'xgb_feature_columns.joblib')),
                'lgb_feats': joblib.load(os.path.join(self.models_path, 'lgb_feature_columns.joblib')),
                'lr_feats': joblib.load(os.path.join(self.models_path, 'lr_feature_columns.joblib')),
            }
        except Exception as e:
            print(f"⚠️ Error loading decision engine artifacts: {e}")
            return None

    def predict_pd(self, X):
        """Generate the ensemble PD score."""
        if self.artifacts is None: return None
        
        # 1. Individual predictions
        pd_xgb = self.artifacts['xgb_model'].predict_proba(X[self.artifacts['xgb_feats']])[:, 1]
        pd_lgb = self.artifacts['lgb_model'].predict_proba(X[self.artifacts['lgb_feats']])[:, 1]
        
        X_lr_scaled = self.artifacts['lr_scaler'].transform(X[self.artifacts['lr_feats']])
        pd_lr = self.artifacts['lr_model'].predict_proba(X_lr_scaled)[:, 1]
        
        # 2. Weighted Blend
        w = self.artifacts['ensemble_config']['optimal_weights']
        y_prob = (w['xgboost'] * pd_xgb) + (w['lightgbm'] * pd_lgb) + (w['logistic'] * pd_lr)
        
        return y_prob

    def decide_tier(self, pd_score, ext_source_min):
        """
        Apply the v6.2 Strategic Waterfall.
        Returns: (Tier Name, Logic Description)
        """
        # Tier 1: Elite (Golden Consensus)
        if pd_score < 0.05 and ext_source_min > 0.6:
            return 'Elite', "Golden Consensus: Both Bureaus and Model agree on ultra-low risk."
            
        # Tier 2: Bureau Prime (Low-Risk Prime)
        # Includes the 'Deceptive Risk' ceiling (PD < 0.25)
        if ext_source_min > 0.7 and pd_score < 0.25:
            return 'Bureau Prime', "Consistently high Bureau scores with acceptable ML risk profile."
            
        # Tier 3: Watchlist (High Risk / Policy Reject)
        if pd_score > 0.35:
            return 'Watchlist', "High Default Risk: Model-predicted PD exceeds risk appetite ceiling (0.35)."
            
        # Tier 4: Engine Core (Standard Scoring)
        return 'Engine Core', "Standard Risk Profile: Relying on optimized ML Pricing Engine."

    def process_applicant(self, X_row):
        """
        Full decision path for a single applicant.
        X_row should be a DataFrame with 1 row.
        """
        pd_score = self.predict_pd(X_row)[0]
        
        # Calculate ext_source_min if not explicitly passed
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in X_row.columns]
        ext_source_min = X_row[ext_cols].min(axis=1).values[0] if ext_cols else 0.5
        
        tier, description = self.decide_tier(pd_score, ext_source_min)
        
        return {
            'pd_score': float(pd_score),
            'tier': tier,
            'description': description,
            'ext_source_min': float(ext_source_min)
        }

if __name__ == "__main__":
    # Quick Test
    engine = DecisionEngine()
    print("✅ Decision Engine Initialized.")
