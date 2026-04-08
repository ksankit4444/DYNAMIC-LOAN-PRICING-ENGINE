"""
expected_loss.py — Expected Loss Calculation (Layer 2)
=====
Implements the Basel II/III Expected Loss framework:
    EL = PD × LGD × EAD

From EL, derives the Minimum Rate Floor:
    Min Rate = (EL / Loan Amount) + Cost of Capital + Operational Margin

This module is used by:
  - The optimization engine (Layer 4) as a hard constraint
  - The Streamlit dashboard for rate justification breakdown
  - The LangGraph agent's calculate_expected_loss() tool

Usage:
    from src.models.expected_loss import ExpectedLossCalculator
    el_calc = ExpectedLossCalculator(config)
    results = el_calc.calculate(pd=0.15, loan_amount=1500000)
"""

import os
import sys
import yaml
import numpy as np
import pandas as pandas_df

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ExpectedLossCalculator:
    """
    Expected Loss calculation engine.

    EL = PD × LGD × EAD

    Where:
      PD  = Probability of Default (output from Layer 1 ensemble)
      LGD = Loss Given Default (business assumption, typically 60-65% for unsecured)
      EAD = Exposure at Default (≈ loan amount for term loans)
    """

    def __init__(self, config=None):
        if config is None:
            config = load_config()

        business = config['business']
        self.lgd = business.get('lgd', 0.65)
        self.cost_of_capital = business.get('cost_of_capital', 0.085)
        self.operational_margin = business.get('operational_margin', 0.015)
        self.regulatory_ceiling = business.get('regulatory_ceiling', 0.36)
        self.fairness_max_spread = business.get('fairness_max_rate_spread', 0.12)

    def calculate(self, pd, loan_amount, lgd_override=None, loan_term_months=60):
        """
        Calculate Expected Loss and Minimum Rate Floor for a single applicant.

        Args:
            pd: Probability of Default (0 to 1)
            loan_amount: Total loan/credit amount (EAD proxy)
            lgd_override: Override default LGD assumption (optional)
            loan_term_months: Loan term in months (for amortization-based EAD)

        Returns:
            dict with EL, rate floor, and full breakdown
        """
        lgd = lgd_override if lgd_override is not None else self.lgd

        # EAD — For term loans, use full loan amount
        # (In production, you'd discount by expected amortization at default timing)
        ead = loan_amount

        # Core formula: EL = PD × LGD × EAD
        expected_loss = pd * lgd * ead
        el_rate = expected_loss / max(loan_amount, 1)  # EL as annualized rate

        # Minimum Rate Floor = EL rate + Cost of Capital + Operational Margin
        min_rate_floor = el_rate + self.cost_of_capital + self.operational_margin

        # Cap at regulatory ceiling
        min_rate_floor = min(min_rate_floor, self.regulatory_ceiling)

        return {
            # Core outputs
            'expected_loss': round(expected_loss, 2),
            'expected_loss_rate': round(el_rate, 6),
            'min_rate_floor': round(min_rate_floor, 6),

            # Inputs used
            'pd': round(pd, 6),
            'lgd': lgd,
            'ead': ead,
            'loan_amount': loan_amount,

            # Rate breakdown (for dashboard justification panel)
            'rate_breakdown': {
                'el_component': round(el_rate, 6),
                'cost_of_capital': self.cost_of_capital,
                'operational_margin': self.operational_margin,
                'total_floor': round(min_rate_floor, 6),
            },

            # Formatted for display
            'formatted': {
                'expected_loss': f"₹{expected_loss:,.0f}",
                'el_rate': f"{el_rate * 100:.2f}%",
                'min_rate_floor': f"{min_rate_floor * 100:.2f}%",
                'cost_of_capital': f"{self.cost_of_capital * 100:.2f}%",
                'operational_margin': f"{self.operational_margin * 100:.2f}%",
            }
        }

    def calculate_batch(self, df, pd_col='pd_ensemble', loan_col='AMT_CREDIT'):
        """
        Calculate EL for an entire DataFrame of applicants.

        Args:
            df: DataFrame with PD scores and loan amounts
            pd_col: Column name for PD score
            loan_col: Column name for loan amount

        Returns:
            DataFrame with EL columns added
        """
        result = df.copy()

        result['ead'] = result[loan_col]
        result['expected_loss'] = result[pd_col] * self.lgd * result['ead']
        result['el_rate'] = result['expected_loss'] / result[loan_col].replace(0, np.nan)
        result['min_rate_floor'] = (
            result['el_rate'] + self.cost_of_capital + self.operational_margin
        )
        result['min_rate_floor'] = result['min_rate_floor'].clip(upper=self.regulatory_ceiling)

        return result

    def sensitivity_analysis(self, pd, loan_amount, lgd_range=None):
        """
        Run EL sensitivity analysis across LGD assumptions.
        The readme says: "sensitivity test across range" for LGD.

        Args:
            pd: Probability of Default
            loan_amount: Loan amount
            lgd_range: List of LGD values to test (default: 0.40 to 0.80)

        Returns:
            DataFrame with EL at each LGD level
        """
        if lgd_range is None:
            lgd_range = np.arange(0.40, 0.85, 0.05)

        results = []
        for lgd in lgd_range:
            calc = self.calculate(pd, loan_amount, lgd_override=lgd)
            results.append({
                'lgd': round(lgd, 2),
                'expected_loss': calc['expected_loss'],
                'el_rate': calc['expected_loss_rate'],
                'min_rate_floor': calc['min_rate_floor'],
            })

        return pandas_df.DataFrame(results)


# ──────────────────────────────────────────────
# CLI Runner
# ──────────────────────────────────────────────
if __name__ == '__main__':
    calc = ExpectedLossCalculator()

    print("=" * 60)
    print("  EXPECTED LOSS CALCULATOR — Layer 2")
    print("=" * 60)

    # Example calculations across risk bands
    test_cases = [
        {'label': 'Low Risk',    'pd': 0.05, 'loan': 1500000},
        {'label': 'Medium Risk', 'pd': 0.18, 'loan': 1000000},
        {'label': 'High Risk',   'pd': 0.40, 'loan': 500000},
    ]

    for case in test_cases:
        result = calc.calculate(case['pd'], case['loan'])
        print(f"\n  📊 {case['label']} (PD={case['pd']:.0%}, Loan=₹{case['loan']:,.0f})")
        print(f"     Expected Loss:    {result['formatted']['expected_loss']}")
        print(f"     EL Rate:          {result['formatted']['el_rate']}")
        print(f"     + Cost of Capital: {result['formatted']['cost_of_capital']}")
        print(f"     + Op. Margin:      {result['formatted']['operational_margin']}")
        print(f"     ─────────────────────────────")
        print(f"     Min Rate Floor:   {result['formatted']['min_rate_floor']}")

    # Sensitivity analysis
    print(f"\n\n  📈 LGD Sensitivity (PD=0.18, Loan=₹10L)")
    sens = calc.sensitivity_analysis(pd=0.18, loan_amount=1000000)
    print(sens.to_string(index=False))

    print("\n" + "=" * 60)
