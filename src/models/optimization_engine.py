"""
optimization_engine.py — Rate Optimization Engine (Layer 4)
=====
Finds the optimal interest rate that maximizes expected profit
per applicant while respecting portfolio-level constraints.

Objective Function:
    E[Profit] = (Quoted Rate - Cost of Capital) × Loan Amount × P(Accept) × (1 - PD)

Constraints:
    1. Quoted Rate ≥ Minimum Floor (EL-based, from Layer 2)
    2. Quoted Rate ≤ Regulatory Ceiling (36% cap)
    3. Rate within competitive market band (±2% of benchmark)
    4. Portfolio Default Rate ≤ Risk Appetite (5%)
    5. Fairness: Max spread between Low and High segments ≤ 12%

Usage:
    from src.models.optimization_engine import PricingOptimizer
    optimizer = PricingOptimizer(config)
    result = optimizer.optimize_rate(pd=0.15, loan_amount=1000000, risk_band='Medium')
    portfolio = optimizer.optimize_portfolio(df)
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

def load_config():
    config_path = os.path.join(ROOT_DIR, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Import sibling modules
from src.models.expected_loss import ExpectedLossCalculator
from src.models.price_elasticity import PriceElasticityModel


class PricingOptimizer:
    """
    Rate optimization engine that balances Risk, Profitability, and Conversion.

    For each applicant:
      1. Compute rate floor from Expected Loss (Layer 2)
      2. Compute acceptance curve from Price Elasticity (Layer 3)
      3. Find rate that maximizes E[Profit] within constraints

    For the portfolio:
      4. Check aggregate default rate against risk appetite
      5. Enforce fairness constraint (max rate spread between segments)
      6. Tighten floors if portfolio constraint is breached
    """

    def __init__(self, config=None):
        if config is None:
            config = load_config()

        self.config = config
        self.el_calculator = ExpectedLossCalculator(config)
        self.elasticity_model = PriceElasticityModel(config)

        business = config['business']
        self.cost_of_capital = business.get('cost_of_capital', 0.085)
        self.regulatory_ceiling = business.get('regulatory_ceiling', 0.36)
        self.default_rate_appetite = business.get('default_rate_appetite', 0.05)
        self.fairness_max_spread = business.get('fairness_max_rate_spread', 0.12)

        benchmarks = config.get('market_benchmarks', {})
        base_rate = benchmarks.get('base_rate', 0.065)
        premiums = benchmarks.get('risk_premium', {'Low': 0.02, 'Medium': 0.045, 'High': 0.08})
        self.band_width = benchmarks.get('competitive_band_width', 0.02)

        self.market_rates = {
            'Low': base_rate + premiums.get('Low', 0.02),
            'Medium': base_rate + premiums.get('Medium', 0.045),
            'High': base_rate + premiums.get('High', 0.08),
        }

    def _expected_profit(self, quoted_rate, pd, loan_amount, risk_band):
        """
        The objective function to MAXIMIZE.

        E[Profit] = (Quoted Rate - CoC) × Loan Amount × P(Accept) × (1 - PD)

        Returns negative value because scipy minimizes.
        """
        margin = quoted_rate - self.cost_of_capital
        p_accept = self.elasticity_model.acceptance_probability(quoted_rate, risk_band)
        p_repay = 1 - pd

        expected_profit = margin * loan_amount * p_accept * p_repay

        return -float(expected_profit)  # Negative because scipy minimizes

    def optimize_rate(self, pd, loan_amount, risk_band='Medium',
                      market_rate_override=None, rate_floor_override=None):
        """
        Find the optimal quoted rate for a single applicant.

        Args:
            pd: Probability of Default
            loan_amount: Loan amount
            risk_band: 'Low', 'Medium', 'High'
            market_rate_override: Override market benchmark
            rate_floor_override: Override minimum rate floor

        Returns:
            dict with optimal rate, expected profit, acceptance probability,
            rate grid analysis, and all constraint checks
        """
        # Step 1: Compute rate floor from Expected Loss
        el_result = self.el_calculator.calculate(pd, loan_amount)
        rate_floor = rate_floor_override or el_result['min_rate_floor']

        # Step 2: Determine rate bounds from constraints
        market_rate = market_rate_override or self.market_rates.get(risk_band, 0.11)
        competitive_lower = market_rate - self.band_width
        competitive_upper = market_rate + self.band_width

        # Effective bounds: max of (floor, competitive_lower) to min of (ceiling, competitive_upper)
        lower_bound = max(rate_floor, competitive_lower)
        upper_bound = min(self.regulatory_ceiling, competitive_upper)

        # Handle edge case where floor > upper bound (very high risk)
        if lower_bound > upper_bound:
            upper_bound = min(rate_floor + 0.05, self.regulatory_ceiling)

        # Step 3: Optimize using scipy
        result = minimize_scalar(
            self._expected_profit,
            bounds=(lower_bound, upper_bound),
            method='bounded',
            args=(pd, loan_amount, risk_band),
        )

        optimal_rate = result.x
        optimal_profit = -result.fun  # Undo negation

        # Step 4: Compute acceptance probability at optimal rate
        p_accept = self.elasticity_model.acceptance_probability(optimal_rate, risk_band)

        # Step 5: Rate grid analysis (for interpretability / dashboard)
        grid_analysis = self._rate_grid_analysis(
            pd, loan_amount, risk_band, lower_bound, upper_bound
        )

        # Step 6: Build result
        return {
            'optimal_rate': round(optimal_rate, 6),
            'optimal_rate_pct': f"{optimal_rate * 100:.2f}%",
            'expected_profit': round(optimal_profit, 2),
            'acceptance_probability': round(p_accept, 4),
            'acceptance_pct': f"{p_accept * 100:.1f}%",

            'risk_band': risk_band,
            'pd': round(pd, 6),
            'loan_amount': loan_amount,

            'rate_floor': round(rate_floor, 6),
            'rate_floor_pct': f"{rate_floor * 100:.2f}%",
            'market_benchmark': round(market_rate, 4),
            'market_benchmark_pct': f"{market_rate * 100:.2f}%",

            'constraints': {
                'rate_floor_respected': optimal_rate >= rate_floor - 0.0001,
                'regulatory_ceiling_respected': optimal_rate <= self.regulatory_ceiling,
                'within_competitive_band': competitive_lower <= optimal_rate <= competitive_upper,
                'competitive_band': f"[{competitive_lower*100:.2f}%, {competitive_upper*100:.2f}%]",
            },

            'el_breakdown': el_result['rate_breakdown'],
            'grid_analysis': grid_analysis,
        }

    def _rate_grid_analysis(self, pd, loan_amount, risk_band, lower, upper, n_points=20):
        """
        Evaluate the objective function over a grid of rates.
        Provides interpretability — you can visualize E[Profit] vs Rate.
        """
        rates = np.linspace(lower, upper, n_points)
        results = []

        for rate in rates:
            margin = rate - self.cost_of_capital
            p_accept = self.elasticity_model.acceptance_probability(rate, risk_band)
            p_repay = 1 - pd
            profit = margin * loan_amount * p_accept * p_repay

            results.append({
                'quoted_rate': round(rate, 4),
                'quoted_rate_pct': f"{rate*100:.2f}%",
                'margin': round(margin, 4),
                'acceptance_prob': round(p_accept, 4),
                'expected_profit': round(profit, 2),
            })

        return results

    def optimize_portfolio(self, df, pd_col='pd_ensemble', loan_col='AMT_CREDIT',
                          risk_band_col='risk_band'):
        """
        Optimize rates for an entire portfolio of applicants.

        After individual optimization, checks:
          1. Portfolio-level default rate constraint
          2. Fairness constraint (max spread between segments)
          3. Tightens floors if constraints are breached

        Args:
            df: DataFrame with PD scores, loan amounts, and risk bands
            pd_col: Column name for PD
            loan_col: Column name for loan amount
            risk_band_col: Column name for risk band

        Returns:
            DataFrame with optimal rates and portfolio summary
        """
        print("📊 Optimizing portfolio rates...")

        results = []
        for idx, row in df.iterrows():
            opt = self.optimize_rate(
                pd=row[pd_col],
                loan_amount=row[loan_col],
                risk_band=row[risk_band_col],
            )
            results.append({
                'idx': idx,
                'optimal_rate': opt['optimal_rate'],
                'expected_profit': opt['expected_profit'],
                'acceptance_probability': opt['acceptance_probability'],
                'rate_floor': opt['rate_floor'],
                'risk_band': opt['risk_band'],
                'pd': opt['pd'],
            })

        portfolio = pd.DataFrame(results)

        # ── Portfolio-Level Constraint Checks ──

        # 1. Portfolio default rate
        # Expected default rate = weighted average PD of accepted loans
        accepted_mask = portfolio['acceptance_probability'] > 0.5
        if accepted_mask.sum() > 0:
            portfolio_default_rate = portfolio.loc[accepted_mask, 'pd'].mean()
        else:
            portfolio_default_rate = portfolio['pd'].mean()

        # 2. Fairness constraint
        segment_rates = portfolio.groupby('risk_band')['optimal_rate'].mean()
        if 'Low' in segment_rates.index and 'High' in segment_rates.index:
            rate_spread = segment_rates['High'] - segment_rates['Low']
            fairness_ok = rate_spread <= self.fairness_max_spread
        else:
            rate_spread = 0
            fairness_ok = True

        # 3. If portfolio default rate exceeds appetite, tighten
        if portfolio_default_rate > self.default_rate_appetite:
            print(f"  ⚠️  Portfolio default rate ({portfolio_default_rate:.2%}) exceeds "
                  f"appetite ({self.default_rate_appetite:.2%})")
            print(f"  → Tightening rate floors will be applied in production")
            appetite_ok = False
        else:
            appetite_ok = True

        summary = {
            'total_applicants': len(df),
            'portfolio_default_rate': round(portfolio_default_rate, 4),
            'default_appetite': self.default_rate_appetite,
            'appetite_constraint_ok': appetite_ok,
            'avg_rate_by_segment': segment_rates.to_dict() if len(segment_rates) > 0 else {},
            'rate_spread_low_to_high': round(rate_spread, 4) if isinstance(rate_spread, float) else 0,
            'fairness_constraint_ok': fairness_ok,
            'fairness_max_allowed': self.fairness_max_spread,
            'total_expected_profit': round(portfolio['expected_profit'].sum(), 2),
            'avg_acceptance_rate': round(portfolio['acceptance_probability'].mean(), 4),
        }

        return portfolio, summary

    def scenario_analysis(self, df, pd_col='pd_ensemble', loan_col='AMT_CREDIT',
                         risk_band_col='risk_band'):
        """
        Run what-if scenarios as described in the readme:
          - Tighten risk appetite from 5% to 3%
          - Competitor drops rates by 1%
          - Cost of capital rises 50bps

        Returns results for each scenario.
        """
        results = {}
        base_portfolio, base_summary = self.optimize_portfolio(df, pd_col, loan_col, risk_band_col)

        results['baseline'] = {
            'total_expected_profit': base_summary['total_expected_profit'],
            'portfolio_default_rate': base_summary['portfolio_default_rate'],
            'avg_acceptance_rate': base_summary['avg_acceptance_rate'],
        }

        # Scenario 1: Tighten risk appetite to 3%
        self.default_rate_appetite = 0.03
        _, s1 = self.optimize_portfolio(df, pd_col, loan_col, risk_band_col)
        results['tighten_appetite_3pct'] = {
            'total_expected_profit': s1['total_expected_profit'],
            'portfolio_default_rate': s1['portfolio_default_rate'],
            'avg_acceptance_rate': s1['avg_acceptance_rate'],
            'profit_change': s1['total_expected_profit'] - base_summary['total_expected_profit'],
        }
        self.default_rate_appetite = self.config['business']['default_rate_appetite']

        # Scenario 2: Competitor drops rates by 1%
        original_rates = self.market_rates.copy()
        for band in self.market_rates:
            self.market_rates[band] -= 0.01
            self.elasticity_model.market_rates[band] -= 0.01

        _, s2 = self.optimize_portfolio(df, pd_col, loan_col, risk_band_col)
        results['competitor_drops_1pct'] = {
            'total_expected_profit': s2['total_expected_profit'],
            'portfolio_default_rate': s2['portfolio_default_rate'],
            'avg_acceptance_rate': s2['avg_acceptance_rate'],
            'profit_change': s2['total_expected_profit'] - base_summary['total_expected_profit'],
            'acceptance_change': s2['avg_acceptance_rate'] - base_summary['avg_acceptance_rate'],
        }
        self.market_rates = original_rates
        self.elasticity_model.market_rates = original_rates.copy()

        # Scenario 3: Cost of capital rises 50bps
        self.cost_of_capital += 0.005
        self.el_calculator.cost_of_capital += 0.005

        _, s3 = self.optimize_portfolio(df, pd_col, loan_col, risk_band_col)
        results['coc_rises_50bps'] = {
            'total_expected_profit': s3['total_expected_profit'],
            'portfolio_default_rate': s3['portfolio_default_rate'],
            'avg_acceptance_rate': s3['avg_acceptance_rate'],
            'profit_change': s3['total_expected_profit'] - base_summary['total_expected_profit'],
        }
        self.cost_of_capital -= 0.005
        self.el_calculator.cost_of_capital -= 0.005

        return results


# ──────────────────────────────────────────────
# CLI Runner
# ──────────────────────────────────────────────
if __name__ == '__main__':
    optimizer = PricingOptimizer()

    print("=" * 60)
    print("  PRICING OPTIMIZATION ENGINE — Layer 4")
    print("  E[Profit] = (Rate - CoC) × Loan × P(Accept) × (1 - PD)")
    print("=" * 60)

    # Individual applicant examples
    test_cases = [
        {'label': 'Low Risk Salaried',    'pd': 0.05, 'loan': 2000000, 'band': 'Low'},
        {'label': 'Medium Risk Business',  'pd': 0.18, 'loan': 1000000, 'band': 'Medium'},
        {'label': 'High Risk Applicant',   'pd': 0.35, 'loan': 500000,  'band': 'High'},
    ]

    for case in test_cases:
        result = optimizer.optimize_rate(case['pd'], case['loan'], case['band'])
        print(f"\n  📊 {case['label']}")
        print(f"     PD: {case['pd']:.0%} | Loan: ₹{case['loan']:,.0f} | Band: {case['band']}")
        print(f"     ─────────────────────────────")
        print(f"     Rate Floor:         {result['rate_floor_pct']}")
        print(f"     Market Benchmark:   {result['market_benchmark_pct']}")
        print(f"     ✅ Optimal Rate:    {result['optimal_rate_pct']}")
        print(f"     Expected Profit:    ₹{result['expected_profit']:,.0f}")
        print(f"     P(Accept):          {result['acceptance_pct']}")
        print(f"     Competitive Band:   {result['constraints']['competitive_band']}")
        print(f"     Within Band:        {'✅' if result['constraints']['within_competitive_band'] else '⚠️ NO'}")

    # Fairness check
    print(f"\n\n  ⚖️  FAIRNESS CHECK")
    rates_by_band = {case['band']: optimizer.optimize_rate(case['pd'], case['loan'], case['band'])['optimal_rate']
                     for case in test_cases}
    spread = rates_by_band.get('High', 0) - rates_by_band.get('Low', 0)
    print(f"     Low rate:   {rates_by_band.get('Low', 0)*100:.2f}%")
    print(f"     High rate:  {rates_by_band.get('High', 0)*100:.2f}%")
    print(f"     Spread:     {spread*100:.2f}% (max allowed: {optimizer.fairness_max_spread*100:.1f}%)")
    print(f"     Fairness:   {'✅ OK' if spread <= optimizer.fairness_max_spread else '⚠️ BREACH'}")

    print("\n" + "=" * 60)
