# Validation Suite

Reliability and auditability are non-negotiable in financial machine learning. This directory contains the automated verification pipeline for the loan pricing engine.

## Coverage Areas

### 1. Mathematical Validation (`unit/`)
Ensures that the financial calculators are logically sound.
* **Expected Loss Consistency**: Validates that $EL = PD \times LGD \times EAD$ holds across different edge cases.
* **Optimizer Convergence**: Ensures the `scipy` optimizer correctly identifies the global maximum profit within the regulatory bounds.

### 2. Feature Integrity
* **WoE Monotonicity**: Checks that the Weight of Evidence (WoE) transformation correctly preserves the relationship between feature bins and the target variable.
* **Null Handling**: Verifies that the engineering pipeline can handle missing data without crashing.

### 3. Integration & System Tests (`integration/`)
Validates the handoff between layers.
* **Feature-to-Score**: Checks that a raw dictionary of applicant data can flow through engineering and produce a valid PD score.
* **Score-to-Rate**: Verifies that a PD score correctly results in an optimized interest rate recommendation.

## Running the Tests

Ensure you have your environment activated and dependencies installed:

```bash
# Run all tests
pytest github/tests/

# Run specifically for the optimization engine
pytest github/tests/unit/test_optimization.py

# Run with verbose output for debugging
pytest -v github/tests/
```

## Continuous Integration
In a professional environment, these tests should be configured to run on every Pull Request via GitHub Actions to ensure no regressions are introduced to the pricing logic.
