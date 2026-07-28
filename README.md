# Insurance Risk Analytics

[![CI](https://github.com/rah-salah/insurance-risk-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/rah-salah/insurance-risk-analytics/actions/workflows/ci.yml)

Risk-based insurance pricing for AlphaCare Insurance Solutions (ACIS), built on 1 million real South African auto insurance policies.

## Business Problem

ACIS is currently paying out more in claims than it collects in premiums (overall loss ratio of 1.05), meaning it loses money on the average policy. Without accurate, data-driven risk pricing, ACIS either overcharges safe customers - losing them to competitors - or undercharges risky ones, losing money on every claim they file. This project builds a machine learning system that predicts expected claim severity per policy, so ACIS can price each policy according to its actual risk rather than a flat rate.

## Solution Overview

1. **Explore & clean** 1,000,098 policies across 9 South African provinces and 46 vehicle makes, with the cleaned dataset tracked via DVC for reproducibility.
2. **Test hypotheses** about what actually drives risk (province, vehicle type, driver demographics) using statistical A/B testing rather than assumption.
3. **Train and compare 3 models** (Linear Regression, Random Forest, XGBoost) to predict claim severity given a claim occurs.
4. **Explain predictions with SHAP**, so pricing decisions are transparent and defensible - not a black box - to underwriters, regulators, and customers.
5. **Convert predictions into a risk-based premium**: `optimal_premium = (P(claim) x predicted_severity) x (1 + expense_loading + profit_margin)`.

## Key Results

- **Overall loss ratio: 1.0477** - ACIS pays out more than it collects on the average policy today
- **Gauteng** is the riskiest province (loss ratio 1.163); **Northern Cape** is the safest (0.283)
- **Heavy Commercial** vehicles cost ACIS the most per policy (loss ratio 1.612)
- 3 models compared on real held-out test data, with the best model selected by MAE
- SHAP explainability identifies which features drive each individual pricing decision, not just global averages
- Refactored modeling pipeline is covered by **27 passing unit tests** and enforced by CI on every push

## Quick Start

```bash
git clone https://github.com/rah-salah/insurance-risk-analytics
cd insurance-risk-analytics
pip install -r requirements.txt
dvc pull                      # pulls the tracked dataset (requires DVC remote access)
python -m pytest tests/ -v    # runs the full test suite
```

To run the modeling pipeline on your own data:

```python
from src.modeling import ModelConfig, run_full_pipeline
import pandas as pd

df = pd.read_csv("data/processed/clean_insurance.csv")
result = run_full_pipeline(df, ModelConfig())
print(result["comparison"])
```

## Project Structure

```
insurance-risk-analytics/
|-- data/
|   |-- raw/            # original data (DVC-tracked)
|   `-- processed/      # cleaned data (DVC-tracked)
|-- notebooks/          # 01 EDA, 02 DVC setup, 03 modeling & SHAP
|-- src/
|   |-- data_loader.py  # loading, cleaning, summary functions
|   |-- eda_utils.py    # loss ratio, outlier, correlation plotting
|   |-- modeling.py     # refactored, type-hinted, tested modeling pipeline
|   `-- utils.py        # shared helper functions
|-- tests/               # 27 unit tests covering data_loader.py and modeling.py
`-- .github/workflows/  # CI: pytest + flake8 on every push
```

## Demo

Interactive Streamlit dashboard: *in progress - see Future Improvements below.*

## Technical Details

- **Data**: `MachineLearningRating_v3.txt`, 1,000,098 rows, pipe-separated, DVC-tracked
- **Preprocessing**: negative premium/claim removal, loss ratio and margin calculation, label encoding for categorical features (Province, VehicleType, Gender, MaritalStatus, make)
- **Models**: Linear Regression (baseline), Random Forest (100 trees, max depth 10), XGBoost (200 trees, max depth 6, learning rate 0.1) - all trained on an 80/20 split (`random_state=42`)
- **Evaluation**: RMSE, MAE, R2 on held-out test data; best model selected by lowest MAE
- **Explainability**: SHAP TreeExplainer on the best tree-based model, global feature importance and per-prediction explanations
- **Pricing formula**: `risk_premium = P(claim) x predicted_severity`; `optimal_premium = risk_premium x (1 + expense_loading + profit_margin)`. Note: this corrects a bug present in the original notebook, where expense/profit loading was applied to raw predicted severity instead of the risk premium.

## Future Improvements

- Interactive Streamlit dashboard: key metrics, loss-ratio trends by province/vehicle, a live risk-prediction tool, and SHAP explanation plots
- Persist trained models and fitted encoders to disk so predictions don't require retraining
- Expand hypothesis testing (Task 3) into a fuller statistical report with confidence intervals
- Explore additional engineered features (vehicle age, policy duration) that were investigated but not yet integrated into the modeling pipeline
- Hyperparameter tuning (grid/random search) for the Random Forest and XGBoost models

## Author

Rahma Salah
10 Academy KAIM 9 - Week 3 (capstone-improved Week 12)
