# ACIS Insurance Risk Analytics

![CI](https://github.com/rah-salah/insurance-risk-analytics/actions/workflows/ci.yml/badge.svg)

A risk analytics and pricing tool for AlphaCare Insurance Solutions (ACIS) — analyzes ~1 million South African motor insurance policies to identify where the book is losing money, and serves an explainable claim-severity pricing model through an interactive dashboard.

## Business Problem

Auto insurers have to price a policy before they know what it will actually cost in claims, and every price has to be defensible to regulators, underwriters, and customers. ACIS's book was running an overall loss ratio of **104.75%** — paying out slightly more in claims than it collects in premium — with risk concentrated unevenly across provinces and vehicle types. The business needs both (1) a clear picture of where flat-rate pricing is losing money, and (2) a severity model whose recommendations can be explained and trusted, not a black box.

## Solution Overview

This project combines exploratory risk analysis with a machine-learning pricing model, delivered through an interactive Streamlit dashboard with four views:

- **Overview** — portfolio-level stats (rows, missingness, total premium/claims, overall loss ratio)
- **Loss Ratio Analysis** — loss ratio broken down by province, to surface where risk-based pricing matters most
- **Model Comparison** — Linear Regression, Random Forest, and XGBoost evaluated on claim severity, with an honest full-dataset comparison rather than a small-sample leaderboard
- **Price a Policy / SHAP** — a live quote form that returns a predicted premium _and_ a SHAP explanation of what drove that specific number

Random Forest was selected as the production model: it matched Linear Regression's full-dataset accuracy while additionally supporting SHAP-based explainability.

## Key Results

- **Overall loss ratio: 104.75%** across 999,805 cleaned policy records
- **Gauteng, KwaZulu-Natal, and Western Cape** are running loss ratios above 1.0; several other provinces sit well under break-even — a clear case for risk-based rather than flat pricing
- **Model comparison (full dataset):** Linear Regression R² 0.28, Random Forest R² 0.25, XGBoost R² 0.04 — XGBoost's apparent edge on a small test subset did not hold at full scale, an important overfitting lesson baked into the final model choice
- **SHAP explainability:** global feature importance (SumInsured, cubic capacity, TotalPremium as top drivers) is consistent with local per-quote explanations, e.g. a sample Bus policy in Gauteng priced at 182.33 with VehicleType and CalculatedPremiumPerTerm as the largest upward drivers

## Quick Start

```bash
git clone https://github.com/rah-salah/insurance-risk-analytics.git
cd insurance-risk-analytics
pip install -r requirements.txt
dvc pull
streamlit run src/dashboard.py
```

### Run tests

```bash
pytest tests/ -v
```

## Project Structure

```
insurance-risk-analytics/
├── .github/workflows/ci.yml     # CI: lint + test on push
├── data/
│   ├── raw/                     # DVC-tracked raw data
│   └── processed/               # Cleaned dataset (999,805 rows)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── eda_utils.py
│   └── dashboard.py              # Streamlit app (Overview / Loss Ratio / Model Comparison / SHAP)
├── tests/
│   └── test_*.py
├── reports/
├── .dvc/
├── requirements.txt
└── README.md
```

## Demo

Live dashboard screenshots (Overview, Loss Ratio by Province, Model Comparison, and Price a Policy / SHAP) are included in the full technical report — see `reports/` or the linked PDF below.

## Technical Details

- **Data:** ACIS motor insurance policy transactions, Oct 2013–Aug 2015, South Africa. 1,000,098 raw rows → 999,805 after cleaning, ~26,000 with nonzero claims.
- **Models:** Linear Regression, Random Forest, XGBoost — trained on policy-level features (TotalPremium, SumInsured, CalculatedPremiumPerTerm, kilowatts, cubic capacity, NumberOfDoors, Province, VehicleType, Gender, MaritalStatus, make) to predict claim severity.
- **Evaluation:** R² on the full dataset, not a small held-out subset, after an early full-vs-subset discrepancy revealed XGBoost was overfitting.
- **Explainability:** SHAP (tree explainer) on the selected Random Forest model, surfaced both as global feature importance and per-quote local explanations in the dashboard.

## Future Improvements

- Engineer incident-level or telematics features to improve on the current R² ceiling
- Expand `tests/` with full unit + integration coverage for data loading and model scoring, wired into the existing CI workflow
- Add confidence intervals around the predicted premium, not just a point estimate
- Support side-by-side SHAP comparison across multiple candidate quotes

## Author

Rahma Salah — [GitHub](https://github.com/rah-salah)
