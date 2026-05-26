# Insurance Risk Analytics
## 10 Academy KAIM 9 - Week 3 | AlphaCare Insurance Solutions

Analyzing 1 million South African insurance policies to identify risk patterns
and build predictive models for risk-based pricing.

## Business Objective
Help ACIS identify low-risk customers and price insurance policies more accurately
using machine learning and statistical analysis of Google Play reviews.

## Key Findings
- **Overall Loss Ratio: 1.0477** — ACIS is paying more in claims than collecting
- **Gauteng** is the riskiest province (Loss Ratio: 1.163)
- **Northern Cape** is the safest province (Loss Ratio: 0.283)
- **Heavy Commercial** vehicles cost ACIS most (Loss Ratio: 1.612)
- **1,000,098 policies** analyzed across 9 provinces and 46 vehicle makes

## Project Structure

```
insurance-risk-analytics/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   │   ├── MachineLearningRating_v3.txt.dvc
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_loader.py
│   └── eda_utils.py
├── tests/
│   └── test_placeholder.py
├── reports/
├── .dvc/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/rah-salah/insurance-risk-analytics.git
cd insurance-risk-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull data with DVC
```bash
dvc pull
```

### 4. Run the notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 5. Run tests
```bash
pytest tests/ -v
```

## Tasks
- **Task 1** - Exploratory Data Analysis (loss ratio, provinces, vehicles, zip codes)
- **Task 2** - Data Version Control with DVC
- **Task 3** - A/B Hypothesis Testing (in progress)
- **Task 4** - Machine Learning Models (in progress)

## Data Source
- AlphaCare Insurance Solutions (ACIS)
- 1,000,098 policy transactions
- Period: October 2013 to August 2015
- South Africa, 9 provinces

## Reproducing the Data Pipeline

### Pull data with DVC
```bash
dvc pull
```

### Run full pipeline
```bash
python scripts/scrape_reviews.py  # Not applicable
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_hypothesis_testing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

### Data versions tracked
- v1 Raw: data/raw/MachineLearningRating_v3.txt (1,000,098 rows)
- v2 Clean: data/processed/clean_insurance.csv (999,805 rows)
