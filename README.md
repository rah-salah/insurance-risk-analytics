# Insurance Risk Analytics
## 10 Academy KAIM 9 - Week 3 | AlphaCare Insurance Solutions

Analyzing 1 million South African car insurance policy transactions
to identify low-risk customers and build a risk-based pricing model.

## Business Question
Which customers are low-risk and how should we price their insurance accordingly?

## Key Findings
- **Overall Loss Ratio: 1.048** — ACIS is paying MORE in claims than collecting
- **Gauteng** is the highest risk province (loss ratio: 1.163)
- **Northern Cape** is the lowest risk province (loss ratio: 0.283)
- **Heavy Commercial** vehicles are the riskiest (loss ratio: 1.612)
- **3 provinces** are losing money: Gauteng, Western Cape, KwaZulu-Natal

## Project Structure
insurance-risk-analytics/
├── .github/workflows/unittests.yml
├── data/
│   └── raw/
│       └── MachineLearningRating_v3.txt.dvc
├── notebooks/
│   └── eda.ipynb
├── src/
│   └── utils.py
├── tests/
│   └── test_placeholder.py
├── .dvc/
├── .gitignore
├── requirements.txt
└── README.md
~## Setup

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

### 4. Run tests
```bash
pytest tests/ -v
```

## Tasks
- **Task 1** - Exploratory Data Analysis (EDA)
- **Task 2** - Data Version Control (DVC)
- **Task 3** - A/B Hypothesis Testing
- **Task 4** - Machine Learning Models

## Data Source
- 1,000,098 insurance policy transactions
- Period: October 2013 to August 2015
- South Africa, 9 provinces
- 52 columns including TotalPremium and TotalClaims
