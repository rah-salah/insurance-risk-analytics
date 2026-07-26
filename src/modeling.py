"""
Modeling module for ACIS risk-based insurance pricing.

Extracted and refactored from notebooks/03_modeling.ipynb into reusable,
type-hinted, unit-testable functions. Trains and compares three claim
severity models (Linear Regression, Random Forest, XGBoost), computes
SHAP explainability, and applies a risk-based pricing formula.

Note on the pricing formula: this refactor fixes a bug present in the
original notebook, where expense_loading and profit_margin were applied
directly to predicted_severity instead of to the risk premium
(p_claim * predicted_severity). The corrected, actuarially standard form
loads expenses and profit as percentages ON TOP of the risk premium:
    risk_premium = p_claim * predicted_severity
    optimal_premium = risk_premium * (1 + expense_loading + profit_margin)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Named constants (replacing magic numbers scattered through the original notebook)
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_EXPENSE_LOADING: float = 0.10
DEFAULT_PROFIT_MARGIN: float = 0.15
DEFAULT_SHAP_SAMPLE_SIZE: int = 200

NUMERIC_FEATURES: List[str] = [
    "TotalPremium", "SumInsured", "CalculatedPremiumPerTerm",
    "kilowatts", "cubiccapacity", "NumberOfDoors",
]
CATEGORICAL_FEATURES: List[str] = ["Province", "VehicleType", "Gender", "MaritalStatus", "make"]
TARGET_COLUMN: str = "TotalClaims"


@dataclass
class ModelConfig:
    """Configuration for the claim severity modeling pipeline.

    Grouping these as a dataclass (rather than loose module-level variables
    or function arguments) makes every hyperparameter explicit, type-checked,
    and easy to vary in tests or experiments without touching pipeline code.
    """

    numeric_features: List[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    categorical_features: List[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    target: str = TARGET_COLUMN
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE

    rf_n_estimators: int = 100
    rf_max_depth: int = 10

    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    expense_loading: float = DEFAULT_EXPENSE_LOADING
    profit_margin: float = DEFAULT_PROFIT_MARGIN
    shap_sample_size: int = DEFAULT_SHAP_SAMPLE_SIZE

    @property
    def features(self) -> List[str]:
        return self.numeric_features + self.categorical_features


@dataclass
class ModelMetrics:
    """Evaluation metrics for a single trained model."""

    name: str
    rmse: float
    mae: float
    r2: float


def filter_claims(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Keep only policies with a nonzero claim, drop rows missing required features.

    Severity models predict the claim amount GIVEN a claim occurred, so
    zero-claim policies (the majority of rows) are excluded here; claim
    frequency (P(claim)) is estimated separately in calculate_p_claim().
    """
    if config.target not in df.columns:
        raise ValueError(f"Target column '{config.target}' not found in dataframe.")

    missing_features = [c for c in config.features if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature column(s): {missing_features}")

    df_claims = df[df[config.target] > 0].copy()
    df_model = df_claims[config.features + [config.target]].dropna()
    return df_model


def encode_categoricals(
    df: pd.DataFrame, categorical_features: List[str]
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns, returning the fitted encoders.

    Returning the per-column encoders (unlike the original notebook, which
    reused one LabelEncoder instance without keeping the fitted mappings)
    lets downstream code - e.g. a dashboard's live prediction tool - encode
    new categorical input consistently with what the model was trained on.
    """
    df_encoded = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_features:
        if col not in df_encoded.columns:
            raise ValueError(f"Categorical column '{col}' not found in dataframe.")
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        encoders[col] = encoder
    return df_encoded, encoders


def split_data(
    df: pd.DataFrame, config: ModelConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronologically-agnostic random train/test split for the severity model."""
    X = df[config.features]
    y = df[config.target]
    return train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        random_state=config.random_state,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> ModelMetrics:
    """Compute RMSE, MAE, and R2 for a fitted model on held-out test data."""
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return ModelMetrics(name=name, rmse=rmse, mae=mae, r2=r2)


def compare_models(metrics: List[ModelMetrics]) -> pd.DataFrame:
    """Return a comparison table of model metrics, best (lowest MAE) first."""
    df = pd.DataFrame([m.__dict__ for m in metrics])
    return df.sort_values("mae").reset_index(drop=True)


def get_feature_importance(model: RandomForestRegressor, features: List[str]) -> pd.DataFrame:
    """Feature importances from a fitted tree-based model, most important first."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model of type {type(model).__name__} has no feature_importances_.")
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    })
    return importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)


def calculate_p_claim(df_full: pd.DataFrame, df_claims: pd.DataFrame) -> float:
    """Overall claim frequency: share of ALL policies (not just claims) that had a claim."""
    if len(df_full) == 0:
        raise ValueError("df_full is empty; cannot compute claim frequency.")
    return len(df_claims) / len(df_full)


def calculate_risk_based_premium(
    p_claim: float, predicted_severity: np.ndarray, config: ModelConfig
) -> np.ndarray:
    """Risk-based premium: risk premium plus expense and profit loadings.

    risk_premium = P(claim) x predicted_severity
    optimal_premium = risk_premium x (1 + expense_loading + profit_margin)

    This corrects a bug in the original notebook, where expense_loading and
    profit_margin were multiplied directly against predicted_severity rather
    than against the risk premium, understating the effect of P(claim) on
    the final loaded premium.
    """
    risk_premium = p_claim * predicted_severity
    loading_factor = 1.0 + config.expense_loading + config.profit_margin
    return risk_premium * loading_factor


def compute_shap_values(model, X_sample: pd.DataFrame):
    """Compute SHAP values for a tree-based model using TreeExplainer."""
    import shap

    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_sample)


def run_full_pipeline(df_raw: pd.DataFrame, config: ModelConfig | None = None) -> dict:
    """End-to-end pipeline: filter, encode, split, train all 3 models, evaluate, price.

    Returns a dict with trained models, metrics, the comparison table,
    feature importances, fitted encoders, and pricing outputs - everything
    a dashboard or report needs, without re-running notebook cells.
    """
    config = config or ModelConfig()

    df_claims = filter_claims(df_raw, config)
    df_encoded, encoders = encode_categoricals(df_claims, config.categorical_features)
    X_train, X_test, y_train, y_test = split_data(df_encoded, config)

    lr = train_linear_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train, config)
    xgb_model = train_xgboost(X_train, y_train, config)

    metrics = [
        evaluate_model(lr, X_test, y_test, "Linear Regression"),
        evaluate_model(rf, X_test, y_test, "Random Forest"),
        evaluate_model(xgb_model, X_test, y_test, "XGBoost"),
    ]
    comparison = compare_models(metrics)
    best_model_name = comparison.iloc[0]["name"]
    models = {"Linear Regression": lr, "Random Forest": rf, "XGBoost": xgb_model}
    best_model = models[best_model_name]

    feature_importance = (
        get_feature_importance(best_model, config.features)
        if hasattr(best_model, "feature_importances_")
        else get_feature_importance(rf, config.features)
    )

    p_claim = calculate_p_claim(df_raw, df_claims)
    predicted_severity = best_model.predict(X_test) if hasattr(best_model, "predict") else rf.predict(X_test)
    optimal_premium = calculate_risk_based_premium(p_claim, predicted_severity, config)

    return {
        "models": models,
        "best_model_name": best_model_name,
        "metrics": metrics,
        "comparison": comparison,
        "feature_importance": feature_importance,
        "encoders": encoders,
        "p_claim": p_claim,
        "optimal_premium": optimal_premium,
        "X_test": X_test,
        "y_test": y_test,
        "config": config,
    }
