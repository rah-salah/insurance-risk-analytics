"""Unit tests for src/modeling.py"""

import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    ModelConfig,
    ModelMetrics,
    calculate_p_claim,
    calculate_risk_based_premium,
    compare_models,
    encode_categoricals,
    filter_claims,
    get_feature_importance,
    run_full_pipeline,
    split_data,
    train_random_forest,
)


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig()


@pytest.fixture
def raw_df() -> pd.DataFrame:
    np.random.seed(0)
    n = 300
    return pd.DataFrame({
        "TotalPremium": np.random.uniform(200, 2000, n),
        "SumInsured": np.random.uniform(50000, 500000, n),
        "CalculatedPremiumPerTerm": np.random.uniform(200, 2000, n),
        "kilowatts": np.random.uniform(50, 200, n),
        "cubiccapacity": np.random.uniform(1000, 3000, n),
        "NumberOfDoors": np.random.choice([2, 4, 5], n),
        "Province": np.random.choice(["Gauteng", "Western Cape", "KwaZulu-Natal"], n),
        "VehicleType": np.random.choice(["Sedan", "SUV", "Hatchback"], n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "MaritalStatus": np.random.choice(["Married", "Single"], n),
        "make": np.random.choice(["Toyota", "VW", "Ford"], n),
        "TotalClaims": np.where(
            np.random.rand(n) < 0.3, np.random.uniform(1000, 50000, n), 0
        ),
    })


def test_filter_claims_keeps_only_nonzero_claims(raw_df, config):
    result = filter_claims(raw_df, config)
    assert (result[config.target] > 0).all()


def test_filter_claims_raises_on_missing_target(config):
    df = pd.DataFrame({"TotalPremium": [1, 2]})
    with pytest.raises(ValueError, match="Target column"):
        filter_claims(df, config)


def test_filter_claims_raises_on_missing_features(config):
    df = pd.DataFrame({"TotalClaims": [100, 200]})
    with pytest.raises(ValueError, match="Missing expected feature"):
        filter_claims(df, config)


def test_encode_categoricals_produces_numeric_columns(raw_df, config):
    df_claims = filter_claims(raw_df, config)
    df_encoded, encoders = encode_categoricals(df_claims, config.categorical_features)
    for col in config.categorical_features:
        assert pd.api.types.is_numeric_dtype(df_encoded[col])
        assert col in encoders


def test_encode_categoricals_raises_on_missing_column(config):
    df = pd.DataFrame({"SomeOtherColumn": [1, 2, 3]})
    with pytest.raises(ValueError, match="not found"):
        encode_categoricals(df, config.categorical_features)


def test_split_data_respects_test_size(raw_df, config):
    df_claims = filter_claims(raw_df, config)
    df_encoded, _ = encode_categoricals(df_claims, config.categorical_features)
    X_train, X_test, y_train, y_test = split_data(df_encoded, config)
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - config.test_size) < 0.05


def test_calculate_p_claim_is_fraction_between_0_and_1(raw_df, config):
    df_claims = filter_claims(raw_df, config)
    p = calculate_p_claim(raw_df, df_claims)
    assert 0.0 <= p <= 1.0


def test_calculate_p_claim_raises_on_empty_full_df(config):
    with pytest.raises(ValueError, match="empty"):
        calculate_p_claim(pd.DataFrame(), pd.DataFrame())


def test_calculate_risk_based_premium_applies_loading_correctly(config):
    """The fixed formula: risk_premium * (1 + expense_loading + profit_margin)."""
    p_claim = 0.3
    severity = np.array([1000.0, 2000.0])
    premium = calculate_risk_based_premium(p_claim, severity, config)

    expected_risk_premium = p_claim * severity
    expected_premium = expected_risk_premium * (1 + config.expense_loading + config.profit_margin)
    np.testing.assert_allclose(premium, expected_premium)


def test_calculate_risk_based_premium_higher_severity_gives_higher_premium(config):
    p_claim = 0.3
    low = calculate_risk_based_premium(p_claim, np.array([1000.0]), config)
    high = calculate_risk_based_premium(p_claim, np.array([5000.0]), config)
    assert high[0] > low[0]


def test_compare_models_sorts_by_mae_ascending():
    metrics = [
        ModelMetrics(name="A", rmse=100, mae=80, r2=0.5),
        ModelMetrics(name="B", rmse=90, mae=50, r2=0.6),
        ModelMetrics(name="C", rmse=110, mae=95, r2=0.4),
    ]
    result = compare_models(metrics)
    assert result.iloc[0]["name"] == "B"
    assert result["mae"].is_monotonic_increasing


def test_get_feature_importance_sums_reasonable(raw_df, config):
    df_claims = filter_claims(raw_df, config)
    df_encoded, _ = encode_categoricals(df_claims, config.categorical_features)
    X_train, X_test, y_train, y_test = split_data(df_encoded, config)
    rf = train_random_forest(X_train, y_train, config)
    importance = get_feature_importance(rf, config.features)
    assert len(importance) == len(config.features)
    assert importance["Importance"].sum() == pytest.approx(1.0, abs=0.01)


def test_get_feature_importance_raises_for_model_without_importances(config):
    class DummyModel:
        pass
    with pytest.raises(ValueError, match="feature_importances_"):
        get_feature_importance(DummyModel(), config.features)


def test_run_full_pipeline_end_to_end(raw_df, config):
    result = run_full_pipeline(raw_df, config)
    assert set(result["models"].keys()) == {"Linear Regression", "Random Forest", "XGBoost"}
    assert result["best_model_name"] in result["models"]
    assert len(result["comparison"]) == 3
    assert result["optimal_premium"].shape[0] == len(result["X_test"])
