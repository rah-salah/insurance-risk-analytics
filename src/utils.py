"""
Utility functions for insurance risk analytics pipeline.
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """Load the insurance dataset from a pipe-separated file."""
    try:
        df = pd.read_csv(filepath, sep='|', low_memory=False)
        print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return pd.DataFrame()


def calculate_loss_ratio(df, group_col):
    """Calculate loss ratio grouped by a column."""
    df_valid = df[df['TotalPremium'] > 0].copy()
    stats = df_valid.groupby(group_col).agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum'),
        PolicyCount=('PolicyID', 'count')
    ).round(2)
    stats['LossRatio'] = (stats['TotalClaims'] / stats['TotalPremium']).round(4)
    return stats.sort_values('LossRatio', ascending=False)


def get_risk_label(loss_ratio):
    """Label a loss ratio as high, medium or low risk."""
    if loss_ratio > 1.0:
        return 'High Risk'
    elif loss_ratio > 0.7:
        return 'Medium Risk'
    else:
        return 'Low Risk'
