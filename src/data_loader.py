"""
Data loading utilities for ACIS Insurance Risk Analytics.
"""

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_insurance_data(filepath: str, nrows: int = None) -> pd.DataFrame:
    """
    Load the insurance dataset from a pipe-separated text file.

    Args:
        filepath: path to the MachineLearningRating_v3.txt file
        nrows: number of rows to load (None = all rows)

    Returns:
        pandas DataFrame with insurance data
    """
    try:
        df = pd.read_csv(
            filepath,
            sep='|',
            low_memory=False,
            nrows=nrows
        )
        logger.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get a quick summary of the dataframe.

    Args:
        df: input DataFrame

    Returns:
        dict with shape, missing values, and dtypes info
    """
    try:
        missing = df.isnull().sum()
        summary = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_columns': int(missing[missing > 0].count()),
            'total_missing': int(missing.sum()),
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        logger.info(f"Summary: {summary['rows']:,} rows, {summary['columns']} columns")
        return summary
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return {}


def clean_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean TotalPremium and TotalClaims columns.
    Removes negative values and calculates LossRatio and Margin.

    Args:
        df: raw insurance DataFrame

    Returns:
        cleaned DataFrame with LossRatio and Margin columns added
    """
    try:
        df = df.copy()

        before = len(df)
        df = df[df['TotalPremium'] >= 0]
        df = df[df['TotalClaims'] >= 0]
        removed = before - len(df)
        logger.info(f"Removed {removed:,} rows with negative values")

        df['LossRatio'] = 0.0
        mask = df['TotalPremium'] > 0
        df.loc[mask, 'LossRatio'] = (
            df.loc[mask, 'TotalClaims'] / df.loc[mask, 'TotalPremium']
        )

        df['Margin'] = df['TotalPremium'] - df['TotalClaims']

        logger.info(f"After cleaning: {len(df):,} rows remain")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        return df
