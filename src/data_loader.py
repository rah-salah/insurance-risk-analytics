"""
Data loading utilities for ACIS Insurance Risk Analytics.
"""

import pandas as pd
import os


def load_insurance_data(filepath, nrows=None):
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
        print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return pd.DataFrame()


def get_data_summary(df):
    """
    Get a quick summary of the dataframe.
    
    Returns:
        dict with shape, missing values, and dtypes info
    """
    try:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        summary = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_columns': missing[missing > 0].count(),
            'total_missing': missing.sum(),
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        return summary
    except Exception as e:
        print(f"ERROR getting summary: {e}")
        return {}


def clean_financial_columns(df):
    """
    Clean TotalPremium and TotalClaims columns.
    Removes negative values and calculates LossRatio.
    
    Returns:
        cleaned DataFrame
    """
    try:
        df = df.copy()
        
        # Remove negative premiums and claims
        df = df[df['TotalPremium'] >= 0]
        df = df[df['TotalClaims'] >= 0]
        
        # Calculate loss ratio where premium > 0
        df['LossRatio'] = 0.0
        mask = df['TotalPremium'] > 0
        df.loc[mask, 'LossRatio'] = (
            df.loc[mask, 'TotalClaims'] / df.loc[mask, 'TotalPremium']
        )
        
        # Calculate margin
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        
        print(f"After cleaning: {len(df):,} rows remain")
        return df
    except Exception as e:
        print(f"ERROR cleaning data: {e}")
        return df
