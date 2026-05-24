"""
EDA utility functions for ACIS Insurance Risk Analytics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_ratio_by_column(df, column, title=None, figsize=(12, 6)):
    """
    Plot loss ratio grouped by a categorical column.
    
    Args:
        df: DataFrame with TotalPremium, TotalClaims columns
        column: column to group by
        title: chart title
        figsize: figure size
    """
    try:
        df_valid = df[df['TotalPremium'] > 0].copy()
        
        stats = df_valid.groupby(column).agg(
            TotalPremium=('TotalPremium', 'sum'),
            TotalClaims=('TotalClaims', 'sum'),
            Count=('PolicyID', 'count')
        )
        stats['LossRatio'] = stats['TotalClaims'] / stats['TotalPremium']
        stats = stats.sort_values('LossRatio', ascending=False)

        fig, ax = plt.subplots(figsize=figsize)
        colors = ['#e74c3c' if x > 1 else '#2ecc71'
                  for x in stats['LossRatio']]
        bars = ax.barh(stats.index, stats['LossRatio'],
                       color=colors, edgecolor='black')
        ax.axvline(x=1, color='black', linestyle='--',
                   linewidth=2, label='Break-even (1.0)')
        ax.set_title(title or f'Loss Ratio by {column}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Loss Ratio')
        ax.legend()

        for bar, val in zip(bars, stats['LossRatio']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10)

        plt.tight_layout()
        return fig, stats
    except Exception as e:
        print(f"ERROR plotting: {e}")
        return None, None


def plot_outliers(df, columns, figsize=(16, 6)):
    """
    Plot box plots to detect outliers in numerical columns.
    
    Args:
        df: DataFrame
        columns: list of column names to plot
        figsize: figure size
    """
    try:
        fig, axes = plt.subplots(1, len(columns), figsize=figsize)
        if len(columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, columns):
            df_clean = df[df[col] > 0][col]
            q99 = df_clean.quantile(0.99)
            df_clean = df_clean[df_clean <= q99]

            ax.boxplot(df_clean, vert=True)
            ax.set_title(f'{col}\n(99th percentile)',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Amount (Rand)')

        plt.suptitle('Outlier Detection — Box Plots',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR plotting outliers: {e}")
        return None


def plot_correlation_matrix(df, columns, figsize=(10, 8)):
    """
    Plot correlation matrix for numerical columns.
    
    Args:
        df: DataFrame
        columns: list of numerical column names
        figsize: figure size
    """
    try:
        corr_data = df[columns].dropna()
        corr_matrix = corr_data.corr()

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            linewidths=0.5
        )
        ax.set_title('Correlation Matrix',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR plotting correlation: {e}")
        return None
