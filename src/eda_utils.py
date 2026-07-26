"""
EDA utility functions for ACIS Insurance Risk Analytics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import logging
from typing import Optional, Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_loss_ratio_by_column(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Tuple[Optional[matplotlib.figure.Figure], Optional[pd.DataFrame]]:
    """
    Plot loss ratio grouped by a categorical column.

    Args:
        df: DataFrame with TotalPremium and TotalClaims columns
        column: column to group by
        title: chart title
        figsize: figure size tuple

    Returns:
        tuple of (figure, stats DataFrame)
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
        logger.info(f"Loss ratio plot created for column: {column}")
        return fig, stats
    except Exception as e:
        logger.error(f"Error plotting loss ratio: {e}")
        return None, None


def plot_outliers(
    df: pd.DataFrame,
    columns: List[str],
    figsize: Tuple[int, int] = (16, 6)
) -> Optional[matplotlib.figure.Figure]:
    """
    Plot box plots to detect outliers in numerical columns.

    Args:
        df: input DataFrame
        columns: list of column names to plot
        figsize: figure size tuple

    Returns:
        matplotlib Figure object
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

        plt.suptitle('Outlier Detection  -  Box Plots',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        logger.info(f"Outlier plots created for: {columns}")
        return fig
    except Exception as e:
        logger.error(f"Error plotting outliers: {e}")
        return None


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[matplotlib.figure.Figure]:
    """
    Plot correlation matrix for numerical columns.

    Args:
        df: input DataFrame
        columns: list of numerical column names
        figsize: figure size tuple

    Returns:
        matplotlib Figure object
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
        logger.info("Correlation matrix created")
        return fig
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {e}")
        return None
