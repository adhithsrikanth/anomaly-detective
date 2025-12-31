"""
Anomaly detection module using rolling statistics and z-scores.
Defines normal behavior using rolling mean and standard deviation,
then flags anomalies when z-scores exceed a threshold.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_rolling_statistics(
    df: pd.DataFrame,
    value_column: str,
    window_size: int
) -> pd.DataFrame:
    """
    Compute rolling mean and standard deviation for the time-series.
    
    Args:
        df: DataFrame with time-series data
        value_column: Name of the numeric value column
        window_size: Size of the rolling window
        
    Returns:
        DataFrame with added 'rolling_mean' and 'rolling_std' columns
    """
    df = df.copy()
    
    # Ensure window size is valid
    if window_size < 2:
        raise ValueError("Window size must be at least 2")
    
    if window_size > len(df):
        window_size = len(df)
        print(f"Warning: Window size adjusted to {window_size} (data length)")
    
    # Compute rolling statistics
    df['rolling_mean'] = df[value_column].rolling(window=window_size, min_periods=1).mean()
    df['rolling_std'] = df[value_column].rolling(window=window_size, min_periods=1).std()
    
    # Handle edge case: if std is 0 (constant values), set to small epsilon
    df['rolling_std'] = df['rolling_std'].replace(0, np.finfo(float).eps)
    
    return df


def compute_z_scores(
    df: pd.DataFrame,
    value_column: str
) -> pd.DataFrame:
    """
    Compute z-scores for each data point.
    z = (value - rolling_mean) / rolling_std
    
    Args:
        df: DataFrame with rolling statistics already computed
        value_column: Name of the numeric value column
        
    Returns:
        DataFrame with added 'z_score' column
    """
    df = df.copy()
    
    # Compute z-scores
    df['z_score'] = (df[value_column] - df['rolling_mean']) / df['rolling_std']
    
    return df


def detect_anomalies(
    df: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    """
    Flag anomalies based on z-score threshold.
    Anomaly is flagged when |z_score| > threshold.
    
    Args:
        df: DataFrame with z-scores already computed
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with added 'is_anomaly' boolean column
    """
    df = df.copy()
    
    # Flag anomalies
    df['is_anomaly'] = np.abs(df['z_score']) > threshold
    
    return df


def compute_anomaly_bounds(
    df: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    """
    Compute upper and lower bounds for visualization.
    bounds = rolling_mean ± (threshold * rolling_std)
    
    Args:
        df: DataFrame with rolling statistics already computed
        threshold: Z-score threshold for bounds
        
    Returns:
        DataFrame with added 'upper_bound' and 'lower_bound' columns
    """
    df = df.copy()
    
    df['upper_bound'] = df['rolling_mean'] + (threshold * df['rolling_std'])
    df['lower_bound'] = df['rolling_mean'] - (threshold * df['rolling_std'])
    
    return df


def detect_anomalies_in_time_series(
    df: pd.DataFrame,
    value_column: str,
    window_size: int,
    threshold: float
) -> pd.DataFrame:
    """
    Complete anomaly detection pipeline:
    1. Compute rolling statistics
    2. Compute z-scores
    3. Flag anomalies
    4. Compute bounds for visualization
    
    Args:
        df: DataFrame with time-series data
        value_column: Name of the numeric value column
        window_size: Size of the rolling window
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with all computed statistics and anomaly flags
    """
    # Step 1: Compute rolling statistics
    df = compute_rolling_statistics(df, value_column, window_size)
    
    # Step 2: Compute z-scores
    df = compute_z_scores(df, value_column)
    
    # Step 3: Flag anomalies
    df = detect_anomalies(df, threshold)
    
    # Step 4: Compute bounds
    df = compute_anomaly_bounds(df, threshold)
    
    return df

