"""
Time-series data loading and preprocessing module.
Handles CSV loading, sorting by timestamp, and missing value handling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_time_series_data(
    file_path: str,
    timestamp_column: str,
    value_column: str
) -> pd.DataFrame:
    """
    Load time-series data from a CSV file.
    
    Args:
        file_path: Path to the CSV file or file-like object
        timestamp_column: Name of the timestamp column
        value_column: Name of the numeric value column
        
    Returns:
        DataFrame with sorted time-series data
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    try:
        # Handle both file paths (strings) and file-like objects (e.g., Streamlit uploads)
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Validate required columns exist
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in CSV. "
                        f"Available columns: {list(df.columns)}")
    
    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found in CSV. "
                        f"Available columns: {list(df.columns)}")
    
    # Select only required columns
    df = df[[timestamp_column, value_column]].copy()
    
    # Convert timestamp to datetime
    try:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    except Exception as e:
        raise ValueError(f"Error parsing timestamp column '{timestamp_column}': {str(e)}")
    
    # Validate value column is numeric
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        try:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        except Exception:
            raise ValueError(f"Value column '{value_column}' cannot be converted to numeric")
    
    # Sort by timestamp
    df = df.sort_values(by=timestamp_column).reset_index(drop=True)
    
    # Handle missing values gracefully
    # Forward fill for missing values, then backward fill if needed
    initial_missing = df[value_column].isna().sum()
    df[value_column] = df[value_column].ffill().bfill()
    
    # If still missing (all NaN), fill with 0 as last resort
    df[value_column] = df[value_column].fillna(0)
    
    if initial_missing > 0:
        print(f"Warning: {initial_missing} missing values were handled using forward/backward fill")
    
    return df


def validate_time_series_data(df: pd.DataFrame, timestamp_column: str, value_column: str) -> bool:
    """
    Validate that the DataFrame has valid time-series structure.
    
    Args:
        df: DataFrame to validate
        timestamp_column: Name of the timestamp column
        value_column: Name of the value column
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < 2:
        raise ValueError("Time-series data must have at least 2 data points")
    
    if df[value_column].isna().all():
        raise ValueError("All values in the value column are missing")
    
    return True

