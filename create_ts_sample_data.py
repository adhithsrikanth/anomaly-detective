"""
Generate sample time-series data with some anomalies for testing.
Creates a CSV file with timestamp and value columns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def create_sample_time_series(
    n_points: int = 500,
    start_date: str = "2024-01-01",
    output_path: str = "data/sample_ts_data.csv",
    include_anomalies: bool = True
) -> pd.DataFrame:
    """
    Create sample time-series data with optional anomalies.
    
    Args:
        n_points: Number of data points to generate
        start_date: Start date for the time series
        output_path: Path to save the CSV file
        include_anomalies: Whether to inject some anomalies
        
    Returns:
        DataFrame with timestamp and value columns
    """
    np.random.seed(42)
    
    # Create timestamp range
    start = pd.to_datetime(start_date)
    timestamps = [start + timedelta(hours=i) for i in range(n_points)]
    
    # Generate base time series with trend and seasonality
    t = np.arange(n_points)
    
    # Trend component
    trend = 0.01 * t
    
    # Seasonal component (daily pattern)
    seasonal = 5 * np.sin(2 * np.pi * t / 24)  # 24-hour cycle
    
    # Random noise
    noise = np.random.normal(0, 2, n_points)
    
    # Base values
    values = 50 + trend + seasonal + noise
    
    # Inject anomalies if requested
    if include_anomalies:
        # Random spike anomalies
        n_anomalies = n_points // 50  # ~2% anomalies
        anomaly_indices = np.random.choice(n_points, size=n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Randomly choose spike up or down
            if np.random.random() > 0.5:
                values[idx] += np.random.uniform(15, 30)  # Spike up
            else:
                values[idx] -= np.random.uniform(15, 30)  # Spike down
        
        # Add a few gradual drift anomalies
        drift_indices = np.random.choice(n_points, size=n_points // 100, replace=False)
        for idx in drift_indices:
            # Create a small drift
            drift_length = min(10, n_points - idx)
            drift = np.linspace(0, np.random.uniform(10, 20), drift_length)
            values[idx:idx+drift_length] += drift
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Sample time-series data created: {output_path}")
    print(f"  - Data points: {n_points}")
    print(f"  - Date range: {timestamps[0]} to {timestamps[-1]}")
    print(f"  - Value range: {values.min():.2f} to {values.max():.2f}")
    if include_anomalies:
        print(f"  - Anomalies injected: ~{n_anomalies} spike anomalies")
    
    return df


if __name__ == '__main__':
    create_sample_time_series()

