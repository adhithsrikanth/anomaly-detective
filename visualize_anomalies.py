"""
Standalone script to visualize time-series anomaly detection.
Run this to see the results without the Streamlit dashboard.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ts_data_loader import load_time_series_data
from anomaly_detector import detect_anomalies_in_time_series
from ts_visualizer import create_time_series_plot, create_z_score_plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    # Configuration
    data_file = 'data/sample_ts_data.csv'
    timestamp_col = 'timestamp'
    value_col = 'value'
    window_size = 20
    threshold = 2.5
    
    print("Loading time-series data...")
    df = load_time_series_data(data_file, timestamp_col, value_col)
    print(f"✓ Loaded {len(df)} data points")
    
    print(f"\nDetecting anomalies (window={window_size}, threshold={threshold})...")
    df_analyzed = detect_anomalies_in_time_series(
        df,
        value_column=value_col,
        window_size=window_size,
        threshold=threshold
    )
    
    n_anomalies = df_analyzed['is_anomaly'].sum()
    print(f"✓ Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Main time-series plot
    fig_main = create_time_series_plot(
        df_analyzed,
        timestamp_column=timestamp_col,
        value_column=value_col,
        show_bounds=True,
        show_rolling_mean=True
    )
    
    # Z-score plot
    fig_zscore = create_z_score_plot(
        df_analyzed,
        timestamp_column=timestamp_col,
        threshold=threshold
    )
    
    # Show plots
    print("\n" + "="*60)
    print("Opening visualizations in your browser...")
    print("="*60)
    print("\nMain Time-Series Plot:")
    fig_main.show()
    
    print("\nZ-Score Analysis Plot:")
    fig_zscore.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total data points: {len(df_analyzed)}")
    print(f"Anomalies detected: {n_anomalies}")
    print(f"Anomaly rate: {n_anomalies/len(df)*100:.2f}%")
    print(f"Window size: {window_size}")
    print(f"Threshold: {threshold}σ")
    print(f"\nValue range: {df_analyzed[value_col].min():.2f} to {df_analyzed[value_col].max():.2f}")
    print(f"Mean value: {df_analyzed[value_col].mean():.2f}")
    print(f"Std deviation: {df_analyzed[value_col].std():.2f}")
    
    if n_anomalies > 0:
        anomaly_df = df_analyzed[df_analyzed['is_anomaly']]
        print(f"\nAnomaly z-scores range: {anomaly_df['z_score'].abs().min():.2f} to {anomaly_df['z_score'].abs().max():.2f}")
        print(f"\nTop 5 anomalies by z-score:")
        # Sort by absolute z-score
        anomaly_df_sorted = anomaly_df.copy()
        anomaly_df_sorted['abs_z_score'] = anomaly_df_sorted['z_score'].abs()
        top_anomalies = anomaly_df_sorted.nlargest(5, 'abs_z_score')[[timestamp_col, value_col, 'z_score']]
        for idx, row in top_anomalies.iterrows():
            print(f"  {row[timestamp_col]}: value={row[value_col]:.2f}, z-score={row['z_score']:.2f}")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()

