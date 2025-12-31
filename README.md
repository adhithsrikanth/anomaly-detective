# Time-Series Anomaly Detection

A statistical approach to identifying and visualizing unusual behavior in time-series data using rolling statistics and z-scores.

## Overview

This project provides a clean, interpretable method for detecting anomalies in time-series data without requiring machine learning models. It uses statistical methods to define "normal" behavior and flags deviations that exceed a configurable threshold.

## Features

- **CSV Data Loading**: Load time-series data from CSV files with automatic timestamp parsing
- **Missing Value Handling**: Gracefully handles missing values using forward/backward fill
- **Rolling Statistics**: Computes adaptive rolling mean and standard deviation
- **Z-Score Anomaly Detection**: Flags anomalies when z-scores exceed a threshold
- **Interactive Visualization**: 
  - Time-series plot with anomaly highlighting
  - Z-score analysis plot
  - Configurable display options (bounds, rolling mean)
- **Parameter Tuning**: Adjustable rolling window size and anomaly threshold
- **Export Results**: Download detected anomalies and full analysis results

## Data Format

Your CSV file should contain:
- **Timestamp column**: Dates/times in a format pandas can parse (e.g., "2024-01-01 00:00:00")
- **Numeric value column**: Integer or float values

## Sample Dataset

A sample dataset is available for testing. Download it here:

**[Download sample_ts_data.csv](https://github.com/adhithsrikanth/anomaly-detective/raw/main/data/sample_ts_data.csv)**

This sample dataset contains 500 data points with hourly timestamps and numeric values. The data includes some injected anomalies that can be detected using the tool. The format is:

```csv
timestamp,value
2024-01-01 00:00:00,50.99
2024-01-01 01:00:00,51.03
...
```

## How It Works

### What is Anomaly Detection?

Anomaly detection identifies unusual or unexpected patterns in data that deviate significantly from normal behavior. In time-series data, anomalies might represent:
- Equipment failures
- System errors
- Unusual events
- Data quality issues

### How Normal Behavior is Defined

This tool uses **rolling statistics** to define normal behavior:

1. **Rolling Mean**: The average value over a sliding window of recent data points
   - Example: With a window size of 20, the rolling mean at time `t` is the average of points `t-19` through `t`
   - This adapts to trends and gradual changes in the data

2. **Rolling Standard Deviation**: The variability over the same window
   - Measures how much values typically deviate from the rolling mean
   - Higher standard deviation = more variability in normal behavior

These statistics are computed for each point in the time series, creating an adaptive baseline that changes over time.

### How Anomalies are Identified

Anomalies are detected using **z-scores**:

```
z-score = (value - rolling_mean) / rolling_std
```

A z-score measures how many standard deviations a value is from the rolling mean:
- **z-score ≈ 0**: Value is close to the rolling mean (normal)
- **|z-score| > threshold**: Value is unusually far from the mean (anomaly)

**Threshold Selection**:
- **Lower threshold (e.g., 2.0)**: More sensitive, flags more anomalies
  - In a normal distribution, ~5% of values would have |z| > 2.0
- **Higher threshold (e.g., 3.0)**: Less sensitive, only extreme anomalies
  - In a normal distribution, ~0.3% of values would have |z| > 3.0

### Example

Consider a time series with values around 50:
- Rolling mean = 50, Rolling std = 2
- A value of 55 has z-score = (55 - 50) / 2 = 2.5
- With threshold = 2.5, this is flagged as an anomaly
- With threshold = 3.0, this is considered normal

## Project Structure

```
ts_anomaly_detection/
├── src/
│   ├── __init__.py
│   ├── ts_data_loader.py      # CSV loading and preprocessing
│   ├── anomaly_detector.py    # Rolling stats and z-score computation
│   └── ts_visualizer.py       # Plotly visualizations
├── data/
│   └── sample_ts_data.csv     # Sample data (generated)
├── outputs/                   # Output directory
├── anomaly_dashboard.py       # Main Streamlit app
├── create_ts_sample_data.py   # Sample data generator
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Parameters

### Rolling Window Size
- **Smaller windows (e.g., 5-10)**: 
  - More responsive to recent changes
  - Better for detecting quick shifts
  - May be more sensitive to noise
  
- **Larger windows (e.g., 30-50)**: 
  - More stable baseline
  - Better for data with trends
  - Less sensitive to temporary fluctuations

### Anomaly Threshold
- **2.0 - 2.5**: Standard threshold, flags moderate deviations
- **3.0 - 3.5**: Conservative threshold, only extreme anomalies
- **1.5 - 2.0**: Sensitive threshold, flags many potential issues

## Limitations

- Assumes data is approximately normally distributed within rolling windows
- May miss gradual drift anomalies if they occur slowly relative to window size
- Requires sufficient data points (at least window_size + a few more)
- Not suitable for highly non-stationary data without preprocessing

## Future Enhancements

Potential improvements:
- Multiple detection methods (IQR, isolation forest)
- Seasonal adjustment
- Multi-variate anomaly detection
- Real-time streaming detection
- Custom anomaly scoring functions

## License

This project is provided as-is for educational and practical use.

