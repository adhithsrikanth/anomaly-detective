"""
Visualization module for time-series anomaly detection.
Creates line charts with anomaly highlighting and optional statistical bounds.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def create_time_series_plot(
    df: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
    show_bounds: bool = True,
    show_rolling_mean: bool = True
) -> go.Figure:
    """
    Create an interactive time-series plot with anomaly highlighting.
    
    Args:
        df: DataFrame with time-series data and anomaly flags
        timestamp_column: Name of the timestamp column
        value_column: Name of the numeric value column
        show_bounds: Whether to show upper/lower bounds
        show_rolling_mean: Whether to show rolling mean line
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Separate normal and anomaly points
    normal_df = df[~df['is_anomaly']]
    anomaly_df = df[df['is_anomaly']]
    
    # Plot normal points
    fig.add_trace(go.Scatter(
        x=normal_df[timestamp_column],
        y=normal_df[value_column],
        mode='lines+markers',
        name='Normal',
        line=dict(color='steelblue', width=2),
        marker=dict(size=4, color='steelblue'),
        hovertemplate='<b>Normal</b><br>' +
                      'Time: %{x}<br>' +
                      'Value: %{y:.2f}<br>' +
                      'Z-score: %{customdata:.2f}<extra></extra>',
        customdata=normal_df['z_score']
    ))
    
    # Plot anomalies with different color and larger markers
    if len(anomaly_df) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_df[timestamp_column],
            y=anomaly_df[value_column],
            mode='markers',
            name='Anomaly',
            marker=dict(
                size=10,
                color='red',
                symbol='x',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='<b>ANOMALY</b><br>' +
                          'Time: %{x}<br>' +
                          'Value: %{y:.2f}<br>' +
                          'Z-score: %{customdata:.2f}<extra></extra>',
            customdata=anomaly_df['z_score']
        ))
    
    # Plot rolling mean if requested
    if show_rolling_mean:
        fig.add_trace(go.Scatter(
            x=df[timestamp_column],
            y=df['rolling_mean'],
            mode='lines',
            name='Rolling Mean',
            line=dict(color='green', width=2, dash='dash'),
            hovertemplate='Rolling Mean: %{y:.2f}<extra></extra>'
        ))
    
    # Plot bounds if requested
    if show_bounds:
        fig.add_trace(go.Scatter(
            x=df[timestamp_column],
            y=df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='orange', width=1, dash='dot'),
            hovertemplate='Upper Bound: %{y:.2f}<extra></extra>',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=df[timestamp_column],
            y=df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='orange', width=1, dash='dot'),
            hovertemplate='Lower Bound: %{y:.2f}<extra></extra>',
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.1)'
        ))
    
    # Update layout
    fig.update_layout(
        title='Time-Series Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_z_score_plot(
    df: pd.DataFrame,
    timestamp_column: str,
    threshold: float
) -> go.Figure:
    """
    Create a plot showing z-scores over time with threshold lines.
    
    Args:
        df: DataFrame with z-scores
        timestamp_column: Name of the timestamp column
        threshold: Z-score threshold
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Separate normal and anomaly z-scores
    normal_df = df[~df['is_anomaly']]
    anomaly_df = df[df['is_anomaly']]
    
    # Plot normal z-scores
    fig.add_trace(go.Scatter(
        x=normal_df[timestamp_column],
        y=normal_df['z_score'],
        mode='lines+markers',
        name='Z-score (Normal)',
        line=dict(color='steelblue', width=2),
        marker=dict(size=4, color='steelblue'),
        hovertemplate='Time: %{x}<br>Z-score: %{y:.2f}<extra></extra>'
    ))
    
    # Plot anomaly z-scores
    if len(anomaly_df) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_df[timestamp_column],
            y=anomaly_df['z_score'],
            mode='markers',
            name='Z-score (Anomaly)',
            marker=dict(
                size=10,
                color='red',
                symbol='x',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='<b>ANOMALY</b><br>Time: %{x}<br>Z-score: %{y:.2f}<extra></extra>'
        ))
    
    # Add threshold lines
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: +{threshold}",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=-threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: -{threshold}",
        annotation_position="right"
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        opacity=0.5
    )
    
    # Update layout
    fig.update_layout(
        title='Z-Scores Over Time',
        xaxis_title='Time',
        yaxis_title='Z-Score',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

