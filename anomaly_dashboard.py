"""
Streamlit dashboard for time-series anomaly detection.
Modern UI design with card-based layout.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ts_data_loader import load_time_series_data, validate_time_series_data
from anomaly_detector import detect_anomalies_in_time_series
from ts_visualizer import create_time_series_plot, create_z_score_plot


st.set_page_config(
    page_title="Anomaly Detective",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
    <style>
    /* Main background - light green */
    .stApp {
        background: #d4edda;
        background-attachment: fixed;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        box-shadow: none !important;
    }
    
    /* Remove shadows from all Streamlit containers */
    .main .block-container > div,
    .main .block-container > div > div {
        box-shadow: none !important;
    }
    
    /* Header styling - normal text */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: normal;
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(15px);
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(30, 30, 50, 0.95);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #fff;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Slider styling - black */
    .stSlider [data-baseweb="slider-track"] {
        background-color: #000000 !important;
    }
    
    .stSlider [data-baseweb="slider-fill"] {
        background-color: #000000 !important;
    }
    
    .stSlider [data-baseweb="slider-handle"] {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
    }
    
    .stSlider [data-baseweb="slider-tick"] {
        background-color: #000000 !important;
    }
    
    /* File uploader */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Metrics - normal styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666;
        font-weight: normal;
    }
    
    /* Tabs - remove all shadows from every possible element */
    .stTabs,
    .stTabs *,
    .stTabs > div,
    .stTabs > div > div,
    .stTabs > div > div > div,
    .stTabs [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab-list"] > div,
    .stTabs [data-baseweb="tab-list"] > div > div,
    [data-testid="stTabs"],
    [data-testid="stTabs"] > div,
    [data-testid="stTabs"] > div > div,
    div[data-baseweb="tabs"],
    .element-container:has(.stTabs),
    .element-container:has([data-testid="stTabs"]) {
        box-shadow: none !important;
        filter: none !important;
        -webkit-box-shadow: none !important;
        -moz-box-shadow: none !important;
        text-shadow: none !important;
    }
    
    /* Specifically target the tab bar container */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: none !important;
        -webkit-box-shadow: none !important;
        -moz-box-shadow: none !important;
    }
    
    /* Remove shadow from parent containers of tabs */
    div:has([data-baseweb="tab-list"]),
    div:has([data-testid="stTabs"]) {
        box-shadow: none !important;
        -webkit-box-shadow: none !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #333;
        border-radius: 10px;
    }
    
    /* Tab indicator - change from red to black */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #000000 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] [data-baseweb="tab-highlight"] {
        background-color: #000000 !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #000000 !important;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #000000 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* Checkbox styling - transparent background */
    .stCheckbox > label {
        color: #333 !important;
    }
    
    .stCheckbox [data-baseweb="checkbox"] {
        background-color: transparent !important;
        border-color: #000000 !important;
    }
    
    .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] {
        background-color: transparent !important;
    }
    
    .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] [data-baseweb="checkmark"] {
        color: #000000 !important;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Aggressively remove all shadows from tabs and their parents */
    div[class*="stTabs"],
    div[data-baseweb="tabs"],
    div[data-baseweb="tab-list"],
    [data-testid="stTabs"],
    [data-testid="stTabs"] *,
    .element-container:has([data-baseweb="tab-list"]),
    .element-container:has([data-testid="stTabs"]),
    /* Target the immediate wrapper around tabs */
    .stTabsContainer,
    div[class*="element-container"]:has(.stTabs),
    /* Remove shadow from any div containing tabs */
    div:has(> [data-baseweb="tab-list"]),
    div:has(> [data-testid="stTabs"]) {
        box-shadow: none !important;
        -webkit-box-shadow: none !important;
        -moz-box-shadow: none !important;
        filter: drop-shadow(none) !important;
        text-shadow: none !important;
    }
    
    /* Force remove shadow from the tab bar area specifically */
    .main .block-container > div:first-child,
    .main .block-container > div:first-child > div {
        box-shadow: none !important;
        -webkit-box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)


def create_metric_card(title, value, delta=None):
    """Create a styled metric card"""
    delta_html = f'<span style="color: #10b981; font-size: 0.9rem;">{delta}</span>' if delta else ""
    return f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem; font-weight: 600;">
            {title}
        </div>
        <div style="font-size: 2.5rem; font-weight: 700; color: #667eea; margin-bottom: 0.3rem;">
            {value}
        </div>
        {delta_html}
    </div>
    """


def main():
    # Hero section
    st.markdown('<div class="main-header">Anomaly Detective</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Uncover hidden patterns • Detect the unexpected • Visualize anomalies</div>', unsafe_allow_html=True)
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Analysis", "Settings"])
    
    with tab1:
        # File upload in main area
        col_upload, col_info = st.columns([2, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Your Time-Series Data",
                type=['csv'],
                help="Upload a CSV file with timestamp and value columns",
                key="main_uploader"
            )
        
        with col_info:
            st.markdown("""
            <div class="info-card">
                <h3 style="color: #667eea; margin-top: 0;">Quick Tips</h3>
                <ul style="color: #333; line-height: 1.8;">
                    <li>Use sample data: <code>data/sample_ts_data.csv</code></li>
                    <li>Ensure timestamp column is parseable</li>
                    <li>Value column must be numeric</li>
                    <li>More data = better detection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Column selection
            st.markdown("### Column Selection")
            
            try:
                # Read preview and reset file pointer
                preview_df = pd.read_csv(uploaded_file, nrows=5)
                uploaded_file.seek(0)  # Reset file pointer to beginning
                
                col1, col2 = st.columns(2)
                
                with col1:
                    timestamp_col = st.selectbox(
                        "Timestamp Column",
                        options=preview_df.columns.tolist(),
                        help="Select the column containing timestamps",
                        key="ts_col"
                    )
                
                with col2:
                    # Default to second column (index 1) if available, otherwise first column
                    default_value_index = 1 if len(preview_df.columns) >= 2 else 0
                    value_col = st.selectbox(
                        "Value Column",
                        options=preview_df.columns.tolist(),
                        index=default_value_index,
                        help="Select the column containing numeric values",
                        key="val_col"
                    )
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                timestamp_col = None
                value_col = None
            
            if timestamp_col and value_col:
                # Get parameters from session state or defaults
                window_size = st.session_state.get('window_size', 20)
                threshold = st.session_state.get('threshold', 2.5)
                show_bounds = st.session_state.get('show_bounds', True)
                show_rolling_mean = st.session_state.get('show_rolling_mean', True)
                
                try:
                    # Load data
                    with st.spinner("Loading and preprocessing data..."):
                        df = load_time_series_data(
                            uploaded_file,
                            timestamp_column=timestamp_col,
                            value_column=value_col
                        )
                        validate_time_series_data(df, timestamp_col, value_col)
                    
                    # Data overview cards
                    st.markdown("### Data Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(create_metric_card("Total Points", f"{len(df):,}"), unsafe_allow_html=True)
                    
                    with col2:
                        date_range = f"{df[timestamp_col].min().date()}<br>to {df[timestamp_col].max().date()}"
                        st.markdown(create_metric_card("Date Range", date_range), unsafe_allow_html=True)
                    
                    with col3:
                        val_range = f"{df[value_col].min():.1f} - {df[value_col].max():.1f}"
                        st.markdown(create_metric_card("Value Range", val_range), unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(create_metric_card("Mean Value", f"{df[value_col].mean():.2f}"), unsafe_allow_html=True)
                    
                    # Detect anomalies
                    with st.spinner("Detecting anomalies..."):
                        df_analyzed = detect_anomalies_in_time_series(
                            df,
                            value_column=value_col,
                            window_size=window_size,
                            threshold=threshold
                        )
                    
                    # Results summary
                    n_anomalies = df_analyzed['is_anomaly'].sum()
                    anomaly_percentage = (n_anomalies / len(df_analyzed)) * 100
                    
                    st.markdown("### Detection Results")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        anomaly_color = "#ef4444" if n_anomalies > 0 else "#10b981"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem; font-weight: 600;">
                                Anomalies Detected
                            </div>
                            <div style="font-size: 2.5rem; font-weight: 700; color: {anomaly_color}; margin-bottom: 0.3rem;">
                                {n_anomalies}
                            </div>
                            <div style="color: #666; font-size: 0.85rem;">
                                {anomaly_percentage:.2f}% of data
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(create_metric_card("Anomaly Rate", f"{anomaly_percentage:.2f}%"), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(create_metric_card("Window Size", str(window_size)), unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(create_metric_card("Threshold", f"{threshold:.1f}σ"), unsafe_allow_html=True)
                    
                    # Visualizations
                    st.markdown("### Visualizations")
                    
                    # Main plot
                    fig_main = create_time_series_plot(
                        df_analyzed,
                        timestamp_column=timestamp_col,
                        value_column=value_col,
                        show_bounds=show_bounds,
                        show_rolling_mean=show_rolling_mean
                    )
                    fig_main.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.95)',
                        paper_bgcolor='rgba(255,255,255,0.95)',
                        font=dict(color='#333')
                    )
                    st.plotly_chart(fig_main, use_container_width=True)
                    
                    # Z-score plot
                    st.markdown("#### Z-Score Analysis")
                    fig_zscore = create_z_score_plot(
                        df_analyzed,
                        timestamp_column=timestamp_col,
                        threshold=threshold
                    )
                    fig_zscore.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.95)',
                        paper_bgcolor='rgba(255,255,255,0.95)',
                        font=dict(color='#333')
                    )
                    st.plotly_chart(fig_zscore, use_container_width=True)
                    
                    # Anomaly details
                    if n_anomalies > 0:
                        st.markdown("### Anomaly Details")
                        
                        anomaly_df = df_analyzed[df_analyzed['is_anomaly']].copy()
                        anomaly_df = anomaly_df[[timestamp_col, value_col, 'z_score', 'rolling_mean']].copy()
                        anomaly_df.columns = ['Timestamp', 'Value', 'Z-Score', 'Rolling Mean']
                        anomaly_df['abs_z'] = anomaly_df['Z-Score'].abs()
                        anomaly_df = anomaly_df.sort_values('abs_z', ascending=False).drop('abs_z', axis=1)
                        
                        st.dataframe(
                            anomaly_df,
                            use_container_width=True,
                            hide_index=True,
                            height=300
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            csv_anomalies = anomaly_df.to_csv(index=False)
                            st.download_button(
                                "Download Anomalies (CSV)",
                                data=csv_anomalies,
                                file_name="detected_anomalies.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        with col2:
                            download_df = df_analyzed[[timestamp_col, value_col, 'rolling_mean', 'rolling_std', 
                                                      'z_score', 'is_anomaly', 'upper_bound', 'lower_bound']].copy()
                            csv_full = download_df.to_csv(index=False)
                            st.download_button(
                                "Download Full Results (CSV)",
                                data=csv_full,
                                file_name="anomaly_detection_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    else:
                        st.info("No anomalies detected with current parameters. Try adjusting the threshold or window size in the Settings tab.")
                
                except ValueError as e:
                    st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
    
    with tab2:
        st.markdown("### Detection Parameters")
        
        window_size = st.slider(
            "Rolling Window Size",
            min_value=2,
            max_value=100,
            value=st.session_state.get('window_size', 20),
            help="Size of the rolling window for computing mean and standard deviation. Smaller = more sensitive, Larger = more stable.",
            key="window_slider"
        )
        st.session_state.window_size = window_size
        
        threshold = st.slider(
            "Anomaly Threshold (Z-Score)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.get('threshold', 2.5),
            step=0.1,
            help="Points with |z-score| > threshold will be flagged as anomalies. Lower = more sensitive, Higher = only extreme anomalies.",
            key="threshold_slider"
        )
        st.session_state.threshold = threshold
        
        st.markdown("---")
        st.markdown("### Visualization Options")
        
        show_bounds = st.checkbox(
            "Show Upper/Lower Bounds",
            value=st.session_state.get('show_bounds', True),
            help="Display the statistical bounds based on the threshold",
            key="bounds_check"
        )
        st.session_state.show_bounds = show_bounds
        
        show_rolling_mean = st.checkbox(
            "Show Rolling Mean",
            value=st.session_state.get('show_rolling_mean', True),
            help="Display the rolling mean line",
            key="mean_check"
        )
        st.session_state.show_rolling_mean = show_rolling_mean


if __name__ == '__main__':
    main()
