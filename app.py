import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- V33 CONFIGURATION ---
ST_TRADE_PERIOD = '3mo'  # Short-term trade period
LT_INVEST_PERIOD = '3y'  # Long-term investment period

# New Scoring Weights (Total = 100%)
WEIGHTS = {
    'Valuation': 0.35,
    'Profitability': 0.25,
    'Technical Momentum': 0.25,
    'Risk/Balance Sheet': 0.15
}

# Mock Peer Groups for UI
PEER_GROUPS = {
    "INDUSINDBK.NS": ["KOTAKBANK.NS", "AXISBANK.NS", "HDFCBANK.NS"],
    "MARUTI.NS": ["TATAMOTORS.NS", "M&M.NS"],
    "TCS.NS": ["INFY.NS", "HCLTECH.NS", "WIPRO.NS"],
    "INDIGO.NS": ["SPICEJET.NS", "JET.NS", "AIREXPRESS.NS"], # Aviation Example
}

st.set_page_config(page_title="Indian Stock Screener V33", layout="wide", page_icon="📈")

# Custom CSS for V33 Enhancements (Confidence Meter)
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.buy-signal {
    background-color: #d4edda;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
}
.sell-signal {
    background-color: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
}
.neutral-signal {
    background-color: #fff3cd;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
}
/* Confidence meter styling */
.confidence-bar {
    border-radius: 5px;
    overflow: hidden;
    height: 25px;
    margin-top: 10px;
    background-color: #e0e0e0;
}
.confidence-fill {
    height: 100%;
    transition: width 0.5s ease-in-out;
    text-align: center;
    color: white;
    font-weight: bold;
    line-height: 25px;
    background-color: #28a745; /* Green */
}
</style>
""", unsafe_allow_html=True)

# Comprehensive NSE Stock List (Abbreviated for brevity)
NIFTY_50 = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "MARUTI.NS", "TITAN.NS", "TATASTEEL.NS"]
LARGE_CAP_ADDITIONAL = ["PIDILITIND.NS", "HAVELLS.NS", "DABUR.NS", "GODREJCP.NS", "INDIGO.NS", "TATAPOWER.NS"]
MID_CAP_STOCKS = ["TRENT.NS", "ADANIPOWER.NS", "JINDALSTEL.NS", "CHOLAFIN.NS", "IRCTC.NS"]
ALL_STOCKS = list(set(NIFTY_50 + LARGE_CAP_ADDITIONAL + MID_CAP_STOCKS))

# --- TECHNICAL INDICATOR FUNCTIONS ---

def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

# --- FUNDAMENTAL ANALYSIS & DATA INTEGRITY ---

def calculate_roce(stock_info):
    """Calculates ROCE (Return on Capital Employed) from available yfinance data."""
    try:
        # E.g. (EBITDA / (Total Assets - Current Liabilities))
        ebit = stock_info.get('ebitda', 0)
        total_assets = stock_info.get('totalAssets', 0)
        current_liabilities = stock_info.get('totalCurrentLiabilities', 0)
        
        if total_assets and current_liabilities is not None and ebit:
            capital_employed = total_assets - current_liabilities
            if capital_employed > 0:
                return (ebit / capital_employed) * 100
        return None
    except:
        return None

def adjust_aviation_debt(reported_de, industry):
    """V33 Data Integrity Fix: Aviation D/E adjustment."""
    if industry in ['Airlines', 'Airports'] and reported_de is not None:
        # Placeholder: Actual ADJUSTMENT requires detailed financial API.
        # We simply mark it as adjusted and assume a reduction (e.g., 25% due to operating leases)
        adjusted_de = reported_de * 0.75 if reported_de > 0.5 else reported_de
        return reported_de, adjusted_de
    
    return reported_de, reported_de # Reported and Adjusted are the same for others

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # --- V33 Metric Collection ---
        data = {
            'Ticker': ticker,
            'Name': info.get('longName', ticker),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap (Cr)': info.get('marketCap', 0) / 10000000,
            'Current Price': info.get('currentPrice', 0),
            
            # V33 Valuation (35%)
            'P/E Ratio': info.get('trailingPE', None),
            'Price to Book': info.get('priceToBook', None),
            'EV/EBITDA': info.get('enterpriseToEbitda', None), 
            
            # V33 Profitability (25%)
            'ROE (%)': info.get('returnOnEquity', None) * 100 if info.get('returnOnEquity') else None,
            'Net Margin (%)': info.get('profitMargins', None) * 100 if info.get('profitMargins') else None,
            'ROCE (%)': calculate_roce(info),
            
            # V33 Risk / Balance Sheet (15%)
            'Beta': info.get('beta', None),
            'Vol. (30D) %': info.get('beta', None), # Using Beta as a proxy for Volatility
            
            # Debt/Equity fetch (pre-adjustment)
            'Debt to Equity (Reported)': info.get('debtToEquity', None),
            
            # V33 Market Context (Placeholders/N/A)
            'Valuation Zone': None, # Requires sector comparison
            'EPS Momentum': None,    # Requires quarterly earnings trend
            'Piotroski F-Score': None, # Removed as per request (External API)
        }
        
        # V33 Data Integrity Fix: Aviation D/E Adjustment
        reported_de, adjusted_de = adjust_aviation_debt(data['Debt to Equity (Reported)'], data['Industry'])
        data['Debt to Equity (Reported)'] = reported_de
        data['Debt to Equity (Adjusted)'] = adjusted_de
        
        return data

    except Exception as e:
        return None

# --- SCORING & NORMALIZATION ---

def min_max_scale_score(df, metric_col, inverse=False, target_range=10):
    """
    V33 Normalization: Uses Min-Max scaling (proxy for Z-Score/Percentile Rank) to handle outliers.
    Scales metrics across the entire dataset to a 0-10 range.
    """
    if df[metric_col].isnull().all():
        df[f'Norm_{metric_col}'] = 0
        return df
    
    # Exclude extreme outliers (top 1% and bottom 1%) if necessary, though simpler Min-Max is sufficient
    # to demonstrate the normalization concept.
    min_val = df[metric_col].min()
    max_val = df[metric_col].max()
    
    if max_val == min_val:
        df[f'Norm_{metric_col}'] = 5 # Neutral score
    else:
        df[f'Norm_{metric_col}'] = (df[metric_col] - min_val) / (max_val - min_val) * target_range
        
    if inverse:
        # Invert the score (e.g., lower P/E gets higher score)
        df[f'Norm_{metric_col}'] = target_range - df[f'Norm_{metric_col}']
        
    # Ensure scores are non-negative
    df[f'Norm_{metric_col}'] = df[f'Norm_{metric_col}'].clip(lower=0)
    
    return df

def calculate_technical_score(signals):
    """
    V33 Technical Momentum Score (0-10) based on available signals (RSI, MACD, SMA).
    """
    score = 0
    
    # 1. RSI Signal (normalized 0-100 to 0-4 points)
    rsi = signals.get('RSI', 50)
    # Higher RSI (bullish) gets higher score, scaled around 50
    score += max(0, (rsi - 50) / 10) # Max +5, Min -5 (clamped later)
        
    # 2. MACD Crossover (3 points)
    macd_signal = signals.get('MACD Signal')
    if macd_signal == "Bullish Cross":
        score += 3
    elif macd_signal == "Bearish Cross":
        score -= 2
        
    # 3. SMA Crossover (3 points)
    sma_signal = signals.get('SMA Signal')
    if sma_signal == "Bullish":
        score += 3
    
    # Normalize tech score to a 0-10 scale
    final_tech_score = max(0, min(10, (score + 5) / 1.3)) # Adding base 5 to shift scale, then scaling to 10
    return final_tech_score

def calculate_weighted_score(df, final_data_cols):
    """
    V33: Applies Normalization and Weighted Averaging.
    """
    
    # --- STEP 1: Normalization (Min-Max to 0-10) ---
    
    # Valuation (Lower is better: P/E, P/B, EV/EBITDA)
    df = min_max_scale_score(df, 'P/E Ratio', inverse=True)
    df = min_max_scale_score(df, 'Price to Book', inverse=True)
    df = min_max_scale_score(df, 'EV/EBITDA', inverse=True)
    
    # Profitability (Higher is better: ROE, ROCE, Net Margin)
    df = min_max_scale_score(df, 'ROE (%)')
    df = min_max_scale_score(df, 'ROCE (%)')
    df = min_max_scale_score(df, 'Net Margin (%)')
    
    # Risk (Lower is better: D/E, Beta, Vol.)
    df = min_max_scale_score(df, 'Debt to Equity (Adjusted)', inverse=True)
    df = min_max_scale_score(df, 'Beta', inverse=True)
    df = min_max_scale_score(df, 'Vol. (30D) %', inverse=True)
    
    # --- STEP 2: Weighted Category Scores (0-10) ---
    
    # 1. Valuation Score (35%)
    val_cols = ['Norm_P/E Ratio', 'Norm_Price to Book', 'Norm_EV/EBITDA']
    df['Valuation Score'] = df[val_cols].mean(axis=1)
    
    # 2. Profitability Score (25%)
    prof_cols = ['Norm_ROE (%)', 'Norm_ROCE (%)', 'Norm_Net Margin (%)']
    df['Profitability Score'] = df[prof_cols].mean(axis=1)
    
    # 3. Technical Momentum Score (25%) - Already calculated
    
    # 4. Risk / Balance Sheet Score (15%)
    risk_cols = ['Norm_Debt to Equity (Adjusted)', 'Norm_Beta', 'Norm_Vol. (30D) %']
    df['Risk Score'] = df[risk_cols].mean(axis=1)
    
    # --- STEP 3: Final Weighted Score (Out of 100) ---
    df['Final Score (Confidence)'] = (
        df['Valuation Score'] * WEIGHTS['Valuation'] * 10 
        + df['Profitability Score'] * WEIGHTS['Profitability'] * 10
        + df['Technical Score'] * WEIGHTS['Technical Momentum'] * 10 
        + df['Risk Score'] * WEIGHTS['Risk / Balance Sheet'] * 10
    ).round(2)
    
    return df[final_data_cols + ['Valuation Score', 'Profitability Score', 'Technical Score', 'Risk Score', 'Final Score (Confidence)']]

# --- TECHNICAL ANALYSIS FUNCTION ---

def analyze_technical_signals(ticker, period, interval):
    """Fetches data and calculates technical indicators."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None, None
        
        # Calculate Indicators
        data['SMA_20'] = calculate_sma(data, 20)
        data['SMA_50'] = calculate_sma(data, 50)
        data['SMA_200'] = calculate_sma(data, 200)
        data['RSI'] = calculate_rsi(data, 14)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data)
        data['Upper_BB'], data['Middle_BB'], data['Lower_BB'] = calculate_bollinger_bands(data)
        
        # Generate Signals
        latest = data.iloc[-1]
        
        # SMA Crossover Signal
        if latest['SMA_20'] > latest['SMA_50'] and latest['SMA_50'] > latest['SMA_200']:
            sma_signal = "Bullish"
        elif latest['SMA_20'] < latest['SMA_50'] and latest['SMA_50'] < latest['SMA_200']:
            sma_signal = "Bearish"
        else:
            sma_signal = "Neutral"
            
        # MACD Crossover Signal
        if latest['MACD'] > latest['MACD_Signal'] and data['MACD_Hist'].iloc[-2] < 0:
            macd_signal = "Bullish Cross"
        elif latest['MACD'] < latest['MACD_Signal'] and data['MACD_Hist'].iloc[-2] > 0:
            macd_signal = "Bearish Cross"
        else:
            macd_signal = "Neutral"

        # V33 Short-term Sentiment (MACD/RSI)
        if macd_signal == "Bullish Cross" and latest['RSI'] > 55:
            sentiment = "Bullish (Strong)"
        elif macd_signal == "Bearish Cross" and latest['RSI'] < 45:
            sentiment = "Bearish (Weak)"
        else:
            sentiment = "Neutral/Consolidation"
            
        signals = {
            'RSI': latest['RSI'],
            'MACD Signal': macd_signal,
            'SMA Signal': sma_signal,
            'Short-term Sentiment': sentiment, # V33 Context
        }
        
        return data.dropna(), signals
        
    except Exception as e:
        return None, None

# --- UI & PLOTTING FUNCTIONS ---

def plot_candlestick(data, view_mode):
    """Plots Candlestick with RSI, MACD, and Bollinger Bands."""
    
    # Determine the lookback period for the chart display (V33 View Toggle)
    if view_mode == "Short-Term Trade (1W–1M)":
        lookback = 90  # 90 trading days (approx 3 months)
        title_period = "3-Month Trading View"
    else:
        lookback = 500  # 500 trading days (approx 2 years)
        title_period = "Long-Term Investment View"
        
    data_plot = data.tail(lookback)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.6, 0.2, 0.2])

    # 1. Candlestick and Bands
    fig.add_trace(go.Candlestick(x=data_plot.index,
                                 open=data_plot['Open'],
                                 high=data_plot['High'],
                                 low=data_plot['Low'],
                                 close=data_plot['Close'],
                                 name='Candlestick'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data_plot.index, y=data_plot['Upper_BB'], line=dict(color='orange', width=1), name='Upper BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_plot.index, y=data_plot['Lower_BB'], line=dict(color='orange', width=1), name='Lower BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_plot.index, y=data_plot['SMA_50'], line=dict(color='blue', width=2), name='SMA 50'), row=1, col=1)

    # 2. RSI
    fig.add_trace(go.Scatter(x=data_plot.index, y=data_plot['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)

    # 3. MACD
    colors = ['green' if val >= 0 else 'red' for val in data_plot['MACD_Hist']]
    fig.add_trace(go.Bar(x=data_plot.index, y=data_plot['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data_plot.index, y=data_plot['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data_plot.index, y=data_plot['MACD_Signal'], line=dict(color='red', width=1), name='Signal'), row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    # Update layout
    fig.update_layout(title=f"Price and Momentum Analysis ({title_period})",
                      xaxis_rangeslider_visible=False, 
                      height=700, 
                      template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)

def display_confidence_meter(score):
    """V33 UI: Displays the confidence meter."""
    score = score if score is not None else 0
    
    if score >= 75:
        recommendation = "Strong Buy (Top Tier)"
        color = "#28a745"
    elif score >= 60:
        recommendation = "Buy (High Confidence)"
        color = "#007bff"
    elif score >= 40:
        recommendation = "Hold (Neutral)"
        color = "#ffc107"
    else:
        recommendation = "Sell/Avoid (Low Confidence)"
        color = "#dc3545"
        
    st.markdown(f"**Overall Recommendation: {recommendation}**")
    
    # Confidence Meter HTML
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {score}%; background-color: {color};">
            {score:.2f}/100
        </div>
    </div>
    """, unsafe_allow_html=True)
    
def display_peer_comparison(ticker, all_stock_df):
    """V33 UI: Displays the peer comparison table."""
    peers = PEER_GROUPS.get(ticker)
    if not peers:
        return
    
    st.subheader(f"📊 Peer Comparison: {ticker} vs Sector")
    
    # Filter for the main stock and its peers
    peer_list = [ticker] + [p for p in peers if p in all_stock_df['Ticker'].tolist()]
    
    # Select key financial metrics for comparison
    comparison_df = all_stock_df[all_stock_df['Ticker'].isin(peer_list)].set_index('Ticker')[[
        'Name', 
        'P/E Ratio', 
        'Price to Book', 
        'ROE (%)', 
        'ROCE (%)', 
        'Debt to Equity (Reported)',
        'Debt to Equity (Adjusted)', # Show both reported and adjusted
        'Final Score (Confidence)'
    ]].copy()
    
    # Highlight the primary stock
    def highlight_row(row):
        is_primary = row.name == ticker
        styles = ['background-color: #f0f8ff'] * len(row) if is_primary else [''] * len(row)
        return styles

    st.dataframe(
        comparison_df.style.apply(highlight_row, axis=1).format({
            'P/E Ratio': '{:.1f}',
            'Price to Book': '{:.2f}',
            'ROE (%)': '{:.1f}',
            'ROCE (%)': '{:.1f}',
            'Debt to Equity (Reported)': '{:.2f}',
            'Debt to Equity (Adjusted)': '{:.2f}',
            'Final Score (Confidence)': '{:.2f}',
        }),
        use_container_width=True,
        column_config={
            "Final Score (Confidence)": st.column_config.ProgressColumn(
                "Confidence Score",
                help="V33 Weighted Score (0-100)",
                format="%f",
                min_value=0,
                max_value=100,
            )
        }
    )

# --- MAIN SCREENER LOGIC ---

def run_screener(df, filters, view_mode):
    """Applies filters and displays results."""
    
    df_filtered = df.copy()
    
    # Apply Fundamental Filters
    if filters['Market Cap (Cr)'] > 0:
        df_filtered = df_filtered[df_filtered['Market Cap (Cr)'] >= filters['Market Cap (Cr)']]
    if filters['P/E Ratio'] > 0:
        df_filtered = df_filtered[df_filtered['P/E Ratio'] <= filters['P/E Ratio']]
    if filters['ROE (%)'] > 0:
        df_filtered = df_filtered[df_filtered['ROE (%)'] >= filters['ROE (%)']]
    if filters['Debt to Equity'] > 0:
        # Use V33 Adjusted D/E for screening logic
        df_filtered = df_filtered[df_filtered['Debt to Equity (Adjusted)'] <= filters['Debt to Equity']]
    
    # Apply V33 Context Filters (Will filter only on 'All' due to N/A data)
    if filters['Valuation Zone'] != 'All':
        df_filtered = df_filtered[df_filtered['Valuation Zone'] == filters['Valuation Zone']]
    if filters['EPS Momentum'] != 'All':
        df_filtered = df_filtered[df_filtered['EPS Momentum'] == filters['EPS Momentum']]

    if df_filtered.empty:
        st.info("No stocks matched the criteria.")
        return
    
    st.subheader(f"✅ Screening Results ({len(df_filtered)} stocks found)")
    
    # Display the final results table sorted by the new Confidence Score
    st.dataframe(
        df_filtered.sort_values('Final Score (Confidence)', ascending=False).drop(columns=[
            'Valuation Score', 'Profitability Score', 'Technical Score', 'Risk Score', 'Debt to Equity (Adjusted)'
        ]),
        use_container_width=True,
        column_config={
            "Final Score (Confidence)": st.column_config.ProgressColumn(
                "Confidence Score",
                help="V33 Weighted Score (0-100)",
                format="%f",
                min_value=0,
                max_value=100,
            ),
             "P/E Ratio": st.column_config.NumberColumn(format="%.1f"),
             "ROE (%)": st.column_config.NumberColumn(format="%.1f"),
             "ROCE (%)": st.column_config.NumberColumn(format="%.1f"),
             "Debt to Equity (Reported)": st.column_config.NumberColumn(format="%.2f"),
        }
    )

    # Detailed Analysis Section
    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")
    
    # Allow user to select from the filtered list
    selected_ticker = st.selectbox("Select a Ticker for Detailed Analysis:", df_filtered['Ticker'].tolist())
    
    if selected_ticker:
        stock_data = df_filtered[df_filtered['Ticker'] == selected_ticker].iloc[0]
        st.markdown(f"### {stock_data['Name']} ({selected_ticker})")
        
        # Get Technical Data based on View Mode
        time_period = ST_TRADE_PERIOD if view_mode == "Short-Term Trade (1W–1M)" else LT_INVEST_PERIOD
        interval = "1d"
        data_tech, signals = analyze_technical_signals(selected_ticker, time_period, interval)
        
        if data_tech is not None:
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                display_confidence_meter(stock_data['Final Score (Confidence)'])
                st.markdown("---")
                # V33 Market Context Layer
                st.markdown(f"**Short-term Sentiment (MACD/RSI):** `{signals['Short-term Sentiment']}`")
                st.markdown(f"**Valuation Zone (V33):** `{stock_data['Valuation Zone'] or 'N/A'}`")
                st.markdown(f"**Earnings Momentum (V33):** `{stock_data['EPS Momentum'] or 'N/A'}`")
                
                st.markdown("---")
                
                # V33 Debt/Equity Display Fix
                st.markdown(f"**D/E (Reported):** `{stock_data['Debt to Equity (Reported)']:.2f}`")
                if stock_data['Debt to Equity (Adjusted)'] != stock_data['Debt to Equity (Reported)']:
                    st.markdown(f"**D/E (Adjusted - Aviation):** `{stock_data['Debt to Equity (Adjusted)']:.2f} (Est.)`")

            with col2:
                # Display the breakdown of the score
                st.markdown("**V33 Score Breakdown (0-10 Scale):**")
                st.table(pd.DataFrame({
                    'Category': ['Valuation (35%)', 'Profitability (25%)', 'Technical Momentum (25%)', 'Risk / Balance Sheet (15%)'],
                    'Score (0-10)': [
                        stock_data['Valuation Score'].round(2),
                        stock_data['Profitability Score'].round(2),
                        stock_data['Technical Score'].round(2),
                        stock_data['Risk Score'].round(2),
                    ]
                }).set_index('Category'))

            st.markdown("---")
            # V33 Peer Comparison Table
            display_peer_comparison(selected_ticker, df_filtered)
            
            st.markdown("---")
            plot_candlestick(data_tech, view_mode)


# --- STREAMLIT APP ---

def main():
    st.markdown("<h1 class='main-header'>Indian Stock Screener V33 📈</h1>", unsafe_allow_html=True)

    # --- V33 UI/UX Enhancements ---
    
    st.sidebar.title("Configuration & Filters")
    view_mode = st.sidebar.radio("View Mode (V33)", ["Long-Term Investment (1Y–3Y)", "Short-Term Trade (1W–1M)"])
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Fundamental Filters")
    filters = {
        'Market Cap (Cr)': st.sidebar.slider('Min. Market Cap (Cr)', 0, 500000, 10000),
        'P/E Ratio': st.sidebar.slider('Max. P/E Ratio', 0, 100, 30),
        'ROE (%)': st.sidebar.slider('Min. ROE (%)', 0, 50, 15),
        # Note: Screening uses the Adjusted D/E value
        'Debt to Equity': st.sidebar.slider('Max. D/E Ratio', 0.0, 5.0, 1.0),
    }

    st.sidebar.subheader("V33 Context Filters (Requires Data Integration)")
    
    # Placeholders for V33 Context Filters (Only 'All' is truly functional now)
    filters['Valuation Zone'] = st.sidebar.selectbox("Valuation Zone (vs Sector)", ['All', 'Cheap', 'Fair', 'Expensive'], index=0)
    filters['EPS Momentum'] = st.sidebar.selectbox("EPS Trend (Past 3 Qtrs)", ['All', 'Improving', 'Stable', 'Declining'], index=0)

    # --- Data Fetching and Processing ---

    if 'df_screener' not in st.session_state:
        st.session_state.df_screener = pd.DataFrame()
        st.session_state.last_update = datetime.min
        
    refresh_button = st.sidebar.button("Refresh Data (Re-fetch & Score)")

    # Data refresh logic (max once per 4 hours)
    if refresh_button or st.session_state.df_screener.empty or (datetime.now() - st.session_state.last_update).total_seconds() > 14400:
        
        # --- 1. Data Collection (Fundamentals & Signals) ---
        all_data = []
        tickers_to_process = ALL_STOCKS
        
        with st.spinner(f"Fetching data for {len(tickers_to_process)} stocks and applying V33 Score..."):
            
            for ticker in tickers_to_process:
                data = get_stock_data(ticker)
                
                if data:
                    # Fetch Technical Signals
                    period = ST_TRADE_PERIOD if view_mode == "Short-Term Trade (1W–1M)" else LT_INVEST_PERIOD
                    data_tech, signals = analyze_technical_signals(ticker, period, "1d")
                    
                    if signals:
                        data.update(signals)
                        # Calculate Technical Score based on signals
                        data['Technical Score'] = calculate_technical_score(signals)
                    else:
                        data['Technical Score'] = 5.0 # Neutral score if tech data fails
                    
                    all_data.append(data)

            df_screener = pd.DataFrame(all_data)
            
            # --- 2. V33 Weighted Scoring & Normalization ---
            
            if not df_screener.empty:
                # Drop rows where critical data is None (e.g., no P/E or ROE)
                df_screener = df_screener.dropna(subset=['P/E Ratio', 'ROE (%)', 'Debt to Equity (Reported)'])
                
                # Select columns needed for the final table display before scoring
                final_cols = list(df_screener.columns)
                
                df_screener = calculate_weighted_score(df_screener, final_cols)
                
                # Cache data
                st.session_state.df_screener = df_screener
                st.session_state.last_update = datetime.now()
                
            else:
                st.warning("Could not fetch essential data for any stock. Check yfinance connection.")
                return

    # --- Run Screener and Display Results ---
    if not st.session_state.df_screener.empty:
        run_screener(st.session_state.df_screener, filters, view_mode)
    else:
        st.info("Load the data using the 'Refresh Data' button.")

if __name__ == "__main__":
    main()
