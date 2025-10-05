import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Indian Stock Screener", layout="wide", page_icon="📈")

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Comprehensive NSE Stock List (NIFTY 50 + Large Cap + Mid Cap)
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "POWERGRID.NS", "NTPC.NS",
    "TATASTEEL.NS", "ADANIPORTS.NS", "ONGC.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "BAJAJFINSV.NS", "DRREDDY.NS", "EICHERMOT.NS", "HINDALCO.NS", "JSWSTEEL.NS",
    "M&M.NS", "BRITANNIA.NS", "CIPLA.NS", "GRASIM.NS", "HEROMOTOCO.NS",
    "INDUSINDBK.NS", "APOLLOHOSP.NS", "ADANIENT.NS", "TATAMOTORS.NS",
    "BAJAJ-AUTO.NS", "TATACONSUM.NS", "SHREECEM.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "BPCL.NS", "LTIM.NS"
]

LARGE_CAP_ADDITIONAL = [
    "PIDILITIND.NS", "HAVELLS.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS",
    "VEDL.NS", "TORNTPHARM.NS", "DLF.NS", "GAIL.NS", "AMBUJACEM.NS",
    "ADANIGREEN.NS", "SIEMENS.NS", "BEL.NS", "BANKBARODA.NS", "IOC.NS",
    "INDIGO.NS", "DMART.NS", "BERGEPAINT.NS", "BOSCHLTD.NS", "LUPIN.NS",
    "HDFCAMC.NS", "PAGEIND.NS", "ABB.NS", "HINDPETRO.NS", "SAIL.NS",
    "TATAPOWER.NS", "NMDC.NS", "BAJAJHLDNG.NS", "MUTHOOTFIN.NS", "ZOMATO.NS"
]

MID_CAP_STOCKS = [
    "TRENT.NS", "ADANIPOWER.NS", "JINDALSTEL.NS", "CANBK.NS", "VOLTAS.NS",
    "IRCTC.NS", "CHOLAFIN.NS", "ESCORTS.NS", "MOTHERSON.NS", "LICHSGFIN.NS",
    "GUJGASLTD.NS", "UNIONBANK.NS", "GODREJPROP.NS", "PETRONET.NS", "INDUSTOWER.NS",
    "PIIND.NS", "OBEROIRLTY.NS", "IDEA.NS", "OFSS.NS", "MPHASIS.NS",
    "L&TFH.NS", "AUROPHARMA.NS", "IPCALAB.NS", "BALKRISIND.NS", "CROMPTON.NS",
    "ASTRAL.NS", "CONCOR.NS", "COFORGE.NS", "PERSISTENT.NS", "LALPATHLAB.NS",
    "POLYCAB.NS", "BATAINDIA.NS", "MRF.NS", "COLPAL.NS", "SUNPHARMA.NS",
    "LTTS.NS", "TORNTPOWER.NS", "METROPOLIS.NS", "DIXON.NS", "SYNGENE.NS"
]

ALL_STOCKS = list(set(NIFTY_50 + LARGE_CAP_ADDITIONAL + MID_CAP_STOCKS))

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=period).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    """Calculate MACD"""
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_atr(data, period=14):
    """Calculate Average True Range for volatility"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def identify_support_resistance(data, window=20):
    """Identify support and resistance levels"""
    highs = data['High'].rolling(window=window, center=True).max()
    lows = data['Low'].rolling(window=window, center=True).min()
    
    resistance = data[data['High'] == highs]['High'].dropna().unique()
    support = data[data['Low'] == lows]['Low'].dropna().unique()
    
    return sorted(support)[-3:] if len(support) > 0 else [], sorted(resistance)[-3:] if len(resistance) > 0 else []

def analyze_technical_signals(ticker, period='6mo'):
    """Comprehensive technical analysis"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty or len(data) < 50:
            return None
        
        # Calculate indicators
        data['SMA_20'] = calculate_sma(data, 20)
        data['SMA_50'] = calculate_sma(data, 50)
        data['SMA_200'] = calculate_sma(data, 200)
        data['EMA_12'] = calculate_ema(data, 12)
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data)
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)
        data['ATR'] = calculate_atr(data)
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = {
            'data': data,
            'current_price': current['Close'],
            'sma_20': current['SMA_20'],
            'sma_50': current['SMA_50'],
            'sma_200': current['SMA_200'],
            'rsi': current['RSI'],
            'macd': current['MACD'],
            'macd_signal': current['MACD_Signal'],
            'bb_position': (current['Close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower']) * 100,
            'volume': current['Volume'],
            'avg_volume': data['Volume'].tail(20).mean(),
            'atr': current['ATR'],
            'volatility': (current['ATR'] / current['Close']) * 100
        }
        
        # Support and Resistance
        support_levels, resistance_levels = identify_support_resistance(data)
        signals['support_levels'] = support_levels
        signals['resistance_levels'] = resistance_levels
        
        # Generate signals
        buy_signals = []
        sell_signals = []
        score = 0
        
        # Trend Analysis
        if current['Close'] > current['SMA_20'] > current['SMA_50']:
            buy_signals.append("Strong uptrend: Price > SMA20 > SMA50")
            score += 2
        elif current['Close'] > current['SMA_20']:
            buy_signals.append("Price above SMA20 (short-term bullish)")
            score += 1
        elif current['Close'] < current['SMA_20'] < current['SMA_50']:
            sell_signals.append("Strong downtrend: Price < SMA20 < SMA50")
            score -= 2
        
        # Golden Cross / Death Cross
        if current['SMA_50'] > current['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
            buy_signals.append("🌟 Golden Cross: SMA50 crossed above SMA200")
            score += 3
        elif current['SMA_50'] < current['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
            sell_signals.append("⚠️ Death Cross: SMA50 crossed below SMA200")
            score -= 3
        
        # RSI Analysis
        if current['RSI'] < 30:
            buy_signals.append(f"RSI Oversold: {current['RSI']:.1f} (potential bounce)")
            score += 2
        elif current['RSI'] > 70:
            sell_signals.append(f"RSI Overbought: {current['RSI']:.1f} (potential correction)")
            score -= 2
        elif 40 <= current['RSI'] <= 60:
            buy_signals.append(f"RSI Neutral: {current['RSI']:.1f} (healthy)")
            score += 1
        
        # MACD Analysis
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            buy_signals.append("MACD bullish crossover")
            score += 2
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            sell_signals.append("MACD bearish crossover")
            score -= 2
        
        # Bollinger Bands
        if signals['bb_position'] < 20:
            buy_signals.append("Price near lower Bollinger Band (oversold)")
            score += 1
        elif signals['bb_position'] > 80:
            sell_signals.append("Price near upper Bollinger Band (overbought)")
            score -= 1
        
        # Volume Analysis
        if current['Volume'] > signals['avg_volume'] * 1.5:
            if current['Close'] > prev['Close']:
                buy_signals.append("High volume with price increase (accumulation)")
                score += 1
            else:
                sell_signals.append("High volume with price decrease (distribution)")
                score -= 1
        
        # Support/Resistance
        if support_levels and current['Close'] <= min(support_levels) * 1.02:
            buy_signals.append(f"Price near support level: ₹{min(support_levels):.2f}")
            score += 1
        if resistance_levels and current['Close'] >= max(resistance_levels) * 0.98:
            sell_signals.append(f"Price near resistance level: ₹{max(resistance_levels):.2f}")
            score -= 1
        
        signals['buy_signals'] = buy_signals
        signals['sell_signals'] = sell_signals
        signals['score'] = score
        
        # Overall recommendation
        if score >= 4:
            signals['recommendation'] = "STRONG BUY"
            signals['rec_color'] = "green"
        elif score >= 2:
            signals['recommendation'] = "BUY"
            signals['rec_color'] = "lightgreen"
        elif score <= -4:
            signals['recommendation'] = "STRONG SELL"
            signals['rec_color'] = "red"
        elif score <= -2:
            signals['recommendation'] = "SELL"
            signals['rec_color'] = "lightcoral"
        else:
            signals['recommendation'] = "HOLD/NEUTRAL"
            signals['rec_color'] = "yellow"
        
        return signals
        
    except Exception as e:
        st.warning(f"Technical analysis error for {ticker}: {str(e)}")
        return None

def plot_candlestick_chart(ticker, signals):
    """Create interactive candlestick chart with indicators"""
    data = signals['data'].tail(100)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Indicators', 'RSI', 'MACD')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], name='SMA 200', line=dict(color='purple', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red', width=2)), row=3, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram'), row=3, col=1)
    
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def calculate_roce(stock_info):
    """Calculate Return on Capital Employed"""
    try:
        ebit = stock_info.get('ebitda', 0)
        total_assets = stock_info.get('totalAssets', 0)
        current_liabilities = stock_info.get('totalCurrentLiabilities', 0)
        
        if total_assets > 0 and current_liabilities is not None:
            capital_employed = total_assets - current_liabilities
            if capital_employed > 0:
                roce = (ebit / capital_employed) * 100
                return roce
    except:
        pass
    return None

def get_quarterly_cashflow(ticker):
    """Get quarterly free cash flow data"""
    try:
        stock = yf.Ticker(ticker)
        cf = stock.quarterly_cashflow
        if not cf.empty and 'Free Cash Flow' in cf.index:
            fcf = cf.loc['Free Cash Flow'].head(4)
            return fcf.tolist()
    except:
        pass
    return []

def check_positive_fcf(fcf_list):
    """Check if at least 3 of last 4 quarters have positive FCF"""
    if len(fcf_list) >= 4:
        positive_count = sum(1 for x in fcf_list[:4] if x > 0)
        return positive_count >= 3
    return False

def get_profit_growth(ticker):
    """Calculate 3-year profit growth"""
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        if not financials.empty and 'Net Income' in financials.index:
            net_income = financials.loc['Net Income']
            if len(net_income) >= 3:
                recent = net_income.iloc[0]
                old = net_income.iloc[min(2, len(net_income)-1)]
                if old != 0:
                    growth = ((recent - old) / abs(old)) * 100
                    return growth
    except:
        pass
    return None

def get_stock_data(ticker):
    """Fetch comprehensive stock data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data = {
            'Ticker': ticker,
            'Name': info.get('longName', ticker),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap (Cr)': info.get('marketCap', 0) / 10000000,
            'Current Price': info.get('currentPrice', 0),
            'P/E Ratio': info.get('trailingPE', None),
            'Debt to Equity': info.get('debtToEquity', None),
            'Dividend Yield (%)': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'ROE (%)': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else None,
            'Profit Margin (%)': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else None,
            'Revenue Growth (%)': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else None,
        }
        
        roce = calculate_roce(info)
        data['ROCE (%)'] = roce
        
        fcf_list = get_quarterly_cashflow(ticker)
        data['Positive FCF (3/4Q)'] = check_positive_fcf(fcf_list) if fcf_list else None
        data['FCF Data'] = fcf_list
        
        profit_growth = get_profit_growth(ticker)
        data['3Y Profit Growth (%)'] = profit_growth
        
        return data
    except Exception as e:
        return None

def apply_filters(df, filters):
    """Apply filtering criteria"""
    filtered_df = df.copy()
    
    if filters['roce']:
        filtered_df = filtered_df[
            (filtered_df['ROCE (%)'].notna()) & 
            (filtered_df['ROCE (%)'] >= filters['roce_min'])
        ]
    
    if filters['fcf']:
        filtered_df = filtered_df[filtered_df['Positive FCF (3/4Q)'] == True]
    
    if filters['pe']:
        filtered_df = filtered_df[
            (filtered_df['P/E Ratio'].notna()) & 
            (filtered_df['P/E Ratio'] < filters['pe_max'])
        ]
    
    if filters['profit_growth']:
        filtered_df = filtered_df[
            (filtered_df['3Y Profit Growth (%)'].notna()) & 
            (filtered_df['3Y Profit Growth (%)'] >= filters['profit_growth_min'])
        ]
    
    if filters['debt_equity']:
        filtered_df = filtered_df[
            (filtered_df['Debt to Equity'].notna()) & 
            (filtered_df['Debt to Equity'] < filters['debt_equity_max'])
        ]
    
    if filters['dividend']:
        filtered_df = filtered_df[filtered_df['Dividend Yield (%)'] > 0]
    
    if filters['roe']:
        filtered_df = filtered_df[
            (filtered_df['ROE (%)'].notna()) & 
            (filtered_df['ROE (%)'] >= filters['roe_min'])
        ]
    
    return filtered_df

def main():
    st.markdown('<h1 class="main-header">📈 Indian Stock Screener with Technical Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Filter Indian stocks (NIFTY 50 + Large Cap + Mid Cap) with fundamental & technical analysis")
    
    # Sidebar
    st.sidebar.header("🎯 Filtering Criteria")
    
    st.sidebar.subheader("✅ Mandatory Filters")
    
    filters = {}
    filters['roce'] = st.sidebar.checkbox("ROCE Filter", value=True)
    filters['roce_min'] = st.sidebar.slider("Minimum ROCE (%)", 10, 30, 15, 1) if filters['roce'] else 15
    
    filters['fcf'] = st.sidebar.checkbox("Positive FCF (3/4 Quarters)", value=True)
    
    filters['pe'] = st.sidebar.checkbox("P/E Ratio Filter", value=True)
    filters['pe_max'] = st.sidebar.slider("Maximum P/E", 10, 100, 40, 5) if filters['pe'] else 40
    
    filters['profit_growth'] = st.sidebar.checkbox("Profit Growth Filter", value=True)
    filters['profit_growth_min'] = st.sidebar.slider("Min 3Y Profit Growth (%)", 0, 30, 10, 1) if filters['profit_growth'] else 10
    
    filters['debt_equity'] = st.sidebar.checkbox("Debt/Equity Filter", value=True)
    filters['debt_equity_max'] = st.sidebar.slider("Max Debt/Equity", 0.0, 2.0, 1.0, 0.1) if filters['debt_equity'] else 1.0
    
    st.sidebar.subheader("🎯 Preferred Filters")
    
    filters['dividend'] = st.sidebar.checkbox("Dividend Payer", value=False)
    filters['roe'] = st.sidebar.checkbox("ROE Filter", value=False)
    filters['roe_min'] = st.sidebar.slider("Minimum ROE (%)", 10, 30, 15, 1) if filters['roe'] else 15
    
    st.sidebar.subheader("📊 Stock Universe")
    stock_category = st.sidebar.multiselect(
        "Select Categories",
        ["NIFTY 50", "Large Cap", "Mid Cap"],
        default=["NIFTY 50"]
    )
    
    stock_list = []
    if "NIFTY 50" in stock_category:
        stock_list.extend(NIFTY_50)
    if "Large Cap" in stock_category:
        stock_list.extend(LARGE_CAP_ADDITIONAL)
    if "Mid Cap" in stock_category:
        stock_list.extend(MID_CAP_STOCKS)
    
    stock_list = list(set(stock_list))
    
    use_custom = st.sidebar.checkbox("Add custom tickers")
    if use_custom:
        custom_stocks = st.sidebar.text_area(
            "Enter ticker symbols (one per line, with .NS suffix)",
            "Example:\nINFY.NS\nTCS.NS"
        )
        if custom_stocks:
            custom_list = [s.strip() for s in custom_stocks.split('\n') if s.strip()]
            stock_list.extend(custom_list)
            stock_list = list(set(stock_list))
    
    # Main content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stocks to Scan", len(stock_list))
    with col2:
        st.metric("NIFTY 50", len([s for s in stock_list if s in NIFTY_50]))
    with col3:
        st.metric("Large+Mid Cap", len([s for s in stock_list if s not in NIFTY_50]))
    
    if st.button("🔍 Start Screening", type="primary"):
        st.info("Fetching stock data... This may take several minutes.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_data = []
        for idx, ticker in enumerate(stock_list):
            status_text.text(f"Processing {ticker}... ({idx+1}/{len(stock_list)})")
            data = get_stock_data(ticker)
            if data:
                all_data.append(data)
            progress_bar.progress((idx + 1) / len(stock_list))
            time.sleep(0.3)
        
        status_text.empty()
        progress_bar.empty()
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            st.success(f"✅ Successfully fetched data for {len(df)} stocks")
            
            filtered_df = apply_filters(df, filters)
            
            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric("Stocks Passing Filters", len(filtered_df), f"{len(filtered_df)/len(df)*100:.1f}%")
            
            if not filtered_df.empty:
                st.subheader("✅ Filtered Stocks (Meeting Criteria)")
                
                display_df = filtered_df.drop('FCF Data', axis=1).copy()
                
                st.dataframe(
                    display_df.style.format({
                        'Market Cap (Cr)': '{:.0f}',
                        'Current Price': '{:.2f}',
                        'P/E Ratio': '{:.2f}',
                        'Debt to Equity': '{:.2f}',
                        'ROCE (%)': '{:.2f}',
                        'ROE (%)': '{:.2f}',
                        'Profit Margin (%)': '{:.2f}',
                        'Revenue Growth (%)': '{:.2f}',
                        '3Y Profit Growth (%)': '{:.2f}',
                        'Dividend Yield (%)': '{:.2f}'
                    }, na_rep='N/A'),
                    height=400
                )
                
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Results as CSV",
                    data=csv,
                    file_name=f"filtered_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Technical Analysis Section
                st.markdown("---")
                st.header("📊 Technical Analysis & Trading Signals")
                
                selected_stock = st.selectbox(
                    "Select a stock for detailed technical analysis",
                    filtered_df['Ticker'].tolist()
                )
                
                if selected_stock:
                    with st.spinner(f"Analyzing {selected_stock}..."):
                        tech_signals = analyze_technical_signals(selected_stock)
                        
                        if tech_signals:
                            stock_data = filtered_df[filtered_df['Ticker'] == selected_stock].iloc[0]
                            
                            # Recommendation Banner
                            rec_class = "buy-signal" if "BUY" in tech_signals['recommendation'] else "sell-signal" if "SELL" in tech_signals['recommendation'] else "neutral-signal"
                            st.markdown(f'<div class="{rec_class}">Overall Recommendation: {tech_signals["recommendation"]} (Score: {tech_signals["score"]})</div>', unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # Key Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Current Price", f"₹{tech_signals['current_price']:.2f}")
                                st.metric("RSI", f"{tech_signals['rsi']:.1f}", 
                                         "Oversold" if tech_signals['rsi'] < 30 else "Overbought" if tech_signals['rsi'] > 70 else "Neutral")
                            
                            with col2:
                                st.metric("SMA 20", f"₹{tech_signals['sma_20']:.2f}")
                                st.metric("SMA 50", f"₹{tech_signals['sma_50']:.2f}")
                            
                            with col3:
                                st.metric("MACD", f"{tech_signals['macd']:.2f}")
                                st.metric("Volatility", f"{tech_signals['volatility']:.2f}%")
                            
                            with col4:
                                volume_change = ((tech_signals['volume'] - tech_signals['avg_volume']) / tech_signals['avg_volume'] * 100)
                                st.metric("Volume vs Avg", f"{volume_change:+.1f}%")
                                st.metric("BB Position", f"{tech_signals['bb_position']:.1f}%")
                            
                            st.markdown("---")
                            
                            # Buy and Sell Signals
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("🟢 Buy Signals")
                                if tech_signals['buy_signals']:
                                    for signal in tech_signals['buy_signals']:
                                        st.success(f"✓ {signal}")
                                else:
                                    st.info("No strong buy signals detected")
                            
                            with col2:
                                st.subheader("🔴 Sell Signals")
                                if tech_signals['sell_signals']:
                                    for signal in tech_signals['sell_signals']:
                                        st.error(f"✗ {signal}")
                                else:
                                    st.info("No strong sell signals detected")
                            
                            # Support and Resistance
                            if tech_signals['support_levels'] or tech_signals['resistance_levels']:
                                st.markdown("---")
                                st.subheader("📍 Key Levels")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if tech_signals['support_levels']:
                                        st.write("**Support Levels:**")
                                        for level in tech_signals['support_levels']:
                                            st.write(f"₹{level:.2f}")
                                
                                with col2:
                                    if tech_signals['resistance_levels']:
                                        st.write("**Resistance Levels:**")
                                        for level in tech_signals['resistance_levels']:
                                            st.write(f"₹{level:.2f}")
                            
                            # Candlestick Chart
                            st.markdown("---")
                            st.subheader("📈 Interactive Chart")
                            
                            chart = plot_candlestick_chart(selected_stock, tech_signals)
                            st.plotly_chart(chart, use_container_width=True)
                            
                            # Trading Strategy Recommendations
                            st.markdown("---")
                            st.subheader("💡 Trading Strategy Recommendations")
                            
                            if tech_signals['score'] >= 3:
                                st.success("""
                                **Bullish Strategy:**
                                - Consider buying in tranches if price dips to support levels
                                - Set stop loss below recent support
                                - Target: Next resistance level
                                - Watch for volume confirmation on breakouts
                                """)
                            elif tech_signals['score'] <= -3:
                                st.error("""
                                **Bearish Strategy:**
                                - Consider booking profits if holding
                                - Avoid fresh buying positions
                                - Wait for price to stabilize near support
                                - Watch for reversal signals (RSI divergence, MACD crossover)
                                """)
                            else:
                                st.info("""
                                **Neutral Strategy:**
                                - Wait for clearer signals before taking position
                                - Consider selling at resistance if holding
                                - Look for breakout or breakdown confirmation
                                - Monitor volume and momentum indicators
                                """)
                            
                            # Key Technical Insights
                            st.markdown("---")
                            st.subheader("🔍 Key Technical Insights")
                            
                            insights = []
                            
                            # Trend strength
                            if tech_signals['current_price'] > tech_signals['sma_200']:
                                insights.append("✅ **Long-term uptrend** - Price above 200 SMA")
                            else:
                                insights.append("⚠️ **Long-term downtrend** - Price below 200 SMA")
                            
                            # Momentum
                            if tech_signals['macd'] > tech_signals['macd_signal']:
                                insights.append("✅ **Positive momentum** - MACD above signal line")
                            else:
                                insights.append("⚠️ **Negative momentum** - MACD below signal line")
                            
                            # Volatility
                            if tech_signals['volatility'] > 3:
                                insights.append("⚠️ **High volatility** - Use wider stop losses")
                            elif tech_signals['volatility'] < 1.5:
                                insights.append("✅ **Low volatility** - Stable price movement")
                            
                            # Volume
                            if tech_signals['volume'] > tech_signals['avg_volume'] * 1.5:
                                insights.append("📊 **Above average volume** - Strong participation")
                            
                            for insight in insights:
                                st.markdown(insight)
                            
                            # Fundamental Data
                            st.markdown("---")
                            st.subheader("📊 Fundamental Metrics")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("ROCE", f"{stock_data['ROCE (%)']:.2f}%" if pd.notna(stock_data['ROCE (%)']) else "N/A")
                                st.metric("P/E Ratio", f"{stock_data['P/E Ratio']:.2f}" if pd.notna(stock_data['P/E Ratio']) else "N/A")
                                st.metric("Debt/Equity", f"{stock_data['Debt to Equity']:.2f}" if pd.notna(stock_data['Debt to Equity']) else "N/A")
                            
                            with col2:
                                st.metric("3Y Profit Growth", f"{stock_data['3Y Profit Growth (%)']:.2f}%" if pd.notna(stock_data['3Y Profit Growth (%)']) else "N/A")
                                st.metric("ROE", f"{stock_data['ROE (%)']:.2f}%" if pd.notna(stock_data['ROE (%)']) else "N/A")
                                st.metric("Profit Margin", f"{stock_data['Profit Margin (%)']:.2f}%" if pd.notna(stock_data['Profit Margin (%)']) else "N/A")
                            
                            with col3:
                                st.metric("Market Cap", f"₹{stock_data['Market Cap (Cr)']:.0f} Cr")
                                st.metric("Dividend Yield", f"{stock_data['Dividend Yield (%)']:.2f}%")
                                st.metric("Revenue Growth", f"{stock_data['Revenue Growth (%)']:.2f}%" if pd.notna(stock_data['Revenue Growth (%)']) else "N/A")
                        else:
                            st.error("Unable to fetch technical analysis data for this stock.")
            else:
                st.warning("⚠️ No stocks passed the filtering criteria. Try relaxing some filters.")
        else:
            st.error("❌ Unable to fetch data for any stocks. Please check your internet connection and try again.")
    
    # Information sections
    with st.expander("ℹ️ Understanding Technical Indicators"):
        st.markdown("""
        ### 📊 Key Technical Indicators Explained
        
        **1. Moving Averages (SMA)**
        - **SMA 20**: Short-term trend (1 month)
        - **SMA 50**: Medium-term trend (2.5 months)
        - **SMA 200**: Long-term trend (10 months)
        - **Golden Cross**: When SMA 50 crosses above SMA 200 (bullish)
        - **Death Cross**: When SMA 50 crosses below SMA 200 (bearish)
        
        **2. RSI (Relative Strength Index)**
        - Measures momentum on a scale of 0-100
        - **Below 30**: Oversold (potential buying opportunity)
        - **Above 70**: Overbought (potential selling opportunity)
        - **40-60**: Healthy neutral zone
        
        **3. MACD (Moving Average Convergence Divergence)**
        - Shows relationship between two moving averages
        - **Bullish**: MACD line crosses above signal line
        - **Bearish**: MACD line crosses below signal line
        - Histogram shows strength of momentum
        
        **4. Bollinger Bands**
        - Shows volatility and price levels
        - **Near lower band**: Potentially oversold
        - **Near upper band**: Potentially overbought
        - **Bands widening**: Increased volatility
        
        **5. Support & Resistance**
        - **Support**: Price level where buying interest is strong
        - **Resistance**: Price level where selling pressure is strong
        - Breakout above resistance or below support signals strong moves
        
        **6. Volume Analysis**
        - High volume confirms price movements
        - **Rising price + high volume**: Strong buying (bullish)
        - **Falling price + high volume**: Strong selling (bearish)
        
        ### 🎯 When to Buy (Look for multiple signals)
        1. ✅ Price crossing above SMA 20 with volume
        2. ✅ RSI between 30-50 (oversold recovery)
        3. ✅ MACD bullish crossover
        4. ✅ Price bouncing off support level
        5. ✅ Golden Cross forming (SMA 50 > SMA 200)
        
        ### ⚠️ When to Avoid/Sell
        1. ❌ Price below all major moving averages
        2. ❌ RSI above 70 (overbought)
        3. ❌ MACD bearish crossover
        4. ❌ Price at strong resistance with high volume
        5. ❌ Death Cross forming
        
        ### 💡 Risk Management Tips
        - Never invest more than you can afford to lose
        - Always use stop losses (typically 5-8% below entry)
        - Diversify across sectors
        - Don't trade based on single indicators
        - Combine technical + fundamental analysis
        """)
    
    with st.expander("ℹ️ About Fundamental Filtering Criteria"):
        st.markdown("""
        ### ✅ Mandatory Filters
        - **ROCE > 15%**: Return on Capital Employed indicates efficiency of capital utilization
        - **Positive FCF (3/4Q)**: Free Cash Flow shows real cash generation ability
        - **P/E < 40**: Price to Earnings ratio for reasonable valuation
        - **Profit Growth > 10%**: 3-year profit growth indicates business expansion
        - **Debt/Equity < 1**: Lower leverage reduces financial risk
        
        ### 🎯 Preferred Characteristics
        - **Dividend Payer**: Shows confidence and cash generation
        - **High ROE**: Return on Equity indicates profitability efficiency
        
        ### 📊 Stock Universe
        - **NIFTY 50**: Top 50 companies by market cap
        - **Large Cap**: Additional large-cap stocks beyond NIFTY 50
        - **Mid Cap**: Quality mid-cap stocks with growth potential
        
        ### ⚠️ Disclaimer
        - Data is fetched from Yahoo Finance in real-time
        - Some metrics may not be available for all stocks
        - Technical analysis shows probability, not certainty
        - Always do your own research before investing
        - This tool is for screening purposes only, not investment advice
        - Past performance does not guarantee future results
        """)

if __name__ == "__main__":
    main()
