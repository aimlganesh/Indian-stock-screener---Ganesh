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

# Comprehensive NSE Stock List
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
    return data['Close'].rolling(window=period).mean()

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

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
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data)
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = {
            'ticker': ticker,
            'current_price': current['Close'],
            'rsi': current['RSI'],
            'data': data
        }
        
        # Generate signals
        score = 0
        
        # Trend Analysis
        if current['Close'] > current['SMA_20'] > current['SMA_50']:
            score += 2
        elif current['Close'] > current['SMA_20']:
            score += 1
        elif current['Close'] < current['SMA_20'] < current['SMA_50']:
            score -= 2
        
        # Golden Cross / Death Cross
        if current['SMA_50'] > current['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
            score += 3
        elif current['SMA_50'] < current['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
            score -= 3
        
        # RSI Analysis
        if current['RSI'] < 30:
            score += 2
        elif current['RSI'] > 70:
            score -= 2
        elif 40 <= current['RSI'] <= 60:
            score += 1
        
        # MACD Analysis
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            score += 2
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            score -= 2
        
        signals['score'] = score
        
        # Overall recommendation
        if score >= 4:
            signals['recommendation'] = "STRONG BUY"
            signals['signal'] = "BUY"
        elif score >= 2:
            signals['recommendation'] = "BUY"
            signals['signal'] = "BUY"
        elif score <= -4:
            signals['recommendation'] = "STRONG SELL"
            signals['signal'] = "SELL"
        elif score <= -2:
            signals['recommendation'] = "SELL"
            signals['signal'] = "SELL"
        else:
            signals['recommendation'] = "HOLD"
            signals['signal'] = "HOLD"
        
        return signals
        
    except Exception as e:
        return None

def plot_candlestick_chart(ticker, data):
    """Create interactive candlestick chart"""
    chart_data = data.tail(100)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Indicators', 'RSI', 'MACD')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data['Open'],
        high=chart_data['High'],
        low=chart_data['Low'],
        close=chart_data['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['RSI'], name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD'], name='MACD', line=dict(color='blue', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD_Signal'], name='Signal', line=dict(color='red', width=2)), row=3, col=1)
    
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig

def calculate_roce(stock_info):
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
    if len(fcf_list) >= 4:
        positive_count = sum(1 for x in fcf_list[:4] if x > 0)
        return positive_count >= 3
    return False

def get_profit_growth(ticker):
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
    filtered_df = df.copy()
    filtered_df['Filter_Score'] = 0
    
    if filters['roce']:
        roce_mask = (filtered_df['ROCE (%)'].notna()) & (filtered_df['ROCE (%)'] >= filters['roce_min'])
        filtered_df.loc[roce_mask, 'Filter_Score'] += 2
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[roce_mask]
    
    if filters['fcf']:
        fcf_mask = filtered_df['Positive FCF (3/4Q)'] == True
        filtered_df.loc[fcf_mask, 'Filter_Score'] += 2
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[fcf_mask]
    
    if filters['pe']:
        pe_mask = (filtered_df['P/E Ratio'].notna()) & (filtered_df['P/E Ratio'] < filters['pe_max']) & (filtered_df['P/E Ratio'] > 0)
        filtered_df.loc[pe_mask, 'Filter_Score'] += 1
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[pe_mask]
    
    if filters['profit_growth']:
        pg_mask = (filtered_df['3Y Profit Growth (%)'].notna()) & (filtered_df['3Y Profit Growth (%)'] >= filters['profit_growth_min'])
        filtered_df.loc[pg_mask, 'Filter_Score'] += 2
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[pg_mask]
    
    if filters['debt_equity']:
        de_mask = (filtered_df['Debt to Equity'].notna()) & (filtered_df['Debt to Equity'] < filters['debt_equity_max'])
        filtered_df.loc[de_mask, 'Filter_Score'] += 1
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[de_mask]
    
    if filters['dividend']:
        div_mask = filtered_df['Dividend Yield (%)'] > 0
        filtered_df.loc[div_mask, 'Filter_Score'] += 0.5
    
    if filters['roe']:
        roe_mask = (filtered_df['ROE (%)'].notna()) & (filtered_df['ROE (%)'] >= filters['roe_min'])
        filtered_df.loc[roe_mask, 'Filter_Score'] += 1
    
    if not filters.get('strict_mode', False):
        min_score = filters.get('min_score', 3)
        filtered_df = filtered_df[filtered_df['Filter_Score'] >= min_score]
    
    filtered_df = filtered_df.sort_values('Filter_Score', ascending=False)
    
    return filtered_df

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'setup'
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'tech_analysis' not in st.session_state:
        st.session_state.tech_analysis = None
    
    st.markdown('<h1 class="main-header">📈 Indian Stock Screener with Technical Analysis</h1>', unsafe_allow_html=True)
    
    # Setup Page
    if st.session_state.page == 'setup':
        st.markdown("Filter Indian stocks (NIFTY 50 + Large Cap + Mid Cap) with fundamental & technical analysis")
        
        # Sidebar Filters
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
        
        st.sidebar.subheader("⚙️ Filter Mode")
        filter_mode = st.sidebar.radio(
            "Select Filtering Mode",
            ["Flexible (Recommended)", "Strict (All criteria must pass)"]
        )
        filters['strict_mode'] = (filter_mode == "Strict (All criteria must pass)")
        
        if not filters['strict_mode']:
            filters['min_score'] = st.sidebar.slider("Minimum Score (out of 8.5)", 1.0, 8.5, 3.0, 0.5)
        
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
        
        # Custom stocks option
        use_custom = st.sidebar.checkbox("Add custom tickers")
        if use_custom:
            custom_stocks = st.sidebar.text_area(
                "Enter ticker symbols (one per line, with .NS suffix)",
                placeholder="Example:\nRELIANCE.NS\nTCS.NS\nINFY.NS",
                height=100
            )
            if custom_stocks:
                custom_list = [s.strip() for s in custom_stocks.split('\n') if s.strip()]
                stock_list.extend(custom_list)
                stock_list = list(set(stock_list))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks to Scan", len(stock_list))
        with col2:
            st.metric("NIFTY 50", len([s for s in stock_list if s in NIFTY_50]))
        with col3:
            st.metric("Large+Mid Cap", len([s for s in stock_list if s not in NIFTY_50]))
        
        if st.button("🔍 Start Screening", type="primary", use_container_width=True):
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
                filtered_df = apply_filters(df, filters)
                st.session_state.filtered_data = filtered_df
                st.session_state.page = 'results'
                st.rerun()
            else:
                st.error("❌ Unable to fetch data")
    
    # Results Page
    elif st.session_state.page == 'results':
        # Sidebar
        st.sidebar.header("📊 Screening Complete")
        if st.sidebar.button("🔄 New Screening", type="primary", use_container_width=True):
            st.session_state.page = 'setup'
            st.session_state.filtered_data = None
            st.session_state.tech_analysis = None
            st.rerun()
        
        filtered_df = st.session_state.filtered_data
        
        if filtered_df is not None and not filtered_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stocks Passing Filters", len(filtered_df))
            with col2:
                if st.button("📊 Generate Technical Analysis for All Stocks", use_container_width=True):
                    st.session_state.page = 'tech_analysis'
                    st.rerun()
            
            st.subheader("✅ Filtered Stocks")
            display_df = filtered_df.drop('FCF Data', axis=1).copy()
            cols = ['Filter_Score', 'Ticker', 'Name', 'Sector'] + [col for col in display_df.columns if col not in ['Filter_Score', 'Ticker', 'Name', 'Sector']]
            display_df = display_df[cols]
            
            st.dataframe(
                display_df.style.format({
                    'Filter_Score': '{:.1f}',
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
        else:
            st.warning("No stocks passed filters")
    
    # Technical Analysis Page
    elif st.session_state.page == 'tech_analysis':
        st.sidebar.header("📊 Technical Analysis")
        if st.sidebar.button("⬅️ Back to Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
        if st.sidebar.button("🔄 New Screening", use_container_width=True):
            st.session_state.page = 'setup'
            st.session_state.filtered_data = None
            st.session_state.tech_analysis = None
            st.rerun()
        
        filtered_df = st.session_state.filtered_data
        
        # Generate technical analysis if not already done
        if st.session_state.tech_analysis is None:
            st.info("Analyzing technical signals for all stocks...")
            progress_bar = st.progress(0)
            tech_results = []
            
            for idx, row in filtered_df.iterrows():
                ticker = row['Ticker']
                tech = analyze_technical_signals(ticker)
                if tech:
                    tech_results.append({
                        'Ticker': ticker,
                        'Name': row['Name'],
                        'Current Price': tech['current_price'],
                        'RSI': tech['rsi'],
                        'Signal': tech['signal'],
                        'Recommendation': tech['recommendation'],
                        'Score': tech['score'],
                        'data': tech.get('data')
                    })
                progress_bar.progress((len(tech_results)) / len(filtered_df))
                time.sleep(0.2)
            
            progress_bar.empty()
            st.session_state.tech_analysis = pd.DataFrame(tech_results)
            st.success("✅ Technical analysis complete!")
        
        tech_df = st.session_state.tech_analysis
        
        # Summary Table
        st.subheader("📊 Technical Analysis Summary - All Stocks")
        
        summary_df = tech_df[['Ticker', 'Name', 'Current Price', 'RSI', 'Signal', 'Recommendation', 'Score']].copy()
        
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda'
            elif val == 'SELL':
                return 'background-color: #f8d7da'
            else:
                return 'background-color: #fff3cd'
        
        st.dataframe(
            summary_df.style.applymap(color_signal, subset=['Signal']).format({
                'Current Price': '₹{:.2f}',
                'RSI': '{:.1f}',
                'Score': '{:+d}'
            }),
            height=400,
            use_container_width=True
        )
        
        # Download
        csv = summary_df.to_csv(index=False)
        st.download_button(
            "📥 Download Technical Analysis",
            csv,
            f"technical_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
        
        # Detailed Analysis
        st.markdown("---")
        st.subheader("📈 Detailed Stock Analysis")
        
        stock_names = [f"{row['Ticker']} - {row['Name']}" for _, row in tech_df.iterrows()]
        selected = st.selectbox("Select stock for detailed view:", stock_names, key='stock_detail_selector')
        
        if selected:
            ticker = selected.split(' - ')[0]
            stock_tech = tech_df[tech_df['Ticker'] == ticker].iloc[0]
            stock_fund = filtered_df[filtered_df['Ticker'] == ticker].iloc[0]
            
            rec_class = "buy-signal" if "BUY" in stock_tech['Recommendation'] else "sell-signal" if "SELL" in stock_tech['Recommendation'] else "neutral-signal"
            st.markdown(f'<div class="{rec_class}">Recommendation: {stock_tech["Recommendation"]} (Score: {stock_tech["Score"]})</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{stock_tech['Current Price']:.2f}")
            with col2:
                st.metric("RSI", f"{stock_tech['RSI']:.1f}")
            with col3:
                st.metric("P/E Ratio", f"{stock_fund['P/E Ratio']:.2f}" if pd.notna(stock_fund['P/E Ratio']) else "N/A")
            with col4:
                st.metric("ROCE", f"{stock_fund['ROCE (%)']:.2f}%" if pd.notna(stock_fund['ROCE (%)']) else "N/A")
            
            # Chart
            if stock_tech['data'] is not None and not stock_tech['data'].empty:
                st.markdown("---")
                st.subheader("📈 Interactive Chart")
                chart = plot_candlestick_chart(ticker, stock_tech['data'])
                st.plotly_chart(chart, use_container_width=True)
            
            # Fundamental Metrics
            st.markdown("---")
            st.subheader("📊 Fundamental Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filter Score", f"{stock_fund['Filter_Score']:.1f}/8.5")
                st.metric("Debt/Equity", f"{stock_fund['Debt to Equity']:.2f}" if pd.notna(stock_fund['Debt to Equity']) else "N/A")
            with col2:
                st.metric("3Y Profit Growth", f"{stock_fund['3Y Profit Growth (%)']:.2f}%" if pd.notna(stock_fund['3Y Profit Growth (%)']) else "N/A")
                st.metric("ROE", f"{stock_fund['ROE (%)']:.2f}%" if pd.notna(stock_fund['ROE (%)']) else "N/A")
            with col3:
                st.metric("Market Cap", f"₹{stock_fund['Market Cap (Cr)']:.0f} Cr")
                st.metric("Dividend Yield", f"{stock_fund['Dividend Yield (%)']:.2f}%")
    
    # Information sections (always visible)
    with st.expander("ℹ️ Understanding Technical Indicators"):
        st.markdown("""
        ### 📊 Key Technical Indicators
        
        **1. RSI (Relative Strength Index)**
        - **Below 30**: Oversold (potential buying opportunity)
        - **Above 70**: Overbought (potential selling opportunity)
        - **40-60**: Healthy neutral zone
        
        **2. Moving Averages**
        - **SMA 20**: Short-term trend (1 month)
        - **SMA 50**: Medium-term trend (2.5 months)
        - **Golden Cross**: SMA 50 crosses above SMA 200 (bullish)
        - **Death Cross**: SMA 50 crosses below SMA 200 (bearish)
        
        **3. MACD**
        - **Bullish**: MACD line crosses above signal line
        - **Bearish**: MACD line crosses below signal line
        
        **4. Signal Interpretation**
        - **STRONG BUY**: Score ≥ 4 (Multiple bullish signals)
        - **BUY**: Score 2-3 (Some bullish signals)
        - **HOLD**: Score -1 to 1 (Neutral)
        - **SELL**: Score -2 to -3 (Some bearish signals)
        - **STRONG SELL**: Score ≤ -4 (Multiple bearish signals)
        
        ### 🎯 How to Use This Analysis
        
        1. **Check Summary Table**: Get overview of all stocks' signals
        2. **Filter by Signal**: Focus on stocks with BUY signals
        3. **Check RSI**: Avoid stocks with RSI > 70 (overbought)
        4. **Verify Fundamentals**: Ensure Filter Score is high (> 5)
        5. **View Detailed Chart**: Confirm trend and support levels
        
        ### ⚠️ Important Notes
        - Technical analysis shows **probability**, not certainty
        - Always combine with fundamental analysis
        - Use proper risk management (stop losses)
        - Diversify across sectors
        - This is for educational purposes only
        """)
    
    with st.expander("ℹ️ About Fundamental Filtering"):
        st.markdown("""
        ### ✅ Filtering Criteria
        
        **Mandatory Filters:**
        - **ROCE > 15%**: Efficient capital utilization
        - **Positive FCF (3/4Q)**: Real cash generation
        - **P/E < 40**: Reasonable valuation
        - **Profit Growth > 10%**: Business expansion
        - **Debt/Equity < 1**: Lower financial risk
        
        **Scoring System (Flexible Mode):**
        - ROCE: 2 points
        - Positive FCF: 2 points
        - Profit Growth: 2 points
        - P/E Ratio: 1 point
        - Debt/Equity: 1 point
        - ROE: 1 point (if enabled)
        - Dividend: 0.5 points (if enabled)
        
        **Total: 8.5 points maximum**
        
        ### 💡 Investment Strategy
        
        1. **High Filter Score (6-8)** + **BUY Signal** = Strong candidate
        2. **Medium Score (4-5)** + **BUY Signal** = Good candidate
        3. **Any Score** + **SELL Signal** = Avoid or book profits
        4. **High Score** + **HOLD** = Monitor for entry
        
        ### ⚠️ Disclaimer
        - Data from Yahoo Finance (may have limitations)
        - For educational and screening purposes only
        - Always do your own research
        - Consult financial advisor before investing
        - Past performance ≠ future results
        """)

if __name__ == "__main__":
    main()
