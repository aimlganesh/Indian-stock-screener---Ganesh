# V32 Screener - Enhanced (Expert factors, performance & robustness)
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import ta
from ta.volatility import AverageTrueRange

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

# --- Stock lists (unchanged) ---
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
    "GUJGASLTD.NS", "UNIONBANK.NS", "GODREJPROP.NS", "PETRONET.NS",
    "INDUSTOWER.NS", "PIIND.NS", "OBEROIRLTY.NS", "IDEA.NS", "OFSS.NS", "MPHASIS.NS",
    "L&TFH.NS", "AUROPHARMA.NS", "IPCALAB.NS", "BALKRISIND.NS", "CROMPTON.NS",
    "ASTRAL.NS", "CONCOR.NS", "COFORGE.NS", "PERSISTENT.NS", "LALPATHLAB.NS",
    "POLYCAB.NS", "BATAINDIA.NS", "MRF.NS", "COLPAL.NS", "SUNPHARMA.NS",
    "LTTS.NS", "TORNTPOWER.NS", "METROPOLIS.NS", "DIXON.NS", "SYNGENE.NS"
]

ALL_STOCKS = list(set(NIFTY_50 + LARGE_CAP_ADDITIONAL + MID_CAP_STOCKS))

# --- Indicator helpers (unchanged fundamentals + technical) ---
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

# --- robust fetch with retries ---
def safe_yf_fetch(ticker, retries=3, pause=1.5):
    for i in range(retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # Basic validation of info
            if info and (('longName' in info) or ('shortName' in info) or ('regularMarketPrice' in info)):
                return stock, info
            # else retry
        except requests.exceptions.RequestException:
            time.sleep(pause)
        except Exception:
            time.sleep(pause)
    return None, {}

# --- technical analysis (enhanced with ATR and 52W checks) ---
def analyze_technical_signals(ticker, period='1y'):
    """Comprehensive technical analysis with ATR and 52-week checks"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, auto_adjust=False)
        if data.empty or len(data) < 50:
            return None

        # Calculate indicators
        data['SMA_20'] = calculate_sma(data, 20)
        data['SMA_50'] = calculate_sma(data, 50)
        data['SMA_200'] = calculate_sma(data, 200)
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data)
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)

        # ATR using ta
        try:
            atr_series = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14).average_true_range()
            data['ATR'] = atr_series
        except Exception:
            data['ATR'] = np.nan

        # 52-week high/low
        if len(data) >= 252:
            data['52W_High'] = data['Close'].rolling(window=252).max()
            data['52W_Low'] = data['Close'].rolling(window=252).min()
        else:
            data['52W_High'] = data['Close'].rolling(window=len(data)).max()
            data['52W_Low'] = data['Close'].rolling(window=len(data)).min()

        current = data.iloc[-1]
        prev = data.iloc[-2]

        signals = {
            'ticker': ticker,
            'current_price': current['Close'],
            'rsi': float(current['RSI']) if pd.notna(current['RSI']) else None,
            'data': data
        }

        # Generate signals (scoring)
        score = 0

        # Trend Analysis - keep similar weighting
        try:
            if (pd.notna(current['Close']) and pd.notna(current['SMA_20']) and pd.notna(current['SMA_50']) and
                current['Close'] > current['SMA_20'] > current['SMA_50']):
                score += 2
            elif pd.notna(current['Close']) and pd.notna(current['SMA_20']) and current['Close'] > current['SMA_20']:
                score += 1
            elif (pd.notna(current['Close']) and pd.notna(current['SMA_20']) and pd.notna(current['SMA_50']) and
                  current['Close'] < current['SMA_20'] < current['SMA_50']):
                score -= 2
        except Exception:
            pass

        # Golden Cross / Death Cross
        try:
            if pd.notna(current['SMA_50']) and pd.notna(current['SMA_200']) and pd.notna(prev['SMA_50']) and pd.notna(prev['SMA_200']):
                if current['SMA_50'] > current['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
                    score += 3
                elif current['SMA_50'] < current['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
                    score -= 3
        except Exception:
            pass

        # RSI Analysis
        if pd.notna(current['RSI']):
            if current['RSI'] < 30:
                score += 2
            elif current['RSI'] > 70:
                score -= 2
            elif 40 <= current['RSI'] <= 60:
                score += 1

        # MACD Analysis - look for recent cross
        try:
            if pd.notna(current['MACD']) and pd.notna(current['MACD_Signal']) and pd.notna(prev['MACD']) and pd.notna(prev['MACD_Signal']):
                if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                    score += 2
                elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                    score -= 2
        except Exception:
            pass

        # 52-week breakout/breakdown checks (small weight)
        try:
            if pd.notna(current.get('52W_High')) and current['Close'] >= 0.98 * current['52W_High']:
                score += 1
            if pd.notna(current.get('52W_Low')) and current['Close'] <= 1.02 * current['52W_Low']:
                score -= 1
        except Exception:
            pass

        # ATR-based volatility note (no direct scoring, could be used later)
        signals['score'] = score

        # Overall recommendation mapping
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

    except Exception:
        return None

# --- plotting (unchanged aside minor safe-calls) ---
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

    # Moving Averages (safe: only plot if exist)
    if 'SMA_20' in chart_data.columns:
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA_20'], name='SMA 20',
                                 line=dict(width=1)), row=1, col=1)
    if 'SMA_50' in chart_data.columns:
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA_50'], name='SMA 50',
                                 line=dict(width=1)), row=1, col=1)

    # RSI
    if 'RSI' in chart_data.columns:
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['RSI'], name='RSI',
                                 line=dict(width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    if 'MACD' in chart_data.columns and 'MACD_Signal' in chart_data.columns:
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD'], name='MACD',
                                 line=dict(width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD_Signal'], name='Signal',
                                 line=dict(width=2)), row=3, col=1)

    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )

    return fig

# --- ROCE calculation (unchanged but safe) ---
def calculate_roce(stock_info):
    try:
        ebit = stock_info.get('ebitda', 0) or 0
        total_assets = stock_info.get('totalAssets', 0) or 0
        current_liabilities = stock_info.get('totalCurrentLiabilities', 0) or 0

        if total_assets > 0 and current_liabilities is not None:
            capital_employed = total_assets - current_liabilities
            if capital_employed > 0:
                roce = (ebit / capital_employed) * 100
                return roce
    except Exception:
        pass
    return None

# --- quarterly cashflow & FCF check (unchanged) ---
def get_quarterly_cashflow(ticker):
    try:
        stock = yf.Ticker(ticker)
        cf = stock.quarterly_cashflow
        if not cf.empty:
            # Return free cash flow if present, otherwise try operatingCashFlow - capitalExpenditures if indices differ
            if 'Free Cash Flow' in cf.index:
                fcf = cf.loc['Free Cash Flow'].head(4)
                return fcf.tolist()
            # fallback
            elif 'Total Cash From Operating Activities' in cf.index and 'Capital Expenditures' in cf.index:
                op_cf = cf.loc['Total Cash From Operating Activities'].head(4)
                capex = cf.loc['Capital Expenditures'].head(4)
                fcf_est = (op_cf - capex).tolist()
                return fcf_est
    except Exception:
        pass
    return []

def check_positive_fcf(fcf_list):
    if len(fcf_list) >= 4:
        positive_count = sum(1 for x in fcf_list[:4] if x is not None and x > 0)
        return positive_count >= 3
    return False

# --- profit growth (unchanged but robust) ---
def get_profit_growth(ticker):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        if not financials.empty:
            # 'Net Income' might be present; fallback to 'Net Income Applicable To Common Shares'
            idx = None
            if 'Net Income' in financials.index:
                idx = 'Net Income'
            else:
                for candidate in ['Net Income Applicable To Common Shares', 'NetIncome']:
                    if candidate in financials.index:
                        idx = candidate
                        break
            if idx:
                net_income = financials.loc[idx]
                # Use last vs 3rd last if available
                if len(net_income) >= 3:
                    recent = net_income.iloc[0]
                    old = net_income.iloc[min(2, len(net_income)-1)]
                    if old != 0 and pd.notna(recent) and pd.notna(old):
                        growth = ((recent - old) / abs(old)) * 100
                        return growth
    except Exception:
        pass
    return None

# --- stock data collector (enhanced with new factors and safe fetch) ---
def get_stock_data(ticker):
    try:
        stock, info = safe_yf_fetch(ticker)
        if not stock or not info:
            return None

        # Basic info + extra fields
        data = {
            'Ticker': ticker,
            'Name': info.get('longName', info.get('shortName', ticker)),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap (Cr)': (info.get('marketCap', 0) / 10000000) if info.get('marketCap') else 0,
            'Current Price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'P/E Ratio': info.get('trailingPE', None),
            'Forward P/E': info.get('forwardPE', None),
            'Debt to Equity': info.get('debtToEquity', None),
            'Dividend Yield (%)': (info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0,
            'ROE (%)': (info.get('returnOnEquity', 0) * 100) if info.get('returnOnEquity') else None,
            'Profit Margin (%)': (info.get('profitMargins', 0) * 100) if info.get('profitMargins') else None,
            'Revenue Growth (%)': (info.get('revenueGrowth', 0) * 100) if info.get('revenueGrowth') else None,
            'EPS (Trailing)': info.get('trailingEps', None),
            'Book Value': info.get('bookValue', None),
            'Price to Book (P/B)': info.get('priceToBook', None),
            'Beta': info.get('beta', None),
            '52W High': info.get('fiftyTwoWeekHigh', None),
            '52W Low': info.get('fiftyTwoWeekLow', None),
            '1Y Target Price': info.get('targetMeanPrice', None),
            'PEG Ratio': None,  # to compute below if possible
            'Revenue (Cr)': (info.get('totalRevenue', 0) / 10000000) if info.get('totalRevenue') else None
        }

        # compute PEG ratio if earningsGrowth available (earningsGrowth is fraction)
        earnings_growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth')
        trailing_pe = info.get('trailingPE')
        try:
            if trailing_pe and earnings_growth and earnings_growth != 0:
                data['PEG Ratio'] = round(trailing_pe / (earnings_growth * 100 if earnings_growth < 2 else earnings_growth), 3)
        except Exception:
            data['PEG Ratio'] = None

        # history for momentum & volatility & avg volume
        hist = None
        try:
            hist = stock.history(period="1y", auto_adjust=False)
        except Exception:
            hist = None

        if hist is not None and not hist.empty:
            # ensure index sorted
            hist = hist.sort_index()

            # momentum returns
            try:
                if len(hist) > 63:
                    data['3M Return (%)'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63]) * 100
                else:
                    data['3M Return (%)'] = None
                if len(hist) > 126:
                    data['6M Return (%)'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-126]) / hist['Close'].iloc[-126]) * 100
                else:
                    data['6M Return (%)'] = None
                data['12M Return (%)'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100 if len(hist) > 1 else None
            except Exception:
                data['3M Return (%)'] = data['6M Return (%)'] = data['12M Return (%)'] = None

            # volatility (annualized)
            try:
                data['Volatility (%)'] = hist['Close'].pct_change().std() * np.sqrt(252) * 100
            except Exception:
                data['Volatility (%)'] = None

            # avg volume 30d
            try:
                data['Average Volume'] = hist['Volume'].tail(30).mean()
            except Exception:
                data['Average Volume'] = None
        else:
            data['3M Return (%)'] = data['6M Return (%)'] = data['12M Return (%)'] = None
            data['Volatility (%)'] = None
            data['Average Volume'] = None

        # ROCE & FCF & profit growth
        data['ROCE (%)'] = calculate_roce(info)
        fcf_list = get_quarterly_cashflow(ticker)
        data['Positive FCF (3/4Q)'] = check_positive_fcf(fcf_list) if fcf_list else None
        data['FCF Data'] = fcf_list
        data['3Y Profit Growth (%)'] = get_profit_growth(ticker)

        return data
    except Exception:
        return None

# --- apply_filters (keeps existing scoring logic + supports new risk filter) ---
def apply_filters(df, filters):
    filtered_df = df.copy()
    filtered_df['Filter_Score'] = 0.0

    # ROCE
    if filters.get('roce', False):
        roce_mask = (filtered_df['ROCE (%)'].notna()) & (filtered_df['ROCE (%)'] >= filters['roce_min'])
        filtered_df.loc[roce_mask, 'Filter_Score'] += 2
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[roce_mask]

    # FCF
    if filters.get('fcf', False):
        fcf_mask = filtered_df['Positive FCF (3/4Q)'] == True
        filtered_df.loc[fcf_mask, 'Filter_Score'] += 2
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[fcf_mask]

    # PE
    if filters.get('pe', False):
        pe_mask = (filtered_df['P/E Ratio'].notna()) & (filtered_df['P/E Ratio'] < filters['pe_max']) & (filtered_df['P/E Ratio'] > 0)
        filtered_df.loc[pe_mask, 'Filter_Score'] += 1
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[pe_mask]

    # Profit growth
    if filters.get('profit_growth', False):
        pg_mask = (filtered_df['3Y Profit Growth (%)'].notna()) & (filtered_df['3Y Profit Growth (%)'] >= filters['profit_growth_min'])
        filtered_df.loc[pg_mask, 'Filter_Score'] += 2
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[pg_mask]

    # Debt/Equity
    if filters.get('debt_equity', False):
        de_mask = (filtered_df['Debt to Equity'].notna()) & (filtered_df['Debt to Equity'] < filters['debt_equity_max'])
        filtered_df.loc[de_mask, 'Filter_Score'] += 1
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[de_mask]

    # Dividend
    if filters.get('dividend', False):
        div_mask = filtered_df['Dividend Yield (%)'] > 0
        filtered_df.loc[div_mask, 'Filter_Score'] += 0.5

    # ROE
    if filters.get('roe', False):
        roe_mask = (filtered_df['ROE (%)'].notna()) & (filtered_df['ROE (%)'] >= filters['roe_min'])
        filtered_df.loc[roe_mask, 'Filter_Score'] += 1
        if filters.get('strict_mode', False):
            filtered_df = filtered_df[roe_mask]

    # Risk / Volatility filter (optional)
    if filters.get('risk', False):
        if 'Volatility (%)' in filtered_df.columns:
            vol_mask = (filtered_df['Volatility (%)'].notna()) & (filtered_df['Volatility (%)'] <= filters['max_vol'])
            filtered_df.loc[vol_mask, 'Filter_Score'] += 0.5
            if filters.get('strict_mode', False):
                filtered_df = filtered_df[vol_mask]

    if not filters.get('strict_mode', False):
        min_score = filters.get('min_score', 3)
        filtered_df = filtered_df[filtered_df['Filter_Score'] >= min_score]

    # Expert composite internal score (backend only): combination of Filter_Score + growth/ROCE - leverage
    def compute_expert_score(row):
        fs = row.get('Filter_Score', 0) or 0
        g = row.get('3Y Profit Growth (%)', 0) or 0
        roce = row.get('ROCE (%)', 0) or 0
        de = row.get('Debt to Equity', 0) or 0
        # scaled contributions
        expert = fs + (g / 10.0) + (roce / 10.0) - de
        return expert

    filtered_df['Expert_Score'] = filtered_df.apply(compute_expert_score, axis=1)

    filtered_df = filtered_df.sort_values(['Filter_Score', 'Expert_Score'], ascending=False)

    return filtered_df

# --- parallel fetch helper for performance ---
def fetch_all_stocks(stock_list, max_workers=10):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_stock_data, t): t for t in stock_list}
        for f in as_completed(futures):
            try:
                res = f.result()
                if res:
                    results.append(res)
            except Exception:
                pass
    return results

# --- main app flow (kept UI same, added risk slider and performance improvements) ---
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'setup'
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'tech_analysis' not in st.session_state:
        st.session_state.tech_analysis = None
    if 'screening_timestamp' not in st.session_state:
        st.session_state.screening_timestamp = None
    if 'tech_timestamp' not in st.session_state:
        st.session_state.tech_timestamp = None
    if 'individual_refresh_times' not in st.session_state:
        st.session_state.individual_refresh_times = {}

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

        # New: Risk / Volatility filter (optional)
        st.sidebar.subheader("⚖ Risk Filters (Optional)")
        filters['risk'] = st.sidebar.checkbox("Apply Volatility Filter", value=False)
        if filters['risk']:
            filters['max_vol'] = st.sidebar.slider("Max Annual Volatility (%)", 10, 80, 40, 5)

        st.sidebar.subheader("⚙ Filter Mode")
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
            st.info("Fetching stock data... This may take several minutes (we fetch in parallel).")

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Use parallel fetch for speed
            all_data = []
            try:
                results = fetch_all_stocks(stock_list, max_workers=10)
                total = len(results)
                for idx, data in enumerate(results):
                    status_text.text(f"Processing {data.get('Ticker','') if data else '...'}... ({idx+1}/{total})")
                    if data:
                        all_data.append(data)
                    progress_bar.progress((idx + 1) / max(1, total))
                    # no arbitrary sleep here; ThreadPoolExecutor handled concurrency
            except Exception:
                # fallback to sequential if parallel fails
                all_data = []
                for idx, ticker in enumerate(stock_list):
                    status_text.text(f"Processing {ticker}... ({idx+1}/{len(stock_list)})")
                    data = get_stock_data(ticker)
                    if data:
                        all_data.append(data)
                    progress_bar.progress((idx + 1) / len(stock_list))
                    time.sleep(0.2)

            status_text.empty()
            progress_bar.empty()

            if all_data:
                df = pd.DataFrame(all_data)
                # Ensure numeric types where possible
                # Use apply(pd.to_numeric, errors='ignore') carefully
                filtered_df = apply_filters(df, filters)
                st.session_state.filtered_data = filtered_df
                st.session_state.screening_timestamp = datetime.now()
                st.session_state.page = 'results'
                st.rerun()
            else:
                st.error("❌ Unable to fetch data")

    # Results Page
    elif st.session_state.page == 'results':
        # Sidebar
        st.sidebar.header("📊 Screening Complete")

        # Show data freshness
        if st.session_state.screening_timestamp:
            time_elapsed = datetime.now() - st.session_state.screening_timestamp
            minutes_ago = int(time_elapsed.total_seconds() / 60)
            hours_ago = int(minutes_ago / 60)

            if hours_ago > 0:
                freshness_text = f"🕐 Data fetched {hours_ago}h {minutes_ago % 60}m ago"
            else:
                freshness_text = f"🕐 Data fetched {minutes_ago}m ago"

            st.sidebar.info(freshness_text)
            st.sidebar.caption(f"Last updated: {st.session_state.screening_timestamp.strftime('%I:%M %p, %d %b %Y')}")

        if st.sidebar.button("🔄 New Screening", type="primary", use_container_width=True):
            st.session_state.page = 'setup'
            st.session_state.filtered_data = None
            st.session_state.tech_analysis = None
            st.session_state.screening_timestamp = None
            st.session_state.tech_timestamp = None
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

            # Show timestamp
            if st.session_state.screening_timestamp:
                time_elapsed = datetime.now() - st.session_state.screening_timestamp
                minutes_ago = int(time_elapsed.total_seconds() / 60)
                if minutes_ago < 60:
                    st.caption(f"⏱ Fundamental data age: {minutes_ago} minutes old")
                else:
                    hours_ago = int(minutes_ago / 60)
                    st.caption(f"⏱ Fundamental data age: {hours_ago}h {minutes_ago % 60}m old")

            st.subheader("✅ Filtered Stocks")
            display_df = filtered_df.drop('FCF Data', axis=1).copy()
            cols = ['Filter_Score', 'Expert_Score', 'Ticker', 'Name', 'Sector'] + [col for col in display_df.columns if col not in ['Filter_Score', 'Ticker', 'Name', 'Sector', 'Expert_Score']]
            display_df = display_df[cols]

            st.dataframe(
                display_df.style.format({
                    'Filter_Score': '{:.1f}',
                    'Expert_Score': '{:.2f}',
                    'Market Cap (Cr)': '{:.0f}',
                    'Current Price': '{:.2f}',
                    'P/E Ratio': '{:.2f}',
                    'Forward P/E': '{:.2f}',
                    'Debt to Equity': '{:.2f}',
                    'ROCE (%)': '{:.2f}',
                    'ROE (%)': '{:.2f}',
                    'Profit Margin (%)': '{:.2f}',
                    'Revenue Growth (%)': '{:.2f}',
                    '3Y Profit Growth (%)': '{:.2f}',
                    'Dividend Yield (%)': '{:.2f}',
                    'Volatility (%)': '{:.2f}',
                    '3M Return (%)': '{:.2f}',
                    '6M Return (%)': '{:.2f}',
                    '12M Return (%)': '{:.2f}'
                }, na_rep='N/A'),
                height=400
            )
        else:
            st.warning("No stocks passed filters")

    # Technical Analysis Page
    elif st.session_state.page == 'tech_analysis':
        st.sidebar.header("📊 Technical Analysis")

        # Show data freshness in sidebar
        if st.session_state.tech_timestamp:
            time_elapsed = datetime.now() - st.session_state.tech_timestamp
            minutes_ago = int(time_elapsed.total_seconds() / 60)
            hours_ago = int(minutes_ago / 60)

            if hours_ago > 0:
                freshness_text = f"🕐 Technical data: {hours_ago}h {minutes_ago % 60}m ago"
            else:
                freshness_text = f"🕐 Technical data: {minutes_ago}m ago"

            st.sidebar.info(freshness_text)
            st.sidebar.caption(f"Last analyzed: {st.session_state.tech_timestamp.strftime('%I:%M %p, %d %b %Y')}")

        if st.sidebar.button("⬅ Back to Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
        if st.sidebar.button("🔄 New Screening", use_container_width=True):
            st.session_state.page = 'setup'
            st.session_state.filtered_data = None
            st.session_state.tech_analysis = None
            st.session_state.screening_timestamp = None
            st.session_state.tech_timestamp = None
            st.session_state.individual_refresh_times = {}
            st.rerun()

        if st.sidebar.button("🔄 Refresh All Technical Data", use_container_width=True):
            st.session_state.tech_analysis = None
            st.session_state.tech_timestamp = None
            st.session_state.individual_refresh_times = {}
            st.rerun()

        filtered_df = st.session_state.filtered_data

        # Generate technical analysis if not already done
        if st.session_state.tech_analysis is None:
            st.info("Analyzing technical signals for all stocks...")
            progress_bar = st.progress(0)
            tech_results = []

            # iterate rows for technical analysis
            rows = filtered_df.to_dict('records')
            total = len(rows)
            for idx, row in enumerate(rows):
                ticker = row['Ticker']
                tech = analyze_technical_signals(ticker)
                if tech:
                    tech_results.append({
                        'Ticker': ticker,
                        'Name': row.get('Name', ''),
                        'Current Price': tech.get('current_price', None),
                        'RSI': tech.get('rsi', None),
                        'Signal': tech.get('signal', None),
                        'Recommendation': tech.get('recommendation', None),
                        'Score': tech.get('score', 0),
                        'data': tech.get('data')
                    })
                progress_bar.progress((idx + 1) / max(1, total))
                time.sleep(0.05)

            progress_bar.empty()
            st.session_state.tech_analysis = pd.DataFrame(tech_results)
            st.session_state.tech_timestamp = datetime.now()
            st.success("✅ Technical analysis complete!")

        tech_df = st.session_state.tech_analysis

        # Show data age
        if st.session_state.tech_timestamp:
            time_elapsed = datetime.now() - st.session_state.tech_timestamp
            minutes_ago = int(time_elapsed.total_seconds() / 60)

            if minutes_ago < 1:
                age_text = "⏱ Technical data: Less than 1 minute old (Live)"
                age_color = "green"
            elif minutes_ago < 5:
                age_text = f"⏱ Technical data: {minutes_ago} minutes old (Recent)"
                age_color = "green"
            elif minutes_ago < 15:
                age_text = f"⏱ Technical data: {minutes_ago} minutes old"
                age_color = "orange"
            else:
                hours_ago = int(minutes_ago / 60)
                if hours_ago > 0:
                    age_text = f"⏱ Technical data: {hours_ago}h {minutes_ago % 60}m old (Consider refreshing)"
                else:
                    age_text = f"⏱ Technical data: {minutes_ago} minutes old (Consider refreshing)"
                age_color = "red"

            st.markdown(f":{age_color}[{age_text}]")

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

        # Score formatting: show as integer
        st.dataframe(
            summary_df.style.applymap(color_signal, subset=['Signal']).format({
                'Current Price': '₹{:.2f}',
                'RSI': '{:.1f}',
                'Score': '{:.0f}'
            }),
            height=400,
            use_container_width=True
        )

        # Download with metadata
        csv_metadata = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nStocks screened: {len(filtered_df)}\n"
        csv_body = summary_df.to_csv(index=False)
        csv = f"# Indian Stock Screener (Expert Mode)\n# {csv_metadata}\n{csv_body}"
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

        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox("Select stock for detailed view:", stock_names, key='stock_detail_selector')

        if selected:
            ticker = selected.split(' - ')[0]

            # Refresh button for individual stock
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button(f"🔄 Refresh {ticker}", use_container_width=True, key=f'refresh_{ticker}'):
                    # Refresh this specific stock's data
                    with st.spinner(f"Refreshing {ticker}..."):
                        tech = analyze_technical_signals(ticker)
                        if tech:
                            # Update in the dataframe
                            idx = tech_df[tech_df['Ticker'] == ticker].index[0]
                            tech_df.loc[idx, 'Current Price'] = tech['current_price']
                            tech_df.loc[idx, 'RSI'] = tech['rsi']
                            tech_df.loc[idx, 'Signal'] = tech['signal']
                            tech_df.loc[idx, 'Recommendation'] = tech['recommendation']
                            tech_df.loc[idx, 'Score'] = tech['score']
                            tech_df.loc[idx, 'data'] = tech.get('data')

                            st.session_state.tech_analysis = tech_df
                            st.session_state.individual_refresh_times[ticker] = datetime.now()
                            st.success(f"✅ {ticker} refreshed!")
                            st.rerun()

            stock_tech = tech_df[tech_df['Ticker'] == ticker].iloc[0]
            stock_fund = filtered_df[filtered_df['Ticker'] == ticker].iloc[0]

            # Show individual stock data age
            if ticker in st.session_state.individual_refresh_times:
                last_refresh = st.session_state.individual_refresh_times[ticker]
                time_diff = datetime.now() - last_refresh
                seconds_ago = int(time_diff.total_seconds())
                if seconds_ago < 60:
                    st.success(f"🕐 {ticker} data: {seconds_ago} seconds old (Just refreshed!)")
                else:
                    minutes_ago = int(seconds_ago / 60)
                    st.info(f"🕐 {ticker} data: {minutes_ago} minutes old | Last refreshed: {last_refresh.strftime('%I:%M:%S %p')}")
            elif st.session_state.tech_timestamp:
                time_diff = datetime.now() - st.session_state.tech_timestamp
                minutes_ago = int(time_diff.total_seconds() / 60)
                st.info(f"🕐 {ticker} data: {minutes_ago} minutes old (from initial scan)")

            rec_class = "buy-signal" if "BUY" in stock_tech['Recommendation'] else "sell-signal" if "SELL" in stock_tech['Recommendation'] else "neutral-signal"
            st.markdown(f'<div class="{rec_class}">Recommendation: {stock_tech["Recommendation"]} (Score: {stock_tech["Score"]})</div>', unsafe_allow_html=True)

            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{stock_tech['Current Price']:.2f}")
            with col2:
                st.metric("RSI", f"{stock_tech['RSI']:.1f}" if pd.notna(stock_tech['RSI']) else "N/A")
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
    with st.expander("ℹ Understanding Technical Indicators"):
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

        ### ⚠ Important Notes
        - Technical analysis shows **probability**, not certainty
        - Always combine with fundamental analysis
        - Use proper risk management (stop losses)
        - Diversify across sectors
        - This is for educational purposes only
        """)

    with st.expander("ℹ About Fundamental Filtering"):
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

        ### ⚠ Disclaimer
        - Data from Yahoo Finance (may have limitations)
        - For educational and screening purposes only
        - Always do your own research
        - Consult financial advisor before investing
        - Past performance ≠ future results
        """)

if __name__ == "__main__":
    main()
