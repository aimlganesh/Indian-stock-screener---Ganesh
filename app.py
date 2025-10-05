import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import time

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
    .criteria-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .pass-criteria {
        color: #28a745;
        font-weight: bold;
    }
    .fail-criteria {
        color: #dc3545;
        font-weight: bold;
    }
    .warning-criteria {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# NSE Stock List (Top stocks - you can expand this)
DEFAULT_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "POWERGRID.NS", "NTPC.NS",
    "TATASTEEL.NS", "ADANIPORTS.NS", "ONGC.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "BAJAJFINSV.NS", "DRREDDY.NS", "EICHERMOT.NS", "HINDALCO.NS", "JSWSTEEL.NS",
    "M&M.NS", "BRITANNIA.NS", "CIPLA.NS", "GRASIM.NS", "HEROMOTOCO.NS",
    "INDUSINDBK.NS", "APOLLOHOSP.NS", "ADANIENT.NS", "TATAMOTORS.NS",
    "BAJAJ-AUTO.NS", "PIDILITIND.NS", "HAVELLS.NS", "DABUR.NS"
]

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
        
        # Basic info
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
        
        # Calculate ROCE
        roce = calculate_roce(info)
        data['ROCE (%)'] = roce
        
        # Get FCF data
        fcf_list = get_quarterly_cashflow(ticker)
        data['Positive FCF (3/4Q)'] = check_positive_fcf(fcf_list) if fcf_list else None
        data['FCF Data'] = fcf_list
        
        # Get profit growth
        profit_growth = get_profit_growth(ticker)
        data['3Y Profit Growth (%)'] = profit_growth
        
        return data
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None

def apply_filters(df, filters):
    """Apply filtering criteria"""
    filtered_df = df.copy()
    
    # Mandatory filters
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
    
    # Preferred filters
    if filters['dividend']:
        filtered_df = filtered_df[filtered_df['Dividend Yield (%)'] > 0]
    
    if filters['roe']:
        filtered_df = filtered_df[
            (filtered_df['ROE (%)'].notna()) & 
            (filtered_df['ROE (%)'] >= filters['roe_min'])
        ]
    
    return filtered_df

def main():
    st.markdown('<h1 class="main-header">📈 Indian Stock Screener</h1>', unsafe_allow_html=True)
    st.markdown("Filter Indian stocks based on fundamental criteria for quality investing")
    
    # Sidebar - Filters
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
    
    # Stock selection
    st.sidebar.subheader("📊 Stock Selection")
    use_custom = st.sidebar.checkbox("Add custom tickers")
    
    stock_list = DEFAULT_STOCKS.copy()
    
    if use_custom:
        custom_stocks = st.sidebar.text_area(
            "Enter ticker symbols (one per line, must end with .NS or .BO)",
            "Example:\nINFY.NS\nTCS.NS"
        )
        if custom_stocks:
            custom_list = [s.strip() for s in custom_stocks.split('\n') if s.strip()]
            stock_list.extend(custom_list)
    
    # Main content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stocks to Scan", len(stock_list))
    
    if st.button("🔍 Start Screening", type="primary"):
        st.info("Fetching stock data... This may take a few minutes.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_data = []
        for idx, ticker in enumerate(stock_list):
            status_text.text(f"Processing {ticker}... ({idx+1}/{len(stock_list)})")
            data = get_stock_data(ticker)
            if data:
                all_data.append(data)
            progress_bar.progress((idx + 1) / len(stock_list))
            time.sleep(0.5)  # Rate limiting
        
        status_text.empty()
        progress_bar.empty()
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            st.success(f"✅ Successfully fetched data for {len(df)} stocks")
            
            # Display unfiltered data
            st.subheader("📊 All Stocks Data")
            st.dataframe(
                df.drop('FCF Data', axis=1).style.format({
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
            
            # Apply filters
            filtered_df = apply_filters(df, filters)
            
            col1, col2, col3 = st.columns(3)
            with col2:
                st.metric("Stocks Passing Filters", len(filtered_df), f"{len(filtered_df)/len(df)*100:.1f}%")
            
            if not filtered_df.empty:
                st.subheader("✅ Filtered Stocks (Meeting Criteria)")
                
                # Display filtered results
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
                
                # Download option
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Results as CSV",
                    data=csv,
                    file_name=f"filtered_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Detailed view
                st.subheader("🔍 Detailed Stock Analysis")
                selected_stock = st.selectbox(
                    "Select a stock for detailed view",
                    filtered_df['Ticker'].tolist()
                )
                
                if selected_stock:
                    stock_data = filtered_df[filtered_df['Ticker'] == selected_stock].iloc[0]
                    
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
                    
                    # FCF Analysis
                    if stock_data['FCF Data']:
                        st.subheader("Quarterly Free Cash Flow")
                        fcf_df = pd.DataFrame({
                            'Quarter': [f'Q{i+1}' for i in range(len(stock_data['FCF Data']))],
                            'FCF': stock_data['FCF Data']
                        })
                        st.bar_chart(fcf_df.set_index('Quarter'))
            else:
                st.warning("⚠️ No stocks passed the filtering criteria. Try relaxing some filters.")
        else:
            st.error("❌ Unable to fetch data for any stocks. Please check your internet connection and try again.")
    
    # Information section
    with st.expander("ℹ️ About the Filtering Criteria"):
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
        
        ### ⚠️ Note
        - Data is fetched from Yahoo Finance
        - Some metrics may not be available for all stocks
        - Always do your own research before investing
        - This tool is for screening purposes only, not investment advice
        """)

if __name__ == "__main__":
    main()
