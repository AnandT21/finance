import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_market_calendars as pmc
from datetime import datetime, timedelta
from plotly import graph_objects as go
import pytz
from pandas.tseries.offsets import BDay 
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.linear_model import LinearRegression 
import warnings
warnings.filterwarnings("ignore") # Ignore specific statsmodels/sklearn warnings

# --- 1. Configuration and Constants ---
APP_TITLE = "🇮🇳 Advanced NSE Stock Predictor & Market Status"
NSE_EXCHANGE = 'XNSE'
MARKET_TZ = 'Asia/Kolkata' 
DAYS_DEFAULT = 7
WINDOW = 20 # Window for Bollinger Bands and Moving Averages

# Curated list of Top 100 Indian Tickers (for simulation and selection)
TICKER_SYMBOLS = {
    'Nifty 50 Index': '^NSEI',
    'Bank Nifty Index': '^NSEBANK',
    'Reliance Industries': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'State Bank of India': 'SBIN.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Larsen & Toubro': 'LT.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Titan Company': 'TITAN.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'HCL Technologies': 'HCLTECH.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Wipro': 'WIPRO.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS'
}

# --- 2. Data Fetching and Market Status Functions ---

@st.cache_data(ttl=3600)
def check_market_status(exchange_code=NSE_EXCHANGE):
    """Checks the current market status for NSE."""
    try:
        nse = pmc.get_calendar(exchange_code)
        market_tz = pytz.timezone(MARKET_TZ)
        now_in_market_tz = datetime.now(market_tz)

        schedule = nse.schedule(start_date=now_in_market_tz.date(), end_date=now_in_market_tz.date())
        
        if schedule.empty:
            return "Closed (Weekend/Holiday)", MARKET_TZ

        market_open = schedule.iloc[0]['market_open'].tz_convert(market_tz)
        market_close = schedule.iloc[0]['market_close'].tz_convert(market_tz)
        
        if market_open <= now_in_market_tz < market_close:
            return "LIVE (Open)", MARKET_TZ
        else:
            status_type = "Closed (Off-Hours)" if now_in_market_tz.date() == market_open.date() else "Closed"
            return status_type, MARKET_TZ
            
    except Exception as e:
        return f"Status Unavailable ({e})", MARKET_TZ

@st.cache_data(ttl=600)
def get_historical_data(ticker_symbol):
    """Fetches 5y historical data and calculates indicators."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        history_df = ticker.history(period='5y', interval='1d')
        info = ticker.info
        
        if history_df.empty or not info:
             raise ValueError("Data fetching resulted in empty or incomplete data.")

        # Ensure index is timezone-naive for ARIMA
        history_df.index = history_df.index.tz_localize(None) 
        df_clean = history_df.copy().dropna()
        
        # Calculate Technical Indicators
        df_clean['SMA'] = df_clean['Close'].rolling(window=WINDOW).mean()
        df_clean['StdDev'] = df_clean['Close'].rolling(window=WINDOW).std()
        df_clean['Upper_BB'] = df_clean['SMA'] + (df_clean['StdDev'] * 2)
        df_clean['Lower_BB'] = df_clean['SMA'] - (df_clean['StdDev'] * 2)

        return df_clean, info
        
    except Exception as e:
        st.error(f"Error fetching historical data for **{ticker_symbol}**. Error: {e}")
        return None, None

def get_live_data(ticker_symbol):
    """Fetches the latest intraday data for current price and change calculation."""
    try:
        live_df = yf.Ticker(ticker_symbol).history(period='1d', interval='1m')
        return live_df if not live_df.empty else None
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_all_movers_data():
    """Fetches latest data for all pre-defined stocks to simulate top movers."""
    movers = {}
    tickers = list(TICKER_SYMBOLS.values())
    
    # Fetch data for all symbols at once (more efficient)
    data = yf.download(tickers, period="2d", interval="1d", progress=False)
    
    for symbol in tickers:
        # Check if the symbol is in the result (yfinance returns a MultiIndex DataFrame)
        if symbol in data['Close']:
            close_prices = data['Close'][symbol].tail(2)
            if len(close_prices) == 2:
                current_price = close_prices.iloc[-1]
                previous_close = close_prices.iloc[-2]
                change = current_price - previous_close
                percent_change = (change / previous_close) * 100
                
                movers[symbol] = {
                    'Price': current_price,
                    'Change %': percent_change,
                    'Change (₹)': change
                }
    return movers

# --- 3. Advanced Prediction Functions ---

def predict_price_arima(history_df, days_to_predict, order=(5, 1, 0)):
    """ARIMA model for price prediction."""
    series = history_df['Close'].dropna()
    
    if len(series) < 50:
        return predict_price_linear_fallback(history_df, days_to_predict)

    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()

        forecast_start_date = series.index[-1] + BDay(1)
        future_dates = pd.date_range(start=forecast_start_date, periods=days_to_predict, freq='B') 
        
        forecast = model_fit.predict(start=len(series), end=len(series) + days_to_predict - 1, dynamic=False)
        
        prediction_df = pd.DataFrame({
            'Predicted_Close': forecast.values
        }, index=future_dates[:len(forecast)])
        
        return prediction_df

    except Exception:
        return predict_price_linear_fallback(history_df, days_to_predict)

def predict_price_linear_fallback(history_df, days_to_predict):
    """Fallback to Linear Regression."""
    df_pred = history_df.copy()
    df_pred['Day'] = np.arange(len(df_pred))
    
    valid_data = df_pred[['Day', 'Close']].dropna()
    X = valid_data[['Day']]
    y = valid_data['Close']

    model = LinearRegression()
    model.fit(X, y)

    last_day_index = df_pred['Day'].iloc[-1]
    future_days = np.array(range(last_day_index + 1, last_day_index + 1 + days_to_predict)).reshape(-1, 1)

    predicted_prices = model.predict(future_days)
    
    last_date = df_pred.index[-1]
    future_dates = pd.date_range(start=last_date + BDay(1), periods=days_to_predict, freq='B')
    
    prediction_df = pd.DataFrame({
        'Predicted_Close': predicted_prices
    }, index=future_dates)
    
    return prediction_df

# --- 4. Charting with Technical Indicators ---

def create_advanced_candlestick_chart(df, ticker):
    """Creates an interactive Candlestick chart with Technical Indicators."""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))

    # Drop initial NaNs created by rolling window
    df_plot = df.dropna(subset=['Upper_BB', 'Lower_BB', 'SMA'])
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Upper_BB'], line={'color': 'rgba(250, 0, 0, 0.5)'}, name='Upper BB'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Lower_BB'], line={'color': 'rgba(0, 0, 250, 0.5)'}, name='Lower BB', fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA'], line={'color': 'yellow', 'width': 2}, name=f'SMA ({WINDOW})'))

    fig.update_layout(
        title=f'{ticker} Historical Price with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 5. Streamlit UI Components ---

def display_index_metrics(index_symbol, label, all_movers):
    """Displays metric for a major index."""
    data = all_movers.get(index_symbol)
    if data:
        delta_rs = data['Change (₹)']
        delta_pct = data['Change %']
        # Fixed the f-string formatting by isolating the delta text
        delta_text = f"₹{delta_rs:+.2f} ({delta_pct:+.2f}%)"
        st.metric(
            label=label,
            value=f"₹{data['Price']:,.2f}",
            delta=delta_text,
            delta_color="normal"
        )
    else:
        st.metric(label=label, value="Data N/A", delta="N/A")

def display_top_movers(all_movers):
    """Displays Top Gainers and Losers."""
    
    if not all_movers:
        st.warning("Could not fetch data to calculate Top Movers.")
        return

    df_movers = pd.DataFrame.from_dict(all_movers, orient='index')
    
    # Exclude indices from top movers calculation
    df_movers = df_movers[~df_movers.index.isin(['^NSEI', '^NSEBANK'])]

    top_gainers = df_movers.sort_values(by='Change %', ascending=False).head(5)
    top_losers = df_movers.sort_values(by='Change %', ascending=True).head(5)

    col_g, col_l = st.columns(2)

    with col_g:
        st.subheader("🚀 Top Gainers (Simulated)")
        for symbol, row in top_gainers.iterrows():
            name = next((k for k, v in TICKER_SYMBOLS.items() if v == symbol), symbol)
            st.markdown(f"""
                <div style="padding: 10px; margin-bottom: 5px; background-color: #004d00; border-radius: 5px; color: white;">
                    <strong>{name}</strong> ({symbol.replace('.NS', '')}): 
                    <span style="float: right;">₹{row['Price']:,.2f} <span style="color: #64ff64;">({row['Change %']:+.2f}%)</span></span>
                </div>
            """, unsafe_allow_html=True)

    with col_l:
        st.subheader("🔻 Top Losers (Simulated)")
        for symbol, row in top_losers.iterrows():
            name = next((k for k, v in TICKER_SYMBOLS.items() if v == symbol), symbol)
            st.markdown(f"""
                <div style="padding: 10px; margin-bottom: 5px; background-color: #4d0000; border-radius: 5px; color: white;">
                    <strong>{name}</strong> ({symbol.replace('.NS', '')}): 
                    <span style="float: right;">₹{row['Price']:,.2f} <span style="color: #ff6464;">({row['Change %']:+.2f}%)</span></span>
                </div>
            """, unsafe_allow_html=True)

# --- 6. Main Application Logic ---

def main():
    # --- Professional Styling (Dark Theme) ---
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS for a professional dark background and consistent look
    st.markdown("""
        <style>
            /* Overall Dark Background */
            .main, .stApp {
                background-color: #1a1a2e; /* Deep purple/navy blue */
                color: #e4e4e4;
            }
            /* Sidebar Background */
            .css-1d391kg, .stSidebar { 
                background-color: #0f0f1f; /* Darker sidebar */
            }
            /* Header and Title Colors */
            h1, h2, h3, h4, .stMarkdown {
                color: #ffcc00; /* Gold accents */
            }
            /* Metric colors (already handled by Streamlit, but ensuring text visibility) */
            [data-testid="stMetricValue"] {
                color: #e4e4e4;
            }
            /* Info Box background */
            .stAlert.info {
                background-color: #2b3a4d;
                border-left: 5px solid #0099ff;
            }
            /* Divider color */
            hr {
                border-top: 1px solid #4d4d73;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title(APP_TITLE)

    # --- Sidebar for Inputs ---
    st.sidebar.header("Stock Selection (NSE) & Options")
    
    selection_mode = st.sidebar.radio(
        "Select Stock/Index Mode",
        ('Predefined Top Stock', 'Search by Ticker'),
        index=0
    )

    selected_ticker = None
    
    if selection_mode == 'Predefined Top Stock':
        selected_name = st.sidebar.selectbox(
            'Select an Indian Stock/Index',
            list(TICKER_SYMBOLS.keys()),
            index=0
        )
        selected_ticker = TICKER_SYMBOLS[selected_name]
    else: # Search by Ticker
        search_input = st.sidebar.text_input('Enter NSE Ticker Symbol (e.g., TCS.NS, RELIANCE.NS)', value='RELIANCE.NS').upper()
        if search_input:
            selected_ticker = search_input
            selected_name = selected_ticker.replace('.NS', '')
        else:
            st.sidebar.warning("Please enter a valid NSE ticker.")
            return

    # Prediction Inputs
    days_to_predict = st.sidebar.slider('Days to Predict', min_value=1, max_value=30, value=DAYS_DEFAULT)
    arima_p = st.sidebar.number_input('ARIMA Order P (AR)', min_value=1, max_value=10, value=5)
    arima_d = st.sidebar.number_input('ARIMA Order D (Differencing)', min_value=0, max_value=2, value=1)
    arima_q = st.sidebar.number_input('ARIMA Order Q (MA)', min_value=0, max_value=10, value=0)
    
    st.sidebar.caption("Prediction uses ARIMA({}, {}, {})".format(arima_p, arima_d, arima_q))
    st.sidebar.info("Data is cached to reduce API calls. To refresh prices, re-run the app.")
    
    # --- Initial Data Fetch (Movers and Historical) ---
    with st.spinner("Fetching market data..."):
        all_movers = get_all_movers_data()
        market_status, market_tz = check_market_status()
        
    status_icon = "🟢" if "LIVE" in market_status else "🔴"
    st.markdown(f"### {status_icon} NSE Market Status: **{market_status}** ({market_tz})")
    st.divider()

    # --- Nifty 50 and Bank Nifty Display ---
    st.header("Major Indices Overview (Nifty 50 & Bank Nifty)")
    index_col1, index_col2 = st.columns(2)
    with index_col1:
        display_index_metrics('^NSEI', 'Nifty 50 Index', all_movers)
    with index_col2:
        display_index_metrics('^NSEBANK', 'Bank Nifty Index', all_movers)
    
    st.divider()

    # --- Top Gainers and Losers ---
    st.header("Top Gainers & Losers (Current Session Simulation)")
    display_top_movers(all_movers)
    st.divider()

    # --- Individual Stock Analysis ---
    st.header(f"Detailed Analysis & Prediction for {selected_name.split('Index')[0]}")
    
    with st.spinner(f"Fetching historical data for {selected_ticker}..."):
        history_data, info = get_historical_data(selected_ticker)
        live_data = get_live_data(selected_ticker)

    if history_data is None:
        return # Stop if data fetch failed

    col1, col2 = st.columns([1, 2])
    
    # --- Column 1: Financials and Prediction ---
    with col1:
        st.subheader("📊 Key Financials")

        if live_data is not None and not live_data.empty:
            current_price = live_data['Close'].iloc[-1]
        else:
            current_price = history_data['Close'].iloc[-1]
        
        try:
            # Check the change based on the last historical close
            last_close = history_data['Close'].iloc[-2]
            change_percent = ((current_price - last_close) / last_close) * 100
            change_rs = current_price - last_close
        except:
            change_percent = 0
            change_rs = 0
            
        # Fixed the f-string formatting by isolating the delta text
        delta_text = f"₹{change_rs:+.2f} ({change_percent:+.2f}%)"
        st.metric(
            label=f"Current Price ({selected_ticker.replace('.NS', '')})",
            value=f"₹{current_price:,.2f}",
            delta=delta_text,
            delta_color="normal"
        )
        
        # Market Cap and Range
        market_cap_lakh_cr = info.get('marketCap', 0) / 1e11
        st.markdown(f"**Market Cap:** ₹{market_cap_lakh_cr:,.2f} Lakh Crore")
        st.markdown(f"**52-Week Range:** ₹{info.get('fiftyTwoWeekLow', 'N/A')} - ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.divider()

        # Advanced Prediction
        st.subheader(f"🔮 ARIMA Forecast ({days_to_predict} Days)")
        
        with st.spinner("Calculating ARIMA forecast..."):
            prediction_df = predict_price_arima(history_data, days_to_predict, order=(arima_p, arima_d, arima_q))
        
        if prediction_df is not None and not prediction_df.empty:
            final_prediction = prediction_df['Predicted_Close'].iloc[-1]
            prediction_delta = ((final_prediction - current_price) / current_price) * 100
            
            st.metric(
                label=f"Forecasted Close Price on {prediction_df.index[-1].strftime('%b %d, %Y')}",
                value=f"₹{final_prediction:,.2f}",
                delta=f"{prediction_delta:+.2f}% change",
                delta_color="off" 
            )
            
            with st.expander("Show Full Forecast Table"):
                st.dataframe(prediction_df.style.format({"Predicted_Close": "₹{:,.2f}"}), use_container_width=True)
        else:
            st.warning("Prediction model failed. Try adjusting ARIMA parameters.")


    # --- Column 2: Charts ---
    with col2:
        st.header("⏳ Technical Analysis (Last 300 Days)")
        
        recent_data = history_data.tail(300)
        st.plotly_chart(create_advanced_candlestick_chart(recent_data, selected_name), use_container_width=True)
        
        st.subheader("📈 Trend and Forecast Line Chart (90 Days History)")
        
        # Combine historical and predicted data for line chart
        plot_history = history_data['Close'].tail(90).to_frame(name='Historical Price') 

        if prediction_df is not None and not prediction_df.empty:
            last_hist_date = plot_history.index[-1]
            temp_forecast_series = prediction_df['Predicted_Close']
            
            # Create a transition point for a continuous line plot
            transition_point = pd.Series(
                data=plot_history.loc[last_hist_date, 'Historical Price'], 
                index=[last_hist_date], 
                name='Predicted Price'
            )
            
            forecast_series_with_start = pd.concat([transition_point, temp_forecast_series])
            
            combined_df = plot_history.merge(
                forecast_series_with_start.to_frame('Predicted Price'), 
                left_index=True, 
                right_index=True, 
                how='outer'
            )
            
            # Ensure the historical line stops cleanly
            combined_df.loc[combined_df.index > last_hist_date, 'Historical Price'] = np.nan
            
            st.line_chart(combined_df, use_container_width=True)
        else:
             st.line_chart(plot_history, use_container_width=True)
             st.warning("Could not generate forecast line chart.")
        
        st.caption("""
            **Disclaimer:** This tool uses academic models (ARIMA, Technical Indicators). 
            It is provided for informational purposes only and **should NOT be used for actual trading or financial decisions.**
        """)

if __name__ == "__main__":
    main()