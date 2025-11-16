# app.py
"""
Streamlit Stock Website using Google Finance Scraping (with yfinance fallback)
Features:
 - Sprite-card dashboard pulling live quotes from Google Finance (scraped)
 - Detail view with candlestick charts (historic from yfinance fallback)
 - Technical indicators: EMA, SMA, Bollinger Bands, RSI, MACD, VWAP, ADX
 - ARIMA forecasting with linear-regression fallback
 - Simple EMA 9/20 backtest
 - Downloadable CSV outputs
 - Caching, safe rerun, and defensive scraping fallbacks
Note: Scraping Google pages is brittle and may break if page structure changes.
Author: Generated for user
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_market_calendars as pmc
from datetime import datetime, timedelta
from plotly import graph_objects as go
import pytz
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import warnings
import math
import logging
import time

warnings.filterwarnings("ignore")
logger = logging.getLogger("google_finance_app")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ----------------------------
# CONFIGURATION & CONSTANTS
# ----------------------------
APP_TITLE = "📈 Google-Finance Scraper — NSE Stock Website"
MARKET_TZ = "Asia/Kolkata"
NSE_EXCHANGE = "NSE"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)" \
             " Chrome/115.0.0.0 Safari/537.36"

# Default tickers (we will map Yahoo-style .NS -> Google 'SYMBOL:NSE')
DEFAULT_TICKERS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Titan Company": "TITAN.NS",
}

EMA_SHORT = 9
EMA_MEDIUM = 20
EMA_LONG = 50
BB_WINDOW = 20
RSI_PERIOD = 14
ADX_PERIOD = 14
FORECAST_DAYS_DEFAULT = 7

# ----------------------------
# Helper utilities
# ----------------------------
def tz_now():
    return datetime.now(pytz.timezone(MARKET_TZ))

def yahoo_to_google_symbol(yahoo_symbol: str) -> str:
    """
    Convert Yahoo's 'RELIANCE.NS' -> Google format 'RELIANCE:NSE'
    If symbol like '^NSEI' (index), we will try to map but fallback is yfinance.
    """
    if yahoo_symbol.startswith("^"):
        # indices are tricky on Google; leave as-is (we'll fallback to yfinance)
        return yahoo_symbol
    if yahoo_symbol.endswith(".NS"):
        base = yahoo_symbol.replace(".NS", "")
        return f"{base}:NSE"
    # generic fallback return original
    return yahoo_symbol

# ----------------------------
# Google Finance Scraping
# ----------------------------
def build_google_quote_url(google_symbol: str) -> str:
    """
    Example: https://www.google.com/finance/quote/RELIANCE:NSE
    """
    return f"https://www.google.com/finance/quote/{google_symbol}"

def fetch_google_quote(google_symbol: str, timeout=8):
    """
    Scrape Google Finance quote page and return dictionary with:
    { 'price': float, 'change': float, 'change_percent': float, 'time': str, 'currency': str, 'summary': str (optional) }
    This parser uses multiple CSS fallbacks to be resilient.
    """
    url = build_google_quote_url(google_symbol)
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            logger.warning("Google quote request returned %s for %s", r.status_code, google_symbol)
            return None
        soup = BeautifulSoup(r.text, "lxml")

        # Attempt multiple selectors (Google's classes change often)
        # 1) Price: often 'YMlKec fxKbKc' or 'IsqQVc NprOob wT3VGc' or a data-attr
        price = None
        # Try common span class
        selectors = [
            "div.YMlKec.fxKbKc",  # older pattern
            "div.IsqQVc.NprOob.wT3VGc",  # another pattern
            "div[data-last-price]",  # some structured data
            "div[data-attrid='price']",  # fallback
            "div[data-precision]"  # sometimes
        ]
        text = None
        for sel in selectors:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                text = el.get_text(strip=True)
                break
        if text is None:
            # Try generic search for numeric-looking text
            cand = soup.find_all(text=True)
            for t in cand:
                s = t.strip()
                # price often contains digits and optionally comma or decimal
                if len(s) > 0 and any(ch.isdigit() for ch in s) and ("," in s or "." in s):
                    # Filter improbable text like dates by length and characters
                    if len(s) < 30:
                        text = s
                        break
        if text:
            # Clean text -> extract number
            # Remove currency symbols and commas
            cleaned = text.replace(",", "").replace("₹", "").replace("INR", "").strip()
            # Sometimes text like "+2.34 (0.45%)" or "1,234.56"
            # pick first token that parses as float
            tokens = cleaned.split()
            price_val = None
            for tok in tokens:
                tok_clean = tok.replace(",", "").replace("%", "").replace("(", "").replace(")", "")
                try:
                    price_val = float(tok_clean)
                    break
                except:
                    continue
            if price_val is not None:
                price = price_val

        # Change and change percent
        change_val = None
        change_pct = None
        # Google often places change in a div with classes 'P6K39c'
        try:
            change_selector_candidates = ["div.P6K39c", "span.WlRRw.IsqQVc.fw-price-dn", "span.wlpxIll"] 
            change_text = None
            for sel in change_selector_candidates:
                el = soup.select_one(sel)
                if el and el.get_text(strip=True):
                    change_text = el.get_text(strip=True)
                    break
            if change_text:
                # e.g. "+12.34 (1.02%)"
                if "(" in change_text and ")" in change_text:
                    left, right = change_text.split("(", 1)
                    change_val = float(left.replace(",", "").replace("+", "").replace("₹", "").strip())
                    change_pct = right.replace(")", "").replace("%", "").strip()
                    # convert to float
                    try:
                        change_pct = float(change_pct)
                    except:
                        change_pct = None
                else:
                    # maybe just +12.34
                    try:
                        change_val = float(change_text.replace(",", "").replace("+", "").replace("₹", "").strip())
                    except:
                        change_val = None
        except Exception as e:
            logger.info("change parse fail %s", e)

        # Time / currency
        currency = None
        # Try find currency label
        try:
            cur_el = soup.select_one("div.Kd6bdc") or soup.select_one("div.YvlyV")
            if cur_el and cur_el.get_text(strip=True):
                cur_text = cur_el.get_text(strip=True)
                if "INR" in cur_text or "₹" in cur_text:
                    currency = "INR"
        except:
            pass

        # Summary / short description
        summary = None
        try:
            summ_el = soup.select_one("div.P6K39c.d2Qm6b")
            if summ_el:
                summary = summ_el.get_text(strip=True)
        except:
            pass

        # Pack result
        result = {}
        if price is not None:
            result['price'] = price
        if change_val is not None:
            result['change'] = change_val
        if change_pct is not None:
            result['change_percent'] = change_pct
        if currency:
            result['currency'] = currency
        if summary:
            result['summary'] = summary
        result['source'] = 'google'
        result['scraped_at'] = tz_now().isoformat()
        return result
    except Exception as e:
        logger.exception("Google scrape failed for %s: %s", google_symbol, e)
        return None

# ----------------------------
# Historical data fetch (yfinance fallback)
# ----------------------------
@st.cache_data(ttl=15 * 60)
def fetch_historical_yfinance(ticker: str, period: str = "5y", interval: str = "1d"):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, auto_adjust=False)
        info = t.info or {}
        if hist.empty:
            return pd.DataFrame(), info
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(MARKET_TZ).tz_localize(None)
        return hist, info
    except Exception as e:
        logger.exception("yfinance fetch failed: %s", e)
        return pd.DataFrame(), {}

# ----------------------------
# Technical Indicators
# ----------------------------
def add_ema(df, span):
    return df['Close'].ewm(span=span, adjust=False).mean()

def add_bollinger(df, window=BB_WINDOW):
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return sma, upper, lower

def add_rsi(df, period=RSI_PERIOD):
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_macd(df, fast=12, slow=26, sig=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def add_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (tp * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9)
    return vwap

def add_adx(df, n=ADX_PERIOD):
    high = df['High']
    low = df['Low']
    close = df['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx, plus_di, minus_di

def compute_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    df[f'EMA_{EMA_SHORT}'] = add_ema(df, EMA_SHORT)
    df[f'EMA_{EMA_MEDIUM}'] = add_ema(df, EMA_MEDIUM)
    df[f'EMA_{EMA_LONG}'] = add_ema(df, EMA_LONG)
    df['SMA_BB'], df['Upper_BB'], df['Lower_BB'] = add_bollinger(df)
    df['RSI'] = add_rsi(df)
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = add_macd(df)
    df['VWAP'] = add_vwap(df)
    df['ADX'], df['+DI'], df['-DI'] = add_adx(df)
    return df

# ----------------------------
# Forecasting: ARIMA + Linear fallback
# ----------------------------
def forecast_arima(series: pd.Series, days: int = FORECAST_DAYS_DEFAULT, order=(5,1,0)):
    series = series.dropna()
    if series.shape[0] < 50:
        return forecast_linear(series, days)
    try:
        model = ARIMA(series, order=order)
        fitted = model.fit()
        start = len(series)
        end = start + days - 1
        preds = fitted.predict(start=start, end=end, dynamic=False)
        last_date = series.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
        preds = pd.Series(preds).iloc[:len(future_dates)]
        preds.index = future_dates[:len(preds)]
        preds = preds.rename("Predicted_Close")
        return preds.to_frame()
    except Exception as e:
        logger.info("ARIMA failed: %s", e)
        return forecast_linear(series, days)

def forecast_linear(series: pd.Series, days: int = FORECAST_DAYS_DEFAULT):
    series = series.dropna()
    if series.empty:
        future_dates = pd.bdate_range(start=datetime.today(), periods=days)
        return pd.DataFrame({'Predicted_Close': [np.nan]*days}, index=future_dates)
    df = series.reset_index().reset_index()
    df.rename(columns={'index':'X', series.name:'y'}, inplace=True)
    X = df[['X']].values
    y = df['y'].values
    model = LinearRegression()
    model.fit(X, y)
    last_x = X[-1,0]
    future_x = np.arange(last_x+1, last_x+1+days).reshape(-1,1)
    preds = model.predict(future_x)
    last_date = series.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
    return pd.DataFrame({'Predicted_Close': preds}, index=future_dates)

# ----------------------------
# Backtest: simple EMA 9/20
# ----------------------------
def backtest_ema(df, short=EMA_SHORT, long=EMA_MEDIUM, initial_capital=100000.0):
    if df.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    data = df.copy().dropna(subset=['Close'])
    data['EMA_short'] = data['Close'].ewm(span=short, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long, adjust=False).mean()
    data['Signal'] = 0
    data.loc[data['EMA_short'] > data['EMA_long'], 'Signal'] = 1
    data['Position'] = data['Signal'].diff().fillna(0)
    cash = initial_capital
    holdings = 0
    trades = []
    equity = []
    for idx, row in data.iterrows():
        price = row['Close']
        pos = row['Position']
        if pos == 1:
            shares = math.floor(cash / price)
            if shares > 0:
                cost = shares * price
                cash -= cost
                holdings += shares
                trades.append({'Date': idx, 'Type': 'BUY', 'Price': price, 'Shares': shares})
        elif pos == -1:
            if holdings > 0:
                proceeds = holdings * price
                cash += proceeds
                trades.append({'Date': idx, 'Type': 'SELL', 'Price': price, 'Shares': holdings})
                holdings = 0
        total = cash + holdings * price
        equity.append({'Date': idx, 'Equity': total})
    if holdings > 0:
        last_price = data['Close'].iloc[-1]
        cash += holdings * last_price
        trades.append({'Date': data.index[-1], 'Type': 'SELL', 'Price': last_price, 'Shares': holdings})
        holdings = 0
    equity_df = pd.DataFrame(equity).set_index('Date')
    returns = equity_df['Equity'].pct_change().fillna(0)
    total_return = (equity_df['Equity'].iloc[-1] - initial_capital) / initial_capital * 100
    max_dd = ((equity_df['Equity'].cummax() - equity_df['Equity']) / equity_df['Equity'].cummax()).max() * 100
    trades_df = pd.DataFrame(trades).set_index('Date') if trades else pd.DataFrame()
    summary = {'initial_capital': initial_capital, 'final_equity': equity_df['Equity'].iloc[-1] if not equity_df.empty else initial_capital,
               'total_return_pct': total_return, 'max_drawdown_pct': max_dd, 'trades': len(trades)}
    return summary, trades_df, equity_df

# ----------------------------
# Plotting (Plotly)
# ----------------------------
def candlestick_figure(df, title="Technical Chart"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    if f'EMA_{EMA_SHORT}' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{EMA_SHORT}'], name=f'EMA {EMA_SHORT}', line=dict(width=1)))
    if f'EMA_{EMA_MEDIUM}' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{EMA_MEDIUM}'], name=f'EMA {EMA_MEDIUM}', line=dict(width=1)))
    if 'Upper_BB' in df.columns and 'Lower_BB' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], name='Upper BB', line=dict(dash='dot', width=1), opacity=0.6))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], name='Lower BB', line=dict(dash='dot', width=1), opacity=0.6, fill='tonexty', fillcolor='rgba(90,90,150,0.06)'))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', marker=dict(opacity=0.4)))
    if 'ADX' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', yaxis='y3', line=dict(width=1)))
        if '+DI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], name='+DI', yaxis='y3', line=dict(dash='dash')))
        if '-DI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], name='-DI', yaxis='y3', line=dict(dash='dash')))
    fig.update_layout(
        title=title,
        xaxis=dict(rangeselector=dict(buttons=list([
            dict(count=7, label='7d', step='day', stepmode='backward'),
            dict(count=1, label='1m', step='month', stepmode='backward'),
            dict(count=3, label='3m', step='month', stepmode='backward'),
            dict(step='all')
        ])), rangeslider=dict(visible=False)),
        yaxis=dict(domain=[0.25, 1.0], title='Price'),
        yaxis2=dict(domain=[0.2, 0.25], overlaying='y', side='left', title='Volume'),
        yaxis3=dict(domain=[0.0, 0.18], overlaying='y', side='right', title='ADX/DI'),
        template='plotly_dark', height=820, legend=dict(orientation='h', y=1.02, x=0.8)
    )
    return fig

# ----------------------------
# UI Components
# ----------------------------
def sprite_card(name, yahoo_symbol, days=180):
    google_symbol = yahoo_to_google_symbol(yahoo_symbol)
    # Try Google scrape
    quote = None
    try:
        quote = fetch_google_quote(google_symbol)
    except Exception:
        quote = None
    # If scraping failed, fall back to yfinance quick fetch for current price
    quick_price = None
    info = {}
    if quote is None:
        # fallback to yfinance current price
        try:
            t = yf.Ticker(yahoo_symbol)
            info = t.info or {}
            intr = t.history(period="1d", interval="1m")
            if not intr.empty:
                quick_price = intr['Close'].iloc[-1]
            else:
                hist, _ = fetch_historical_yfinance(yahoo_symbol, period=f"{max(90, days)}d", interval="1d")
                if not hist.empty:
                    quick_price = hist['Close'].iloc[-1]
        except Exception:
            quick_price = None
    price = quote.get('price') if quote and 'price' in quote else quick_price
    change_pct = quote.get('change_percent') if quote and 'change_percent' in quote else None

    # Pull basic historical to compute ADX & EMAs for card
    hist, info2 = fetch_historical_yfinance(yahoo_symbol, period=f"{max(90, days)}d", interval="1d")
    if not hist.empty:
        hist = compute_indicators(hist)
        adx_val = hist['ADX'].iloc[-1] if 'ADX' in hist.columns else np.nan
        adx_txt = f"{adx_val:.1f}" if not pd.isna(adx_val) else "N/A"
    else:
        adx_txt = "N/A"

    with st.container():
        st.markdown(f"**{name}**  `{yahoo_symbol}`")
        col1, col2 = st.columns([2,1])
        with col1:
            if price is not None:
                st.markdown(f"<span style='font-size:16px;'>**₹{price:,.2f}**</span>", unsafe_allow_html=True)
                if change_pct is not None:
                    st.markdown(f"<span style='color:lightgreen'>{change_pct:+.2f}%</span>", unsafe_allow_html=True)
            else:
                st.write("Price unavailable")
            st.write(f"ADX: {adx_txt}")
        with col2:
            st.write(info.get('sector', info2.get('sector', '')))
            st.write(f"52W: {safe_get(info2, 'fiftyTwoWeekLow', 'N/A')} - {safe_get(info2, 'fiftyTwoWeekHigh', 'N/A')}")
        if st.button("Analyze", key=f"analyze_{yahoo_symbol}", use_container_width=True):
            st.session_state['selected_name'] = name
            st.session_state['selected_ticker'] = yahoo_symbol
            st.rerun()

def render_dashboard(tickers_dict, days_display=180):
    st.title(APP_TITLE)
    st.caption(f"Data source: Google Finance (scraped) — fallback to yfinance for historical data. Time: {tz_now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.divider()
    cols = st.columns(3)
    idx = 0
    for name, yahoo_symbol in tickers_dict.items():
        with cols[idx % 3]:
            sprite_card(name, yahoo_symbol, days=days_display)
        idx += 1

def render_detail(yahoo_symbol, name):
    st.header(f"🔎 {name} — {yahoo_symbol}")
    if st.button("⬅ Back to Dashboard"):
        st.session_state['selected_ticker'] = None
        st.session_state['selected_name'] = None
        st.rerun()

    google_symbol = yahoo_to_google_symbol(yahoo_symbol)
    quote = fetch_google_quote(google_symbol)
    hist, info = fetch_historical_yfinance(yahoo_symbol, period="5y", interval="1d")
    intraday = fetch_intraday(yahoo_symbol, period="1d", interval="1m")
    if hist.empty:
        st.error("Historical data unavailable. Cannot show charts.")
        return
    hist = compute_indicators(hist)
    current_price = quote.get('price') if quote and 'price' in quote else (intraday['Close'].iloc[-1] if not intraday.empty else hist['Close'].iloc[-1])
    st.metric("Current Price (INR)", f"₹{current_price:,.2f}")

    # KPIs
    signal_text, signal_col = ema_signal_text(hist) if True else ("N/A","gray")
    adx_val = hist['ADX'].iloc[-1] if 'ADX' in hist.columns else np.nan
    adx_str = adx_text(adx_val, hist.get('+DI', pd.Series([np.nan])).iloc[-1] if '+DI' in hist.columns else np.nan,
                       hist.get('-DI', pd.Series([np.nan])).iloc[-1] if '-DI' in hist.columns else np.nan)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Signal:** <span style='color:{signal_col}'>{signal_text}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Trend (ADX):** {adx_str}")

    st.divider()

    st.subheader("Technical Chart (last 300 days)")
    recent = hist.tail(300)
    fig = candlestick_figure(recent, title=f"{name} — Price & Indicators")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Historical Close + Forecast (combined)")
    hist_short = hist['Close'].tail(90).to_frame().rename(columns={'Close': 'Historical'})
    with st.expander("Forecast settings"):
        days_f = st.number_input("Forecast days (business days)", min_value=1, max_value=30, value=FORECAST_DAYS_DEFAULT)
        p = st.number_input("ARIMA p", min_value=0, max_value=10, value=5)
        d = st.number_input("ARIMA d", min_value=0, max_value=2, value=1)
        q = st.number_input("ARIMA q", min_value=0, max_value=10, value=0)
        run_forecast = st.button("Run Forecast")
    if run_forecast:
        with st.spinner("Forecasting..."):
            preds = forecast_arima(hist['Close'], days=days_f, order=(int(p), int(d), int(q)))
            if not preds.empty:
                # Combine series for plotting
                combined = pd.concat([hist_short, preds.rename(columns={'Predicted_Close':'Predicted'})], axis=0, sort=False)
                combined = combined[~combined.index.duplicated(keep='first')]
                st.line_chart(combined)
                st.dataframe(preds.style.format({"Predicted_Close": "₹{:,.2f}"}))
                st.download_button("Download forecast CSV", preds.to_csv().encode(), file_name=f"{yahoo_symbol}_forecast.csv", mime="text/csv")
            else:
                st.info("Forecast unavailable.")
    st.divider()

    # Backtest
    st.subheader("Backtest EMA 9/20")
    with st.expander("Backtest options"):
        years = st.number_input("Backtest years", min_value=1, max_value=10, value=2)
        initial_cap = st.number_input("Initial capital (₹)", min_value=1000.0, value=100000.0)
        run_bt = st.button("Run Backtest")
    if run_bt:
        hist_bt, _ = fetch_historical_yfinance(yahoo_symbol, period=f"{int(years)*365}d", interval="1d")
        if hist_bt.empty:
            st.error("Not enough history for backtest.")
        else:
            summary, trades_df, equity_df = backtest_ema(hist_bt, initial_capital=float(initial_cap))
            if summary:
                st.write("Summary:")
                st.write({
                    "Initial capital": f"₹{summary['initial_capital']:,.2f}",
                    "Final equity": f"₹{summary['final_equity']:,.2f}",
                    "Total return (%)": f"{summary['total_return_pct']:+.2f}%",
                    "Max drawdown (%)": f"{summary['max_drawdown_pct']:+.2f}%",
                    "Trades": summary['trades']
                })
                if not trades_df.empty:
                    st.write("Trades sample:")
                    st.dataframe(trades_df.head(50))
                if not equity_df.empty:
                    st.line_chart(equity_df['Equity'])

    st.divider()
    st.subheader("Company / Instrument Info (yfinance fallback)")
    try:
        t = yf.Ticker(yahoo_symbol)
        info = t.info or {}
        show_keys = ['longName','sector','industry','website','shortName','currency']
        info_display = {k: safe_get(info, k, 'N/A') for k in show_keys}
        st.write(info_display)
    except Exception:
        st.write("No extra info available.")

# ----------------------------
# Small helpers for signals
# ----------------------------
def ema_signal_text(df):
    if df.empty or f'EMA_{EMA_MEDIUM}' not in df.columns:
        return "NO DATA", "gray"
    s = df[f'EMA_{EMA_SHORT}'].iloc[-1]
    m = df[f'EMA_{EMA_MEDIUM}'].iloc[-1]
    s_prev = df[f'EMA_{EMA_SHORT}'].iloc[-2] if len(df) > 1 else s
    m_prev = df[f'EMA_{EMA_MEDIUM}'].iloc[-2] if len(df) > 1 else m
    if s_prev <= m_prev and s > m:
        return "STRONG BUY (9/20 CROSS)", "green"
    if s_prev >= m_prev and s < m:
        return "STRONG SELL (9/20 CROSS)", "red"
    if s > m:
        return "BUY (Uptrend)", "blue"
    return "HOLD/CAUTION", "orange"

def adx_text(adx_val, plus, minus):
    if pd.isna(adx_val):
        return "N/A"
    if adx_val >= 35:
        strength = "VERY STRONG"
    elif adx_val >= 25:
        strength = "STRONG"
    elif adx_val >= 20:
        strength = "DEVELOPING"
    else:
        strength = "WEAK/CONSOLIDATION"
    dirn = "BULLISH" if plus > minus else "BEARISH"
    return f"{strength} ({adx_val:.1f} | {dirn})"

# ----------------------------
# Main App
# ----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.sidebar.title("Controls & Settings")

    # Market status
    try:
        cal = pmc.get_calendar("NSE")
        status = "OK"
        try:
            now = tz_now()
            schedule = cal.schedule(start_date=now.date(), end_date=now.date())
            if schedule.empty:
                status = "Closed (Holiday/Weekend)"
            else:
                open_t = schedule.iloc[0]['market_open'].tz_convert(MARKET_TZ)
                close_t = schedule.iloc[0]['market_close'].tz_convert(MARKET_TZ)
                if open_t <= tz_now() < close_t:
                    status = "LIVE (Open)"
                else:
                    status = "Closed (Off Hours)"
        except:
            status = "Calendar OK"
    except Exception:
        status = "Market calendar unavailable"
    st.sidebar.markdown(f"**Market status:** {status}")

    # Ticker selection (sidebar)
    all_names = list(DEFAULT_TICKERS.keys())
    default_sel = all_names[:8] if len(all_names) >= 8 else all_names
    selected_names = st.sidebar.multiselect("Tickers to show", all_names, default=default_sel)
    if not selected_names:
        selected_names = default_sel

    days_display = st.sidebar.slider("Days for card history (used for ADX/EMA)", min_value=60, max_value=730, value=180, step=30)
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear all caches & refresh"):
        st.cache_data.clear()
        st.rerun()

    # Session state for selected instrument
    if 'selected_ticker' not in st.session_state:
        st.session_state['selected_ticker'] = None
        st.session_state['selected_name'] = None

    # Main area - dashboard or detail
    if st.session_state.get('selected_ticker'):
        render_detail(st.session_state['selected_ticker'], st.session_state['selected_name'])
    else:
        # Prepare tickers dict
        tickers_to_show = {name: DEFAULT_TICKERS[name] for name in selected_names}
        render_dashboard(tickers_to_show, days_display)

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    main()
