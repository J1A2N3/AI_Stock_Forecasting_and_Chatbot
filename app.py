import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime
import time

st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

# Sidebar for ticker selection
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    st.subheader("📊 Select Stock Ticker")
    
    # Popular tickers
    popular_tickers = {
        "TCS (Tata Consultancy)": "TCS.NS",
        "INFY (Infosys)": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Axis Bank": "AXISBANK.NS",
        "SBI (State Bank)": "SBIN.NS",
        "Wipro": "WIPRO.NS",
        "HCL Technologies": "HCLTECH.NS",
        "HDFC Corp": "HDFC.NS",
        "ITC Limited": "ITC.NS",
        "Apple (US)": "AAPL",
        "Microsoft (US)": "MSFT",
        "Google (US)": "GOOGL",
        "Tesla (US)": "TSLA",
        "Amazon (US)": "AMZN",
    }
    
    # Tabs for selection
    tab1, tab2 = st.tabs(["Popular", "Custom"])
    
    with tab1:
        selected_company = st.selectbox(
            "Choose from popular stocks:",
            list(popular_tickers.keys()),
            index=0
        )
        ticker = popular_tickers[selected_company]
    
    with tab2:
        ticker = st.text_input(
            "Enter ticker symbol:",
            value="TCS.NS",
            help="e.g., INFY.NS for Infosys, AAPL for Apple"
        )
    
    st.markdown("---")
    st.subheader("💡 Suggested Questions")
    
    suggested_questions = [
        "What is the current price?",
        "What will be the price tomorrow?",
        "Should I buy or invest now?",
        "What is the trend right now?",
        "Show me next week prediction",
        "What is next month's forecast?",
        "How volatile is the stock?",
        "What's the 52-week high and low?",
        "How is the performance this month?",
        "What is the trading volume today?",
    ]
    
    for i, question in enumerate(suggested_questions):
        if st.button(f"❓ {question}", key=f"suggested_{i}", use_container_width=True):
            st.session_state.user_question = question
            st.session_state.trigger_ask = True
    
    st.markdown("---")
    st.subheader("❔ FAQ")
    
    with st.expander("📊 What is this tool?"):
        st.write("""
        This is an AI-powered stock prediction tool that uses Machine Learning 
        (Random Forest) to analyze 5 years of historical data and predict future prices.
        """)
    
    with st.expander("🤖 How accurate?"):
        st.write("""
        The model uses technical indicators and historical patterns. However, 
        stock markets are unpredictable. Use this as ONE tool in your analysis.
        """)
    
    with st.expander("📈 What does RSI mean?"):
        st.write("""
        RSI (Relative Strength Index) measures momentum:
        - RSI > 70: Overbought (might fall)
        - RSI < 30: Oversold (might rise)
        - RSI 30-70: Neutral range
        """)
    
    with st.expander("⚠️ Is this financial advice?"):
        st.write("""
        **NO!** Educational tool only. Always consult a certified 
        financial advisor before investing.
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Display Settings")
    
    show_chart_history = st.checkbox("Show 2-year history", value=False)
    show_volume_chart = st.checkbox("Show volume chart", value=True)
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & ML")

st.title(f"📈 AI Stock Predictor - {ticker} (Machine Learning)")

# ============ DATA LOADING WITH RATE LIMIT HANDLING ============
@st.cache_data
def load_data(ticker_symbol):
    """Load stock data with rate limit handling and retries"""
    max_retries = 4
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                remaining = retry_delay * (2 ** (attempt - 1))
                st.warning(f"⏳ Retrying in {remaining}s... (Attempt {attempt}/{max_retries})")
                time.sleep(remaining)
            
            if attempt == 0:
                st.info(f"📥 Loading {ticker_symbol} data from Yahoo Finance...")
            
            # Download with timeout
            data = yf.download(
                ticker_symbol,
                period="5y",
                progress=False,
                timeout=30
            )
            
            if data is None or data.empty:
                if attempt < max_retries - 1:
                    st.warning("⚠️ No data returned. Retrying...")
                    continue
                else:
                    st.error(f"❌ No data available for {ticker_symbol}")
                    return None
            
            data = data.dropna()
            
            if len(data) > 100:
                st.success(f"✅ Successfully loaded {len(data)} days of data for {ticker_symbol}!")
                return data
            else:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Insufficient data. Retrying...")
                    continue
                else:
                    st.error(f"❌ Not enough data for {ticker_symbol}")
                    return None
        
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg or "too many requests" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    st.warning(f"⏳ Rate limited. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    st.error(f"❌ Yahoo Finance rate limit exceeded. Try another ticker or wait a few minutes.")
                    st.info("💡 **Try these alternative tickers:**\n- TCS.NS (Tata Consultancy)\n- INFY.NS (Infosys)\n- AAPL (Apple)\n- MSFT (Microsoft)")
                    return None
            
            elif "timeout" in error_msg or "connection" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Connection issue. Retrying...")
                    time.sleep(retry_delay)
                else:
                    st.error(f"❌ Could not connect to Yahoo Finance")
                    return None
            
            else:
                st.error(f"❌ Error: {type(e).__name__}")
                return None
    
    return None

# ============ FEATURE ENGINEERING ============
def create_features(df):
    """Create technical indicators and features"""
    df = df.copy()
    
    df['Returns'] = df['Close'].pct_change()
    df['High_Low'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Lag features
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    return df.dropna()

# ============ ML MODEL ============
@st.cache_resource
def train_model(data):
    """Train Random Forest model"""
    df = create_features(data)
    
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'Returns', 'High_Low', 
                    'Price_Change', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 
                    'Volatility', 'RSI', 'Close_Lag_1', 'Close_Lag_2', 
                    'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5']
    
    X = df[feature_cols].values  # Convert to numpy array
    y = df['Close'].values.ravel()  # 1D array
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, feature_cols, feature_importance, df

# ============ PREDICTION ============
def predict_future(model, df, feature_cols, days=30):
    """Predict future prices - FIXED: Proper scalar handling"""
    predictions = []
    
    # Get last row as starting point
    last_row_data = df.iloc[-1].copy()
    
    for _ in range(days):
        # Prepare features - ensure all are scalars
        X_pred = []
        for col in feature_cols:
            val = last_row_data[col]
            # Convert Series to scalar if needed
            if isinstance(val, (pd.Series, np.ndarray)):
                val = float(val.flat[0])
            X_pred.append(float(val))
        
        X_pred = np.array([X_pred])
        
        # Predict next price
        next_price = float(model.predict(X_pred)[0])
        predictions.append(next_price)
        
        # Update row for next iteration
        last_row_data['Close'] = next_price
        last_row_data['Returns'] = (next_price - last_row_data['Close']) / last_row_data['Close']
        last_row_data['Close_Lag_5'] = float(last_row_data['Close_Lag_4'])
        last_row_data['Close_Lag_4'] = float(last_row_data['Close_Lag_3'])
        last_row_data['Close_Lag_3'] = float(last_row_data['Close_Lag_2'])
        last_row_data['Close_Lag_2'] = float(last_row_data['Close_Lag_1'])
        last_row_data['Close_Lag_1'] = next_price
        
        # Update moving average
        ma_5_vals = [
            last_row_data['Close_Lag_1'],
            last_row_data['Close_Lag_2'],
            last_row_data['Close_Lag_3'],
            last_row_data['Close_Lag_4'],
            last_row_data['Close_Lag_5']
        ]
        last_row_data['MA_5'] = float(np.mean(ma_5_vals))
        last_row_data['Price_Change'] = next_price - last_row_data['Open']
    
    return np.array(predictions)

# ============ MAIN APP LOGIC ============
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""
if 'trigger_ask' not in st.session_state:
    st.session_state.trigger_ask = False

# Load data
st.subheader("⏳ Loading and Training Model...")
progress_placeholder = st.empty()

with st.spinner(f"Loading {ticker} and training model..."):
    data = load_data(ticker)
    
    if data is None:
        st.stop()

try:
    progress_placeholder.info("🤖 Training machine learning model...")
    model, feature_cols, feature_importance, df_processed = train_model(data)
    close_series = data['Close'].dropna()

    if len(close_series) < 2:
        st.error("❌ Not enough valid data for analysis")
        st.stop()

    current_price = float(close_series.iloc[-1])
    prev_price = float(close_series.iloc[-2])
    
    progress_placeholder.success("✅ Model trained successfully!")
    
except Exception as e:
    st.error(f"❌ Error during model training: {str(e)}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

progress_placeholder.empty()

# ============ DISPLAY METRICS ============
col1, col2, col3, col4 = st.columns(4)

change = current_price - prev_price
change_pct = (change / prev_price) * 100
week_high = float(data['High'].tail(7).max())
week_low = float(data['Low'].tail(7).min())

currency = "₹" if "NS" in ticker else "$"

with col1:
    st.metric("Current Price", f"{currency}{current_price:.2f}")

with col2:
    st.metric("Today's Change", f"{currency}{change:.2f}", f"{change_pct:.2f}%")

with col3:
    st.metric("Week High", f"{currency}{week_high:.2f}")

with col4:
    st.metric("Week Low", f"{currency}{week_low:.2f}")

# Generate predictions
future_prices = predict_future(model, df_processed, feature_cols, days=90)
df_with_features = create_features(data)

# ============ CHARTS ============
tab1, tab2, tab3 = st.tabs(["📊 Historical Data", "🔮 Predictions", "📈 Technical Analysis"])

with tab1:
    st.subheader("Stock Price History")
    history_days = 504 if show_chart_history else 252
    chart_data = data['Close'].tail(history_days)
    st.line_chart(chart_data, height=400)
    
    if show_volume_chart:
        st.subheader("Trading Volume")
        st.bar_chart(data['Volume'].tail(history_days), height=200)

with tab2:
    pred_days = st.slider("Prediction Days", 7, 90, 30)
    
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_days)
    
    st.subheader(f"📈 Price Prediction (Next {pred_days} Days)")
    
    historical_tail = data['Close'].tail(60)
    combined_df = pd.DataFrame({
        'Historical': historical_tail,
        'Predicted': pd.Series(dtype=float)
    })
    
    pred_df = pd.DataFrame({
        'Historical': pd.Series(dtype=float),
        'Predicted': future_prices[:pred_days]
    }, index=future_dates)
    
    combined = pd.concat([combined_df, pred_df])
    st.line_chart(combined, height=400)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Price (Next Day)", f"{currency}{future_prices[0]:.2f}")
    with col2:
        avg_pred = np.mean(future_prices[:pred_days])
        st.metric(f"Average ({pred_days} days)", f"{currency}{avg_pred:.2f}")
    with col3:
        change_pred = ((future_prices[pred_days-1] - current_price) / current_price) * 100
        st.metric("Expected Change", f"{change_pred:.2f}%")

with tab3:
    st.subheader("Technical Indicators")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Moving Averages**")
        ma_chart = df_with_features[['Close', 'MA_5', 'MA_10', 'MA_20', 'MA_50']].tail(100)
        st.line_chart(ma_chart, height=300)
    
    with col2:
        st.write("**RSI (Relative Strength Index)**")
        rsi_chart = df_with_features['RSI'].tail(100)
        st.line_chart(rsi_chart, height=300)
        
        current_rsi = float(df_with_features['RSI'].iloc[-1])
        if current_rsi > 70:
            st.warning("⚠️ RSI indicates overbought condition")
        elif current_rsi < 30:
            st.info("ℹ️ RSI indicates oversold condition")
        else:
            st.success("✅ RSI is in normal range")
    
    st.subheader("Feature Importance (Top 10)")
    st.bar_chart(feature_importance.head(10).set_index('feature'), height=300)

# ============ AI CHATBOT ============
st.markdown("---")
st.subheader("💬 AI Assistant - Ask Anything")

def answer_question(question, data, future_prices, model):
    """Intelligent Q&A system"""
    q = question.lower()
    
    if any(word in q for word in ['current', 'price', 'now', 'today']):
        return f"📊 Current price: **{currency}{current_price:.2f}**\nToday's change: {currency}{change:.2f} ({change_pct:+.2f}%)"
    
    elif 'tomorrow' in q or 'next day' in q:
        return f"🔮 Predicted price for tomorrow: **{currency}{future_prices[0]:.2f}**"
    
    elif 'next week' in q or '7 days' in q or 'week' in q:
        avg_week = np.mean(future_prices[:7])
        return f"📅 Average for next week: **{currency}{avg_week:.2f}**\n\nRange: {currency}{future_prices[:7].min():.2f} to {currency}{future_prices[:7].max():.2f}"
    
    elif 'next month' in q or '30 days' in q or 'month' in q:
        avg_month = np.mean(future_prices[:30])
        return f"📆 Average for next month: **{currency}{avg_month:.2f}**\n\nChange: {((avg_month - current_price) / current_price * 100):+.2f}%"
    
    elif 'trend' in q:
        ma_5 = data['Close'].tail(5).mean()
        ma_20 = data['Close'].tail(20).mean()
        if current_price > ma_5 > ma_20:
            return f"📈 **Strong upward trend**\n\nPrice above both MAs:\n- 5-day: {currency}{ma_5:.2f}\n- 20-day: {currency}{ma_20:.2f}"
        elif current_price < ma_5 < ma_20:
            return f"📉 **Downward trend**\n\nPrice below both moving averages"
        else:
            return f"➡️ **Mixed/Sideways trend**\n\nConsolidating around {currency}{current_price:.2f}"
    
    elif 'high' in q:
        if 'week' in q:
            return f"📊 This week's high: **{currency}{week_high:.2f}**"
        elif 'month' in q:
            month_high = data['High'].tail(30).max()
            return f"📊 This month's high: **{currency}{month_high:.2f}**"
        else:
            year_high = data['High'].tail(252).max()
            return f"📊 52-week high: **{currency}{year_high:.2f}**"
    
    elif 'low' in q:
        if 'week' in q:
            return f"📊 This week's low: **{currency}{week_low:.2f}**"
        elif 'month' in q:
            month_low = data['Low'].tail(30).min()
            return f"📊 This month's low: **{currency}{month_low:.2f}**"
        else:
            year_low = data['Low'].tail(252).min()
            return f"📊 52-week low: **{currency}{year_low:.2f}**"
    
    elif 'buy' in q or 'invest' in q or 'should i' in q:
        current_rsi = float(df_with_features['RSI'].iloc[-1])
        trend = "bullish 📈" if current_price > data['Close'].tail(20).mean() else "bearish 📉"
        prediction_trend = "Upward ⬆️" if future_prices[30] > current_price else "Downward ⬇️"
        
        return f"""📊 **Market Analysis:**
- Trend: {trend}
- RSI: {current_rsi:.2f} {"(Overbought 🔴)" if current_rsi > 70 else "(Oversold 🟢)" if current_rsi < 30 else "(Neutral 🟡)"}
- 30-day Prediction: {prediction_trend}

⚠️ NOT financial advice. Consult a certified advisor."""
    
    elif 'volatile' in q or 'risk' in q:
        volatility = float(df_with_features['Volatility'].iloc[-1]) * 100
        risk_level = 'High 🔴' if volatility > 2 else 'Moderate 🟡' if volatility > 1 else 'Low 🟢'
        return f"📊 Volatility: **{volatility:.2f}%**\nRisk level: **{risk_level}**"
    
    elif 'volume' in q:
        avg_vol = data['Volume'].tail(20).mean()
        today_vol = float(data['Volume'].iloc[-1])
        vol_status = '📈 Above average' if today_vol > avg_vol else '📉 Below average'
        return f"📊 Trading Volume:\nToday: **{today_vol:,.0f}**\n20-day avg: {avg_vol:,.0f}\nStatus: {vol_status}"
    
    elif 'performance' in q or 'return' in q:
        if 'week' in q:
            week_return = ((current_price - data['Close'].iloc[-7]) / data['Close'].iloc[-7]) * 100
            emoji = "🟢" if week_return > 0 else "🔴"
            return f"{emoji} This week: {week_return:+.2f}%"
        elif 'month' in q:
            month_return = ((current_price - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100
            emoji = "🟢" if month_return > 0 else "🔴"
            return f"{emoji} This month: {month_return:+.2f}%"
        else:
            year_return = ((current_price - data['Close'].iloc[-252]) / data['Close'].iloc[-252]) * 100
            emoji = "🟢" if year_return > 0 else "🔴"
            return f"{emoji} This year: {year_return:+.2f}%"
    
    else:
        return """🤖 Ask about:
✅ Current price and changes
✅ Future predictions (tomorrow, week, month)
✅ Trend analysis
✅ High/Low prices
✅ Volatility and risk
✅ Trading volume
✅ Returns and performance

Click **suggested questions** on the left! 👈"""

col1, col2 = st.columns([4, 1])

with col1:
    user_question_input = st.text_input(
        "Type your question here:", 
        value=st.session_state.user_question,
        key="question_input",
        placeholder="e.g., What is the current price?"
    )

with col2:
    st.write("")
    st.write("")
    ask_button = st.button("🚀 Ask", use_container_width=True)

if ask_button or st.session_state.trigger_ask:
    current_question = user_question_input or st.session_state.user_question
    
    if current_question:
        st.session_state.chat_history.append({"role": "user", "content": current_question})
        answer = answer_question(current_question, data, future_prices, model)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        st.session_state.trigger_ask = False
        st.session_state.user_question = ""
        st.rerun()

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Chat History")
    
    for message in reversed(st.session_state.chat_history[-10:]):
        if message["role"] == "user":
            st.info(f"**🧑 You:** {message['content']}")
        else:
            st.success(f"**🤖 AI:** {message['content']}")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("---")
st.caption("⚠️ Educational tool only. Not financial advice. Consult certified advisors before investing.")
