import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import datetime
import time

st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

# ---------------- SIDEBAR WITH SUGGESTED QUESTIONS ----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8f/Reliance_Industries_Logo.svg/1200px-Reliance_Industries_Logo.svg.png", width=200)
    st.title("🤖 AI Assistant")
    
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
    st.subheader("❔ Frequently Asked Questions")
    
    with st.expander("📊 What is this tool?"):
        st.write("""
        This is an AI-powered stock prediction tool that uses Machine Learning 
        (Random Forest) to analyze 5 years of historical data and predict future 
        prices for Reliance Industries stock.
        """)
    
    with st.expander("🤖 How accurate are the predictions?"):
        st.write("""
        The model uses technical indicators and historical patterns. However, 
        stock markets are influenced by many unpredictable factors. Use these 
        predictions as one of many tools in your analysis, not as sole guidance.
        """)
    
    with st.expander("📈 What does RSI mean?"):
        st.write("""
        RSI (Relative Strength Index) measures momentum:
        - RSI > 70: Overbought (might fall)
        - RSI < 30: Oversold (might rise)
        - RSI 30-70: Neutral range
        """)
    
    with st.expander("🎯 What questions can I ask?"):
        st.write("""
        You can ask about:
        - Current prices and changes
        - Future predictions (days/weeks/months)
        - Trend analysis
        - High/Low prices
        - Volatility and risk levels
        - Trading volume
        - Returns and performance
        - Investment insights
        """)
    
    with st.expander("⚠️ Is this financial advice?"):
        st.write("""
        **NO!** This is an educational tool only. Always consult with a 
        certified financial advisor before making investment decisions. 
        Past performance doesn't guarantee future results.
        """)
    
    with st.expander("📱 How to use this tool?"):
        st.write("""
        1. View the charts and predictions on the main page
        2. Click on suggested questions or type your own
        3. Explore different tabs for detailed analysis
        4. Adjust prediction days using the slider
        5. Check technical indicators for insights
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    show_chart_history = st.checkbox("Show 2-year history", value=False)
    show_volume_chart = st.checkbox("Show volume chart", value=True)
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & ML")

st.title("📈 AI Stock Predictor - RELIANCE (Machine Learning)")

# ---------------- LOAD DATA WITH RATE LIMIT HANDLING ----------------
@st.cache_data
def load_data():
    """Load stock data with rate limit handling, retries, and timeout"""
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Add delay between retries to avoid rate limiting
            if attempt > 0:
                remaining = retry_delay * (attempt + 1)
                st.warning(f"⏳ Rate limited. Waiting {remaining} seconds before retry (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(remaining)
            
            # Show loading message on first attempt
            if attempt == 0:
                st.info("📥 Loading RELIANCE stock data from Yahoo Finance...")
            
            # Download with explicit timeout and error handling
            data = yf.download(
                "RELIANCE.NS", 
                period="5y", 
                progress=False,
                timeout=30
            )
            
            if data is None or data.empty:
                if attempt < max_retries - 1:
                    st.warning("⚠️ No data returned. Retrying...")
                    continue
                else:
                    st.error("❌ No data available after all retry attempts")
                    return None
            
            # Clean data
            data = data.dropna()
            
            if len(data) > 100:  # Ensure we have enough data
                st.success(f"✅ Successfully loaded {len(data)} days of historical data!")
                return data
            else:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Insufficient data ({len(data)} records). Retrying...")
                    continue
                else:
                    st.error("❌ Could not load sufficient data")
                    return None
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # Rate limit error handling
            if "rate limit" in error_msg or "too many requests" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    st.warning(f"⏳ Yahoo Finance rate limit hit. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    st.error("❌ Yahoo Finance rate limit exceeded after multiple retries")
                    st.info("💡 **Solutions:**\n- Wait a few minutes and refresh the page\n- Check your internet connection\n- Try again during off-peak hours")
                    return None
            
            # Network/timeout errors
            elif "timeout" in error_msg or "connection" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Connection issue. Retrying (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    st.error(f"❌ Could not connect to Yahoo Finance after {max_retries} attempts")
                    return None
            
            # Other errors
            else:
                st.error(f"❌ Error loading data: {type(e).__name__} - {str(e)[:100]}")
                return None
    
    return None

# ---------------- FEATURE ENGINEERING ----------------
def create_features(df):
    """Create technical indicators and features for ML model"""
    df = df.copy()
    
    # Price features
    df['Returns'] = df['Close'].pct_change()
    df['High_Low'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Lag features (previous days)
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    return df.dropna()

# ---------------- ML MODEL ----------------
@st.cache_resource
def train_model(data):
    """Train Random Forest model on historical data"""
    df = create_features(data)
    
    # Features for prediction
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'Returns', 'High_Low', 
                    'Price_Change', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 
                    'Volatility', 'RSI', 'Close_Lag_1', 'Close_Lag_2', 
                    'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5']
    
    X = df[feature_cols]
    y = df['Close']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, feature_cols, feature_importance

# ---------------- PREDICTION ----------------
def predict_future(model, data, feature_cols, days=30):
    """Predict future prices using trained model"""
    df = create_features(data)
    predictions = []
    
    # Start with the last known row
    last_row = df.iloc[-1:].copy()
    
    for _ in range(days):
        # Prepare features
        X_pred = last_row[feature_cols]
        
        # Predict next day
        next_price = model.predict(X_pred)[0]
        predictions.append(next_price)
        
        # Update features for next prediction (simplified)
        new_row = last_row.copy()
        new_row['Close'] = next_price
        new_row['Close_Lag_5'] = new_row['Close_Lag_4'].values
        new_row['Close_Lag_4'] = new_row['Close_Lag_3'].values
        new_row['Close_Lag_3'] = new_row['Close_Lag_2'].values
        new_row['Close_Lag_2'] = new_row['Close_Lag_1'].values
        new_row['Close_Lag_1'] = next_price
        new_row['MA_5'] = np.mean([new_row['Close_Lag_1'].values[0], 
                                    new_row['Close_Lag_2'].values[0],
                                    new_row['Close_Lag_3'].values[0],
                                    new_row['Close_Lag_4'].values[0],
                                    new_row['Close_Lag_5'].values[0]])
        
        last_row = new_row
    
    return np.array(predictions)

# ============ MAIN APP LOGIC ============
# Initialize session state first
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""
if 'trigger_ask' not in st.session_state:
    st.session_state.trigger_ask = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load and process data with status messages
st.subheader("⏳ Loading and Training Model...")
progress_placeholder = st.empty()

with st.spinner("Loading data and training model..."):
    data = load_data()
    
    if data is None:
        st.stop()

try:
    progress_placeholder.info("🤖 Training machine learning model...")
    model, feature_cols, feature_importance = train_model(data)
    close_series = data['Close'].dropna()

    if len(close_series) < 2:
        st.error("❌ Not enough valid data for analysis")
        st.stop()

    current_price = float(close_series.iloc[-1])
    prev_price = float(close_series.iloc[-2])
    
    progress_placeholder.success("✅ Model trained successfully!")
    st.session_state.data_loaded = True
    
except Exception as e:
    st.error(f"❌ Error during model training: {str(e)}")
    st.stop()

# Clear loading message
progress_placeholder.empty()

# ---------------- DISPLAY METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

change = current_price - prev_price
change_pct = (change / prev_price) * 100
week_high = float(data['High'].tail(7).max())
week_low = float(data['Low'].tail(7).min())

with col1:
    st.metric("Current Price", f"₹{current_price:.2f}")

with col2:
    st.metric("Today's Change", f"₹{change:.2f}", f"{change_pct:.2f}%")

with col3:
    st.metric("Week High", f"₹{week_high:.2f}")

with col4:
    st.metric("Week Low", f"₹{week_low:.2f}")

# Generate predictions
future_prices = predict_future(model, data, feature_cols, days=90)
df_with_features = create_features(data)

# ---------------- CHARTS ----------------
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
    
    # Create date range for predictions
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_days)
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_prices[:pred_days]
    }).set_index('Date')
    
    st.subheader(f"📈 Price Prediction (Next {pred_days} Days)")
    
    # Combine historical and predicted data for visualization
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
    
    # Prediction statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Price (Next Day)", f"₹{future_prices[0]:.2f}")
    with col2:
        avg_pred = np.mean(future_prices[:pred_days])
        st.metric(f"Average ({pred_days} days)", f"₹{avg_pred:.2f}")
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

# ---------------- AI CHATBOT ----------------
st.markdown("---")
st.subheader("💬 AI Assistant - Ask Anything About RELIANCE")

def answer_question(question, data, future_prices, model):
    """Intelligent question answering system"""
    q = question.lower()
    
    # Current price questions
    if any(word in q for word in ['current', 'price', 'now', 'today']):
        return f"📊 The current price of Reliance is **₹{current_price:.2f}**. Today's change is ₹{change:.2f} ({change_pct:+.2f}%)."
    
    # Future predictions
    elif 'tomorrow' in q or 'next day' in q:
        return f"🔮 Predicted price for tomorrow: **₹{future_prices[0]:.2f}**"
    
    elif 'next week' in q or '7 days' in q or 'week' in q:
        avg_week = np.mean(future_prices[:7])
        return f"📅 Average predicted price for next week: **₹{avg_week:.2f}**\n\nPrices range from ₹{future_prices[:7].min():.2f} to ₹{future_prices[:7].max():.2f}"
    
    elif 'next month' in q or '30 days' in q or 'month' in q:
        avg_month = np.mean(future_prices[:30])
        return f"📆 Average predicted price for next month: **₹{avg_month:.2f}**\n\nExpected change: {((avg_month - current_price) / current_price * 100):+.2f}%"
    
    # Trend analysis
    elif 'trend' in q:
        ma_5 = data['Close'].tail(5).mean()
        ma_20 = data['Close'].tail(20).mean()
        if current_price > ma_5 > ma_20:
            return f"📈 **Strong upward trend**\n\nCurrent price (₹{current_price:.2f}) is above both:\n- 5-day MA: ₹{ma_5:.2f}\n- 20-day MA: ₹{ma_20:.2f}"
        elif current_price < ma_5 < ma_20:
            return f"📉 **Downward trend**\n\nCurrent price (₹{current_price:.2f}) is below both moving averages."
        else:
            return f"➡️ **Mixed/Sideways trend**\n\nPrice is consolidating around ₹{current_price:.2f}"
    
    # High/Low questions
    elif 'high' in q:
        if 'week' in q:
            return f"📊 This week's high: **₹{week_high:.2f}**"
        elif 'month' in q:
            month_high = data['High'].tail(30).max()
            return f"📊 This month's high: **₹{month_high:.2f}**"
        else:
            year_high = data['High'].tail(252).max()
            return f"📊 52-week high: **₹{year_high:.2f}**"
    
    elif 'low' in q:
        if 'week' in q:
            return f"📊 This week's low: **₹{week_low:.2f}**"
        elif 'month' in q:
            month_low = data['Low'].tail(30).min()
            return f"📊 This month's low: **₹{month_low:.2f}**"
        else:
            year_low = data['Low'].tail(252).min()
            return f"📊 52-week low: **₹{year_low:.2f}**"
    
    # Investment advice
    elif 'buy' in q or 'invest' in q or 'should i' in q:
        current_rsi = float(df_with_features['RSI'].iloc[-1])
        trend = "bullish 📈" if current_price > data['Close'].tail(20).mean() else "bearish 📉"
        prediction_trend = "Upward ⬆️" if future_prices[30] > current_price else "Downward ⬇️"
        
        return f"""📊 **Market Analysis:**

**Current Status:**
- Trend: {trend}
- RSI: {current_rsi:.2f} {"(Overbought 🔴)" if current_rsi > 70 else "(Oversold 🟢)" if current_rsi < 30 else "(Neutral 🟡)"}
- 30-day Prediction: {prediction_trend} movement expected

**Signals:**
{'- ✅ Positive indicators for entry' if current_rsi < 50 and prediction_trend.startswith('Up') else '- ⚠️ Mixed signals - exercise caution'}

⚠️ **IMPORTANT DISCLAIMER**: This is an AI prediction model for educational purposes only. NOT financial advice. Always consult a certified financial advisor before investing."""
    
    # Volatility
    elif 'volatile' in q or 'risk' in q:
        volatility = float(df_with_features['Volatility'].iloc[-1]) * 100
        risk_level = 'High 🔴' if volatility > 2 else 'Moderate 🟡' if volatility > 1 else 'Low 🟢'
        return f"📊 **Volatility Analysis:**\n\nCurrent volatility: **{volatility:.2f}%**\nRisk level: **{risk_level}**"
    
    # Volume
    elif 'volume' in q:
        avg_vol = data['Volume'].tail(20).mean()
        today_vol = float(data['Volume'].iloc[-1])
        vol_status = '📈 Above average' if today_vol > avg_vol else '📉 Below average'
        return f"📊 **Trading Volume:**\n\nToday: **{today_vol:,.0f}**\n20-day avg: {avg_vol:,.0f}\nStatus: {vol_status}"
    
    # Performance
    elif 'performance' in q or 'return' in q:
        if 'week' in q:
            week_return = ((current_price - data['Close'].iloc[-7]) / data['Close'].iloc[-7]) * 100
            emoji = "🟢" if week_return > 0 else "🔴"
            return f"{emoji} **This week's return:** {week_return:+.2f}%"
        elif 'month' in q:
            month_return = ((current_price - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100
            emoji = "🟢" if month_return > 0 else "🔴"
            return f"{emoji} **This month's return:** {month_return:+.2f}%"
        else:
            year_return = ((current_price - data['Close'].iloc[-252]) / data['Close'].iloc[-252]) * 100
            emoji = "🟢" if year_return > 0 else "🔴"
            return f"{emoji} **This year's return:** {year_return:+.2f}%"
    
    # Default response
    else:
        return """🤖 **I can help you with:**

✅ Current price and changes
✅ Future predictions (tomorrow, week, month)
✅ Trend analysis
✅ High/Low prices
✅ Volatility and risk assessment
✅ Trading volume analysis
✅ Performance and returns
✅ Investment insights

💡 **Try asking:**
- "What's the current price?"
- "Should I invest now?"
- "What will be the price next week?"
- "What is the trend?"
- "How volatile is the stock?"

Click on **suggested questions** in the sidebar! 👈"""

# Chat interface
col1, col2 = st.columns([4, 1])

with col1:
    user_question_input = st.text_input(
        "Type your question here:", 
        value=st.session_state.user_question,
        key="question_input",
        placeholder="e.g., What is the current price?"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    ask_button = st.button("🚀 Ask", use_container_width=True)

# Handle question submission
if ask_button or st.session_state.trigger_ask:
    current_question = user_question_input or st.session_state.user_question
    
    if current_question:
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": current_question})
        
        # Generate answer
        answer = answer_question(current_question, data, future_prices, model)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Reset trigger
        st.session_state.trigger_ask = False
        st.session_state.user_question = ""
        st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Chat History")
    
    # Display in reverse order (most recent first)
    for i, message in enumerate(reversed(st.session_state.chat_history[-10:])):
        if message["role"] == "user":
            st.info(f"**🧑 You:** {message['content']}")
        else:
            st.success(f"**🤖 AI:** {message['content']}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This is an AI-based prediction model for educational purposes only. Not financial advice. Past performance doesn't guarantee future results. Always do your own research and consult with certified financial advisors before making investment decisions.")
