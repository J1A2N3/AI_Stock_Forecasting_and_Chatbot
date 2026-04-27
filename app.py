import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# ============ LOAD DATA ============
@st.cache_data
def load_data():
    try:
        data = yf.download("TCS.NS", period="5y", progress=False, timeout=30)
        if data is None or data.empty:
            time.sleep(2)
            data = yf.download("TCS.NS", period="5y", progress=False, timeout=30)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        return data if len(data) > 100 else None
    except Exception as e:
        st.error(f"Error loading data: {str(e)[:80]}")
        return None

# ============ FEATURES ============
def create_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Returns'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['High_Low'] = df['High'] - df['Low']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    for i in range(1, 4):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    return df.dropna()

# ============ TRAIN MODEL ============
@st.cache_resource
def train_model(data):
    df = create_features(data)
    cols = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 'RSI', 'High_Low', 'Lag_1', 'Lag_2', 'Lag_3']
    X = df[cols].values
    y = df['Close'].values.ravel()
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, cols, df

# ============ PREDICT ============
def predict_prices(model, df, cols, days=30):
    predictions = []
    last_row = {}
    for col in df.columns:
        val = df[col].iloc[-1]
        if hasattr(val, 'item'):
            val = val.item()
        last_row[col] = float(val)

    for _ in range(days):
        features = [last_row[col] for col in cols]
        X = np.array([features])
        next_price = float(model.predict(X)[0])
        predictions.append(next_price)
        last_row['Lag_3'] = last_row['Lag_2']
        last_row['Lag_2'] = last_row['Lag_1']
        last_row['Lag_1'] = last_row['Close']
        last_row['Close'] = next_price
        last_row['MA_5'] = float(np.mean([
            last_row['Close'], last_row['Lag_1'],
            last_row['Lag_2'], last_row['Lag_3'], next_price
        ]))
    return np.array(predictions)

# ============ ANSWER ENGINE ============
def get_answer(question, current_price, prev_price, change, change_pct,
               future_prices, current_rsi, week_high, week_low, ma5, ma20, data):
    q = question.lower()

    # Current price
    if any(x in q for x in ['current price', 'price now', 'what is price', 'price today', 'trading at', 'how much']):
        direction = "🟢 up" if change > 0 else "🔴 down"
        return (f"**TCS is currently trading at ₹{current_price:.2f}**\n\n"
                f"It's {direction} by ₹{abs(change):.2f} ({abs(change_pct):.2f}%) from yesterday's close of ₹{prev_price:.2f}.")

    # Tomorrow prediction
    elif any(x in q for x in ['tomorrow', 'next day', 'predict tomorrow', 'price tomorrow']):
        pred = future_prices[0]
        diff = pred - current_price
        pct = (diff / current_price) * 100
        direction = "rise 📈" if diff > 0 else "fall 📉"
        return (f"**Tomorrow's predicted price: ₹{pred:.2f}**\n\n"
                f"The model expects TCS to {direction} by ₹{abs(diff):.2f} ({abs(pct):.2f}%) from today's ₹{current_price:.2f}.\n\n"
                f"⚠️ This is a model estimate, not a guarantee.")

    # Week prediction
    elif any(x in q for x in ['week', 'next 7', '7 days', 'weekly']):
        avg = np.mean(future_prices[:7])
        high = future_prices[:7].max()
        low = future_prices[:7].min()
        trend = "upward 📈" if future_prices[6] > current_price else "downward 📉"
        return (f"**Next 7-day outlook for TCS:**\n\n"
                f"- Average predicted price: ₹{avg:.2f}\n"
                f"- Expected high: ₹{high:.2f}\n"
                f"- Expected low: ₹{low:.2f}\n"
                f"- Overall trend: {trend}\n\n"
                f"⚠️ Predictions become less reliable further out.")

    # Month prediction
    elif any(x in q for x in ['month', '30 days', 'next month', 'monthly']):
        avg = np.mean(future_prices[:30])
        high = future_prices[:30].max()
        low = future_prices[:30].min()
        end = future_prices[29]
        trend = "📈 Bullish" if end > current_price else "📉 Bearish"
        return (f"**30-day outlook for TCS:**\n\n"
                f"- Average predicted price: ₹{avg:.2f}\n"
                f"- Expected high: ₹{high:.2f}\n"
                f"- Expected low: ₹{low:.2f}\n"
                f"- End of month estimate: ₹{end:.2f}\n"
                f"- Sentiment: {trend}\n\n"
                f"⚠️ Long-range predictions carry significant uncertainty.")

    # Buy / sell signal
    elif any(x in q for x in ['buy', 'sell', 'invest', 'should i', 'good time', 'worth buying']):
        rsi_msg = ""
        if current_rsi > 70:
            rsi_msg = "🔴 RSI is above 70 — stock may be **overbought**. Consider waiting for a pullback."
        elif current_rsi < 30:
            rsi_msg = "🟢 RSI is below 30 — stock may be **oversold**. Could be a buying opportunity."
        else:
            rsi_msg = "🟡 RSI is in the **neutral zone** (neither overbought nor oversold)."

        ma_signal = "above" if current_price > ma20 else "below"
        ma_msg = f"Price is {ma_signal} the 20-day moving average (₹{ma20:.2f})."
        trend_30 = "upward 📈" if future_prices[29] > current_price else "downward 📉"

        return (f"**TCS Buy/Sell Analysis:**\n\n"
                f"- {rsi_msg}\n"
                f"- {ma_msg}\n"
                f"- 30-day model trend: {trend_30}\n"
                f"- Current RSI: {current_rsi:.1f}\n\n"
                f"⚠️ This is not financial advice. Always consult a SEBI-registered advisor.")

    # RSI
    elif any(x in q for x in ['rsi', 'overbought', 'oversold', 'momentum']):
        if current_rsi > 70:
            status = "Overbought 🔴 — momentum is high, possible correction ahead."
        elif current_rsi < 30:
            status = "Oversold 🟢 — momentum is low, possible bounce ahead."
        else:
            status = "Neutral 🟡 — no extreme momentum signals."
        return (f"**RSI (Relative Strength Index): {current_rsi:.1f}**\n\n"
                f"Status: {status}\n\n"
                f"RSI ranges: Below 30 = oversold, 30–70 = neutral, Above 70 = overbought.")

    # Moving averages
    elif any(x in q for x in ['moving average', 'ma', 'trend', 'ma5', 'ma20']):
        ma_trend = "bullish 📈" if ma5 > ma20 else "bearish 📉"
        return (f"**TCS Moving Averages:**\n\n"
                f"- 5-day MA: ₹{ma5:.2f}\n"
                f"- 20-day MA: ₹{ma20:.2f}\n"
                f"- Signal: MA5 is {'above' if ma5 > ma20 else 'below'} MA20 → {ma_trend}\n\n"
                f"A bullish crossover (MA5 > MA20) suggests short-term upward momentum.")

    # 52-week / high / low
    elif any(x in q for x in ['52 week', '52week', 'year high', 'year low', 'all time', 'highest', 'lowest']):
        high_52 = float(data['High'].tail(252).max())
        low_52 = float(data['Low'].tail(252).min())
        from_high = ((current_price - high_52) / high_52) * 100
        from_low = ((current_price - low_52) / low_52) * 100
        return (f"**TCS 52-Week Range:**\n\n"
                f"- 52-week High: ₹{high_52:.2f} (current is {abs(from_high):.1f}% {'below' if from_high < 0 else 'above'})\n"
                f"- 52-week Low: ₹{low_52:.2f} (current is {from_low:.1f}% above)\n"
                f"- Current Price: ₹{current_price:.2f}")

    # Volume
    elif any(x in q for x in ['volume', 'traded', 'liquidity']):
        vol_today = float(data['Volume'].iloc[-1])
        vol_avg = float(data['Volume'].tail(20).mean())
        vol_signal = "above" if vol_today > vol_avg else "below"
        return (f"**TCS Volume Analysis:**\n\n"
                f"- Today's volume: {vol_today:,.0f} shares\n"
                f"- 20-day avg volume: {vol_avg:,.0f} shares\n"
                f"- Today's volume is {vol_signal} average.\n\n"
                f"Higher-than-average volume on a price rise suggests strong buying interest.")

    # About TCS
    elif any(x in q for x in ['what is tcs', 'about tcs', 'tata consultancy', 'company']):
        return ("**About TCS (Tata Consultancy Services):**\n\n"
                "TCS is India's largest IT services company and a subsidiary of Tata Group. "
                "Listed on NSE/BSE as TCS.NS, it operates in over 46 countries, serving clients in banking, "
                "retail, telecom, and more. It is a constituent of the Nifty 50 and Sensex indices.\n\n"
                f"Current market price: ₹{current_price:.2f}")

    # Help / what can you do
    elif any(x in q for x in ['help', 'what can you', 'what do you', 'options', 'queries']):
        return ("**I can answer questions like:**\n\n"
                "- What is the current TCS price?\n"
                "- What will TCS price be tomorrow?\n"
                "- What's the next week/month outlook?\n"
                "- Should I buy TCS now?\n"
                "- What is the RSI?\n"
                "- Show moving averages\n"
                "- What is the 52-week high/low?\n"
                "- What is today's trading volume?\n"
                "- Tell me about TCS\n\n"
                "Just type your question naturally!")

    else:
        return ("I'm not sure about that. Try asking:\n\n"
                "- *Current price* / *Tomorrow's prediction* / *Next week outlook*\n"
                "- *Should I buy?* / *RSI* / *Moving averages*\n"
                "- *52-week high* / *Volume* / *About TCS*\n\n"
                "Or type **help** to see all I can answer.")

# ============ MAIN ============
st.title("📈 TCS Stock Assistant")

# Load data
with st.spinner("Loading TCS data and training model..."):
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please refresh.")
        st.stop()
    model, cols, df_proc = train_model(data)

# Compute values
close_prices = data['Close'].values.flatten()
current_price = float(close_prices[-1])
prev_price = float(close_prices[-2])
change = current_price - prev_price
change_pct = (change / prev_price) * 100
week_high = float(data['High'].tail(7).max())
week_low = float(data['Low'].tail(7).min())
df_feat = create_features(data)
current_rsi = float(df_feat['RSI'].values.flatten()[-1])
ma5 = float(df_feat['MA_5'].values.flatten()[-1])
ma20 = float(df_feat['MA_20'].values.flatten()[-1])
future_prices = predict_prices(model, df_proc, cols, days=90)

# Top metrics strip
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("TCS Price", f"₹{current_price:.2f}", f"{change_pct:+.2f}%")
with col2:
    st.metric("RSI", f"{current_rsi:.1f}")
with col3:
    st.metric("MA5 / MA20", f"₹{ma5:.0f} / ₹{ma20:.0f}")
with col4:
    st.metric("Tomorrow Forecast", f"₹{future_prices[0]:.2f}")

st.divider()

# Sidebar quick buttons
with st.sidebar:
    st.title("📊 Quick Questions")
    st.caption("Click to ask instantly")
    questions = [
        "What is the current price?",
        "Price tomorrow?",
        "Next week trend?",
        "Next month outlook?",
        "Should I buy TCS?",
        "What is the RSI?",
        "Show moving averages",
        "52-week high and low?",
        "What is today's volume?",
        "Tell me about TCS",
    ]
    for q in questions:
        if st.button(q, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            answer = get_answer(q, current_price, prev_price, change, change_pct,
                                future_prices, current_rsi, week_high, week_low, ma5, ma20, data)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": (
            f"👋 Hello! I'm your TCS Stock Assistant.\n\n"
            f"TCS is currently trading at **₹{current_price:.2f}** "
            f"({'🟢 +' if change >= 0 else '🔴 '}{change_pct:.2f}% today).\n\n"
            "Ask me anything about TCS — price, predictions, RSI, moving averages, buy/sell signals and more!"
        )}
    ]

# Render chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything about TCS stock..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = get_answer(prompt, current_price, prev_price, change, change_pct,
                        future_prices, current_rsi, week_high, week_low, ma5, ma20, data)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

st.divider()
st.caption("⚠️ Educational tool only. Not financial advice. Data from Yahoo Finance.")
