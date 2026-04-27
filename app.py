import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# Simple sidebar
with st.sidebar:
    st.title("📊 Stock Predictor")
    st.markdown("**Ticker:** TCS.NS")
    st.markdown("**Type:** Indian Stock")
    st.divider()
    
    st.subheader("Questions")
    if st.button("What is the current price?", use_container_width=True):
        st.session_state.question = "current price"
    if st.button("Price tomorrow?", use_container_width=True):
        st.session_state.question = "tomorrow"
    if st.button("Next week trend?", use_container_width=True):
        st.session_state.question = "week"
    if st.button("Buy now?", use_container_width=True):
        st.session_state.question = "buy"

st.title("📈 TCS Stock Predictor")

# ============ LOAD DATA ============
@st.cache_data
def load_data():
    try:
        st.info("📥 Loading TCS data...")
        data = yf.download("TCS.NS", period="5y", progress=False, timeout=30)
        
        if data is None or data.empty:
            st.warning("No data. Retrying...")
            time.sleep(2)
            data = yf.download("TCS.NS", period="5y", progress=False, timeout=30)
        
        if data is None or data.empty:
            st.error("Could not load data")
            return None
            
        data = data.dropna()
        if len(data) > 100:
            st.success(f"✅ Loaded {len(data)} days")
            return data
        else:
            st.error("Not enough data")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)[:50]}")
        return None

# ============ FEATURES ============
def create_features(df):
    df = df.copy()
    
    # Flatten any MultiIndex columns from yfinance
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
    
    # Extract last row as a plain dict of Python scalars to avoid Series issues
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
        
        # Update the dict with new scalar values
        last_row['Lag_3'] = last_row['Lag_2']
        last_row['Lag_2'] = last_row['Lag_1']
        last_row['Lag_1'] = last_row['Close']
        last_row['Close'] = next_price
        last_row['MA_5'] = float(np.mean([
            last_row['Close'],
            last_row['Lag_1'],
            last_row['Lag_2'],
            last_row['Lag_3'],
            next_price
        ]))
    
    return np.array(predictions)

# ============ MAIN ============
if 'question' not in st.session_state:
    st.session_state.question = ""

with st.spinner("Loading model..."):
    data = load_data()
    if data is None:
        st.stop()
    
    model, cols, df_proc = train_model(data)

# Flatten MultiIndex if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Get current price
close_prices = data['Close'].values.flatten()
current_price = float(close_prices[-1])
prev_price = float(close_prices[-2])
change = current_price - prev_price
change_pct = (change / prev_price) * 100

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"₹{current_price:.2f}")
with col2:
    st.metric("Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
with col3:
    week_high = float(data['High'].tail(7).values.flatten().max())
    st.metric("Week High", f"₹{week_high:.2f}")

st.divider()

# Predictions
future_prices = predict_prices(model, df_proc, cols, days=90)

# Chart
st.subheader("📊 Price Chart & Prediction")
history = data['Close'].tail(60)
pred_dates = pd.date_range(data.index[-1], periods=31)[1:]

chart_df = pd.DataFrame({
    'Historical': list(history.values.flatten()) + [None]*30,
    'Prediction': [None]*60 + list(future_prices[:30])
})
chart_df.index = pd.date_range(history.index[0], periods=len(chart_df))

st.line_chart(chart_df, height=400)

st.divider()

# Analysis
df_feat = create_features(data)
current_rsi = float(df_feat['RSI'].values.flatten()[-1])

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**Tomorrow's Pred:** ₹{future_prices[0]:.2f}")
with col2:
    avg_week = np.mean(future_prices[:7])
    st.info(f"**Week Avg:** ₹{avg_week:.2f}")
with col3:
    st.info(f"**RSI:** {current_rsi:.1f}")

st.divider()

# Q&A
st.subheader("💬 Quick Answer")

question = st.session_state.get('question', '').lower()

if 'current' in question or 'price' in question:
    st.write(f"📊 Current price: **₹{current_price:.2f}**")
    st.write(f"Change: ₹{change:.2f} ({change_pct:+.2f}%)")

elif 'tomorrow' in question:
    st.write(f"🔮 Tomorrow's price: **₹{future_prices[0]:.2f}**")
    change_tomorrow = ((future_prices[0] - current_price) / current_price) * 100
    st.write(f"Expected change: {change_tomorrow:+.2f}%")

elif 'week' in question:
    avg_week = np.mean(future_prices[:7])
    st.write(f"📅 Next week average: **₹{avg_week:.2f}**")
    st.write(f"Range: ₹{future_prices[:7].min():.2f} - ₹{future_prices[:7].max():.2f}")

elif 'buy' in question or 'invest' in question:
    if current_rsi > 70:
        st.warning("⚠️ RSI > 70 (Overbought)")
    elif current_rsi < 30:
        st.info("ℹ️ RSI < 30 (Oversold)")
    else:
        st.success("✅ RSI in normal range")
    
    trend = "📈 Upward" if future_prices[30] > current_price else "📉 Downward"
    st.write(f"30-day trend: {trend}")
    st.write("⚠️ Not financial advice. Consult advisor.")

else:
    st.write("Click a question on the left sidebar! 👈")

st.divider()
st.caption("⚠️ Educational tool only. Not financial advice.")
