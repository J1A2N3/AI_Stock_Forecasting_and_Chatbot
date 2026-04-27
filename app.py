import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

st.title("📈 AI Stock Predictor (RELIANCE)")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    try:
        data = yf.download("RELIANCE.NS", period="1y", progress=False)

        if data is None or data.empty:
            return None

        if 'Close' not in data.columns:
            return None

        data = data[['Close']]
        data.dropna(inplace=True)
        return data

    except:
        return None

data = load_data()

# ---------------- Handle Error ----------------
if data is None:
    st.error("❌ Failed to load RELIANCE data from Yahoo Finance.")
    st.info("👉 This happens due to API blocking on cloud.\nTry refreshing or redeploying.")
    st.stop()

# ---------------- Chart ----------------
st.subheader("📊 RELIANCE Stock Price")
st.line_chart(data['Close'])   # FIXED

# ---------------- Prediction ----------------
def predict_future(days=30):
    last_prices = data['Close'].tail(60).values

    predictions = []
    temp = list(last_prices)

    for _ in range(days):
        next_val = sum(temp[-5:]) / 5
        temp.append(next_val)
        predictions.append(next_val)

    return np.array(predictions)

future_prices = predict_future(30)

st.subheader("📈 Future Prediction")
st.line_chart(future_prices)

# ---------------- Chatbot ----------------
st.subheader("💬 Ask about RELIANCE")

user_input = st.text_input("Ask something:")

def answer_question(q):
    q = q.lower()

    if "trend" in q:
        return "Trend is visible in the chart above."

    elif "future" in q or "price" in q:
        return f"Next predicted price: {future_prices[0]:.2f}"

    elif "next 5 days" in q:
        return str(future_prices[:5])

    elif "invest" in q:
        return "⚠️ Demo model — not financial advice"

    else:
        return "Try: trend, future price, next 5 days"

if st.button("Ask"):
    if user_input:
        st.success(answer_question(user_input))
