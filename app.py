import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ---------------- UI ----------------
st.title("📈 AI Stock Predictor + Chatbot")

ticker = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")

# ---------------- Load Data ----------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
    data = data[['Close']]
    data.dropna(inplace=True)
    return data

data = load_data(ticker)

if data.empty:
    st.error("Invalid stock symbol")
    st.stop()

st.subheader("📊 Stock Price")
st.line_chart(data)

# ---------------- Prediction (LIGHTWEIGHT MODEL) ----------------
def predict_future(days=30):
    last_prices = data['Close'].tail(60).values
    
    predictions = []
    temp = list(last_prices)

    for _ in range(days):
        next_val = sum(temp[-5:]) / 5   # moving average
        temp.append(next_val)
        predictions.append(next_val)

    return np.array(predictions).reshape(-1,1)

future_prices = predict_future(30)

# ---------------- Show Predictions ----------------
st.subheader("📈 Future Predictions")
st.line_chart(future_prices)

# ---------------- Chatbot ----------------
st.subheader("💬 Ask AI about stock")

user_input = st.text_input("Ask something:")

def answer_question(question):
    question = question.lower()

    if "trend" in question:
        return "Check the chart above for trend 📊"

    elif "price" in question or "future" in question:
        return f"Predicted next price: {future_prices[0][0]:.2f}"

    elif "next 5 days" in question:
        return str(future_prices[:5].flatten())

    elif "invest" in question:
        return "⚠️ Demo model — not financial advice"

    else:
        return "Ask about trend, future price, or next 5 days"

if st.button("Ask"):
    response = answer_question(user_input)
    st.success(response)
