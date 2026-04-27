import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ---------------- UI ----------------
st.title("📈 AI Stock Predictor + Chatbot")

ticker = st.text_input("Enter Stock Symbol", "RELIANCE.NS")

# ---------------- Load Data ----------------
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, period="1y", progress=False)
        
        if data is None or data.empty:
            return None
            
        data = data[['Close']]
        data.dropna(inplace=True)
        return data
        
    except:
        return None

data = load_data(ticker)

# ---------------- Error Handling ----------------
if data is None:
    st.warning("⚠️ Could not fetch data. Try another stock symbol.")
    st.stop()

# ---------------- Chart ----------------
st.subheader("📊 Stock Price (Last 1 Year)")
st.line_chart(data)

# ---------------- Prediction ----------------
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
st.subheader("📈 Future Predictions (Next 30 Days)")
st.line_chart(future_prices)

# ---------------- Chatbot ----------------
st.subheader("💬 Ask AI about stock")

user_input = st.text_input("Ask something:")

def answer_question(question):
    question = question.lower()

    if "trend" in question:
        return "Check the chart above for the trend 📊"

    elif "price" in question or "future" in question:
        return f"Predicted next price: {future_prices[0][0]:.2f}"

    elif "next 5 days" in question:
        return str(future_prices[:5].flatten())

    elif "invest" in question:
        return "⚠️ This is a demo model — not financial advice"

    else:
        return "Try asking: trend, future price, next 5 days"

if st.button("Ask"):
    if user_input:
        response = answer_question(user_input)
        st.success(response)
