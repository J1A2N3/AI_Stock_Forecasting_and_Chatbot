import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

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

# ---------------- Preprocessing ----------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ---------------- Model ----------------
@st.cache_resource
def train_model(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=2, batch_size=32, verbose=0)  # reduced epochs for speed
    return model

model = train_model(X, y)

# ---------------- Prediction ----------------
def predict_future(days=30):
    temp_input = scaled_data[-60:].reshape(1, -1)
    temp_input = temp_input[0].tolist()

    output = []

    for i in range(days):
        x_input = np.array(temp_input[-60:])
        x_input = x_input.reshape(1, 60, 1)

        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        output.append(yhat[0][0])

    return scaler.inverse_transform(np.array(output).reshape(-1,1))

future_prices = predict_future(30)

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
