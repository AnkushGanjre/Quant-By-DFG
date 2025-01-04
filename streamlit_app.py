import os
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from DataFetch import update_historical_data  # Import your update function

# Set page configuration
st.set_page_config(page_title="Quant by DFG", layout="wide", initial_sidebar_state="collapsed")

st.write("### Welcome to Quant by DFG!")
st.write("##### Dashboard")

# Folder containing the stock data files
DATA_FOLDER = "historical_data"

# Fetch all stock symbols from the folder
@st.cache_data
def get_stock_symbols(data_folder):
    return [file.replace(".parquet", "") for file in os.listdir(data_folder) if file.endswith(".parquet")]

# Load stock data from the selected file
@st.cache_data
def load_stock_data(data_folder, stock_symbol):
    file_path = os.path.join(data_folder, f"{stock_symbol}.parquet")
    return pd.read_parquet(file_path)

# Create candlestick chart
def create_candlestick_chart(stock_data, stock_symbol):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=stock_data['Date'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close']
            )
        ]
    )
    fig.update_layout(
        title=f"Candlestick Chart for {stock_symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    return fig

# Get last close and delta for metrics
def get_last_close_and_delta(data_folder, stock_symbol):
    data = load_stock_data(data_folder, stock_symbol)
    last_close = data['Close'].iloc[-1]
    delta = last_close - data['Close'].iloc[-2]
    return last_close, delta

# Get last date from Nifty50 file
def get_latest_date(data_folder, stock_symbol="^NSEI"):
    data = load_stock_data(data_folder, stock_symbol)
    return data['Date'].max()

# Metrics for Sensex, Nifty, BankNifty
col1, col2, col3 = st.columns(3)
with col1:
    sensex_close, sensex_delta = get_last_close_and_delta(DATA_FOLDER, "^BSESN")
    st.metric(label="Sensex", value=f"{sensex_close:.2f}", delta=f"{sensex_delta:.2f}")
with col2:
    nifty_close, nifty_delta = get_last_close_and_delta(DATA_FOLDER, "^NSEI")
    st.metric(label="Nifty", value=f"{nifty_close:.2f}", delta=f"{nifty_delta:.2f}")
with col3:
    banknifty_close, banknifty_delta = get_last_close_and_delta(DATA_FOLDER, "^NSEBANK")
    st.metric(label="BankNifty", value=f"{banknifty_close:.2f}", delta=f"{banknifty_delta:.2f}")

# Get list of chart symbols
stock_symbols = get_stock_symbols(DATA_FOLDER)

# Dropdown for stock chart
col4, col5, col6 = st.columns(3)
with col4:
    selected_chart_name = st.selectbox("Select Chart:", stock_symbols)
with col5:
    st.write("")  # Spacer for alignment
with col6:
    # st.write("")  # Spacer for alignment
    time_frame = st.selectbox("Select Timeframe:", ["Daily", "Weekly", "Monthly"])  # Timeframe dropdown

# Display candlestick chart for selected stock
if selected_chart_name:
    stock_data = load_stock_data(DATA_FOLDER, selected_chart_name)
    stock_data = stock_data.sort_values(by="Date")  # Ensure data is sorted
    st.plotly_chart(create_candlestick_chart(stock_data, selected_chart_name))

# Update data button
latest_date = get_latest_date(DATA_FOLDER, "^NSEI")
st.write(f"Latest Data: {latest_date}")
if st.button("Update data"):
    update_historical_data()
    st.rerun()  # Refresh the app to reflect updated data