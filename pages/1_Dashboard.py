import streamlit as st
import pandas as pd
from logzero import logger
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# Check if user is logged in
if "authToken" not in st.session_state:
    st.warning("Please log in to access the Dashboard.")
    st.switch_page("streamlit_app.py")

# Initialize session state for metrics if not already done
if 'sensex' not in st.session_state:
    st.session_state.sensex = 0
if 'nifty' not in st.session_state:
    st.session_state.nifty = 0
if 'banknifty' not in st.session_state:
    st.session_state.banknifty = 0

# Display dashboard content
st.write("### Dashboard")
st.write("Welcome to Quant by DFG!")

# Sensex, Nifty, bank Nifty placeholders for metrics
col1, col2, col3 = st.columns(3)

with col1:
    sensex_placeholder = st.empty()
with col2:
    nifty_placeholder = st.empty()
with col3:
    banknifty_placeholder = st.empty()

# Function to update metrics
def update_metrics(sensex, nifty, banknifty):
    sensex_placeholder.metric(label="Sensex", value=f"{sensex:,.2f}", delta="+1")
    nifty_placeholder.metric(label="Nifty", value=f"{nifty:,.2f}", delta="+2")
    banknifty_placeholder.metric(label="BankNifty", value=f"{banknifty:,.2f}", delta="+3")

# Update initial metrics display
update_metrics(st.session_state.sensex, st.session_state.nifty, st.session_state.banknifty)



# Top 10 Stocks Section
st.write("#### Top 10 Stocks")

stocks = [ "Reliance", "HDFC Bank", "ICICI Bank", "TCS", "Bharti Airtel", "Infosys", "SBI", "ITC", "HUL", "LIC India" ]
dummy_values = ["2400", "1600", "940", "3300", "800", "1520", "620", "440", "2600", "750"]

# Create table for stocks and values
for stock, value in zip(stocks, dummy_values):
    col1, col2 = st.columns(2)
    col1.write(stock)
    col2.write(value)




AUTH_TOKEN = st.session_state.authToken
API_KEY = st.session_state.api_key
CLIENT_CODE = st.session_state.username
FEED_TOKEN = st.session_state.feedToken
correlation_id = "abc123"
action = 1
mode = 3

token_list = [
    # {
    #     "action": 1,
    #     "exchangeType": 1,
    #     "tokens": ["1"]
    # },
    {
        "action": 1,
        "exchangeType": 1,
        "tokens": ["26000"]
    },
    {
        "action": 1,
        "exchangeType": 1,
        "tokens": ["26009"]
    },
]
token_list1 = [
    # {
    #     "action": 0,
    #     "exchangeType": 1,
    #     "tokens": ["1"]
    # },
    {
        "action": 0,
        "exchangeType": 1,
        "tokens": ["26000"]
    },
    {
        "action": 0,
        "exchangeType": 1,
        "tokens": ["26009"]
    },
]

sws = SmartWebSocketV2(AUTH_TOKEN, API_KEY, CLIENT_CODE, FEED_TOKEN)
live_data = {}

def on_data(wsapp, message):
    print(message)
    global live_data
    token = message.get('token')    # Assuming 'token' is present in the message
    last_traded_price = message.get('last_traded_price')

    if token is not None and last_traded_price is not None:
        # Update live_data
        live_data[token] = {'ltp': last_traded_price}
        
        # Example token mapping to index for updating metrics (adjust based on actual token values)
        if token == "1":  # Example token for Sensex
            st.session_state.sensex = last_traded_price
        elif token == "26000":  # Example token for Nifty
            st.session_state.nifty = last_traded_price
        elif token == "26009":  # Example token for Nifty Bank
            st.session_state.banknifty = last_traded_price
        
        # Update metrics display
        update_metrics(st.session_state.sensex, st.session_state.nifty, st.session_state.banknifty)
    else:
        print("Token or last traded price is missing in the message")

def on_open(wsapp):
    logger.info("on open")
    sws.subscribe(correlation_id, mode, token_list)
    # sws.unsubscribe(correlation_id, mode, token_list1)


def on_error(wsapp, error):
    logger.error(error)


def on_close(wsapp):
    logger.info("Close")



def close_connection():
    sws.close_connection()


# Assign the callbacks.
sws.on_open = on_open
sws.on_data = on_data
sws.on_error = on_error
sws.on_close = on_close

sws.connect()