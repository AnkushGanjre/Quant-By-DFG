import os
import pandas as pd
from taipy.gui import Gui
from plotly import graph_objects as go
from DataFetch import update_historical_data, symbol_to_name

# Folder containing the stock data files
DATA_FOLDER = "historical_data"

# Fetch all stock symbols from the folder
def get_stock_symbols(data_folder):
    return [file.replace(".parquet", "") for file in os.listdir(data_folder) if file.endswith(".parquet")]

# Load stock data from the selected file
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

# Resample chart data based on selected timeframe
def resample_chart_data(stock_data, timeframe):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.set_index('Date')

    if timeframe == "Weekly":
        stock_data = stock_data.resample("W").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        }).dropna().reset_index()
    elif timeframe == "Monthly":
        stock_data = stock_data.resample("M").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        }).dropna().reset_index()
    return stock_data

# State variables
selected_company_name = ""
selected_chart_symbol = ""
chart_timeframe = "Daily"
stock_data = pd.DataFrame()

# Create dropdown mappings
stock_symbols = get_stock_symbols(DATA_FOLDER)
company_names = list(symbol_to_name.values())
company_to_symbol = {v: k for k, v in symbol_to_name.items()}

# Taipy GUI Layout
layout = """
# Quant by DFG Dashboard

## Select Stock
<|{selected_company_name}|selector|lov={company_names}|label=Company|on_change=update_symbol|>
<|{chart_timeframe}|selector|lov=["Daily", "Weekly", "Monthly"]|label=Timeframe|>

## Metrics
<|Sensex: {sensex_close:.2f} Δ {sensex_delta:.2f}|>
<|Nifty: {nifty_close:.2f} Δ {nifty_delta:.2f}|>
<|BankNifty: {banknifty_close:.2f} Δ {banknifty_delta:.2f}|>

## Stock Chart
<|part|render={plot_stock_chart}|>

## Update Data
<|button|label=Update Data|on_action=update_data|>
"""

def get_last_close_and_delta(data_folder, stock_symbol):
    data = load_stock_data(data_folder, stock_symbol)
    last_close = data['Close'].iloc[-1]
    delta = last_close - data['Close'].iloc[-2]
    return last_close, delta


def update_symbol(state):
    state.selected_chart_symbol = company_to_symbol[state.selected_company_name]
    if state.selected_chart_symbol.endswith(".NS"):
        state.selected_chart_symbol = state.selected_chart_symbol.replace(".NS", "")
    update_stock_data(state)

def update_stock_data(state):
    if state.selected_chart_symbol:
        state.stock_data = load_stock_data(DATA_FOLDER, state.selected_chart_symbol)
        if state.chart_timeframe != "Daily":
            state.stock_data = resample_chart_data(state.stock_data, state.chart_timeframe)

def plot_stock_chart(state):
    if not state.stock_data.empty:
        return create_candlestick_chart(state.stock_data, state.selected_chart_symbol)
    return "No data available."

def update_data(state):
    update_historical_data()
    state.update()

# Metrics Calculation
def calculate_metrics(state):
    state.sensex_close, state.sensex_delta = get_last_close_and_delta(DATA_FOLDER, "^BSESN")
    state.nifty_close, state.nifty_delta = get_last_close_and_delta(DATA_FOLDER, "^NSEI")
    state.banknifty_close, state.banknifty_delta = get_last_close_and_delta(DATA_FOLDER, "^NSEBANK")

Gui(page=layout).run()
