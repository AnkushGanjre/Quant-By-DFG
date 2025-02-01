from taipy.gui import Gui, State, invoke_long_callback, notify, builder as tgb
import yfinance as yf
import pandas as pd

# Initialize state variables
input_ticker = ""
input_qty = 1
stock_list = []  # Store portfolio stocks as a list of dictionaries

# Callback function to validate the ticker and handle the input
def OnTickerValidate(state):
    ticker: str = state.input_ticker.upper()
    quantity: int = int(state.input_qty)
    
    # Reset the InputField
    state.input_ticker = ""
    state.input_qty = 1

    # Validate ticker and quantity
    if isinstance(ticker, str) and ticker.strip() != "":
        if isinstance(quantity, int) and quantity > 0:
            notify(state, "info", f"Fetching {ticker}...")
            GetStockData(ticker.upper(), quantity, state)
        else:
            notify(state, "error", "Invalid Quantity")
    else:
        notify(state, "error", "Invalid Ticker")


# Fetch stock data and add to portfolio
def GetStockData(symbol, qty, state):
    stock = yf.Ticker(symbol)
    
    # Validate symbol
    if not stock.info.get("longName"):
        notify(state, "error", f"'{symbol}' is not a valid stock symbol")
        return
    
    # Fetch historical data (last 1 year, 1-day interval)
    df = stock.history(period="1y", interval="1d")

    # Verify data not empty
    if df.empty:
        notify(state, "error", f"No data found for '{symbol}'")
        return

    # Reset index to get Date as a column
    df.reset_index(inplace=True)

    # Format the Date column
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    # Storing Stock Name, symbol, qty, price and DataFrame in a dictionary
    stock_data = {
        "Name": stock.info.get("longName"),
        "Symbol": symbol,
        "Quantity": qty,
        "Current Price": df['Close'].iloc[-1],
        "DataFrame": df
    }

    # Add the stock_data dictionary to the stock_list
    stock_list.append(stock_data)

    # Optionally, you can print stock_list to verify
    print(stock_list)

    # Notify Success
    notify(state, "success", f"Added {symbol}, Quantity: {qty}")
    Gui.reload(state)
    

# GUI Layout
with tgb.Page() as VaR_page:
    tgb.text("## Value at Risk **(VaR)** & Conditional Value at Risk **(CVaR)**", mode="md")
    tgb.html("br")  # blank spacer
    tgb.text("##### **Create Your Portfolio** &nbsp;&nbsp;&nbsp;&nbsp; Get the stock symbol from [Yahoo Finance](https://finance.yahoo.com)", mode="md")

    # First row: Input fields and button
    with tgb.layout(columns="5 2 1", gap="30px", class_name="card"):
        with tgb.part():
            tgb.input(value="{input_ticker}", label="Yahoo Finance Ticker", class_name="fullwidth")

        with tgb.part():
            tgb.input(value="{input_qty}", label="Quantity", type="number", class_name="fullwidth")
            
        with tgb.part(class_name="text-center"):
            tgb.button("ADD", class_name="plain fullwidth", on_action=OnTickerValidate)

    tgb.html("br")  # blank spacer

    # Portfolio Table
    with tgb.part(class_name="card"):
        print("Second Card Called")
        if isinstance(stock_list, list):
            print("List count zero")
            tgb.text("Empty Here, Add some stocks to your portfolio", mode="md", class_name="fullwidth text-center")
        else:
            print("List count *************")
            tgb.text("List not empty anymore", mode="md", class_name="fullwidth text-center")
    
    



# Store stock_list Locally on the browser (Does not erase when reload the page)
# Show in table Stock NAME, stock symbol, current price, Qty, Investment amount
# Total Portfolio value & Calculate VaR & CVaR Button

# Variance & Covariance Method
# calculate daily return & mean return of stock
# calculate weightage of each stock
# Calculate mean return of portfolio (Average of (w1*r1 + w2*r2 + w3*r3)) (in %)

# Historical Method

# Monte Carlo Method

# Visually show the result
# Final look- (Input field at top, Table showing portfolio, below result (Charts & Graphs))