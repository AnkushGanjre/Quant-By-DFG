from taipy.gui import notify, builder as tgb
import yfinance as yf
import pandas as pd


# Initialize state variables
input_ticker = ""
input_qty = 1
render_stock_table = False

# Store portfolio data (table) and full stock data (dict)
stock_table_data = pd.DataFrame(columns=["Stock Name", "Stock Symbol", "Qty", "Current Price", "Investment Amount"])
stock_data_dict = {}  # Store full 1-year historical data keyed by symbol
total_portfolio_value = 0


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
    global stock_table_data, stock_data_dict, total_portfolio_value
    
    stock = yf.Ticker(symbol)
    stock_info = stock.info
    
    # Validate symbol
    if not stock_info.get("longName"):
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

    # Store full 1-year data for later access
    stock_data_dict[symbol] = df 

    # last close price & Investment Amount
    current_price: float = round(float(df['Close'].iloc[-1]), 2)
    investment_amount: float = round(current_price*qty, 2)

    # Add the Investment Amount to Total Portfolio Value
    total_portfolio_value += investment_amount
    state.total_portfolio_value = total_portfolio_value  # Ensure state gets updated

    # Create new row for portfolio table
    new_row = pd.DataFrame([{
        "Stock Name": stock_info["longName"],
        "Stock Symbol": symbol,
        "Qty": qty,
        "Current Price": f"{current_price:,.2f}",
        "Investment Amount": f"{investment_amount:,.2f}"
    }])
    
    # Append new stock to portfolio table
    # Check if stock_table_data is empty
    if stock_table_data.empty:
        stock_table_data = new_row
    else:
        stock_table_data = pd.concat([stock_table_data, new_row], ignore_index=True)
    state.stock_table_data = stock_table_data  # Update Taipy state

    # Render the table
    state.render_stock_table = True

    # Notify Success
    notify(state, "success", f"Added {symbol}, Quantity: {qty}")


def OnClearTable(state):
    state.render_stock_table = False
    print("Now Clearing Table")


def OnVaRCalculate(state):
    print("Now calculating VaR")


# ----------------------------------
# Building Pages with TGB
# ----------------------------------
with tgb.Page() as VaR_page:
    tgb.text("## Value at Risk **(VaR)** & Conditional Value at Risk **(CVaR)**", mode="md")
    tgb.html("br")  # blank spacer
    tgb.text("##### **Create Your Portfolio** &nbsp;&nbsp;&nbsp;&nbsp; Get the stock symbol from [Yahoo Finance](https://finance.yahoo.com)", mode="md")

    # First Card: Input fields and button
    with tgb.layout(columns="5 2 1", gap="30px", class_name="card"):
        with tgb.part():
            tgb.input(value="{input_ticker}", label="Yahoo Finance Ticker", class_name="fullwidth")

        with tgb.part():
            tgb.input(value="{input_qty}", label="Quantity", type="number", class_name="fullwidth")
            
        with tgb.part(class_name="text-center"):
            tgb.button("ADD", class_name="plain fullwidth", on_action=OnTickerValidate)

    tgb.html("br")  # blank spacer

    # Second Card: Portfolio Table
    with tgb.part(class_name="card", render=lambda render_stock_table: render_stock_table):
        tgb.text("### **Portfolio**", class_name="text-center", mode="md")
        tgb.table("{stock_table_data}", page_size=10)
        with tgb.layout(columns="5 1 1", gap="20px"):
            with tgb.part():
                tgb.text("Total Portfolio Value: **{total_portfolio_value:,.2f}**", class_name="text-left", mode="md")

            with tgb.part():
                tgb.button("Clear Table", class_name="plain fullwidth", on_action=OnClearTable)
                
            with tgb.part(class_name="text-center"):
                tgb.button("Calculate VaR", class_name="plain fullwidth", on_action=OnVaRCalculate)

        
    tgb.html("br")  # blank spacer
    tgb.html("br")  # blank spacer
    
    



# Store stock_list Locally on the browser (Does not erase when reload the page)
# Show in table Stock NAME, stock symbol, current price, Qty, Investment amount & delete button
# Total Portfolio value & Calculate VaR & CVaR Button

# Variance & Covariance Method
# Historical Method
# Monte Carlo Method

# Traffic Light Approach
# Kupiec Test

# Visually show the result
# Final look- (Input field at top, Table showing portfolio, below result (Charts & Graphs))