from taipy.gui import notify, builder as tgb
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats

# Initialize state variables
input_ticker = ""
input_qty = 1
render_stock_table = False
render_VaR_results = False

# Store portfolio data (table) and full stock data (dict)
stock_table_data = pd.DataFrame(columns=["Stock Name", "Stock Symbol", "Qty", "Current Price", "Investment Amount"])
stock_data_dict = {}  # Store full 1-year historical data keyed by symbol
total_portfolio_value = 0
confidence_interval = 95
close_prices = None         # Will assign closing price of each stock in pandas dataframe
daily_returns = None        # Will assign daily returns of each stock in pandas dataframe
weightage_arr = None        # Weightage of each stock in a numpy array

def OnTickerValidate(state):                # Callback function to validate the ticker and handle the input
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


def GetStockData(symbol, qty, state):       # Fetch stock data and add to portfolio
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


def OnClearTable(state):                    # Reset to default values
    global render_stock_table, render_VaR_results, stock_table_data, stock_data_dict
    global total_portfolio_value, confidence_interval, close_prices, daily_returns, weightage_arr

    # setting all values to default values
    render_stock_table = False
    render_VaR_results = False
    stock_table_data = stock_table_data.iloc[0:0]
    stock_data_dict = {}  # Reset the dictionary
    total_portfolio_value = 0
    confidence_interval = 95
    close_prices = None
    daily_returns = None
    weightage_arr = None

    # Ensure state gets updated
    state.render_stock_table = render_stock_table
    state.render_VaR_results = render_VaR_results
    state.stock_table_data = stock_table_data
    state.stock_data_dict = stock_data_dict
    state.total_portfolio_value = total_portfolio_value
    state.confidence_interval = confidence_interval


def OnVaRCalculate(state):
    global close_prices, daily_returns, weightage_arr
    if state.render_VaR_results == False:
        state.render_VaR_results = True
    notify(state, "info", "Calculating VaR & CVaR...")

    # Getting Close prices of all stocks in one pandas dataframe
    close_prices = pd.concat({symbol: df.set_index("Date")["Close"] for symbol, df in stock_data_dict.items()},axis=1)
    close_prices.columns = stock_data_dict.keys()       # Rename columns for readability
    daily_returns = close_prices.pct_change().dropna()  # All daily return of each stock

    # Calculate weightage for each stock
    # Convert weightage_arr to a NumPy array and ensure it matches avg_returns index
    weightage_arr = stock_table_data.set_index("Stock Symbol")["Investment Amount"]
    weightage_arr = weightage_arr.loc[close_prices.columns]  # Align with stock symbols
    weightage_arr = weightage_arr.str.replace(",", "").astype(float) / total_portfolio_value  # Normalize
    weightage_arr = weightage_arr.to_numpy()  # Convert to NumPy array

    # Call all VaR & CVaR calculation functions
    print("Parametric VaR: ", CalParametricVaR(state))
    print("Monte Carlo VaR: ", CalMonteCarloVaR(state))


def CalParametricVaR(state):
    global close_prices, daily_returns, weightage_arr
    VaR_results = None

    # Calculate Variance Covariance Matrix
    cov_matrix = daily_returns.cov()

    # Average Return
    avg_returns = daily_returns.mean()

    # Total count of returns
    count = daily_returns.shape[0]

    # Mean (Portfolio Expected Return)
    port_mean = avg_returns @ weightage_arr

    # Calculate standard deviation (Portfolio Risk)
    port_std = np.sqrt(weightage_arr.T @ cov_matrix @ weightage_arr)

    print("Portfolio Mean:", port_mean)
    print("Portfolio Standard Deviation:", port_std)

    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    confidence_level = 1-confidence_level
    print("Confidence Level:", confidence_level)
    VaR_results = stats.norm.ppf(confidence_level, port_mean, port_std)     # (Percent-Point Function) normal distribution function 

    # 2 is z value based on 95% CI
    z = 2
    lower = port_mean - z*port_std / np.sqrt(count)
    higher = port_mean + z*port_std / np.sqrt(count)

    print("Lower:", lower)
    print("Higher:", higher)

    return VaR_results


def CalMonteCarloVaR(state):
    global close_prices, daily_returns, weightage_arr
    VaR_results = None

    n_sims = 1000000
    rfr = 0
    time = 30
    vol = 0.25
    s0 = 1

    d = (rfr - 0.5 * vol**2) * (time / 252)
    a = vol * np.sqrt(time / 252)
    r = np.random.normal(0, 1, (n_sims, 1))

    GBM_returns = s0 * np.exp(d + a*r)

    pers = [0.01, 0.1, 1.0, 2.5, 5.0, 10.0]
    VaR_results = stats.scoreatpercentile(GBM_returns -1, pers)

    return VaR_results



# ----------------------------------
# Building Pages with TGB
# ----------------------------------
with tgb.Page() as valueAtRisk_page:
    tgb.text("## **Value at Risk** (VaR) **& Conditional Value at Risk** (CVaR)", mode="md")
    tgb.text("This application calculates Value at Risk (VaR) and Conditional VaR (CVaR) using the Parametric, Historical, and Monte Carlo methods. It fetches the last 1 year (365 days) of historical data from Yahoo Finance based on entered stock symbols. Specify stock quantities to determine total investment and portfolio weightage. Customize the confidence interval to analyze risk exposure effectively.", mode="md")
    tgb.html("br")  # blank spacer
    tgb.text("##### **Create Your Portfolio** &nbsp;&nbsp;&nbsp;&nbsp; Get the stock symbol from [Yahoo Finance](https://finance.yahoo.com)", mode="md")

    # First Card: Input fields and button
    with tgb.layout(columns="5 1 1", gap="30px", class_name="card"):
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
                tgb.button("Clear Results", class_name="plain fullwidth", on_action=OnClearTable)
                
            with tgb.part(class_name="text-center"):
                tgb.button("Calculate VaR", class_name="plain fullwidth", on_action=OnVaRCalculate)

    tgb.html("br")  # blank spacer
    tgb.html("br")  # blank spacer
    
    with tgb.part(class_name="card", render=lambda render_VaR_results: render_VaR_results):
        tgb.text("## **VaR** & **CVaR**", class_name="text-center", mode="md")

        with tgb.layout(columns="1 2", gap="10px"):  # Wider layout for better alignment
            with tgb.part():
                tgb.text("Confidence Interval: **{confidence_interval}%**", class_name="text-center", mode="md")

            with tgb.part(class_name="fullwidth"):
                tgb.slider(value="{confidence_interval}", min=80, max=99, step=1, on_change=OnVaRCalculate)   # Default value = 95

    
    tgb.html("br")  # blank spacer
    tgb.html("br")  # blank spacer




# Variance & Covariance Method
# Historical Method
# Monte Carlo Method

# Traffic Light Approach
# Kupiec Test

# Visually show the result
# Final look- (Input field at top, Table showing portfolio, below result (Charts & Graphs))