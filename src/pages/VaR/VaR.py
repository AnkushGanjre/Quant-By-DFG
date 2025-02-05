from taipy.gui import notify, builder as tgb
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import concurrent.futures


# Initialize state variables
input_ticker = ""
input_qty = 1
render_stock_table = False
render_VaR_data = False

# Store portfolio data (table) and full stock data (dict)
stock_table_data = pd.DataFrame(columns=["Stock Name", "Stock Symbol", "Qty", "Current Price", "Investment Amount"])
stock_data_dict = {}  # Store full 1-year historical data keyed by symbol
total_portfolio_value = 0
confidence_interval = 95


def CalParametricVaR(state): return "Parametric VaR result"
def CalParametricCVaR(state): return "Parametric CVaR result"
def CalHistoricalVaR(state): return "Historical VaR result"
def CalHistoricalCVaR(state): return "Historical CVaR result"
def CalMonteCarloVaR(state): return "Monte Carlo VaR result"
def CalMonteCarloCVaR(state): return "Monte Carlo CVaR result"
def CalTrafficLightVaR(state): return "Traffic Light VaR result"
def CalTrafficLightCVaR(state): return "Traffic Light CVaR result"
def CalKupiecVaR(state): return "Kupiec VaR result"
def CalKupiecCVaR(state): return "Kupiec CVaR result"


# VaR Calculations Function Mapping
VaR_methods = {
    "Parametric VaR": CalParametricVaR,
    "Parametric CVaR": CalParametricCVaR,
    "Historical VaR": CalHistoricalVaR,
    "Historical CVaR": CalHistoricalCVaR,
    "Monte Carlo VaR": CalMonteCarloVaR,
    "Monte Carlo CVaR": CalMonteCarloCVaR,
    "Traffic Light VaR": CalTrafficLightVaR,
    "Traffic Light CVaR": CalTrafficLightCVaR,
    "Kupiec VaR": CalKupiecVaR,
    "Kupiec CVaR": CalKupiecCVaR,
}


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


def OnClearTable(state):                    # Clear all data
    global stock_table_data, stock_data_dict, total_portfolio_value
    # setting all values to default values
    state.render_stock_table = False
    state.render_VaR_data = False
    state.confidence_interval = 95

    total_portfolio_value = 0
    state.total_portfolio_value = total_portfolio_value  # Ensure state gets updated
    stock_table_data = stock_table_data.iloc[0:0]
    state.stock_table_data = stock_table_data
    stock_data_dict = {}  # Reset the dictionary
    state.stock_data_dict = stock_data_dict


def OnVaRCalculate(state):
    state.render_VaR_data = True
    notify(state, "info", "Calculating VaR & CVaR...")

    results = {}

    # Run calculations in parallel for better performance
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(func, state): name for name, func in VaR_methods.items()}
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()

    # Store results in state
    state.VaR_results = results    

    notify(state, "success", "VaR & CVaR calculations completed.")


def CalParametricVaR(state):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    z_score = stats.norm.ppf(confidence_level)  # Get Z-score for confidence level

    VaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change()  # Calculate daily returns
        mean_return = np.mean(df["Returns"].dropna())  # Mean return
        std_dev = np.std(df["Returns"].dropna())  # Standard deviation

        # Calculate VaR
        VaR = mean_return - (z_score * std_dev)

        # Convert to absolute portfolio value
        stock_qty = state.stock_table_data.loc[state.stock_table_data["Stock Symbol"] == symbol, "Qty"].values[0]
        stock_investment = state.stock_table_data.loc[state.stock_table_data["Stock Symbol"] == symbol, "Investment Amount"].values[0]
        VaR_abs = round(stock_investment * abs(VaR), 2)

        VaR_results[symbol] = {
            "VaR (%)": round(VaR * 100, 2),
            "VaR (₹)": VaR_abs
        }

    print("Parametric VaR: " + VaR_results)
    return VaR_results


def CalParametricCVaR(state):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    z_score = stats.norm.ppf(confidence_level)  # Get Z-score for confidence level
    pdf_value = stats.norm.pdf(z_score)  # Get PDF value at Z-score

    CVaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change()  # Calculate daily returns
        mean_return = np.mean(df["Returns"].dropna())  # Mean return
        std_dev = np.std(df["Returns"].dropna())  # Standard deviation

        # Calculate CVaR (Expected Shortfall)
        CVaR = mean_return - ((pdf_value / (1 - confidence_level)) * std_dev)

        # Convert to absolute portfolio value
        stock_qty = state.stock_table_data.loc[state.stock_table_data["Stock Symbol"] == symbol, "Qty"].values[0]
        stock_investment = state.stock_table_data.loc[state.stock_table_data["Stock Symbol"] == symbol, "Investment Amount"].values[0]
        CVaR_abs = round(stock_investment * abs(CVaR), 2)

        CVaR_results[symbol] = {
            "CVaR (%)": round(CVaR * 100, 2),
            "CVaR (₹)": CVaR_abs
        }

    print("Parametric CVaR: " + CVaR_results)
    return CVaR_results


def CalHistoricalVaR(state):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    quantile_level = 1 - confidence_level  # E.g., 0.05 for 95% confidence

    Historical_VaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change()  # Calculate daily returns
        sorted_returns = df["Returns"].dropna().sort_values().reset_index(drop=True)  # Sort returns in ascending order
        
        # Find VaR at the specified quantile
        VaR_percent = sorted_returns.quantile(quantile_level)

        # Convert to absolute loss
        stock_investment = state.stock_table_data.loc[state.stock_table_data["Stock Symbol"] == symbol, "Investment Amount"].values[0]
        VaR_abs = round(stock_investment * abs(VaR_percent), 2)

        Historical_VaR_results[symbol] = {
            "VaR (%)": round(VaR_percent * 100, 2),
            "VaR (₹)": VaR_abs
        }

    print("Historical VaR: " + Historical_VaR_results)
    return Historical_VaR_results


def CalHistoricalCVaR(state):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    quantile_level = 1 - confidence_level  # E.g., 0.05 for 95% confidence

    Historical_CVaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change()  # Calculate daily returns
        sorted_returns = df["Returns"].dropna().sort_values().reset_index(drop=True)  # Sort returns in ascending order
        
        # Identify the VaR threshold (quantile level)
        VaR_percent = sorted_returns.quantile(quantile_level)

        # Compute CVaR: Average of all returns worse than the VaR threshold
        CVaR_percent = sorted_returns[sorted_returns <= VaR_percent].mean()

        # Convert to absolute loss
        stock_investment = state.stock_table_data.loc[state.stock_table_data["Stock Symbol"] == symbol, "Investment Amount"].values[0]
        CVaR_abs = round(stock_investment * abs(CVaR_percent), 2)

        Historical_CVaR_results[symbol] = {
            "CVaR (%)": round(CVaR_percent * 100, 2),
            "CVaR (₹)": CVaR_abs
        }

    print("Historical CVaR: " + Historical_CVaR_results)
    return Historical_CVaR_results


def CalMonteCarloVaR(state, num_simulations=10_000, days=1):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    quantile_level = 1 - confidence_level  # E.g., 0.05 for 95% confidence

    MonteCarlo_VaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change().dropna()  # Daily log returns
        mean_return = df["Returns"].mean()  # Average daily return
        std_dev = df["Returns"].std()  # Standard deviation of returns

        # Simulate future price movements
        simulated_returns = np.random.normal(mean_return, std_dev, (num_simulations, days))

        # Compute simulated portfolio values
        last_price = df["Close"].iloc[-1]
        simulated_prices = last_price * (1 + simulated_returns)

        # Compute the simulated losses
        simulated_losses = last_price - simulated_prices

        # Compute Monte Carlo VaR at confidence level
        VaR_simulated = np.percentile(simulated_losses, quantile_level * 100)

        # Convert to absolute loss
        stock_investment = state.stock_table_data.loc[state.stock_table_data["Stock Symbol"] == symbol, "Investment Amount"].values[0]
        VaR_abs = round(stock_investment * abs(VaR_simulated / last_price), 2)

        MonteCarlo_VaR_results[symbol] = {
            "VaR (%)": round(VaR_simulated / last_price * 100, 2),
            "VaR (₹)": VaR_abs
        }

    print("Monte Carlo VaR: " + MonteCarlo_VaR_results)
    return MonteCarlo_VaR_results


# ----------------------------------
# Building Pages with TGB
# ----------------------------------
with tgb.Page() as VaR_page:
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
                tgb.button("Clear Table", class_name="plain fullwidth", on_action=OnClearTable)
                
            with tgb.part(class_name="text-center"):
                tgb.button("Calculate VaR", class_name="plain fullwidth", on_action=OnVaRCalculate)

        
    tgb.html("br")  # blank spacer
    tgb.html("br")  # blank spacer
    
    with tgb.part(class_name="card", render=lambda render_VaR_data: render_VaR_data):
        tgb.text("## **VaR** & **CVaR**", class_name="text-center", mode="md")

        with tgb.layout(columns="1 2", gap="10px"):  # Wider layout for better alignment
            with tgb.part():
                tgb.text("Confidence Interval: **{confidence_interval}%**", class_name="text-center", mode="md")

            with tgb.part(class_name="fullwidth"):
                tgb.slider(value="{confidence_interval}", min=80, max=99, step=1)   # Default value = 95

    
    tgb.html("br")  # blank spacer
    tgb.html("br")  # blank spacer




# Variance & Covariance Method
# Historical Method
# Monte Carlo Method

# Traffic Light Approach
# Kupiec Test

# Visually show the result
# Final look- (Input field at top, Table showing portfolio, below result (Charts & Graphs))