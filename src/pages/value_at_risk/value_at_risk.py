from taipy.gui import notify, builder as tgb
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2

# Initialize state variables
input_ticker = ""
input_qty = 1
render_stock_table = False
render_VaR_results = False

# Store portfolio data (table) and full stock data (dict)
stock_table_data = pd.DataFrame(columns=["Stock Name", "Stock Symbol", "Qty", "Current Price", "Investment Amount"])
stock_data_dict = {}            # Store full 1-year historical data keyed by symbol
total_portfolio_value = 0
confidence_interval = 99
close_prices = None             # Will assign closing price of each stock in pandas dataframe
daily_returns = None            # Will assign daily returns of each stock in pandas dataframe
weightage_arr = None            # Weightage of each stock in a numpy array
portfolio_returns = None        # Matrix multiplication of daily_returns & weightage_arr
n_sims = 500000                 # No. of simulations for Monte Carlo

# Initialize an empty DataFrame with the required columns
all_result_table_data = pd.DataFrame(columns=["Method", "Value at Risk", "Conditional Value at Risk", "Traffic Light", "Kupiec"])
var_chart_data = None
cvar_chart_data = None

# For bar chart
layout = {"barmode": "stack", "hovermode": "x"}
options = {"unselected": {"marker": {"opacity": 0.5}}}


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
    global portfolio_returns, total_portfolio_value
    global all_result_table_data, cvar_chart_data, var_chart_data

    if state.render_VaR_results == False:
        state.render_VaR_results = True
    notify(state, "info", "Calculating VaR & CVaR...")

    # Getting Close prices of all stocks in one pandas dataframe
    close_prices = pd.concat({symbol: df.set_index("Date")["Close"] for symbol, df in stock_data_dict.items()},axis=1)
    close_prices.columns = stock_data_dict.keys()       # Rename columns for readability
    close_price_cleaned = close_prices.ffill()          # forward fill cleaning
    daily_returns = close_price_cleaned.pct_change()    # Calculate daily returns

    # Calculate weightage for each stock
    # Convert weightage_arr to a NumPy array and ensure it matches avg_returns index
    weightage_arr = stock_table_data.set_index("Stock Symbol")["Investment Amount"]
    weightage_arr = weightage_arr.loc[close_prices.columns]  # Align with stock symbols
    weightage_arr = weightage_arr.str.replace(",", "").astype(float) / total_portfolio_value  # Normalize
    weightage_arr = weightage_arr.to_numpy()  # Convert to NumPy array

    # Calculate daily portfolio returns keeping
    portfolio_returns = daily_returns @ weightage_arr  # Matrix multiplication now it is a numpy array
    state.portfolio_returns = portfolio_returns

    # Call the function and print the results
    parametric_var, parametric_cvar = Parametric_VaR_CVaR(state)
    historical_var, historical_cvar = Historical_VaR_CVaR(state)
    montecarlo_var, montecarlo_cvar = MonteCarlo_VaR_CVaR(state)
    parametric_traffic, historical_traffic, montecarlo_traffic = TrafficLightBackTest(parametric_var, historical_var, montecarlo_var)
    parametric_lr, parametric_pval, historical_lr, historical_pval, montecarlo_lr, montecarlo_pval = KupiecBackTest(state, parametric_var, historical_var, montecarlo_var)

    # Determine reliability based on p-value
    def reliability(p_value):
        return "Reliable" if p_value > 0.05 else "Unreliable"

    # Create a new DataFrame with updated values
    all_result_table_data = pd.DataFrame([
        ["Variance-Covariance (Parametric)", f"{parametric_var:,.2f}", f"{parametric_cvar:,.2f}", parametric_traffic, reliability(parametric_pval)],
        ["Historical Method", f"{historical_var:,.2f}", f"{historical_cvar:,.2f}", historical_traffic, reliability(historical_pval)],
        ["Monte Carlo Simulation", f"{montecarlo_var:,.2f}", f"{montecarlo_cvar:,.2f}", montecarlo_traffic, reliability(montecarlo_pval)]
    ], columns=["Method", "Value at Risk", "Conditional Value at Risk", "Traffic Light", "Kupiec"])

    # Update the global table data
    state.all_result_table_data = all_result_table_data

    # Prepare data for visualization (long format for easier plotting)
    var_chart_data = 0
    cvar_chart_data = 0
    state.var_chart_data = var_chart_data
    state.cvar_chart_data = cvar_chart_data


def Parametric_VaR_CVaR(state):
    global close_prices, daily_returns, weightage_arr, total_portfolio_value
    VaR_results = None
    CVaR_results = None

    # Average Return
    avg_returns = daily_returns.mean()

    # Calculate Variance Covariance Matrix
    # Covariance matrix shows how different stocks move together.
    cov_matrix = daily_returns.cov()

    # Mean (Portfolio Expected Return)
    port_mean = avg_returns @ weightage_arr

    # Calculate standard deviation (Portfolio Risk)
    # weightage_arr.T is the transpose of the weight array. 
    # If weightage_arr is a column vector, then weightage_arr.T becomes a row vector.
    # Matrix Multiplication (@ operator)
    # weightage_arr.T @ cov_matrix gives a row vector representing the combined effect of stock covariances on portfolio risk.
    # (weightage_arr.T @ cov_matrix) @ weightage_arr → This results in a single number, which is the portfolio variance.
    # np.sqrt(...) → Finally, we take the square root to get the portfolio standard deviation (port_std), 
    # which is the overall risk of the portfolio.
    # It is also refered to as Portfolio Volatility
    port_std_dev = np.sqrt(weightage_arr.T @ cov_matrix @ weightage_arr)

    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    alpha = 1-confidence_level

    # This is the percent-point function (PPF)
    # (also known as the inverse cumulative distribution function) 
    # for the standard normal distribution.
    z_score = stats.norm.ppf(alpha)

    # Parametric VaR = (mean + Z score * std_dev) * portfolio val
    VaR_results = (port_mean + z_score * port_std_dev) * total_portfolio_value

    # Standard normal PDF at z-score
    pdf_z = stats.norm.pdf(z_score)

    # Conditional VaR (Expected Shortfall)
    CVaR_results = (port_mean - (pdf_z / alpha) * port_std_dev) * total_portfolio_value
    return VaR_results, CVaR_results


def Historical_VaR_CVaR(state):
    global portfolio_returns, total_portfolio_value
    VaR_results = None
    CVaR_results = None

    # Sort portfolio returns from lowest to highest
    sorted_returns = np.sort(portfolio_returns)
    count = portfolio_returns.shape[0]

    # Confidence Level
    confidence_level = state.confidence_interval / 100      # Convert to decimal (e.g., 95 → 0.95)
    alpha = 1-confidence_level

    # Find VaR rank index
    rank = round(count * alpha)
    value_at_rank = sorted_returns[rank - 1]       # (zero-based index)

    # VaR
    VaR_results = total_portfolio_value * value_at_rank

    # Extract all returns beyond VaR (more negative than VaR)
    tail_losses = sorted_returns[:rank]  # Losses worse than VaR

    # Calculate Conditional VaR (Expected Shortfall)
    if len(tail_losses) > 0:
        CVaR_results = np.mean(tail_losses) * total_portfolio_value
    else:
        CVaR_results = VaR_results         # Fallback if no values in tail

    return VaR_results, CVaR_results


def MonteCarlo_VaR_CVaR(state):
    global close_prices, daily_returns, weightage_arr
    global total_portfolio_value
    VaR_results = None
    CVaR_results = None

    n_sims = state.n_sims         # Number of Monte Carlo simulations
    t = 1                   # Forecasting for 1 day
    
    # Portfolio statistics
    covariance_matrix = daily_returns.cov()         # Covariance Matrix
    correlation_matrix = daily_returns.corr()       # Correlation Matrix
    avg_returns = daily_returns.mean()
    port_mean = avg_returns @ weightage_arr
    port_std_dev = np.sqrt(weightage_arr.T @ covariance_matrix @ weightage_arr)     # Standard Deviation (Volatility)


    ### STEP 1 -> CREATE CORRELATED RANDOM VARIABLES (Cholesky Decomposition)

    # Cholesky Decomposition
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)

    # Generate uncorrelated random variables (Count = n_sims)
    uncorrelated_randoms = stats.norm.ppf(np.random.rand(n_sims, daily_returns.shape[1]))

    # Matrix multiply Cholesky Decomposition with the transpose of uncorrelated random variables
    correlated_randoms = cholesky_matrix @ uncorrelated_randoms.T
    correlated_randoms = correlated_randoms.T  # Transpose back to match shape


    ### STEP 2 -> USE GBM TO SIMULATED STOCK PRICE USING CORRELATED RANDOM VARIABLES

    # Geometric Brownian Motion Formula 
    # S_t = S0 * exp((μ - 0.5 * σ²) * t + σW_t)
    # 
    # S_t​= Stock price at time 𝑡    (Price tomorrow for 1 day var)
    # S_0= inital stock price       (Price Today i.e., last traded price)
    # μ (meu)= Expected return (drift)
    # σ (Sigma)= Volatility (standard deviation of returns)
    # W_t= Wiener process (standard Brownian motion)
    # t= Time step

    # GBM parameters
    meu = port_mean - 0.5 * (port_std_dev ** 2)     # Drift
    sigma = daily_returns.std().values              # Individual stock volatilities
    S_0 = close_prices.iloc[-1].values              # Latest stock prices

    # Simulated stock prices using GBM
    sim_stock_prices = S_0 * np.exp((meu - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * correlated_randoms)

    # Compute simulated portfolio returns for ONE DAY
    sim_returns = (sim_stock_prices - S_0) / S_0  # (P_t - P_0) / P_0
    sim_portfolio_returns = sim_returns @ weightage_arr
    sorted_sim_returns = np.sort(sim_portfolio_returns)     # Sort portfolio returns


    ### STEP 3 -> Calculate VaR from stock prices simulated {n_sims} times

    # Now last calculation
    count = sorted_sim_returns.shape[0]
    confidence_level = state.confidence_interval / 100      # Convert to decimal (e.g., 95 → 0.95)
    alpha = 1-confidence_level
    rank = round(count * alpha)
    value_at_rank = sorted_sim_returns[rank]
    VaR_results = total_portfolio_value * value_at_rank

    # Extract all returns worse than or equal to VaR
    tail_losses = sorted_sim_returns[:rank]  # Losses worse than VaR

    # Calculate Conditional VaR (Expected Shortfall)
    if len(tail_losses) > 0:
        CVaR_results = np.mean(tail_losses) * total_portfolio_value
    else:
        CVaR_results = VaR_results        # Fallback if no values in tail
    return VaR_results, CVaR_results


def TrafficLightBackTest(parametric_VaR, historical_VaR, monteCarlo_VaR):
    global portfolio_returns, total_portfolio_value

    # Compute actual losses in currency terms
    actual_losses = portfolio_returns * total_portfolio_value  # Element-wise multiplication

    # Count violations for each VaR model
    parametric_traffic = np.sum(actual_losses < parametric_VaR)
    historical_traffic = np.sum(actual_losses < historical_VaR)
    montecarlo_traffic = np.sum(actual_losses < monteCarlo_VaR)

    return parametric_traffic, historical_traffic, montecarlo_traffic


def KupiecBackTest(state, parametric_VaR, historical_VaR, monteCarlo_VaR):

    # Count violations for each VaR model
    parametric_lr, parametric_pval  = kupiec_pof_test(state, parametric_VaR)
    historical_lr, historical_pval = kupiec_pof_test(state, historical_VaR)
    montecarlo_lr, montecarlo_pval = kupiec_pof_test(state, monteCarlo_VaR)

    return parametric_lr, parametric_pval, historical_lr, historical_pval, montecarlo_lr, montecarlo_pval


def kupiec_pof_test(state, VaR):
    """
    Kupiec Proportion of Failures (POF) Test for VaR backtesting.
    :param returns: Array of actual portfolio returns
    :param VaR: Array of VaR estimates (should match the confidence level)
    :param confidence_level: The confidence level of the VaR
    :return: Kupiec test statistics and p-value
    """
    global portfolio_returns, total_portfolio_value

    # Compute actual losses in currency terms
    actual_losses = portfolio_returns * total_portfolio_value  # Element-wise multiplication
    N = len(actual_losses)      # Total observations
    x = np.sum(actual_losses < VaR)  # Count of VaR breaches
    p_hat = x / N               # Observed failure rate
    confidence_level = state.confidence_interval / 100      # Convert to decimal (e.g., 95 → 0.95)
    p = 1-confidence_level      # Expected failure rate
    
    # Handle edge cases where log(0) occurs
    if x == 0 or x == N:
        return np.inf, 1.0    # LR = infinity, p-value = 1 (not rejectable)

    # Kupiec Likelihood Ratio test statistic
    LR = -2 * np.log(((1 - p) ** (N - x) * p ** x) / ((1 - p_hat) ** (N - x) * p_hat ** x))
    p_value = 1 - chi2.cdf(LR, df=1)  # Chi-square test with 1 degree of freedom
    return LR, p_value


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

        # Heading, Confidence interval Slider & Results Table
        tgb.text("## **VaR** & **CVaR**", class_name="text-center", mode="md")
        with tgb.layout(columns="1 1", gap="10px"):
            with tgb.part():
                tgb.text("Confidence Interval: **{confidence_interval}%**", class_name="text-center", mode="md")

            with tgb.part(class_name="fullwidth"):
                tgb.slider(value="{confidence_interval}", min=80, max=99, step=1, on_change=OnVaRCalculate)
        tgb.table("{all_result_table_data}", page_size=3)

        # VaR & CVaR Bar Chart
        # with tgb.layout(columns="1 1", gap="10px"):
        #     with tgb.part(class_name="fullwidth"):
        #         tgb.chart(
        #             "{var_chart_data}",
        #             type="bar",
        #             x="Returns",
        #             y="Frequency",
        #             layout=layout,
        #             options=options,
        #             title="Value At Risk",
        #         )
        #     with tgb.part(class_name="fullwidth"):
        #         tgb.chart(
        #             "{cvar_chart_data}",
        #             type="bar",
        #             x="Returns",
        #             y="Frequency",
        #             layout=layout,
        #             options=options,
        #             title="Conditional Value At Risk",
        #         )

        # Monte Carlo Simulations below
        tgb.text("## **Monte Carlo** Simulations", class_name="text-center", mode="md")
        with tgb.layout(columns="1 1", gap="10px"):
            with tgb.part():
                tgb.text("Number of Simulations: **{n_sims:,}**", class_name="text-center", mode="md")

            with tgb.part(class_name="fullwidth"):
                tgb.slider(value="{n_sims}", min=10000, max=1000000, step=50000, on_change=OnVaRCalculate)
        # tgb.chart(
        #     "{cvar_chart_data}",
        #     type="bar",
        #     x="Returns",
        #     y="Frequency",
        #     layout=layout,
        #     options=options,
        #     title="Monte Carlo Simulations",
        # )
                    
    tgb.html("br")  # blank spacer
    tgb.html("br")  # blank spacer