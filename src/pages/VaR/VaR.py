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
weightage_arr = None        # Weightage of each stock

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




    # print(CalParametricCVaR(state))
    
    # print(CalHistoricalVaR(state))
    # print(CalHistoricalCVaR(state))
    
    # print(CalMonteCarloVaR(state))
    # print(CalMonteCarloCVaR(state))
    
    # print(CalTrafficLightVaR(state))
    # print(CalTrafficLightCVaR(state))
    
    # print(CalKupiecVaR(state))
    # print(CalKupiecCVaR(state))


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


def CalMonteCarloCVaR(state, num_simulations=10_000, days=1):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    quantile_level = 1 - confidence_level  # E.g., 0.05 for 95% confidence

    MonteCarlo_CVaR_results = {}

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

        # Compute CVaR: Average of all losses beyond VaR
        CVaR_simulated = simulated_losses[simulated_losses >= VaR_simulated].mean()

        # Convert to absolute loss
        stock_investment = state.stock_table_data.loc[state.stock_table_data["Stock Symbol"] == symbol, "Investment Amount"].values[0]
        CVaR_abs = round(stock_investment * abs(CVaR_simulated / last_price), 2)

        MonteCarlo_CVaR_results[symbol] = {
            "CVaR (%)": round(CVaR_simulated / last_price * 100, 2),
            "CVaR (₹)": CVaR_abs
        }

    print("Monte Carlo CVaR: " + MonteCarlo_CVaR_results)
    return MonteCarlo_CVaR_results


def CalTrafficLightVaR(state):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    quantile_level = 1 - confidence_level  # E.g., 0.05 for 95% confidence

    TrafficLight_VaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change().dropna()  # Daily returns
        VaR_historical = np.percentile(df["Returns"], quantile_level * 100)  # Historical VaR

        # Count exceptions (actual losses exceeding VaR)
        df["Exceptions"] = df["Returns"] < VaR_historical
        num_exceptions = df["Exceptions"].sum()

        # Determine the Traffic Light category
        if num_exceptions <= 4:
            risk_category = "🟢 Green (Model OK)"
        elif 5 <= num_exceptions <= 9:
            risk_category = "🟡 Yellow (Monitor Closely)"
        else:
            risk_category = "🔴 Red (Model Unreliable)"

        TrafficLight_VaR_results[symbol] = {
            "VaR (%)": round(VaR_historical * 100, 2),
            "Exceptions": num_exceptions,
            "Risk Category": risk_category
        }

    print("Traffic Light VaR: " + TrafficLight_VaR_results)
    return TrafficLight_VaR_results


def CalTrafficLightCVaR(state):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    quantile_level = 1 - confidence_level  # E.g., 0.05 for 95% confidence

    TrafficLight_CVaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change().dropna()  # Daily returns
        VaR_historical = np.percentile(df["Returns"], quantile_level * 100)  # Historical VaR

        # Compute CVaR: Average of all returns beyond VaR threshold
        CVaR_historical = df["Returns"][df["Returns"] < VaR_historical].mean()

        # Count extreme losses beyond CVaR
        df["Extreme_Losses"] = df["Returns"] < CVaR_historical
        num_extreme_losses = df["Extreme_Losses"].sum()

        # Determine the Traffic Light category based on extreme losses
        if num_extreme_losses <= 4:
            risk_category = "🟢 Green (Model OK)"
        elif 5 <= num_extreme_losses <= 9:
            risk_category = "🟡 Yellow (Monitor Closely)"
        else:
            risk_category = "🔴 Red (Model Unreliable)"

        TrafficLight_CVaR_results[symbol] = {
            "CVaR (%)": round(CVaR_historical * 100, 2),
            "Extreme Losses": num_extreme_losses,
            "Risk Category": risk_category
        }

    print("Traffic Light CVaR: " + TrafficLight_CVaR_results)
    return TrafficLight_CVaR_results


def CalKupiecVaR(state):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    failure_probability = 1 - confidence_level  # Expected failure rate (e.g., 5% for 95% confidence)

    Kupiec_VaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change().dropna()  # Calculate daily returns
        VaR_historical = np.percentile(df["Returns"], failure_probability * 100)  # Historical VaR

        # Count actual VaR breaches (exceptions)
        df["Exceptions"] = df["Returns"] < VaR_historical
        num_exceptions = df["Exceptions"].sum()
        num_obs = len(df)  # Total number of days

        # Compute Kupiec likelihood ratio (LR) test statistic
        if num_exceptions > 0 and num_exceptions < num_obs:
            prob_fail = num_exceptions / num_obs
            LR_POF = -2 * np.log(
                ((1 - failure_probability) ** (num_obs - num_exceptions) * failure_probability ** num_exceptions) /
                ((1 - prob_fail) ** (num_obs - num_exceptions) * prob_fail ** num_exceptions)
            )
        else:
            LR_POF = np.inf  # Invalid case, model is unreliable

        # Compute p-value from Chi-Square distribution
        p_value = 1 - stats.chi2.cdf(LR_POF, df=1)

        # Determine if model passes or fails
        if LR_POF < 3.84:  # Chi-square critical value at 95% confidence
            model_status = "✅ Model Passes (Reliable)"
        else:
            model_status = "❌ Model Fails (Unreliable)"

        Kupiec_VaR_results[symbol] = {
            "VaR (%)": round(VaR_historical * 100, 2),
            "Observed Breaches": num_exceptions,
            "Likelihood Ratio": round(LR_POF, 2),
            "p-value": round(p_value, 4),
            "Model Status": model_status
        }

    print("Kupiec VaR: " + Kupiec_VaR_results)
    return Kupiec_VaR_results


def CalKupiecCVaR(state):
    confidence_level = state.confidence_interval / 100  # Convert to decimal (e.g., 95 → 0.95)
    failure_probability = 1 - confidence_level  # Expected failure rate (e.g., 5% for 95% confidence)

    Kupiec_CVaR_results = {}

    for symbol, df in state.stock_data_dict.items():
        df["Returns"] = df["Close"].pct_change().dropna()  # Calculate daily returns
        VaR_historical = np.percentile(df["Returns"], failure_probability * 100)  # Historical VaR

        # Calculate CVaR (average of worst losses beyond VaR)
        worst_losses = df[df["Returns"] < VaR_historical]["Returns"]
        if not worst_losses.empty:
            CVaR_historical = worst_losses.mean()
        else:
            CVaR_historical = VaR_historical  # If no worst cases, fallback to VaR

        # Count actual CVaR breaches (exceptions)
        df["Exceptions"] = df["Returns"] < CVaR_historical
        num_exceptions = df["Exceptions"].sum()
        num_obs = len(df)  # Total number of days

        # Compute Kupiec likelihood ratio (LR) test statistic
        if num_exceptions > 0 and num_exceptions < num_obs:
            prob_fail = num_exceptions / num_obs
            LR_POF = -2 * np.log(
                ((1 - failure_probability) ** (num_obs - num_exceptions) * failure_probability ** num_exceptions) /
                ((1 - prob_fail) ** (num_obs - num_exceptions) * prob_fail ** num_exceptions)
            )
        else:
            LR_POF = np.inf  # Invalid case, model is unreliable

        # Compute p-value from Chi-Square distribution
        p_value = 1 - stats.chi2.cdf(LR_POF, df=1)

        # Determine if model passes or fails
        if LR_POF < 3.84:  # Chi-square critical value at 95% confidence
            model_status = "✅ Model Passes (Reliable)"
        else:
            model_status = "❌ Model Fails (Unreliable)"

        Kupiec_CVaR_results[symbol] = {
            "CVaR (%)": round(CVaR_historical * 100, 2),
            "Observed Breaches": num_exceptions,
            "Likelihood Ratio": round(LR_POF, 2),
            "p-value": round(p_value, 4),
            "Model Status": model_status
        }

    print("Kupiec CVaR: " + Kupiec_CVaR_results)
    return Kupiec_CVaR_results




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