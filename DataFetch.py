import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

# Directory to save parquet files
output_dir = "historical_data"
os.makedirs(output_dir, exist_ok=True)

# Symbol-to-Name mapping
symbol_to_name = {
    # Indexes
    "^BSESN": "Sensex",
    "^NSEI": "Nifty",
    "^NSEBANK": "NiftyBank",
    
    # Stocks
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "SBIN.NS": "State Bank of India",
    "ITC.NS": "ITC",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "HCLTECH.NS": "HCL Technologies",
    "LT.NS": "Larsen & Toubro",
    "BAJFINANCE.NS": "Bajaj Finance",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries",
    "M&M.NS": "Mahindra & Mahindra",
    "MARUTI.NS": "Maruti Suzuki India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "AXISBANK.NS": "Axis Bank",
    "NTPC.NS": "NTPC",
    "ONGC.NS": "Oil and Natural Gas Corporation",
    "WIPRO.NS": "Wipro",
    "TITAN.NS": "Titan Company",
    "ADANIENT.NS": "Adani Enterprises",
    "POWERGRID.NS": "Power Grid Corporation of India",
    "TATAMOTORS.NS": "Tata Motors",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "TRENT.NS": "Trent",
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "COALINDIA.NS": "Coal India",
    "ASIANPAINT.NS": "Asian Paints",
    "JSWSTEEL.NS": "JSW Steel",
    "NESTLEIND.NS": "Nestlé India",
    "BEL.NS": "Bharat Electronics",
    "TATASTEEL.NS": "Tata Steel",
    "TECHM.NS": "Tech Mahindra",
    "GRASIM.NS": "Grasim Industries",
    "EICHERMOT.NS": "Eicher Motors",
    "SBILIFE.NS": "SBI Life Insurance",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "HINDALCO.NS": "Hindalco Industries",
    "BPCL.NS": "Bharat Petroleum Corporation",
    "CIPLA.NS": "Cipla",
    "BRITANNIA.NS": "Britannia Industries",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "APOLLOHOSP.NS": "Apollo Hospitals Enterprise",
    "TATACONSUM.NS": "Tata Consumer Products",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "INDUSINDBK.NS": "IndusInd Bank"
}

# Function to fetch and save data
def fetch_and_save_data(symbol, name):
    print(f"Fetching data for {name} ({symbol})...")
    try:
        # Download data
        df = yf.download(symbol, period="max", interval="1d")
        if df.empty:
            print(f"No data for {name}")
            return

        # Reset index to have Date as a column
        df.reset_index(inplace=True)

        # Select and rename required columns
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        # Format the Date column
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        # Save to parquet file
        file_path = os.path.join(output_dir, f"{symbol.split('.')[0]}.parquet")
        # file_path = os.path.join(output_dir, f"{symbol.replace('.', '_')}.parquet")
        df.to_parquet(file_path, index=False)
        print(f"Data for {name} saved to {file_path}")

    except Exception as e:
        print(f"Failed to fetch data for {name}: {e}")

# Function to update existing data
def update_historical_data():
    today = datetime.now().strftime("%Y-%m-%d")

    for file in os.listdir(output_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(output_dir, file)
            symbol = file.replace(".parquet", "")

            print(f"Checking updates for {symbol}...")
            try:
                # Read existing data
                df_existing = pd.read_parquet(file_path)
                last_date = df_existing["Date"].max()

                if last_date < today:
                    # Download missing data
                    start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

                    
                    if symbol.startswith("^"):
                        full_symbol = symbol
                    elif not symbol.startswith("^"):
                        full_symbol = f"{symbol}.NS"
                    else:
                        print(f"Unknown symbol format for {symbol}")
                        continue

                    new_data = yf.download(full_symbol, start=start_date, end=today, interval="1d")

                    if not new_data.empty:
                        # Reset index and format data
                        new_data.reset_index(inplace=True)
                        new_data = new_data[["Date", "Open", "High", "Low", "Close", "Volume"]]
                        new_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
                        new_data["Date"] = new_data["Date"].dt.strftime("%Y-%m-%d")

                        # Check if the last_date matches the first date in new_data
                        if new_data["Date"].iloc[0] != last_date:
                            # Append new data and save
                            df_updated = pd.concat([df_existing, new_data], ignore_index=True)
                            df_updated.to_parquet(file_path, index=False)
                            print(f"Data for {symbol} updated up to {today}.")
                        else:
                            print(f"No new data to add for {symbol}, last date is the same as new data.")
                    else:
                        print(f"No new data for {symbol}.")
                else:
                    print(f"Data for {symbol} is already up-to-date.")

            except Exception as e:
                print(f"Failed to update data for {symbol}: {e}")

# Ensure top-level code runs only when executed directly
if __name__ == "__main__":
    # Fetch and save data for all symbols
    for symbol, name in symbol_to_name.items():
        fetch_and_save_data(symbol, name)

    print("Data fetching and update completed.")