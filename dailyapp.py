### daily analysis 03-07-25

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pytz import timezone
import pytz
from gtts import gTTS
import os
import time
from datetime import datetime, time
from time import sleep
from matplotlib.lines import Line2D
import pandas_market_calendars as mcal

#eastern = pytz.timezone("America/New")
eastern = pytz.timezone("US/Eastern")

# Function to calculate RSI
def calculate_rsi(data, window1=14, window2=25):
    # Calculate RSI with the first window (default 14)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window1).mean()
    
    # Avoid division by zero by adding a small epsilon to loss
    rs1 = gain / (loss + 1e-10)
    rsi1 = 100 - (100 / (1 + rs1))
    
    # Add RSI1 to the DataFrame
    data['RSI'] = rsi1
    
    # Calculate RSI with the second window (default 25)
    delta2 = data['Close'].diff(1)
    gain2 = (delta2.where(delta2 > 0, 0)).rolling(window=window2).mean()
    loss2 = (-delta2.where(delta2 < 0, 0)).rolling(window=window2).mean()
    
    # Avoid division by zero by adding a small epsilon to loss
    rs2 = gain2 / (loss2 + 1e-10)
    rsi2 = 100 - (100 / (1 + rs2))
    
    # Add RSI2 to the DataFrame
    data['RSI2'] = rsi2
    
    return data

def calculate_macd(data):
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to fetch stock data with a specified interval
def fetch_stock_data(ticker, interval="5m"):
    # Fetch data for the specified stock with the given interval (including premarket)
    stock = yf.Ticker(ticker)
    data = stock.history(period="5d", interval=interval, prepost=True)  # Include premarket data
    return data

# Function to fetch stock data with a specified 1h interval
def fetch_stock_data1mo(ticker, interval="1h"):
    # Fetch data for the specified stock with the given interval (including premarket)
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo", interval="1h", prepost=True)  # no doest not Include premarket data
    return data

# Function to fetch the previous 5 day's close price
def fetch_daily5(ticker):
    stock = yf.Ticker(ticker)
    daily5 = stock.history(period="3mo")  # Fetch 30 days of data
    if len(daily5) >= 2:
        return daily5['Close'][-5:]  ## return only last 5 days 
    else:
        return None  # Handle cases where there isn't enough data

# Function to fetch 3mo close price
def fetch_3mo(ticker):
    stock = yf.Ticker(ticker)
    daily3mo = stock.history(period="3mo")
    if len(daily3mo) >= 2:
        return daily3mo    
    else:
        return None  # Handle cases where there isn't enough data
    
# Function to fetch 6mo close price
def fetch_6mo(ticker):
    stock = yf.Ticker(ticker)
    daily6mo = stock.history(period="6mo")
    if len(daily6mo) >= 2:
        return daily6mo
    else:
        return None  # Handle cases where there isn't enough data

# Function to fetch the previous day's close price
def fetch_previous_close(ticker):
    close_prices = fetch_daily5(ticker)
    if close_prices is None:
        return None  # Handle cases where there isn't enough data
    
    # Get current time in NY (US Eastern Time)
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    # Define US market hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if now < market_open or now > market_close:  # Pre-market or post-market
        previous_close = close_prices[-1]  # Use the previous day's close
    else:
        previous_close = close_prices[-2]  # Use the most recent close

    return previous_close

# Function to fetch the day before yesterday's close price
def fetch_d2_close(ticker):
    close_prices = fetch_daily5(ticker)
    if close_prices is None:
        return None  # Handle cases where there isn't enough data
    
    # Get current time in NY (US Eastern Time)
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    # Define US market hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if now < market_open or now > market_close:  # Pre-market or post-market
        d2_close = close_prices[-2]  # Use the previous day's close
    else:
        d2_close = close_prices[-3]  # Use the most recent close

    return d2_close

# Function to perform regression analysis
def perform_regression(data, degree=1):
    # Prepare data (use only the most recent 300 points)
    data_recent = data.tail(300)  # Get the most recent 300 data points
    X = np.arange(len(data_recent)).reshape(-1, 1)  # Time as feature
    y = data_recent['Close'].values  # Closing prices as target

    # Polynomial regression
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    # Calculate R-squared
    r2 = r2_score(y, y_pred)

    return X, y, y_pred, r2, data_recent

# Function to calculate percentage change
def calculate_percentage_change(current_price, previous_close):
    return ((current_price - previous_close) / previous_close) * 100

# Function to calculate exponential moving averages
def calculate_emas(data):
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    return data
    
# Function to perform regression analysis and plot
def regression_analysis(data_recent, interval):

    # Ensure index is datetime
     
    data_recent.index = pd.to_datetime(data_recent.index)
    # Instead of using time in seconds, use a simple range index
    data_recent["TimeIndex"] = np.arange(len(data_recent))

    # Create Timestamp column from index
    data_recent["Timestamp"] = data_recent.index  # Use index instead of non-existing 'Timestamp' column

    # Extract the Hour and Minute
    data_recent["Hour"] = data_recent["Timestamp"].dt.strftime("%H:%M")  # Format as HH:MM

    X = data_recent[["TimeIndex"]].values  # Reshape needed for sklearn
    y = data_recent["score"].fillna(0).values

    # Polynomial Regression
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_pred_lin = lin_model.predict(X)  # Ensure this line is present
    r2_lin = r2_score(y, y_pred_lin)

    # Plot the actual total values
    plt.figure(figsize=(10, 5))
    ##assign color 
    color = "blue"

    plt.scatter(data_recent["TimeIndex"], data_recent["score"], color=color, label="Actual score")

    # Select every 10th label
    xticks = data_recent["TimeIndex"][::10]
    xlabels = data_recent["Hour"][::10]

    # Set x-ticks and labels
    plt.xticks(ticks=xticks, labels=xlabels, rotation=45)

    # Plot Linear Regression Line
    plt.plot(data_recent["TimeIndex"], y_pred_lin, color="gray", linestyle="solid", label=f"Linear ( R² = {r2_lin:.2f})")

    # Plot Polynomial Regression Line
    plt.plot(data_recent["TimeIndex"], y_pred_poly, color="blue", linestyle="dashed", label=f"Polynomial ( R² = {r2_poly:.2f})")

    # Add a horizontal line at set y_value
    current = y[-1]
    
    plt.axhline(y=current, color="gray", linestyle="--", label=f"current: {current: .2f}")
    plt.axhline(y=4, color="red", linestyle="--")
    plt.axhline(y=-4, color="green", linestyle="--")
    plt.axhline(y=0, color="gray", linestyle="-", linewidth = 3)
    
    
    # Labels and legend
    plt.xlabel("Time")
    plt.ylabel("Total Score")
    plt.legend()
    plt.title(f"T_score Regression ({interval})  || {datetime.now().strftime('%D:%H:%M')})")

    # Show plot in Streamlit
    st.pyplot(plt)

def plot_bars(price=0):
    # Assuming 'eastern' timezone is defined elsewhere
    eastern = 'US/Eastern'  # Example timezone, replace with actual timezone if needed

    # Read the data
    df = pd.read_csv("scoreT.csv", names=['tFrame', 'ema_trend', 'ema', 'rsi', 'macd', 'total', 'dev_from_std', "y_pred_p_trend", 'score_trend'])
    
    # Define custom order
    timeframe_order = ["1m", "5m", "15m", "30m", "1h", "3mo", "6mo"]
    
    # Convert 'tFrame' to categorical with order
    df["tFrame"] = pd.Categorical(df["tFrame"], categories=timeframe_order, ordered=True)
    df = df.sort_values("tFrame")
    
    # Prepare data
    unique_intervals = df["tFrame"].unique()
    x = np.arange(len(unique_intervals))
    width = 0.15
    
    # Calculate metric values
    ema_trend = [df[df["tFrame"] == interval]["ema_trend"].mean() for interval in unique_intervals]
    ema_values = [df[df["tFrame"] == interval]["ema"].mean() for interval in unique_intervals]
    rsi_values = [df[df["tFrame"] == interval]["rsi"].mean() for interval in unique_intervals]
    macd_values = [df[df["tFrame"] == interval]["macd"].mean() for interval in unique_intervals]
    total_values = [df[df["tFrame"] == interval]["total"].mean() for interval in unique_intervals]
    
    # Define bar positions
    offsets = [-2 * width, -width, 0, width, 2 * width]
    
    # Plot bars
    plt.figure(figsize=(12, 4),facecolor='lightgray')
    
    plt.bar(x + offsets[0], ema_trend, width, color="red", edgecolor="black")
    plt.bar(x + offsets[1], ema_values, width, color="darkred", edgecolor="black")
    plt.bar(x + offsets[2], rsi_values, width, color="navy", edgecolor="black")
    plt.bar(x + offsets[3], macd_values, width, color="orange", edgecolor="black", label="MACD")
    plt.bar(x + offsets[4], total_values, width, color="gray", edgecolor="black")
    
    # Add value labels
    for i, interval in enumerate(unique_intervals):
        for offset, values in zip(offsets, [ema_trend, ema_values, rsi_values, macd_values, total_values]):
            plt.text(x[i] + offset, values[i] + 0.2, f"{values[i]:.1f}", ha='center', fontsize=10)
    
    # Add threshold lines
    plt.axhline(y=4, color="red", linestyle="--", linewidth=1)
    plt.axhline(y=-4, color="green", linestyle="--", linewidth=1)
    plt.axhline(y=0, color="gray", linestyle="-", linewidth=1)
    
    #current_time = datetime.now(eastern).strftime('%m/%d/%Y %H:%M')
    plt.xlabel("Time Frame")
    plt.ylabel("Score")
    plt.title(f"Trend Scores by Interval (pr now: {price:.2f})")
    
    # Format x-axis
    plt.xticks(x, unique_intervals, rotation=45)
    
    # Add legend and adjust layout
    plt.legend()
    plt.tight_layout()

# Display the plot
    st.pyplot(plt)
    plt.close()  # Prevent memory leaks

if st.session_state.get("show_confirmation", False):
        st.success("Text saved successfully!")
        st.button("ClearInput", on_click=clear_text)
        st.session_state.show_confirmation = False

def clear_text():
    st.session_state["text_input"] = "zz"
        
# Streamlit app
def main():
    st.title("Score Regression Analysis")

    # Input box for user to enter stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., SPY, AAPL, TSLA):", value="SPY").upper()

    # Initialize session states
    if 'index' not in st.session_state:
        st.session_state.index = 0
        
    if 'rerun_count' not in st.session_state:
        st.session_state.rerun_count = 0
        
    if "stop_sleep" not in st.session_state:
        st.session_state.stop_sleep = 0

    if "temp_price" not in st.session_state:
        st.session_state.temp_price = 0

    if "sb_status" not in st.session_state:
        st.session_state.sb_status = 0

    if "sleepGap" not in st.session_state:
        st.session_state.sleepGap = 5

    if "setpr" not in st.session_state:
        st.session_state.setpr = 0
 
    if "setnote" not in st.session_state:
        st.session_state.setnote = "zz"

    if "poly_degree" not in st.session_state:
        st.session_state.poly_degree = 5
    
    scoreT_file = f"scoreT.csv"
    pe_file = f"pe.csv"

    # List of intervals
    intervals = ['1m', '5m', '15m', '30m', '1h', '3mo', '6mo']

    # Get the current interval
    interval = intervals[st.session_state.index]

    
    # Add a button group for interval selection
    # Fetch data for the user-specified stock and interval
    if interval == "1h":
        data = fetch_stock_data1mo(ticker, interval="1h")
    elif interval == "3mo":
        data = fetch_3mo(ticker)
    elif interval == "6mo":
        data = fetch_6mo(ticker)
    else:
        data = fetch_stock_data(ticker, interval=interval)

    if data.empty:
        st.error(f"Failed to fetch data for {ticker}. Please check the ticker and try again.")
        return

    #data_recent = data.tail(100 + selected_backtrack)  # Get the most recent 300 + selected_backtrack data points
    data_recent = data.tail(300)  # Use only the first 300 points after backtracking
    #data_recent = data_recent.head(100)  # Use only the first 300 points after backtracking
    columns_to_drop = ['Stock Splits', 'Capital Gains']
    data_recent = data_recent.drop(columns=columns_to_drop)

    # Calculate EMAs
    data_recent = calculate_emas(data_recent)
    
    # Get the current price (last available price in the data)
    current_price = data_recent['Close'].iloc[-1]

    # Fetch the previous day's close price
    previous_close = fetch_previous_close(ticker)
    if previous_close is None:
        st.error("Failed to fetch the previous day's close price. Please try again.")
        return

    change = current_price - previous_close

    # Calculate percentage change
    percentage_change = calculate_percentage_change(current_price, previous_close)

    # Get current local time
    eastern = pytz.timezone("US/Eastern")
    #current_time = datetime.now(eastern).strftime("%H:%M:%S")
    current_time = datetime.now(eastern).strftime("%I:%M:%S %p")

    # Display the percentage change message with current local time
    #st.write("### Current Price vs Previous Close___" f"{ticker}")
    if percentage_change >= 0:
        st.success(f"🟢 {ticker}:  **{current_price:.2f}**, **{change:.2f}**  (**{percentage_change:.2f}%**, previous_close **{previous_close:.2f}**)  |  **___** {current_time} **___**")
    else:
        st.error(f"🔴 {ticker}:  **{current_price:.2f}**, **{change:.2f}**  (**{percentage_change:.2f}%**, prev_close **{previous_close:.2f}**)  |  **......** {current_time}")

    ##############################
    
    degree = st.session_state.poly_degree
    
    # Perform linear regression (using only the most recent 300 points)
    X, y, y_pred_linear, r2_linear, data_recent = perform_regression(data_recent, degree=1)

    # Perform polynomial regression with the selected degree
    X, y, y_pred_poly, r2_poly, _ = perform_regression(data_recent, degree=degree)

    # Calculate residuals for each row
    data_recent["residuals"] = y - y_pred_poly
    # Option 1: Rolling standard deviation (e.g., over 50 time points)
    data_recent["std_dev"] = data_recent["residuals"].rolling(window=30).std()## first 50 time points no data

    std_dev = data_recent["std_dev"].iloc[-1]

    # Determine the trend message
    if current_price > data_recent['EMA_9'].iloc[-1] and data_recent['EMA_9'].iloc[-1] > data_recent['EMA_20'].iloc[-1]:
        trend_message = f"Trend UP"
        trend_color = "green"
    elif current_price < data_recent['EMA_9'].iloc[-1] and data_recent['EMA_9'].iloc[-1] < data_recent['EMA_20'].iloc[-1]:
        trend_message = f"Trend DOWN"
        trend_color = "red"
    else:
        trend_message = f"Trend NEUTRAL"
        trend_color = "gray"

    # Extract time (hours and minutes) for the x-axis
    time_labels = data_recent.index.strftime('%H:%M')  # Format time as HH:MM

    # Simplify x-axis labels based on the interval

    if interval == "15m":
        # For 15-minute interval, show only every 3 hours (e.g., 09:00, 12:00, 15:00)
        simplified_time_labels = [label if label.endswith('00') and int(label.split(':')[0]) % 3 == 0 else '' for label in time_labels]
    elif interval == "30m":
        # For 30-minute interval, show only every 3 hours (e.g., 09:00, 12:00, 15:00)
        simplified_time_labels = [label if label.endswith('00') and int(label.split(':')[0]) % 3 == 0 else '' for label in time_labels]
    elif interval == "1h":
        simplified_time_labels = [label if label.endswith('00') and int(label.split(':')[0]) % 8 == 0 else '' for label in time_labels]

    #3mo and 6mo data has only day information not hours and minute
    elif interval == "3mo":
        data3mo = fetch_3mo(ticker)
        time_labels = data3mo.index.strftime('%Y-%m-%d')  # Format to YYYY-MM-DD
        simplified_time_labels = [label if idx % 9 == 0 else '' for idx, label in enumerate(time_labels)]

    elif interval == "6mo":
        data6mo = fetch_6mo(ticker)
        time_labels = data6mo.index.strftime('%Y-%m-%d')  # Format to YYYY-MM-DD
        simplified_time_labels = [label if idx % 9 == 0 else '' for idx, label in enumerate(time_labels)]    

    else:
        # For 1-minute and 5-minute intervals, show only hours (e.g., 09:00, 10:00)
        simplified_time_labels = [label if label.endswith('00') else '' for label in time_labels]

    # Calculate the deviation of the current price from the polynomial regression model
    current_price_deviation = current_price - y_pred_poly[-1]  # Deviation from the polynomial model
    deviation_in_std = current_price_deviation / std_dev  # Deviation in terms of standard deviations

    # Add a message above the plot showing the price deviation
    if deviation_in_std >= 1:
        deviation_message = f"{ticker}_Deviation from PR: +{deviation_in_std:.2f} std_dev"
        deviation_color = "red"  # Red for >= +2 std_dev
    elif deviation_in_std <= -1:
        deviation_message = f"{ticker}_Deviation from PR: {deviation_in_std:.2f} std_dev"
        deviation_color = "green"  # Green for <= -2 std_dev
    else:
        deviation_message = f"{ticker}_Deviation from PR: {deviation_in_std:.2f} std_dev"
        deviation_color = "gray"  # Default color for other cases

    # Display the deviation message with the appropriate color
    st.markdown(f"<h3 style='color:{deviation_color};'>{deviation_message} ({interval})</h3>", unsafe_allow_html=True)

    # Add a message above the plot showing the trend
    st.markdown(f"<h3 style='color:{trend_color};'>{ticker}_{trend_message} ({interval})</h3>", unsafe_allow_html=True)
        
    # Calculate RSI before plotting
    data_recent = calculate_rsi(data_recent)
    data_recent = calculate_macd(data_recent)

    ## add p.r. model y value
    data_recent['y_pred_poly'] = y_pred_poly
    
    def calculate_scores(row):
    
        score = 0
        
        price = row['Close']
        ema9 = row['EMA_9']
        ema20 = row['EMA_20']
        ema50 = row['EMA_50']
        ema100 = row['EMA_100']
        ema200 = row['EMA_200']
        rsi = row['RSI']
        rsi2 = row['RSI2']
        macd = row['MACD']
        signal = row['Signal_Line']

        ema_trend = 0

        if (ema9 >= ema20):
            ema_trend = 1
        else:
            ema_trend = -1
        
        ema_score = 0

        if (ema100>= ema200):
            ema_score = 1
        else:
            ema_score = -1
            
        rsi_score = 0
        
        if (rsi >= rsi2) and (rsi >= 50):
            rsi_score = 2
        elif (rsi < rsi2) and (rsi >= 50):
            rsi_score = 1
        elif (rsi < rsi2) and (rsi < 50):
            rsi_score = -2
        elif (rsi >= rsi2) and (rsi < 50):
            rsi_score = -1
        else:
            rsi_score = 0
            
        macd_score = 0
        
        if (macd > signal) and (macd > 0):
            macd_score = 2
        elif (macd < signal) and (macd > 0):
            macd_score = 1
        elif (macd < signal) and (macd < 0):
            macd_score = - 2
        elif (macd > signal) and (macd < 0):
            macd_score = - 1
        else:
            macd_score = 0

        pred = row['y_pred_poly']
        
        pred_1 = data_recent["y_pred_poly"].shift(1).loc[row.name]

        pred_score = 0

        if (pred > pred_1):
            pred_score = 1
        elif (pred < pred_1):
            pred_score = - 1
        else :
            pred_score = 0
        
        std_dev = row['std_dev']

        deviation_in_std = (price - pred) / std_dev
        
        std_score = - deviation_in_std

        score = ema_trend + ema_score + rsi_score + macd_score 

        return pd.Series([ema_trend, ema_score, rsi_score, macd_score, score], 
                         index=["ema_trend", "ema_score", "rsi_score", "macd_score", "score"])

    # Apply function to DataFrame
    data_recent[["ema_trend", "ema_score", "rsi_score", "macd_score", "score"]] = data_recent.apply(calculate_scores, axis=1)

    # Define a list of timeframes that support MACD
    valid_macd_timeframes = ["1m","5m","15m","30m","1h", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

    #if interval in valid_macd_timeframes:
    
    ### get scores functions
    def get_scores():
        ema_score = data_recent["ema_score"].iloc[-1]
        ema_trend = data_recent["ema_trend"].iloc[-1]
        rsi_score = data_recent["rsi_score"].iloc[-1]
        macd_score = data_recent["macd_score"].iloc[-1]
        score = data_recent["score"].iloc[-1]
        
        std_dev = data_recent["std_dev"].iloc[-1]
        y_pred_poly = data_recent["y_pred_poly"].iloc[-1]
        delta = current_price - y_pred_poly
        dev_from_std = round(delta/std_dev,0)

        return ema_score, ema_trend, rsi_score, macd_score, score, dev_from_std

    def get_scores_more():
        price = data_recent['Close'].iloc[-1]
        
        ema9= data_recent['EMA_9'].iloc[-1]
        ema20= data_recent['EMA_20'].iloc[-1]
        ema50= data_recent['EMA_50'].iloc[-1]
        ema100= data_recent['EMA_100'].iloc[-1]
        ema200= data_recent['EMA_200'].iloc[-1]
        rsi = data_recent['RSI'].iloc[-1]
        rsi2 = data_recent['RSI2'].iloc[-1]
        macd = data_recent['MACD'].iloc[-1]
        signal = data_recent['Signal_Line'].iloc[-1]
        y_pred_poly = data_recent['y_pred_poly'].iloc[-1]
        y_pred_poly1 = data_recent['y_pred_poly'].iloc[-2]

        return price, ema9, ema20, ema50, ema100, ema200, rsi, rsi2, macd, signal, y_pred_poly, y_pred_poly1

    #get all scores:
    ema_score, ema_trend, rsi_score, macd_score, score, dev_from_std = get_scores()
    price, ema9, ema20, ema50, ema100, ema200, rsi, rsi2, macd, signal, y_pred_poly, y_pred_poly1 = get_scores_more()

    # File path
    file_path = 'scoreT.csv'

    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=None)

        # Check if any row contains {interval} in the first column
        if interval in df[0].values:
            # Remove the row where the first column is {interval}
            df = df[df[0] != interval]

            # Save the updated DataFrame back to the CSV file
            df.to_csv(file_path, index=False, header=False)
        else:
            print("No row with '{interval' found. File remains unchanged.")
    else:
        # If the file doesn't exist or is empty, create a new DataFrame
        print("File does not exist or is empty. Creating a new file.")

        df = pd.DataFrame(columns=['tFrame', 'ema_trend', 'ema', 'rsi','macd', 'score', 'dev_from_std', 'score_trend'])

        # Save the empty DataFrame to the CSV file
        df.to_csv(file_path, index=False, header=False)
        st.success(f"✅ File created successfully as `{file_path}`")

    #ema_score, ema_trend, rsi_score, macd_score, score
    y_pred_p_trend = 0
    if y_pred_poly >= y_pred_poly1:
        y_pred_p_trend = 1
        #y_pred_p_trend = "↑"
    else:
       # y_pred_p_trend = "↓"
        y_pred_p_trend = -1
        
    if ema_score >= 0 and rsi_score >= 0 and macd_score >= 0 and ema_trend >= 0:
        score_trend = 1
    elif ema_score <0 and rsi_score < 0  and macd_score < 0 and ema_trend < 0:
        score_trend = -1
    else:
        score_trend = 0
        
    # total == score (above) 
    new_data = pd.DataFrame([{
        "tFrame": f"{interval}",
        "ema9/20": round(ema_trend, 2),
        "ema100/200": round(ema_score, 2),
        "rsi": round(rsi_score, 2),
        "macd": round(macd_score, 2),
        "score": round(score, 2),
        "dev_from_std": deviation_in_std,
        "y_pred_p_trend": y_pred_p_trend,
        "score_trend": score_trend,
    }])
    #new_data.to_csv(scoreT_file, mode="a", header=False, index=False)
    new_data.to_csv(scoreT_file, mode="a", header=False, index=False, float_format="%.2f") ## chatGPT

    # Read the updated CSV file
    df = pd.read_csv(file_path, header=None)

    # Define the custom order for the first column
    custom_order = ["1m", "5m", "15m", "30m", "1h", "3mo", "6mo"]

    # Convert the first column to a categorical type with the custom order
    df[0] = pd.Categorical(df[0], categories=custom_order, ordered=True)

    # Sort the DataFrame by the first column using the custom order
    df = df.sort_values(by=0)

    #add column names
    df.columns = ['tFrame', 'ema9/20', 'ema100/200', 'rsi', 'macd', 'score', 'dev_from_std', "y_pred_p_trend", 'score_trend']
        
    #display table
    st.dataframe(df, hide_index=True) #original table looks neater

    plot_bars(current_price)

    ################### all control buttons ###########################################################
    current_price = round(data_recent['Close'].iloc[-1], 2)
    now = datetime.now(eastern).strftime('%m-%d %I:%M:%S %p')  # Correct format
  #  ema_trend_1m = df[df["tFrame"] == "1m"]["ema_trend"].values[0]
 #   ema_trend_5m = df[df["tFrame"] == "5m"]["ema_trend"].values[0]
    pr1=df[df["tFrame"] == "1m"]["y_pred_p_trend"].values[0]
    pr5=df[df["tFrame"] == "5m"]["y_pred_p_trend"].values[0]
    dev1=df[df["tFrame"] == "1m"]["dev_from_std"].values[0]
    dev5=df[df["tFrame"] == "5m"]["dev_from_std"].values[0]
    
    st.write(f"interval: {interval}__rerun:{ st.session_state.rerun_count}")
    # Extract "score_trend" for "1m"  ## 
    
    if pr1 ==1:
        message = "___B OK 1"
        color = "green"
    elif pr1==-1:
        message = "___S OK -1"
        color = "red"
    else:
        message = "Hold it"
        color = "orange"
    st.markdown(f'<p style="color:{color}; font-weight:bold;">polynomial 1min: {message}</s></p>', unsafe_allow_html=True)
    if pr5 ==1:
        message = "___B OK 1"
        color = "green"
    elif pr5==-1:
        message = "___S OK -1"
        color = "red"
    else:
        message = "Hold it"
        color = "orange"
    st.markdown(f'<p style="color:{color}; font-weight:bold;">polynomial 5min: {message}</s></p>', unsafe_allow_html=True)

    sum_score_trend_rest = df[~df["tFrame"].isin(["1m", "6mo"])]["score_trend"].sum()
    
    if sum_score_trend_rest >=4:
        message = "___B OK >=4"
        color = "green"
    elif sum_score_trend_rest <= -4:
        message = "___S OK <=-4"
        color = "red"
    else:
        message = "Hold it"
        color = "orange"
    #st.write(f"score_trend_others: ||... {sum_score_trend_rest} ___ {message}")
    st.markdown(f'<p style="color:{color}; font-weight:bold;">score_trend_others: {message}__{sum_score_trend_rest}</s></p>', unsafe_allow_html=True)

    #display message about app status
    sleep_status = 'on' if st.session_state.stop_sleep == 0 else "off"
    updated_data = pd.read_csv(pe_file, names=["type", "B_pr", "S_pr", "pl", "total", "temp_pr", "scoreTrendRest","note"])

    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # delete data button
        if st.button("1min"):
            #st.session_state.rerun_count = 0
            st.session_state.index = 0
           # st.session_state.stop_sleep = 0
            #st.session_state.sbOK = 1
            st.session_state.sleepGap = 6 # !=5: will keep 1min
            st. rerun()

    with col2:
        # delete data button
        if st.button("1m, 5m"):
            #st.session_state.rerun_count = 0
            
            st.session_state.index = 0
            #st.session_state.sbOK = 1
            st.session_state.sleepGap = 7 # !=5 or 6 will keep 5min
            st. rerun()
            
    with col3:
        if st.button("check all"):
            
            st.session_state.index = 0
            st.session_state.rerun_count = 0
            st.session_state.sleepGap = 5
            st.rerun()

    with col4:
        if st.button("stop slp"):
            st.session_state.stop_sleep =1
            st.rerun()

    st.write(f"slp: {st.session_state.sleepGap}_stop:{st.session_state.stop_sleep}")
            
    st.write("---------------------")
    
########################################
    if st.session_state.stop_sleep == 0:
        # Sleep for 8 seconds (simulating some processing)
        sleep(st.session_state.sleepGap)
        if st.session_state.sleepGap == 5:
            # Update the index for the next interval
            if st.session_state.index < len(intervals) - 1:
                st.session_state.index += 1
            else:
                st.session_state.index = 0
            st.session_state.rerun_count +=1
        elif st.session_state.sleepGap == 7:
            if st.session_state.index ==0:
                st.session_state.index = 1
            else:
                st.session_state.index = 0
        elif st.session_state.sleepGap == 6:
            st.session_state.index = 0

        st.rerun()

    fig, ax = plt.subplots()
    ax.set_facecolor("lightgray")  # Set background color to light gray
    ax.plot([1, 2, 3], [4, 5, 2])
    st.pyplot ()
    
        

if __name__ == "__main__":
    main()
